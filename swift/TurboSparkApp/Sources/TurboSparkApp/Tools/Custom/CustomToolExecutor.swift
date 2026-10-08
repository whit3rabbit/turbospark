import Foundation

/// Executor responsible for running user-defined custom tools.
public enum CustomToolExecutor {
    /// The scrubbed baseline a custom tool starts from. The child used to get
    /// only the tool's own `environment`, i.e. no PATH or LANG, so "npm test"
    /// or "brew list" failed with "command not found" while the same command
    /// worked through run_command. Still an allowlist: no secrets inherited.
    static func baseEnvironment(parent: [String: String] = ProcessInfo.processInfo.environment) -> [String: String] {
        var env: [String: String] = [:]
        for key in ["HOME", "LANG", "TMPDIR"] {
            if let value = parent[key] { env[key] = value }
        }
        // A Finder launch inherits only the system PATH; add the Homebrew
        // locations a developer's tools live in.
        var entries: [String] = []
        for dir in (parent["PATH"] ?? "/usr/bin:/bin:/usr/sbin:/sbin").components(separatedBy: ":")
            + ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin"]
        where !dir.isEmpty && !entries.contains(dir) {
            entries.append(dir)
        }
        env["PATH"] = entries.joined(separator: ":")
        env["LANG"] = env["LANG"].flatMap { $0.isEmpty ? nil : $0 } ?? "en_US.UTF-8"
        return env
    }

    /// Executes a custom tool call with argument interpolation and process isolation.
    public static func execute(
        tool: CustomToolDefinition,
        arguments rawArguments: [String: String],
        projectRootURL: URL
    ) async throws -> String {
        // Only declared parameters take part in substitution and the
        // environment. An undeclared model-supplied key (say "home") would
        // otherwise turn a template's `$HOME` into `${TOOL_ARG_HOME}` and let
        // the model choose where the tool writes.
        let declared = Set(tool.parameters.properties?.keys.map { $0 } ?? [])
        let arguments = rawArguments.filter { declared.contains($0.key) }
        switch tool.execution.type {
        case .command:
            guard var cmd = tool.execution.command else {
                throw NSError(domain: "TurboSparkTool", code: 30, userInfo: [NSLocalizedDescriptionKey: "Command string is not configured for '\(tool.name)'."])
            }
            // Argument values are model-controlled. Keep their bytes out of
            // the shell program and expand only environment variables at the
            // placeholder sites. The renderer preserves the template's quote
            // context without ever putting a value into `zsh -c` source.
            cmd = substituteArguments(into: cmd, arguments: arguments)
            var environment = baseEnvironment()
            for (key, val) in tool.execution.environment ?? [:] { environment[key] = val }
            for (key, val) in arguments {
                environment["TOOL_ARG_\(environmentKey(for: key))"] = val
            }
            let timeout = tool.execution.timeoutSeconds ?? ProcessExecutor.defaultTimeoutSeconds
            let result = try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: "/bin/zsh"),
                arguments: ["-c", cmd],
                currentDirectoryURL: projectRootURL,
                environment: environment,
                timeoutSeconds: timeout
            )
            let combined = [result.stdout, result.stderr].filter { !$0.isEmpty }.joined(separator: "\n")
            try failIfUnsuccessful(result, toolName: tool.name, output: combined, timeout: timeout)
            if combined.isEmpty {
                return "(Command finished with exit code \(result.exitCode))"
            }
            return AppToolRegistry.compactOutput(combined)

        case .script:
            guard let script = tool.execution.scriptContent else {
                throw NSError(domain: "TurboSparkTool", code: 31, userInfo: [NSLocalizedDescriptionKey: "Script content is not configured for '\(tool.name)'."])
            }
            let tempDir = FileManager.default.temporaryDirectory
            let scriptFile = tempDir.appendingPathComponent("custom_\(tool.name)_\(UUID().uuidString.prefix(8)).sh")
            try script.write(to: scriptFile, atomically: true, encoding: .utf8)
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: scriptFile.path)
            defer { try? FileManager.default.removeItem(at: scriptFile) }

            var args: [String] = []
            if let customArgs = tool.execution.arguments {
                for arg in customArgs {
                    var rendered = arg
                    for (k, v) in arguments {
                        rendered = rendered.replacingOccurrences(of: "{{\(k)}}", with: v)
                        rendered = rendered.replacingOccurrences(of: "${\(k)}", with: v)
                    }
                    args.append(rendered)
                }
            }

            let interpreter = tool.execution.scriptInterpreter ?? "/bin/zsh"
            let scriptTimeout = tool.execution.timeoutSeconds ?? 30.0
            let result = try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: interpreter),
                arguments: [scriptFile.path] + args,
                currentDirectoryURL: projectRootURL,
                environment: baseEnvironment().merging(tool.execution.environment ?? [:]) { _, declared in declared },
                timeoutSeconds: scriptTimeout
            )
            let combined = [result.stdout, result.stderr].filter { !$0.isEmpty }.joined(separator: "\n")
            try failIfUnsuccessful(result, toolName: tool.name, output: combined, timeout: scriptTimeout)
            if combined.isEmpty {
                return "(Script executed with exit code \(result.exitCode))"
            }
            return AppToolRegistry.compactOutput(combined)

        case .http:
            guard let urlStr = tool.execution.httpURL, let url = URL(string: urlStr) else {
                throw NSError(domain: "TurboSparkTool", code: 32, userInfo: [NSLocalizedDescriptionKey: "Invalid HTTP endpoint configured for '\(tool.name)'."])
            }
            var req = URLRequest(url: url)
            req.httpMethod = tool.execution.httpMethod ?? "POST"
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
            if let headers = tool.execution.httpHeaders {
                for (k, v) in headers { req.setValue(v, forHTTPHeaderField: k) }
            }
            req.httpBody = try JSONSerialization.data(withJSONObject: arguments, options: [])
            // URLSession's defaults are 60 s idle and 7 days overall, so a
            // slow-drip endpoint held the turn; honor the tool's own limit.
            req.timeoutInterval = tool.execution.timeoutSeconds ?? 30.0
            let (data, response) = try await URLSession.shared.data(for: req)
            // A multi-megabyte body would otherwise land whole in the model context.
            let body = AppToolRegistry.compactOutput(String(decoding: data, as: UTF8.self))
            if let http = response as? HTTPURLResponse {
                return "HTTP \(http.statusCode):\n\(body)"
            }
            return body
        }
    }

    /// A non-zero exit or a timeout is a failure the model must see as one;
    /// returning partial output as a normal result read as success.
    private static func failIfUnsuccessful(
        _ result: ProcessExecutor.Output, toolName: String, output: String, timeout: TimeInterval
    ) throws {
        guard result.timedOut || result.exitCode != 0 else { return }
        let headline = result.timedOut
            ? "Custom tool '\(toolName)' timed out after \(Int(timeout))s and was terminated"
            : "Custom tool '\(toolName)' exited with status \(result.exitCode)"
        let detail = output.isEmpty ? "" : ":\n\(AppToolRegistry.compactOutput(output))"
        throw NSError(
            domain: "TurboSparkTool", code: 33,
            userInfo: [NSLocalizedDescriptionKey: headline + detail])
    }

    /// Rewrites `{{key}}`, `${key}` and `$KEY` in a shell command template to
    /// references to the corresponding `TOOL_ARG_<KEY>` environment variable.
    ///
    /// **ONE PASS, NOT ONE PASS PER ARGUMENT.** Repeated
    /// `replacingOccurrences` re-scans text a previous argument's value
    /// already produced, so a value containing `${other}` was itself
    /// substituted into -- which is the same class of bug as the quoting
    /// this fixes, one level of indirection out. A marker naming no argument
    /// is left exactly as it was, so a template referring to a real
    /// environment variable still works.
    static func substituteArguments(into template: String, arguments: [String: String]) -> String {
        let pattern = "\\{\\{([A-Za-z0-9_]+)\\}\\}|\\$\\{([A-Za-z0-9_]+)\\}|\\$([A-Z_][A-Z0-9_]*)"
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return template }
        let ns = template as NSString
        let matches = regex.matches(
            in: template, range: NSRange(location: 0, length: ns.length))
        var result = ""
        var cursor = 0
        var quote: Character?
        for match in matches {
            let prefix = ns.substring(with: NSRange(
                location: cursor, length: match.range.location - cursor))
            result += prefix
            updateQuoteState(with: prefix, quote: &quote)
            cursor = match.range.location + match.range.length
            var argumentName: String?
            if match.range(at: 1).location != NSNotFound {
                argumentName = ns.substring(with: match.range(at: 1))
            } else if match.range(at: 2).location != NSNotFound {
                argumentName = ns.substring(with: match.range(at: 2))
            } else if match.range(at: 3).location != NSNotFound {
                // `$KEY` matches an argument whose name uppercases to it,
                // which is the rule the per-argument loop this replaced used.
                let upper = ns.substring(with: match.range(at: 3))
                argumentName = arguments.keys.first { $0.uppercased() == upper }
            }
            if let name = argumentName, arguments[name] != nil {
                let reference = "${TOOL_ARG_\(environmentKey(for: name))}"
                switch quote {
                case "'":
                    result += "'\"\(reference)\"'"
                case "\"":
                    result += reference
                default:
                    result += "\"\(reference)\""
                }
            } else {
                result += ns.substring(with: match.range)
            }
        }
        result += ns.substring(from: cursor)
        return result
    }

    private static func updateQuoteState(with text: String, quote: inout Character?) {
        var escaped = false
        for character in text {
            if escaped {
                escaped = false
            } else if character == "\\" && quote != "'" {
                escaped = true
            } else if character == "'" && quote != "\"" {
                quote = quote == "'" ? nil : "'"
            } else if character == "\"" && quote != "'" {
                quote = quote == "\"" ? nil : "\""
            }
        }
    }

    /// An argument name as an environment-variable suffix: uppercased, with
    /// anything that is not `A-Z`, `0-9` or `_` replaced, since `setenv`
    /// takes no other characters.
    static func environmentKey(for name: String) -> String {
        String(
            name.uppercased().map { ch in
                (ch.isASCII && (ch.isLetter || ch.isNumber)) || ch == "_" ? ch : "_"
            })
    }
}
