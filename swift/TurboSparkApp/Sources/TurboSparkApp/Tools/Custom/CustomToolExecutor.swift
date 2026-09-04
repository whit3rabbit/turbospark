import Foundation

/// Executor responsible for running user-defined custom tools.
public enum CustomToolExecutor {
    /// Executes a custom tool call with argument interpolation and process isolation.
    public static func execute(
        tool: CustomToolDefinition,
        arguments: [String: String],
        projectRootURL: URL
    ) async throws -> String {
        switch tool.execution.type {
        case .command:
            guard var cmd = tool.execution.command else {
                throw NSError(domain: "TurboSparkTool", code: 30, userInfo: [NSLocalizedDescriptionKey: "Command string is not configured for '\(tool.name)'."])
            }
            // **AN ARGUMENT VALUE IS MODEL-CONTROLLED TEXT GOING INTO A
            // `/bin/zsh -c` STRING** (state#70). It was spliced in raw, so a
            // `{{path}}` of `x; rm -rf ~` ran two commands where the tool's
            // author wrote one -- the hazard state#60 fixed for hook option
            // values, in a second place nobody swept. Two changes, and the
            // first is the one that matters: every value reaches the template
            // as a single-quoted literal, which zsh does not expand,
            // word-split or re-parse. The environment copy is for a tool
            // author who wants the raw bytes with no quoting question at all.
            cmd = substituteArguments(into: cmd, arguments: arguments)
            var environment = tool.execution.environment ?? [:]
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
            let result = try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: interpreter),
                arguments: [scriptFile.path] + args,
                currentDirectoryURL: projectRootURL,
                environment: tool.execution.environment ?? [:],
                timeoutSeconds: tool.execution.timeoutSeconds ?? 30.0
            )
            let combined = [result.stdout, result.stderr].filter { !$0.isEmpty }.joined(separator: "\n")
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
            let (data, response) = try await URLSession.shared.data(for: req)
            let body = String(decoding: data, as: UTF8.self)
            if let http = response as? HTTPURLResponse {
                return "HTTP \(http.statusCode):\n\(body)"
            }
            return body
        }
    }

    /// Substitutes `{{key}}`, `${key}` and `$KEY` in a SHELL command template
    /// with single-quoted literals (state#70).
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
        for match in matches {
            result += ns.substring(with: NSRange(location: cursor, length: match.range.location - cursor))
            cursor = match.range.location + match.range.length
            var replacement: String?
            if match.range(at: 1).location != NSNotFound {
                replacement = arguments[ns.substring(with: match.range(at: 1))]
            } else if match.range(at: 2).location != NSNotFound {
                replacement = arguments[ns.substring(with: match.range(at: 2))]
            } else if match.range(at: 3).location != NSNotFound {
                // `$KEY` matches an argument whose name uppercases to it,
                // which is the rule the per-argument loop this replaced used.
                let upper = ns.substring(with: match.range(at: 3))
                replacement = arguments.first { $0.key.uppercased() == upper }?.value
            }
            result += replacement.map(shellQuoted) ?? ns.substring(with: match.range)
        }
        result += ns.substring(from: cursor)
        return result
    }

    /// A value as one single-quoted shell word.
    ///
    /// Single quotes are the only zsh quoting in which NOTHING is special;
    /// the closing/escaping dance around an embedded `'` is the standard
    /// `'\''` idiom.
    static func shellQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\\''") + "'"
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
