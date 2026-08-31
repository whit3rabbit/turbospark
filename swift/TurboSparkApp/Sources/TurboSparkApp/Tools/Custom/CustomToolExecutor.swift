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
            // Template substitution for {{key}}, ${key}, and $KEY
            for (key, val) in arguments {
                cmd = cmd.replacingOccurrences(of: "{{\(key)}}", with: val)
                cmd = cmd.replacingOccurrences(of: "${\(key)}", with: val)
                cmd = cmd.replacingOccurrences(of: "$\(key.uppercased())", with: val)
            }
            let timeout = tool.execution.timeoutSeconds ?? ProcessExecutor.defaultTimeoutSeconds
            let result = try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: "/bin/zsh"),
                arguments: ["-c", cmd],
                currentDirectoryURL: projectRootURL,
                environment: tool.execution.environment ?? [:],
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
}
