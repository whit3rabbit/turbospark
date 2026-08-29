import Foundation

extension AppHookExecutionEngine {
    func executeCommandHook(
        hook: AppHookCommand,
        event: AppHookEvent,
        sessionID: String,
        toolName: String?,
        toolArguments: [String: String]?,
        toolOutput: String?,
        toolDurationSeconds: Double?,
        isError: Bool?,
        workingDirectory: String?,
        startTime: Date
    ) async -> AppHookExecutionResult {
        let payloadDict: [String: Any] = [
            "event": event.rawValue,
            "session_id": sessionID,
            "tool_name": toolName ?? "",
            "tool_input": toolArguments ?? [:],
            "tool_output": toolOutput ?? "",
            "tool_duration_seconds": toolDurationSeconds ?? 0.0,
            "is_error": isError ?? false,
            "cwd": workingDirectory ?? FileManager.default.currentDirectoryPath,
            "timestamp": ISO8601DateFormatter().string(from: Date())
        ]

        guard let payloadData = try? JSONSerialization.data(withJSONObject: payloadDict, options: []) else {
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 1,
                stdout: "",
                stderr: "Failed to serialize hook input payload",
                durationSeconds: 0.0
            )
        }

        // Variable substitution (${user_config.KEY})
        let store = await AppHookStore.shared
        let allOptions = await store.optionValues
        var commandText = hook.command
        let sourceID = hook.pluginName != nil ? "plugin_\(hook.pluginName!)" : (hook.sourceType == .projectConfig ? "project_config" : "user_config")
        let options = allOptions[sourceID] ?? [:]

        for (k, v) in options {
            commandText = commandText.replacingOccurrences(of: "${user_config.\(k)}", with: v)
        }

        // Run process via Process
        let process = Process()
        let pipeOut = Pipe()
        let pipeErr = Pipe()
        let pipeIn = Pipe()

        switch hook.shell {
        case .bash:
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = ["-c", commandText]
        case .zsh:
            process.executableURL = URL(fileURLWithPath: "/bin/zsh")
            process.arguments = ["-c", commandText]
        case .sh:
            process.executableURL = URL(fileURLWithPath: "/bin/sh")
            process.arguments = ["-c", commandText]
        case .pwsh:
            process.executableURL = URL(fileURLWithPath: "/usr/local/bin/pwsh")
            process.arguments = ["-Command", commandText]
        }

        var env = ProcessInfo.processInfo.environment
        env["TURBOSPARK_HOOK_EVENT"] = event.rawValue
        env["TURBOSPARK_SESSION_ID"] = sessionID
        if let toolName { env["TURBOSPARK_TOOL_NAME"] = toolName }
        if let workingDirectory { env["TURBOSPARK_PROJECT_DIR"] = workingDirectory }

        for (k, v) in options {
            let normalized = k.uppercased().replacingOccurrences(of: "-", with: "_")
            env["TURBOSPARK_OPTION_\(normalized)"] = v
            env["CLAUDE_PLUGIN_OPTION_\(normalized)"] = v
        }

        process.environment = env
        if let workingDirectory, !workingDirectory.isEmpty {
            process.currentDirectoryURL = URL(fileURLWithPath: workingDirectory)
        }

        process.standardInput = pipeIn
        process.standardOutput = pipeOut
        process.standardError = pipeErr

        do {
            try process.run()
            try? pipeIn.fileHandleForWriting.write(contentsOf: payloadData)
            try? pipeIn.fileHandleForWriting.close()

            // Handle timeout
            let timeout = hook.timeoutSeconds > 0 ? hook.timeoutSeconds : 30.0
            let deadline = Date().addingTimeInterval(timeout)

            while process.isRunning && Date() < deadline {
                try? await Task.sleep(nanoseconds: 20_000_000)
            }

            if process.isRunning {
                process.terminate()
                return AppHookExecutionResult(
                    hookID: hook.id,
                    hookName: hook.name,
                    event: event,
                    exitCode: 124, // Timeout exit code
                    stdout: "",
                    stderr: "Hook execution timed out after \(timeout) seconds.",
                    durationSeconds: Date().timeIntervalSince(startTime)
                )
            }

            let outData = pipeOut.fileHandleForReading.readDataToEndOfFile()
            let errData = pipeErr.fileHandleForReading.readDataToEndOfFile()

            let stdout = String(data: outData, encoding: .utf8) ?? ""
            let stderr = String(data: errData, encoding: .utf8) ?? ""
            let exitCode = process.terminationStatus
            let duration = Date().timeIntervalSince(startTime)

            var parsedDecision: AppHookPreToolUseDecision? = nil
            if event == .preToolUse {
                parsedDecision = parsePreToolUseOutput(stdout: stdout, stderr: stderr, exitCode: exitCode, hookName: hook.name)
            }

            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: exitCode,
                stdout: stdout,
                stderr: stderr,
                durationSeconds: duration,
                decision: parsedDecision
            )
        } catch {
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 1,
                stdout: "",
                stderr: error.localizedDescription,
                durationSeconds: Date().timeIntervalSince(startTime)
            )
        }
    }

    func executeHttpHook(
        hook: AppHookCommand,
        event: AppHookEvent,
        sessionID: String,
        toolName: String?,
        toolArguments: [String: String]?,
        toolOutput: String?,
        startTime: Date
    ) async -> AppHookExecutionResult {
        guard let url = URL(string: hook.command) else {
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 1,
                stdout: "",
                stderr: "Invalid webhook URL: \(hook.command)",
                durationSeconds: 0.0
            )
        }

        var request = URLRequest(url: url)
        request.httpMethod = "POST"
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")

        let payloadDict: [String: Any] = [
            "event": event.rawValue,
            "session_id": sessionID,
            "tool_name": toolName ?? "",
            "tool_input": toolArguments ?? [:],
            "tool_output": toolOutput ?? "",
            "timestamp": ISO8601DateFormatter().string(from: Date())
        ]

        request.httpBody = try? JSONSerialization.data(withJSONObject: payloadDict, options: [])

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let httpResponse = response as? HTTPURLResponse
            let statusCode = Int32(httpResponse?.statusCode ?? 200)
            let stdout = String(data: data, encoding: .utf8) ?? ""
            let isSuccess = statusCode >= 200 && statusCode < 300

            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: isSuccess ? 0 : 1,
                stdout: stdout,
                stderr: isSuccess ? "" : "HTTP status \(statusCode)",
                durationSeconds: Date().timeIntervalSince(startTime)
            )
        } catch {
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 1,
                stdout: "",
                stderr: error.localizedDescription,
                durationSeconds: Date().timeIntervalSince(startTime)
            )
        }
    }
}
