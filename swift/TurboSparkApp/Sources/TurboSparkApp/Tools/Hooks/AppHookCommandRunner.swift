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

        let executableURL: URL
        let arguments: [String]
        switch hook.shell {
        case .bash:
            executableURL = URL(fileURLWithPath: "/bin/bash")
            arguments = ["-c", commandText]
        case .zsh:
            executableURL = URL(fileURLWithPath: "/bin/zsh")
            arguments = ["-c", commandText]
        case .sh:
            executableURL = URL(fileURLWithPath: "/bin/sh")
            arguments = ["-c", commandText]
        case .pwsh:
            executableURL = URL(fileURLWithPath: "/usr/local/bin/pwsh")
            arguments = ["-Command", commandText]
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

        let timeout = hook.timeoutSeconds > 0 ? hook.timeoutSeconds : 30.0
        let workingDirectoryURL = (workingDirectory?.isEmpty == false) ? URL(fileURLWithPath: workingDirectory!) : nil

        do {
            // `ProcessExecutor` writes stdin off this call's thread and drains
            // stdout/stderr concurrently with the timeout wait. The previous
            // synchronous `pipeIn.write(contentsOf:)` embedded the tool's own
            // output in the hook payload and could block on it alone, before
            // the timeout loop even started; and reading stdout/stderr only
            // after `waitUntilExit()` meant a hook that wrote more than one
            // pipe buffer to stdout hung until the timeout fired and was then
            // misreported as "timed out" rather than as blocked on IO.
            let result = try await ProcessExecutor.run(
                executableURL: executableURL,
                arguments: arguments,
                currentDirectoryURL: workingDirectoryURL,
                environment: env,
                stdin: payloadData,
                timeoutSeconds: timeout
            )

            if result.timedOut {
                return AppHookExecutionResult(
                    hookID: hook.id,
                    hookName: hook.name,
                    event: event,
                    exitCode: 124, // Timeout exit code
                    stdout: result.stdout,
                    stderr: "Hook execution timed out after \(timeout) seconds.",
                    durationSeconds: Date().timeIntervalSince(startTime)
                )
            }

            let duration = Date().timeIntervalSince(startTime)
            var parsedDecision: AppHookPreToolUseDecision? = nil
            if event == .preToolUse {
                parsedDecision = parsePreToolUseOutput(stdout: result.stdout, stderr: result.stderr, exitCode: result.exitCode, hookName: hook.name)
            }

            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: result.exitCode,
                stdout: result.stdout,
                stderr: result.stderr,
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
