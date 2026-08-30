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
        prompt: String?,
        source: String?,
        reason: String?,
        stopHookActive: Bool?,
        startTime: Date
    ) async -> AppHookExecutionResult {
        let cwd = (workingDirectory?.isEmpty == false) ? workingDirectory! : FileManager.default.currentDirectoryPath
        let payloadDict = AppHookStdinPayload.build(
            event: event,
            sessionID: sessionID,
            transcriptPath: AppHookStdinPayload.transcriptPath,
            cwd: cwd,
            toolName: toolName,
            toolArguments: toolArguments,
            toolOutput: toolOutput,
            toolDurationSeconds: toolDurationSeconds,
            isError: isError,
            prompt: prompt,
            source: source,
            reason: reason,
            stopHookActive: stopHookActive
        )

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

        // Variable substitution (${user_config.KEY}, ${CLAUDE_PROJECT_DIR})
        let store = await AppHookStore.shared
        let allOptions = await store.optionValues
        var commandText = hook.command
        let sourceID = hook.pluginName != nil ? "plugin_\(hook.pluginName!)" : (hook.sourceType == .projectConfig ? "project_config" : "user_config")
        let options = allOptions[sourceID] ?? [:]

        for (k, v) in options {
            commandText = commandText.replacingOccurrences(of: "${user_config.\(k)}", with: v)
        }
        if let workingDirectory, !workingDirectory.isEmpty {
            commandText = commandText.replacingOccurrences(of: "${CLAUDE_PROJECT_DIR}", with: workingDirectory)
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
        if let workingDirectory {
            env["TURBOSPARK_PROJECT_DIR"] = workingDirectory
            env["CLAUDE_PROJECT_DIR"] = workingDirectory
        }

        for (k, v) in options {
            let normalized = k.uppercased().replacingOccurrences(of: "-", with: "_")
            env["TURBOSPARK_OPTION_\(normalized)"] = v
            env["CLAUDE_PLUGIN_OPTION_\(normalized)"] = v
        }

        // Claude Code's own default is 600s. `SessionEnd` is capped far
        // tighter: nothing is waiting on its result, and a slow hook there
        // would hold up chat deletion / clearing for the whole timeout.
        var timeout = hook.timeoutSeconds > 0 ? hook.timeoutSeconds : 600.0
        if event == .sessionEnd { timeout = min(timeout, 1.5) }
        let workingDirectoryURL = (workingDirectory?.isEmpty == false) ? URL(fileURLWithPath: workingDirectory!) : nil

        do {
            // `ProcessExecutor` writes stdin off this call's thread and drains
            // stdout/stderr concurrently with the timeout wait. A previous
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
            let outcome = AppHookResponseParser.parseHookOutput(
                stdout: result.stdout,
                stderr: result.stderr,
                exitCode: result.exitCode,
                event: event
            )

            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: result.exitCode,
                stdout: result.stdout,
                stderr: result.stderr,
                durationSeconds: duration,
                outcome: outcome
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
        workingDirectory: String?,
        prompt: String?,
        source: String?,
        reason: String?,
        stopHookActive: Bool?,
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

        let cwd = (workingDirectory?.isEmpty == false) ? workingDirectory! : FileManager.default.currentDirectoryPath
        let payloadDict = AppHookStdinPayload.build(
            event: event,
            sessionID: sessionID,
            transcriptPath: AppHookStdinPayload.transcriptPath,
            cwd: cwd,
            toolName: toolName,
            toolArguments: toolArguments,
            toolOutput: toolOutput,
            prompt: prompt,
            source: source,
            reason: reason,
            stopHookActive: stopHookActive
        )
        request.httpBody = try? JSONSerialization.data(withJSONObject: payloadDict, options: [])

        do {
            let (data, response) = try await URLSession.shared.data(for: request)
            let httpResponse = response as? HTTPURLResponse
            let statusCode = Int32(httpResponse?.statusCode ?? 200)
            let stdout = String(data: data, encoding: .utf8) ?? ""
            let isSuccess = statusCode >= 200 && statusCode < 300

            // An HTTP hook has no exit code of its own, so a non-2xx status
            // maps to "exit 1" for the purposes of the shared parser -- an
            // HTTP hook cannot use Claude Code's exit-2 blocking convention
            // at all, only the JSON `hookSpecificOutput`/`decision` fields.
            let outcome = AppHookResponseParser.parseHookOutput(
                stdout: stdout,
                stderr: isSuccess ? "" : "HTTP status \(statusCode)",
                exitCode: isSuccess ? 0 : 1,
                event: event
            )

            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: isSuccess ? 0 : 1,
                stdout: stdout,
                stderr: isSuccess ? "" : "HTTP status \(statusCode)",
                durationSeconds: Date().timeIntervalSince(startTime),
                outcome: outcome
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
