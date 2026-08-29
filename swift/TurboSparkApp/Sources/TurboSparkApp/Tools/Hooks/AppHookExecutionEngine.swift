import Foundation

/// Asynchronous execution engine for dispatching lifecycle events and executing hooks.
public final class AppHookExecutionEngine: Sendable {
    public static let shared = AppHookExecutionEngine()

    public init() {}

    /// Dispatches a lifecycle event to all matching, enabled, and trusted hooks.
    public func dispatch(
        event: AppHookEvent,
        sessionID: String,
        toolName: String? = nil,
        toolArguments: [String: String]? = nil,
        toolOutput: String? = nil,
        toolDurationSeconds: Double? = nil,
        isError: Bool? = nil,
        workingDirectory: String? = nil
    ) async -> [AppHookExecutionResult] {
        let store = await AppHookStore.shared
        let allHooks = await store.hooks
        let trustedHashes = await store.trustedHashes

        let candidateHooks = allHooks.filter { hook in
            hook.isEnabled && hook.event == event && (hook.sourceType == .custom || trustedHashes.contains(hook.contentHash))
        }

        var results: [AppHookExecutionResult] = []

        for hook in candidateHooks {
            // Check matcher and if conditions
            guard matchesCondition(hook: hook, toolName: toolName, toolArguments: toolArguments) else {
                continue
            }

            if hook.isAsync {
                Task.detached {
                    _ = await self.executeSingleHook(
                        hook: hook,
                        event: event,
                        sessionID: sessionID,
                        toolName: toolName,
                        toolArguments: toolArguments,
                        toolOutput: toolOutput,
                        toolDurationSeconds: toolDurationSeconds,
                        isError: isError,
                        workingDirectory: workingDirectory
                    )
                }
            } else {
                let result = await executeSingleHook(
                    hook: hook,
                    event: event,
                    sessionID: sessionID,
                    toolName: toolName,
                    toolArguments: toolArguments,
                    toolOutput: toolOutput,
                    toolDurationSeconds: toolDurationSeconds,
                    isError: isError,
                    workingDirectory: workingDirectory
                )
                results.append(result)

                // If this is PreToolUse and the hook made a blocking denial, stop evaluating remaining hooks
                if event == .preToolUse, let dec = result.decision, dec.behavior == .deny || dec.behavior == .ask {
                    break
                }
            }
        }

        return results
    }

    /// Evaluates whether a tool call is permitted by `PreToolUse` hooks.
    public func evaluatePreToolUse(
        sessionID: String,
        toolName: String,
        toolArguments: [String: String],
        workingDirectory: String? = nil
    ) async -> AppHookPreToolUseDecision {
        let results = await dispatch(
            event: .preToolUse,
            sessionID: sessionID,
            toolName: toolName,
            toolArguments: toolArguments,
            workingDirectory: workingDirectory
        )

        for result in results {
            if let decision = result.decision {
                if decision.behavior == .deny {
                    return decision
                } else if decision.behavior == .ask {
                    return decision
                }
            }
            if result.exitCode == 2 {
                let reason = result.stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    ? result.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
                    : result.stderr.trimmingCharacters(in: .whitespacesAndNewlines)
                return AppHookPreToolUseDecision(
                    behavior: .deny,
                    reason: reason.isEmpty ? "Blocked by hook `\(result.hookName)`" : reason,
                    blockedByHookName: result.hookName
                )
            }
        }

        return AppHookPreToolUseDecision(behavior: .allow)
    }

    // MARK: - Single Hook Execution

    private func executeSingleHook(
        hook: AppHookCommand,
        event: AppHookEvent,
        sessionID: String,
        toolName: String?,
        toolArguments: [String: String]?,
        toolOutput: String?,
        toolDurationSeconds: Double?,
        isError: Bool?,
        workingDirectory: String?
    ) async -> AppHookExecutionResult {
        let start = Date()

        switch hook.type {
        case .command:
            return await executeCommandHook(
                hook: hook,
                event: event,
                sessionID: sessionID,
                toolName: toolName,
                toolArguments: toolArguments,
                toolOutput: toolOutput,
                toolDurationSeconds: toolDurationSeconds,
                isError: isError,
                workingDirectory: workingDirectory,
                startTime: start
            )
        case .http:
            return await executeHttpHook(
                hook: hook,
                event: event,
                sessionID: sessionID,
                toolName: toolName,
                toolArguments: toolArguments,
                toolOutput: toolOutput,
                startTime: start
            )
        case .prompt:
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 0,
                stdout: "Evaluated prompt hook: \(hook.command)",
                stderr: "",
                durationSeconds: Date().timeIntervalSince(start),
                decision: AppHookPreToolUseDecision(behavior: .allow)
            )
        }
    }

    private func executeCommandHook(
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

    private func executeHttpHook(
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

    // MARK: - Condition & Output Parsing

    private func matchesCondition(hook: AppHookCommand, toolName: String?, toolArguments: [String: String]?) -> Bool {
        // If matcher is set (e.g. "Write|Edit" or "Bash")
        if let matcher = hook.matcher, !matcher.isEmpty, let toolName {
            let parts = matcher.split(separator: "|").map { $0.trimmingCharacters(in: .whitespaces) }
            let matched = parts.contains("*") || parts.contains { part in
                toolName.localizedCaseInsensitiveContains(part) || part.localizedCaseInsensitiveContains(toolName)
            }
            if !matched { return false }
        }

        // If 'ifCondition' is set (e.g. "Bash(git *)")
        if let ifCond = hook.ifCondition, !ifCond.isEmpty, let toolName {
            if ifCond.contains("(") && ifCond.hasSuffix(")") {
                let toolPrefix = ifCond.prefix(while: { $0 != "(" })
                if !toolName.localizedCaseInsensitiveContains(toolPrefix) {
                    return false
                }
                if let innerStart = ifCond.firstIndex(of: "("), let innerEnd = ifCond.lastIndex(of: ")") {
                    let pattern = String(ifCond[ifCond.index(after: innerStart)..<innerEnd]).trimmingCharacters(in: .whitespaces)
                    let cmdArg = toolArguments?["command"] ?? toolArguments?["cmd"] ?? toolArguments?["path"] ?? ""
                    if pattern != "*" && !cmdArg.localizedCaseInsensitiveContains(pattern.replacingOccurrences(of: "*", with: "")) {
                        return false
                    }
                }
            }
        }

        return true
    }

    private func parsePreToolUseOutput(stdout: String, stderr: String, exitCode: Int32, hookName: String) -> AppHookPreToolUseDecision {
        if exitCode == 2 {
            let reason = stderr.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                ? stdout.trimmingCharacters(in: .whitespacesAndNewlines)
                : stderr.trimmingCharacters(in: .whitespacesAndNewlines)
            return AppHookPreToolUseDecision(
                behavior: .deny,
                reason: reason.isEmpty ? "Blocked by hook `\(hookName)`" : reason,
                blockedByHookName: hookName
            )
        }

        if let data = stdout.data(using: .utf8),
           let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
            if let decision = json["permissionDecision"] as? String {
                let reason = json["permissionDecisionReason"] as? String
                switch decision.lowercased() {
                case "deny", "block":
                    return AppHookPreToolUseDecision(behavior: .deny, reason: reason, blockedByHookName: hookName)
                case "ask":
                    return AppHookPreToolUseDecision(behavior: .ask, reason: reason, blockedByHookName: hookName)
                case "allow":
                    return AppHookPreToolUseDecision(behavior: .allow, reason: reason, blockedByHookName: hookName)
                default:
                    break
                }
            }
        }

        return AppHookPreToolUseDecision(behavior: .allow)
    }
}
