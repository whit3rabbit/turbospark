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
}
