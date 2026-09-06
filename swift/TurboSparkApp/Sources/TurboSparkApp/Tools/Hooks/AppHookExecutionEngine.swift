import Foundation

/// Asynchronous execution engine for dispatching lifecycle events and executing hooks.
public final class AppHookExecutionEngine: Sendable {
    public static let shared = AppHookExecutionEngine()

    public init() {}

    /// Events where an `async: true` command hook is still awaited
    /// synchronously, because the caller is about to act on its decision:
    /// `PreToolUse`/`PermissionRequest` gate whether a call runs, and
    /// `UserPromptSubmit`/`Stop` gate whether the turn proceeds or ends.
    /// Every other event is a notification nothing is waiting on, so
    /// `async: true` there really does run in the background.
    private static let blockingEvents: Set<AppHookEvent> = [.preToolUse, .userPromptSubmit, .stop, .permissionRequest]

    /// Dispatches a lifecycle event to all matching, enabled, and trusted hooks.
    public func dispatch(
        event: AppHookEvent,
        sessionID: String,
        toolName: String? = nil,
        toolArguments: [String: String]? = nil,
        toolOutput: String? = nil,
        toolDurationSeconds: Double? = nil,
        isError: Bool? = nil,
        workingDirectory: String? = nil,
        prompt: String? = nil,
        source: String? = nil,
        reason: String? = nil,
        stopHookActive: Bool? = nil,
        agentID: String? = nil,
        agentType: String? = nil
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

            if hook.isAsync && !Self.blockingEvents.contains(event) {
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
                        workingDirectory: workingDirectory,
                        prompt: prompt,
                        source: source,
                        reason: reason,
                        stopHookActive: stopHookActive,
                        agentID: agentID,
                        agentType: agentType
                    )
                }
            } else {
                // Every matching hook runs (no short-circuit on a deny/ask):
                // Claude Code's own semantics are "deny beats ask beats
                // allow across ALL hooks for the event", which
                // `AppHookDecisionAggregator` implements over the full
                // result set rather than this loop picking one early.
                let result = await executeSingleHook(
                    hook: hook,
                    event: event,
                    sessionID: sessionID,
                    toolName: toolName,
                    toolArguments: toolArguments,
                    toolOutput: toolOutput,
                    toolDurationSeconds: toolDurationSeconds,
                    isError: isError,
                    workingDirectory: workingDirectory,
                    prompt: prompt,
                    source: source,
                    reason: reason,
                    stopHookActive: stopHookActive,
                    agentID: agentID,
                    agentType: agentType
                )
                results.append(result)
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

        let verdict = AppHookDecisionAggregator.aggregate(results, event: .preToolUse)
        return AppHookPreToolUseDecision(
            behavior: verdict.permissionDecision ?? .allow,
            reason: verdict.permissionReason,
            updatedInput: verdict.updatedInput,
            additionalContext: verdict.additionalContext,
            preventContinuation: verdict.preventContinuation,
            continuationStopReason: verdict.continuationStopReason
        )
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
        workingDirectory: String?,
        prompt: String?,
        source: String?,
        reason: String?,
        stopHookActive: Bool?,
        agentID: String? = nil,
        agentType: String? = nil
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
                prompt: prompt,
                source: source,
                reason: reason,
                stopHookActive: stopHookActive,
                agentID: agentID,
                agentType: agentType,
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
                workingDirectory: workingDirectory,
                prompt: prompt,
                source: source,
                reason: reason,
                stopHookActive: stopHookActive,
                agentID: agentID,
                agentType: agentType,
                startTime: start
            )
        case .prompt:
            // Out of scope (root task: "prompt"/"agent" hook types are not
            // evaluated), but a CONFIGURED hook that does nothing is a
            // misconfiguration the user has to be able to see. This is a
            // visible non-blocking outcome rather than the anonymous no-op
            // it used to be; discovery flags the same limitation when the
            // entry is parsed.
            return AppHookExecutionResult(
                hookID: hook.id,
                hookName: hook.name,
                event: event,
                exitCode: 0,
                stdout: "Prompt hook type is not evaluated by this client.",
                stderr: "",
                durationSeconds: Date().timeIntervalSince(start),
                outcome: .nonBlockingError("\(hook.name) is a prompt hook, which this client does not evaluate.")
            )
        }
    }
}
