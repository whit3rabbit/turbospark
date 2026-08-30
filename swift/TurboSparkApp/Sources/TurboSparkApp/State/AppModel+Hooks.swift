import Foundation

extension AppModel {
    /// Dispatches a lifecycle hook event across all registered and trusted hooks.
    @discardableResult
    public func dispatchLifecycleHook(
        event: AppHookEvent,
        toolName: String? = nil,
        toolArguments: [String: String]? = nil,
        toolOutput: String? = nil,
        toolDurationSeconds: Double? = nil,
        isError: Bool? = nil,
        prompt: String? = nil,
        source: String? = nil,
        reason: String? = nil,
        stopHookActive: Bool? = nil
    ) async -> [AppHookExecutionResult] {
        let sessionID = selectedChatID.uuidString
        let projectDir = selectedProject?.rootDirectoryPath

        return await AppHookExecutionEngine.shared.dispatch(
            event: event,
            sessionID: sessionID,
            toolName: toolName,
            toolArguments: toolArguments,
            toolOutput: toolOutput,
            toolDurationSeconds: toolDurationSeconds,
            isError: isError,
            workingDirectory: projectDir,
            prompt: prompt,
            source: source,
            reason: reason,
            stopHookActive: stopHookActive
        )
    }

    /// Evaluates `PreToolUse` hooks before a tool call runs. Returns whether execution is blocked, denied, or allowed.
    public func evaluatePreToolUseHooks(
        toolName: String,
        toolArguments: [String: String]
    ) async -> AppHookPreToolUseDecision {
        let sessionID = selectedChatID.uuidString
        let projectDir = selectedProject?.rootDirectoryPath

        return await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: sessionID,
            toolName: toolName,
            toolArguments: toolArguments,
            workingDirectory: projectDir
        )
    }

    /// Evaluates `UserPromptSubmit` hooks before a turn starts. A hook that
    /// blocks (exit 2, or `decision: "block"`) should stop the turn before
    /// anything is sent to the model; `additionalContext`/plain stdout is
    /// folded into the turn as extra context.
    @discardableResult
    public func evaluateUserPromptSubmit(prompt: String) async -> AppHookVerdict {
        let results = await dispatchLifecycleHook(event: .userPromptSubmit, prompt: prompt)
        return AppHookDecisionAggregator.aggregate(results, event: .userPromptSubmit)
    }

    /// Evaluates `Stop` hooks. `stopHookActive` mirrors Claude Code's own
    /// field: true once this turn has already been re-entered by a Stop
    /// hook's block, so a hook script can tell "the model is about to stop
    /// again" from "this is the first Stop of the turn".
    @discardableResult
    public func evaluateStop(stopHookActive: Bool) async -> AppHookVerdict {
        let results = await dispatchLifecycleHook(event: .stop, stopHookActive: stopHookActive)
        return AppHookDecisionAggregator.aggregate(results, event: .stop)
    }

    /// Dispatches `PostToolUse` (and, on failure, `PostToolUseFailure` too)
    /// and returns the combined verdict: a hook's exit-2 stderr or
    /// `additionalContext` is feedback for the model, never a block --
    /// the tool has already run by the time this fires.
    @discardableResult
    public func dispatchPostToolUseVerdict(
        toolName: String,
        toolArguments: [String: String],
        toolOutput: String,
        toolDurationSeconds: Double,
        isError: Bool
    ) async -> AppHookVerdict {
        var results = await dispatchLifecycleHook(
            event: .postToolUse,
            toolName: toolName,
            toolArguments: toolArguments,
            toolOutput: toolOutput,
            toolDurationSeconds: toolDurationSeconds,
            isError: isError
        )
        if isError {
            results += await dispatchLifecycleHook(
                event: .postToolUseFailure,
                toolName: toolName,
                toolArguments: toolArguments,
                toolOutput: toolOutput,
                toolDurationSeconds: toolDurationSeconds,
                isError: true
            )
        }
        return AppHookDecisionAggregator.aggregate(results, event: .postToolUse)
    }

    /// Evaluates `PermissionRequest` hooks at the point the permission
    /// engine would otherwise show the approval card, so a hook can resolve
    /// `allow`/`deny` without ever surfacing the UI.
    @discardableResult
    public func evaluatePermissionRequest(toolName: String, toolArguments: [String: String]) async -> AppHookVerdict {
        let sessionID = selectedChatID.uuidString
        let projectDir = selectedProject?.rootDirectoryPath
        let results = await AppHookExecutionEngine.shared.dispatch(
            event: .permissionRequest,
            sessionID: sessionID,
            toolName: toolName,
            toolArguments: toolArguments,
            workingDirectory: projectDir
        )
        return AppHookDecisionAggregator.aggregate(results, event: .permissionRequest)
    }

    /// Dispatches `SessionEnd` (chat deletion or clearing). Fire-and-forget
    /// from the caller's point of view -- nothing in this app can meaningfully
    /// block a chat from being cleared on a hook's say-so.
    @discardableResult
    public func dispatchSessionEnd(reason: String) async -> [AppHookExecutionResult] {
        await dispatchLifecycleHook(event: .sessionEnd, reason: reason)
    }

    /// Dispatches `Notification`.
    @discardableResult
    public func dispatchNotification(message: String) async -> [AppHookExecutionResult] {
        await dispatchLifecycleHook(event: .notification, reason: message)
    }
}
