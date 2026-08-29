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
        isError: Bool? = nil
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
            workingDirectory: projectDir
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
}
