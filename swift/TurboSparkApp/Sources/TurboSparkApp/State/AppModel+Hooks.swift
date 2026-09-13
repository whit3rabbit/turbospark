import Foundation

extension AppModel {
    private func isGhostChat(_ chatID: UUID) -> Bool {
        chats.first(where: { $0.id == chatID })?.isGhost == true
    }

    /// Points `AppHookStore` at the directory the hooks about to run belong
    /// to (state#67).
    ///
    /// The store is refreshed by `selectProject` / `createProject` /
    /// `updateProject`, i.e. by the SELECTION, and a turn's project is not
    /// the selection: a call can sit at an approval card for minutes with
    /// `generating` false, which is exactly when a switch is legal. Without
    /// this, a `PostToolUse` hook for a tool that ran in project A is drawn
    /// from project B's `.claude/settings.json`.
    ///
    /// A no-op when the store already points there, which is the common case,
    /// so the ordinary single-project session pays one comparison.
    func rebindHookStore(to projectDirectory: String?) {
        let store = AppHookStore.shared
        guard store.lastProjectDirectory != projectDirectory || !store.didRefreshAtLeastOnce else {
            return
        }
        store.refresh(projectDirectory: projectDirectory)
    }

    /// Dispatches a lifecycle hook event across all registered and trusted hooks.
    ///
    /// **THE CHAT AND THE PROJECT ARE ARGUMENTS, NEVER THE SELECTION**
    /// (state#67). This resolved both from `selectedChatID` /
    /// `selectedProject` while the agent loop threaded a captured `chatID`
    /// and `project` through every other decision it makes (state#30) -- so
    /// a hook fired for a call in chat A carried chat B's session id and ran
    /// in project B's root. The lifecycle callers that genuinely describe the
    /// selection (`sessionStart`, `sessionEnd`) pass it explicitly.
    @discardableResult
    public func dispatchLifecycleHook(
        event: AppHookEvent,
        chatID: UUID,
        project: AppProject?,
        toolName: String? = nil,
        toolArguments: [String: String]? = nil,
        toolOutput: String? = nil,
        toolDurationSeconds: Double? = nil,
        isError: Bool? = nil,
        prompt: String? = nil,
        source: String? = nil,
        reason: String? = nil,
        stopHookActive: Bool? = nil,
        agentID: String? = nil,
        agentType: String? = nil
    ) async -> [AppHookExecutionResult] {
        if isGhostChat(chatID), event.carriesConversationContent {
            return []
        }
        let sessionID = chatID.uuidString
        let projectDir = project?.rootDirectoryPath
        rebindHookStore(to: projectDir)

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
            stopHookActive: stopHookActive,
            agentID: agentID,
            agentType: agentType
        )
    }

    /// Evaluates `PreToolUse` hooks before a tool call runs. Returns whether execution is blocked, denied, or allowed.
    public func evaluatePreToolUseHooks(
        toolName: String,
        toolArguments: [String: String],
        chatID: UUID,
        project: AppProject?
    ) async -> AppHookPreToolUseDecision {
        let projectDir = project?.rootDirectoryPath
        rebindHookStore(to: projectDir)

        if isGhostChat(chatID) {
            let hasGate = AppHookExecutionEngine.shared.hasMatchingHook(
                event: .preToolUse, toolName: toolName, toolArguments: toolArguments)
            return hasGate
                ? AppHookPreToolUseDecision(
                    behavior: .deny,
                    reason: "Tool use is blocked in Ghost Mode because a matching "
                        + "PreToolUse safety hook cannot receive private tool content.")
                : AppHookPreToolUseDecision(behavior: .allow)
        }

        return await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: chatID.uuidString,
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
    public func evaluateUserPromptSubmit(
        prompt: String, chatID: UUID, project: AppProject?
    ) async -> AppHookVerdict {
        let results = await dispatchLifecycleHook(
            event: .userPromptSubmit, chatID: chatID, project: project, prompt: prompt)
        return AppHookDecisionAggregator.aggregate(results, event: .userPromptSubmit)
    }

    /// Evaluates `Stop` hooks. `stopHookActive` mirrors Claude Code's own
    /// field: true once this turn has already been re-entered by a Stop
    /// hook's block, so a hook script can tell "the model is about to stop
    /// again" from "this is the first Stop of the turn".
    @discardableResult
    public func evaluateStop(
        stopHookActive: Bool, chatID: UUID, project: AppProject?
    ) async -> AppHookVerdict {
        let results = await dispatchLifecycleHook(
            event: .stop, chatID: chatID, project: project, stopHookActive: stopHookActive)
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
        isError: Bool,
        chatID: UUID,
        project: AppProject?
    ) async -> AppHookVerdict {
        var results = await dispatchLifecycleHook(
            event: .postToolUse,
            chatID: chatID,
            project: project,
            toolName: toolName,
            toolArguments: toolArguments,
            toolOutput: toolOutput,
            toolDurationSeconds: toolDurationSeconds,
            isError: isError
        )
        if isError {
            results += await dispatchLifecycleHook(
                event: .postToolUseFailure,
                chatID: chatID,
                project: project,
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
    public func evaluatePermissionRequest(
        toolName: String,
        toolArguments: [String: String],
        chatID: UUID,
        project: AppProject?
    ) async -> AppHookVerdict {
        let projectDir = project?.rootDirectoryPath
        rebindHookStore(to: projectDir)
        if isGhostChat(chatID) {
            let hasGate = AppHookExecutionEngine.shared.hasMatchingHook(
                event: .permissionRequest, toolName: toolName, toolArguments: toolArguments)
            return hasGate
                ? AppHookVerdict(
                    permissionDecision: .deny,
                    permissionReason: "Tool use is blocked in Ghost Mode because a matching "
                        + "PermissionRequest safety hook cannot receive private tool content.")
                : AppHookVerdict()
        }
        let results = await AppHookExecutionEngine.shared.dispatch(
            event: .permissionRequest,
            sessionID: chatID.uuidString,
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
    public func dispatchSessionEnd(
        reason: String, chatID: UUID, project: AppProject?
    ) async -> [AppHookExecutionResult] {
        await dispatchLifecycleHook(
            event: .sessionEnd, chatID: chatID, project: project, reason: reason)
    }

    /// Dispatches `Notification`.
    @discardableResult
    public func dispatchNotification(
        message: String, chatID: UUID, project: AppProject?
    ) async -> [AppHookExecutionResult] {
        await dispatchLifecycleHook(
            event: .notification, chatID: chatID, project: project, reason: message)
    }

    /// Dispatches `PermissionDenied` when a tool call is refused, whether
    /// by the user at the approval card or by the permission engine. The
    /// standard non-blocking table applies to its output; a hook's
    /// `hookSpecificOutput.retry` is parsed but deliberately NOT acted on:
    /// automatically re-running a call a human just refused is not a
    /// decision this app makes on a hook's word.
    @discardableResult
    public func dispatchPermissionDenied(
        toolName: String,
        toolArguments: [String: String],
        reason: String,
        chatID: UUID,
        project: AppProject?
    ) async -> [AppHookExecutionResult] {
        await dispatchLifecycleHook(
            event: .permissionDenied, chatID: chatID, project: project,
            toolName: toolName, toolArguments: toolArguments, reason: reason)
    }

    /// Dispatches `SubagentStart` / `SubagentStop`. Notification-grade:
    /// matching hooks run and any feedback is logged, but neither event can
    /// block the subagent, which has no interaction surface to resolve a
    /// block with.
    @discardableResult
    public func dispatchSubagentLifecycle(
        event: AppHookEvent,
        chatID: UUID,
        project: AppProject?,
        agentID: String,
        agentType: String,
        stopHookActive: Bool? = nil
    ) async -> [AppHookExecutionResult] {
        await dispatchLifecycleHook(
            event: event, chatID: chatID, project: project,
            stopHookActive: stopHookActive, agentID: agentID, agentType: agentType)
    }
}

private extension AppHookEvent {
    var carriesConversationContent: Bool {
        switch self {
        case .userPromptSubmit, .preToolUse, .postToolUse, .postToolUseFailure,
             .permissionRequest, .permissionDenied:
            return true
        case .sessionStart, .sessionEnd, .stop, .subagentStart, .subagentStop,
             .notification, .preCompact:
            return false
        }
    }
}
