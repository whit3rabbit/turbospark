import Foundation

/// Asynchronous execution engine for dispatching lifecycle events and executing hooks.
public final class AppHookExecutionEngine: Sendable {
    public static let shared = AppHookExecutionEngine()

    public init() {}

    /// Session ids (chat ids) of Ghost Mode chats.
    ///
    /// The guard lives HERE rather than at each caller because tool calls
    /// reach hooks from several paths that do not go through
    /// `AppModel+Hooks` (batch children, subagent steps, deferred MCP
    /// calls), and each would otherwise have to remember to apply the ghost
    /// rule: content-bearing events never reach a hook for a ghost session,
    /// and a matching gate hook fails closed instead of receiving content.
    private final class GhostSessions: @unchecked Sendable {
        private let lock = NSLock()
        private var ids: Set<String> = []
        func set(_ id: String, _ isGhost: Bool) {
            lock.lock(); defer { lock.unlock() }
            if isGhost { ids.insert(id) } else { ids.remove(id) }
        }
        func contains(_ id: String) -> Bool {
            lock.lock(); defer { lock.unlock() }
            return ids.contains(id)
        }
    }
    private static let ghostSessions = GhostSessions()

    public static func markGhostSession(_ sessionID: String, isGhost: Bool) {
        ghostSessions.set(sessionID, isGhost)
    }

    static func isGhostSession(_ sessionID: String) -> Bool {
        ghostSessions.contains(sessionID)
    }

    /// Whether dispatch would run at least one hook for this event and tool.
    /// Used when callers must preserve a hook as a safety gate without
    /// disclosing the tool payload to it.
    @MainActor
    func hasMatchingHook(
        event: AppHookEvent,
        toolName: String,
        toolArguments: [String: String]
    ) -> Bool {
        let store = AppHookStore.shared
        return store.hooks.contains { hook in
            hook.isEnabled && hook.event == event
                && (hook.sourceType == .custom
                    || store.trustedHashes.contains(hook.contentHash))
                && matchesCondition(
                    hook: hook, toolName: toolName, toolArguments: toolArguments)
        }
    }

    /// Events where an `async: true` command hook is still awaited
    /// synchronously, because the caller is about to act on its decision:
    /// `PreToolUse`/`PermissionRequest` gate whether a call runs, and
    /// `UserPromptSubmit`/`Stop` gate whether the turn proceeds or ends.
    /// Every other event is a notification nothing is waiting on, so
    /// `async: true` there really does run in the background.
    static let blockingEvents: Set<AppHookEvent> = [.preToolUse, .userPromptSubmit, .stop, .permissionRequest]

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
        agentType: String? = nil,
        projectBoundHookDirectory: String? = nil
    ) async -> [AppHookExecutionResult] {
        // Ghost sessions never expose prompt or tool content to a hook. An
        // empty result set also makes a PermissionRequest gate read as "no
        // allow", which the callers treat as a refusal (fail closed).
        if Self.isGhostSession(sessionID), event.carriesConversationContent {
            return []
        }
        let (candidateHooks, snapshot) = await candidates(
            event: event, projectBoundHookDirectory: projectBoundHookDirectory)

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
                        agentType: agentType,
                        hookSnapshot: snapshot
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
                    agentType: agentType,
                    hookSnapshot: snapshot
                )
                results.append(result)
            }
        }

        return results
    }

    /// The enabled, trusted hooks registered for `event`, and the snapshot
    /// they came from when the dispatch is bound to a project directory.
    private func candidates(
        event: AppHookEvent, projectBoundHookDirectory: String?
    ) async -> (hooks: [AppHookCommand], snapshot: AppHookDispatchSnapshot?) {
        let store = await AppHookStore.shared
        let snapshot: AppHookDispatchSnapshot?
        if let projectBoundHookDirectory {
            snapshot = await store.dispatchSnapshot(projectDirectory: projectBoundHookDirectory)
        } else {
            snapshot = nil
        }
        let allHooks: [AppHookCommand]
        let trustedHashes: Set<String>
        if let snapshot {
            allHooks = snapshot.hooks
            trustedHashes = snapshot.trustedHashes
        } else {
            allHooks = await store.hooks
            trustedHashes = await store.trustedHashes
        }
        let hooks = allHooks.filter { hook in
            hook.isEnabled && hook.event == event
                && (hook.sourceType == .custom || trustedHashes.contains(hook.contentHash))
        }
        return (hooks, snapshot)
    }

    /// Evaluates whether a tool call is permitted by `PreToolUse` hooks.
    public func evaluatePreToolUse(
        sessionID: String,
        toolName: String,
        toolArguments: [String: String],
        workingDirectory: String? = nil,
        projectBoundHookDirectory: String? = nil
    ) async -> AppHookPreToolUseDecision {
        if Self.isGhostSession(sessionID) {
            // A safety hook must keep gating, but it cannot be handed the
            // private tool content: block the call when one matches.
            let (hooks, _) = await candidates(
                event: .preToolUse, projectBoundHookDirectory: projectBoundHookDirectory)
            let gated = hooks.contains {
                matchesCondition(hook: $0, toolName: toolName, toolArguments: toolArguments)
            }
            return gated
                ? AppHookPreToolUseDecision(
                    behavior: .deny,
                    reason: "Tool use is blocked in Ghost Mode because a matching "
                        + "PreToolUse safety hook cannot receive private tool content.")
                : AppHookPreToolUseDecision(behavior: .allow)
        }
        let results = await dispatch(
            event: .preToolUse,
            sessionID: sessionID,
            toolName: toolName,
            toolArguments: toolArguments,
            workingDirectory: workingDirectory,
            projectBoundHookDirectory: projectBoundHookDirectory
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
        agentType: String? = nil,
        hookSnapshot: AppHookDispatchSnapshot? = nil
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
                hookSnapshot: hookSnapshot,
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
