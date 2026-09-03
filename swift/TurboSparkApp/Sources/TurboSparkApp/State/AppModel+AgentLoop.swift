import Foundation
import TurboSpark

/// The agent loop's tool-call handling: evaluating `PreToolUse` /
/// `PermissionRequest` hooks and the permission engine for one extracted
/// tool call, running or denying it, and deciding whether to continue the
/// loop or hand off to `Stop`. Split out of `AppModel+Generation.swift`
/// (swift/CLAUDE.md Gotcha 15) to keep that file to the turn's streaming
/// lifecycle alone.
extension AppModel {
    /// Dispatches `Stop` and, if a hook blocks, re-enters the agent loop
    /// with the hook's reason folded in as the next user turn -- Claude
    /// Code's own "a Stop hook can force the turn to continue" behavior.
    /// Capped at 8 consecutive re-entries so a hook that always blocks
    /// cannot loop forever. Returns `true` when it re-entered (the caller
    /// should treat the turn as still in progress, not finished).
    @discardableResult
    func dispatchStopAndContinueIfBlocked(chatID: UUID, resumeStep: Int) async -> Bool {
        let wasActive = stopHookReentryCount > 0
        let verdict = await evaluateStop(stopHookActive: wasActive)
        guard verdict.isBlocked, stopHookReentryCount < 8 else {
            stopHookReentryCount = 0
            return false
        }
        stopHookReentryCount += 1
        let reason = verdict.blockReason ?? "A Stop hook requested the turn continue."
        if let idx = chats.firstIndex(where: { $0.id == chatID }) {
            chats[idx].messages.append(AppChatMessage(role: .user, content: reason))
            chats[idx].updatedAt = Date()
            persistChats()
        }
        continueAgentLoop(step: resumeStep, chatID: chatID)
        return true
    }

    /// Continues the agent loop if steps remain, otherwise asks `Stop`
    /// hooks whether to keep going anyway (bounded by the 8-cap above,
    /// independent of `maxAutonomousSteps`).
    func continueOrStop(afterStep currentStep: Int, chatID: UUID) async {
        let maxSteps = selectedProject?.maxAutonomousSteps ?? 5
        if currentStep + 1 < maxSteps {
            continueAgentLoop(step: currentStep + 1, chatID: chatID)
        } else {
            _ = await dispatchStopAndContinueIfBlocked(chatID: chatID, resumeStep: currentStep + 1)
        }
    }

    // `internal` rather than `private`: exercised directly by
    // AgentLoopRoutingTests / HookDecisionRoutingTests via `@testable
    // import`, since it is otherwise reachable only from inside a live
    // generation `Task` that needs a real model session.
    ///
    /// **`async`, AND THE CALLER MUST AWAIT IT.** This used to spawn a
    /// detached `Task` and return immediately, so the turn's stream loop fell
    /// straight through to its tail and set `generating = false` while the
    /// tool was still running: `canRun` went true mid-loop (Send started a
    /// second turn that overwrote `runTask`), `canCancel` went false (Stop
    /// was disabled for exactly the duration of a shell command), and
    /// `createChat` / `deleteChat` / `selectProject` / model unload were all
    /// open for the same window. Awaiting it keeps the turn's lifecycle
    /// honest, and puts the tool inside `runTask` so `cancel()` reaches it.
    func handleExtractedToolCall(_ call: AppToolCall, fullContent: String, reasoning: String, result: GenerationResult, currentStep: Int, chatID: UUID) async {
        let sessionID = chatID.uuidString

        // Evaluate PreToolUse lifecycle hooks first
        let hookDecision = await self.evaluatePreToolUseHooks(toolName: call.name, toolArguments: call.arguments)

        // `updatedInput` replaces the corresponding argument keys before
        // anything downstream (permission evaluation, execution, the
        // approval card) sees the call.
        var call = call
        if let updated = hookDecision.updatedInput {
            for (key, value) in updated { call.arguments[key] = value }
        }
        let cmd = call.arguments["command"] ?? call.arguments["cmd"]

        if hookDecision.behavior == .deny {
            let reason = hookDecision.reason ?? "Blocked by PreToolUse hook"
            let deniedResult = AppToolResult(
                callID: call.id,
                output: "Tool execution blocked by hook: \(reason)",
                isError: true,
                durationSeconds: 0.0
            )
            var deniedCall = call
            deniedCall.status = .denied
            if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
                self.chats[idx].messages.append(AppChatMessage(
                    role: .assistant,
                    content: fullContent,
                    reasoning: reasoning,
                    stopReason: "tool_use",
                    toolCalls: [deniedCall],
                    toolResults: [deniedResult]
                ))
                self.chats[idx].updatedAt = Date()
                self.persistChats()
            }
            await self.continueOrStop(afterStep: currentStep, chatID: chatID)
            return
        }

        // A hook that asked for confirmation was previously ignored
        // outright -- only `.deny` was checked above, so `.ask` fell
        // through to the ordinary permission evaluation below, which
        // could return `.allow` and run the call with no prompt at all
        // (T10). Route it into the same pending-approval UI the normal
        // engine's own `.ask` uses, folding the hook's reason into the
        // call's risk assessment rather than replacing it.
        if hookDecision.behavior == .ask {
            var pending = call
            pending.status = .pendingApproval
            let baseAssessment = call.riskAssessment ?? ToolRiskClassifier.assessRisk(name: call.name, arguments: call.arguments)
            let hookReason = hookDecision.reason ?? "A PreToolUse hook requested confirmation before this call runs."
            pending.riskAssessment = ToolRiskAssessment(
                level: baseAssessment.level,
                category: baseAssessment.category,
                reasons: baseAssessment.reasons + [hookReason]
            )
            self.pendingToolCall = pending
            self.pendingToolCallChatID = chatID
            self.pendingToolCallStep = currentStep
            self.pendingToolCallProject = self.selectedProject
            if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
                self.chats[idx].messages.append(AppChatMessage(
                    role: .assistant,
                    content: fullContent,
                    reasoning: reasoning,
                    stopReason: "tool_use",
                    toolCalls: [pending]
                ))
                self.chats[idx].updatedAt = Date()
                self.persistChats()
            }
            return
        }

        let sessionApproved = await SessionApprovalStore.shared.isApproved(sessionID: sessionID, toolName: call.name, command: cmd)
        // Captured once. Every branch below -- run, deny, or park for
        // approval -- refers to THIS project, so the policy that produced the
        // decision is the policy the call executes under.
        let decisionProject = self.selectedProject
        let decision = AppToolPermissionEngine.evaluate(call: call, project: decisionProject, sessionApproved: sessionApproved)

        switch decision {
        case .ask(let assessment, _):
            // `PermissionRequest` fires exactly where this app would
            // otherwise show the approval card, so a hook can resolve
            // `allow`/`deny` without ever surfacing the UI.
            let permVerdict = await self.evaluatePermissionRequest(toolName: call.name, toolArguments: call.arguments)

            if permVerdict.permissionDecision == .allow {
                await self.runApprovedCall(call, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep, project: decisionProject)
            } else if permVerdict.permissionDecision == .deny {
                let reason = permVerdict.permissionReason ?? "Denied by PermissionRequest hook"
                await self.recordDeniedCall(call, reason: reason, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep)
            } else {
                var pending = call
                pending.status = .pendingApproval
                pending.riskAssessment = assessment
                self.pendingToolCall = pending
                self.pendingToolCallChatID = chatID
                self.pendingToolCallStep = currentStep
                self.pendingToolCallProject = self.selectedProject
                if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
                    self.chats[idx].messages.append(AppChatMessage(
                        role: .assistant,
                        content: fullContent,
                        reasoning: reasoning,
                        stopReason: "tool_use",
                        toolCalls: [pending]
                    ))
                    self.chats[idx].updatedAt = Date()
                    self.persistChats()
                }
            }

        case .allow:
            await self.runApprovedCall(call, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep, project: decisionProject)

        case .deny(let reason):
            await self.recordDeniedCall(call, reason: reason, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep)
        }
    }

    /// Runs an approved call, folds `PostToolUse`'s feedback/context into the
    /// result, appends the turn, and continues the loop. Shared by the
    /// ordinary `.allow` path and a `PermissionRequest` hook resolving
    /// `allow` in place of the approval card.
    private func runApprovedCall(_ call: AppToolCall, fullContent: String, reasoning: String, chatID: UUID, currentStep: Int, project: AppProject?) async {
        var runningCall = call
        runningCall.status = .running
        var toolResult = await AppToolRegistry.execute(call: runningCall, in: project, chatID: chatID)
        runningCall.status = toolResult.isError ? .failed : .completed

        let postVerdict = await self.dispatchPostToolUseVerdict(
            toolName: runningCall.name,
            toolArguments: runningCall.arguments,
            toolOutput: toolResult.output,
            toolDurationSeconds: toolResult.durationSeconds,
            isError: toolResult.isError
        )
        // Exit-2 stderr (or `decision: "block"`) from a PostToolUse hook is
        // feedback, never a block -- the tool already ran. `additionalContext`
        // is folded in the same way. Both flow to the model on the NEXT turn
        // through the `<tool_response>`/`<tool_error>` tags built from
        // `toolResults` in `executeGenerationTurn`.
        if let note = postVerdict.blockReason ?? postVerdict.feedbackMessage, !note.isEmpty {
            toolResult.output += "\n\n<hook_feedback>\n\(note)\n</hook_feedback>"
        }
        if let ctx = postVerdict.additionalContext, !ctx.isEmpty {
            toolResult.output += "\n\n<hook_context>\n\(ctx)\n</hook_context>"
        }

        if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
            self.chats[idx].messages.append(AppChatMessage(
                role: .assistant,
                content: fullContent,
                reasoning: reasoning,
                stopReason: "tool_use",
                toolCalls: [runningCall],
                toolResults: [toolResult]
            ))
            self.chats[idx].updatedAt = Date()
            self.persistChats()
        }

        await self.continueOrStop(afterStep: currentStep, chatID: chatID)
    }

    /// Records a denied call (permission engine or `PermissionRequest` hook)
    /// and continues the loop.
    private func recordDeniedCall(_ call: AppToolCall, reason: String, fullContent: String, reasoning: String, chatID: UUID, currentStep: Int) async {
        let deniedResult = AppToolResult(
            callID: call.id,
            output: "Tool execution denied: \(reason)",
            isError: true,
            durationSeconds: 0.0
        )
        var deniedCall = call
        deniedCall.status = .denied
        if let idx = self.chats.firstIndex(where: { $0.id == chatID }) {
            self.chats[idx].messages.append(AppChatMessage(
                role: .assistant,
                content: fullContent,
                reasoning: reasoning,
                stopReason: "tool_use",
                toolCalls: [deniedCall],
                toolResults: [deniedResult]
            ))
            self.chats[idx].updatedAt = Date()
            self.persistChats()
        }
        await self.continueOrStop(afterStep: currentStep, chatID: chatID)
    }

    /// Continues multi-turn autonomous loop after tool execution.
    ///
    /// Does NOT gate on `!generating`: by the time anything calls this
    /// (the allow/deny paths above, or an approved/denied pending call),
    /// the CURRENT turn's token stream has already finished -- `generating`
    /// reads `true` here only because the ORIGINAL turn's own `runTask`
    /// tail (which resets it) has not yet run, a race against this very
    /// call (state#10). The `generationEpoch` guard on that tail is what
    /// keeps it from clobbering the state a reentrant call here is about
    /// to set up.
    /// - Parameter chatID: the conversation this loop belongs to. NOT
    ///   `selectedChatID`: a tool can run for seconds with `generating` false
    ///   (state#9), so the user may have selected another chat by now, and
    ///   resolving it here builds the next step from that chat's history and
    ///   replies into it.
    public func continueAgentLoop(step: Int, chatID: UUID) {
        guard session != nil else { return }
        // Cancellation is cooperative: a tool already in flight runs to
        // completion, and what Stop guarantees is that no further turn starts.
        // Without this the loop resumed straight through a cancel.
        guard !isCancellationPending else { return }
        executeGenerationTurn(step: step, chatID: chatID)
    }
}
