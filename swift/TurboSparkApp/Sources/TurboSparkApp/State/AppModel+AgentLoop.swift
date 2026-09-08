import Foundation
import TurboSpark

/// The message a call the model issued alongside another one is answered
/// with. Stating the rule beats dropping the call silently: an unanswered
/// call reads to the model as a tool that produced nothing, and it reissues
/// it until the step cap runs out (state#37).
let TOOL_DEFERRED_MESSAGE =
    "Not executed: this turn issued more than one tool call and only the first is run. "
    + "Issue one tool call per turn; you may call this one again on the next turn."

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
    ///
    /// **A CANCELLED TURN APPENDS NOTHING** (state#38). This ran from
    /// `finishProseTurn` after `cancel()` had already stopped the stream, so
    /// a blocking `Stop` hook left an orphan user turn in the transcript that
    /// `continueAgentLoop` then refused to act on -- a message from nobody,
    /// permanently.
    @discardableResult
    func dispatchStopAndContinueIfBlocked(
        chatID: UUID, resumeStep: Int, project: AppProject?
    ) async -> Bool {
        // **THE GOAL LOOP EVALUATES HERE, BEFORE THE USER'S OWN STOP
        // HOOKS** (swift/docs/SWIFT_GOALS.md). CC's goal is a session-
        // scoped prompt-based Stop hook sitting among the blockable
        // turn-end hooks; this is that position. When the goal continues
        // the loop itself -- a not-met verdict or a due check-in -- this
        // round is not a stop, so the user's Stop hooks wait for the turn
        // the goal actually ends. The Bool is "re-entered" either way.
        if activeGoals[chatID] != nil {
            let goalContinued = await handleGoalAtStop(
                chatID: chatID, project: project)
            if goalContinued { return true }
        }
        let wasActive = stopHookReentryCount > 0
        let verdict = await evaluateStop(
            stopHookActive: wasActive, chatID: chatID, project: project)
        // `continue: false` from a Stop hook's JSON output ends the turn and
        // shows the reason to the USER. It is deliberately NOT the
        // block-and-continue path below (that one feeds its reason to the
        // MODEL as the next user turn), so it must not consume one of the
        // 8 re-entries.
        if verdict.preventContinuation {
            stopHookReentryCount = 0
            if let reason = verdict.continuationStopReason, !reason.isEmpty {
                showToast("Turn stopped by hook: \(reason)", style: .warning)
            }
            return false
        }
        guard verdict.isBlocked, stopHookReentryCount < 8, !isCancellationPending else {
            stopHookReentryCount = 0
            return false
        }
        stopHookReentryCount += 1
        let reason = verdict.blockReason ?? "A Stop hook requested the turn continue."
        mutateTurnMessages(for: chatID) {
            $0.append(AppChatMessage(role: .user, content: reason))
        }
        continueAgentLoop(step: resumeStep, chatID: chatID)
        return true
    }

    /// Continues the agent loop if steps remain, otherwise asks `Stop`
    /// hooks whether to keep going anyway (bounded by the 8-cap above,
    /// independent of `maxAutonomousSteps`).
    ///
    /// `project` is the TURN's project, never `selectedProject` (state#30):
    /// the step cap belongs to the workspace the loop is running in, and the
    /// selection is free to move while a call waits for approval.
    func continueOrStop(afterStep currentStep: Int, chatID: UUID, project: AppProject?) async {
        // **A STEER DELIVERS AT THE NEXT STEP BOUNDARY, AND RESETS THE
        // ALLOWANCE ONCE FOR THE BATCH** (opencode v2's default `steer`
        // mode; `deliverSteersAtBoundary` carries the reasoning). A user's
        // fresh input supersedes both the step cap and the Stop evaluation
        // this boundary would otherwise reach: the loop continues from a
        // fresh step 0 with the steer in the transcript, and no `Stop` hook
        // is consulted about a turn the user just added to. Entries that
        // arrive during the FINAL step catch no boundary here and remain
        // the tail drain's.
        if await deliverSteersAtBoundary(chatID: chatID, project: project) {
            continueAgentLoop(step: 0, chatID: chatID)
            return
        }
        let maxSteps = project?.maxAutonomousSteps ?? 5
        if currentStep + 1 < maxSteps {
            continueAgentLoop(step: currentStep + 1, chatID: chatID)
        } else {
            _ = await dispatchStopAndContinueIfBlocked(
                chatID: chatID, resumeStep: currentStep + 1, project: project)
        }
    }

    /// The call/result pair recorded for every call after the first one in a
    /// turn (state#37).
    ///
    /// `extractToolCalls` returns every block the reply contained and the
    /// loop runs exactly one of them; before this, the others were parsed,
    /// counted, and then dropped on the floor with nothing written anywhere.
    func deferredCallRecords(
        _ calls: [AppToolCall]
    ) -> (calls: [AppToolCall], results: [AppToolResult]) {
        var recorded: [AppToolCall] = []
        var results: [AppToolResult] = []
        for call in calls {
            var denied = call
            denied.status = .denied
            recorded.append(denied)
            results.append(
                AppToolResult(
                    callID: call.id,
                    output: TOOL_DEFERRED_MESSAGE,
                    isError: true,
                    durationSeconds: 0.0))
        }
        return (recorded, results)
    }

    // `internal` rather than `private`: exercised directly by
    // AgentLoopRoutingTests / HookDecisionRoutingTests via `@testable
    // import`, since it is otherwise reachable only from inside a live
    // generation `Task` that needs a real model session.
    ///
    /// **`async`, AND THE CALLER MUST AWAIT IT** (state#16). This used to spawn a
    /// detached `Task` and return immediately, so the turn's stream loop fell
    /// straight through to its tail and set `generating = false` while the
    /// tool was still running: `canRun` went true mid-loop (Send started a
    /// second turn that overwrote `runTask`), `canCancel` went false (Stop
    /// was disabled for exactly the duration of a shell command), and
    /// `createChat` / `deleteChat` / `selectProject` / model unload were all
    /// open for the same window. Awaiting it keeps the turn's lifecycle
    /// honest, and puts the tool inside `runTask` so `cancel()` reaches it.
    func handleExtractedToolCall(_ call: AppToolCall, deferred: [AppToolCall] = [], fullContent: String, reasoning: String, result: GenerationResult, currentStep: Int, chatID: UUID, project: AppProject?) async {
        let sessionID = chatID.uuidString
        // Recorded on whichever message this call's own outcome lands on, so
        // the model is told the rule rather than left to infer it from
        // silence (state#37).
        let extra = self.deferredCallRecords(deferred)

        // Evaluate PreToolUse lifecycle hooks first
        let hookDecision = await self.evaluatePreToolUseHooks(
            toolName: call.name, toolArguments: call.arguments, chatID: chatID, project: project)

        // `updatedInput` replaces the corresponding argument keys before
        // anything downstream (permission evaluation, execution, the
        // approval card) sees the call.
        var call = call
        if let updated = hookDecision.updatedInput {
            for (key, value) in updated { call.arguments[key] = value }
        }
        let cmd = call.arguments["command"] ?? call.arguments["cmd"]

        // `continue: false` from a PreToolUse hook's JSON: the call does not
        // run AND the turn ends here with the reason shown to the user --
        // stronger than `.deny`, which records the refusal and lets the
        // loop carry on.
        if hookDecision.preventContinuation {
            let reason = hookDecision.continuationStopReason
                ?? hookDecision.reason
                ?? "A PreToolUse hook stopped the turn."
            await self.recordDeniedCall(
                call, extra: extra, reason: reason, fullContent: fullContent,
                reasoning: reasoning, chatID: chatID, currentStep: currentStep,
                project: project, continuesLoop: false)
            showToast("Turn stopped by hook: \(reason)", style: .warning)
            return
        }

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
            mutateTurnMessages(for: chatID) {
                $0.append(AppChatMessage(
                    role: .assistant,
                    content: fullContent,
                    reasoning: reasoning,
                    stopReason: "tool_use",
                    toolCalls: [deniedCall] + extra.calls,
                    toolResults: [deniedResult] + extra.results
                ))
            }
            await self.continueOrStop(afterStep: currentStep, chatID: chatID, project: project)
            return
        }

        let sessionApproved = await SessionApprovalStore.shared.isApproved(sessionID: sessionID, toolName: call.name, command: cmd)
        // The TURN's project, threaded in from `executeGenerationTurn`
        // (state#30). Every branch below -- run, deny, or park for approval --
        // refers to it, so the policy that produced the decision is the
        // policy the call executes under even if the selection moves while
        // the card is up.
        let decisionProject = project
        let decision = AppToolPermissionEngine.evaluate(
            call: call, project: decisionProject, sessionApproved: sessionApproved,
            fallbackMode: self.activePermissionMode,
            globalServers: self.globalMcpServers)

        // The engine's refusal wins over a hook that merely asked (state#78).
        if case .deny(let reason) = decision {
            // The engine itself denied the call: that is a `PermissionDenied`
            // in Claude Code's contract, so configured hooks hear about it.
            await self.dispatchPermissionDenied(
                toolName: call.name, toolArguments: call.arguments, reason: reason,
                chatID: chatID, project: decisionProject)
            await self.recordDeniedCall(
                call, extra: extra, reason: reason, fullContent: fullContent,
                reasoning: reasoning, chatID: chatID, currentStep: currentStep,
                project: decisionProject)
            return
        }

        // A hook that asked for confirmation is routed into the same
        // pending-approval UI the engine's own `.ask` uses, folding the
        // hook's reason into the call's risk assessment rather than
        // replacing it (T10).
        //
        // **BELOW THE ENGINE, NOT ABOVE IT** (state#78). This returned
        // before `evaluate` ever ran, so a hook's `ask` OUTRANKED a
        // project's category deny: a Strict Read-Only project got an
        // Approve button for a shell command, and `approvePendingToolCall`
        // re-evaluates nothing. A refusal has to survive a hook that only
        // wanted confirmation.
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
            self.pendingToolCallProject = project
            // A hook ask is not a classifier fallback; never render a stale
            // one beside it.
            self.pendingToolCallClassifierNotice = nil
            mutateTurnMessages(for: chatID) {
                $0.append(AppChatMessage(
                    role: .assistant,
                    content: fullContent,
                    reasoning: reasoning,
                    stopReason: "tool_use",
                    toolCalls: [pending] + extra.calls,
                    toolResults: extra.results
                ))
            }
            return
        }

        switch decision {
        case .ask(let assessment, _):
            // `PermissionRequest` fires exactly where this app would
            // otherwise show the approval card, so a hook can resolve
            // `allow`/`deny` without ever surfacing the UI.
            let permVerdict = await self.evaluatePermissionRequest(
                toolName: call.name, toolArguments: call.arguments, chatID: chatID,
                project: decisionProject)

            // `continue: false` resolves the card the way a deny would, and
            // additionally ends the turn rather than letting the loop carry
            // on -- the same treatment the PreToolUse verdict gets.
            if permVerdict.preventContinuation {
                let reason = permVerdict.continuationStopReason
                    ?? "A PermissionRequest hook stopped the turn."
                await self.recordDeniedCall(
                    call, extra: extra, reason: reason, fullContent: fullContent,
                    reasoning: reasoning, chatID: chatID, currentStep: currentStep,
                    project: decisionProject, continuesLoop: false)
                showToast("Turn stopped by hook: \(reason)", style: .warning)
                return
            }

            if permVerdict.permissionDecision == .allow {
                await self.runApprovedCall(call, extra: extra, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep, project: decisionProject)
            } else if permVerdict.permissionDecision == .deny {
                let reason = permVerdict.permissionReason ?? "Denied by PermissionRequest hook"
                await self.recordDeniedCall(call, extra: extra, reason: reason, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep, project: decisionProject)
            } else {
                // **AGENT MODE ROUTES HERE, BEFORE THE CARD** (see
                // `swift/docs/SWIFT_AGENT_MODE.md`). The hook above already
                // declined to decide; the router turns the ask into a run
                // (fast path or classifier allow), a policy refusal, or the
                // ordinary card -- with a fallback notice when the
                // classifier, not the policy, is why a human is seeing it.
                let outcome = await resolveAskUnderAgentMode(
                    call, assessment: assessment, chatID: chatID, project: decisionProject)
                switch outcome {
                case .run(let byClassifier):
                    var approved = call
                    if byClassifier {
                        approved.autoApprovedBy = "classifier"
                    }
                    await self.runApprovedCall(approved, extra: extra, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep, project: decisionProject)
                case .denyWithReason(let reason):
                    await self.dispatchPermissionDenied(
                        toolName: call.name, toolArguments: call.arguments, reason: reason,
                        chatID: chatID, project: decisionProject)
                    await self.recordDeniedCall(
                        call, extra: extra, reason: reason, fullContent: fullContent,
                        reasoning: reasoning, chatID: chatID, currentStep: currentStep,
                        project: decisionProject)
                case .parkCard(let notice):
                    var pending = call
                    pending.status = .pendingApproval
                    pending.riskAssessment = assessment
                    self.pendingToolCall = pending
                    self.pendingToolCallChatID = chatID
                    self.pendingToolCallStep = currentStep
                    self.pendingToolCallProject = decisionProject
                    self.pendingToolCallClassifierNotice = notice
                    mutateTurnMessages(for: chatID) {
                        $0.append(AppChatMessage(
                            role: .assistant,
                            content: fullContent,
                            reasoning: reasoning,
                            stopReason: "tool_use",
                            toolCalls: [pending] + extra.calls,
                            toolResults: extra.results
                        ))
                    }
                }
            }

        case .allow:
            await self.runApprovedCall(call, extra: extra, fullContent: fullContent, reasoning: reasoning, chatID: chatID, currentStep: currentStep, project: decisionProject)

        case .deny:
            // Handled above the hook's `ask`, which is the whole point of
            // state#78. Kept so the switch stays exhaustive over the enum
            // rather than over what this function happens to reach.
            break
        }
    }

    /// Runs an approved call, folds `PostToolUse`'s feedback/context into the
    /// result, appends the turn, and continues the loop. Shared by the
    /// ordinary `.allow` path and a `PermissionRequest` hook resolving
    /// `allow` in place of the approval card.
    private func runApprovedCall(_ call: AppToolCall, extra: (calls: [AppToolCall], results: [AppToolResult]) = ([], []), fullContent: String, reasoning: String, chatID: UUID, currentStep: Int, project: AppProject?) async {
        var runningCall = call
        runningCall.status = .running
        var toolResult = await AppToolRegistry.execute(call: runningCall, in: project, chatID: chatID)
        runningCall.status = toolResult.isError ? .failed : .completed

        let postVerdict = await self.dispatchPostToolUseVerdict(
            toolName: runningCall.name,
            toolArguments: runningCall.arguments,
            toolOutput: toolResult.output,
            toolDurationSeconds: toolResult.durationSeconds,
            isError: toolResult.isError,
            chatID: chatID,
            project: project
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

        mutateTurnMessages(for: chatID) {
            $0.append(AppChatMessage(
                role: .assistant,
                content: fullContent,
                reasoning: reasoning,
                stopReason: "tool_use",
                toolCalls: [runningCall] + extra.calls,
                toolResults: [toolResult] + extra.results
            ))
        }

        // `continue: false` from a PostToolUse hook: the result above is
        // recorded, but the loop does not go on to the next model step --
        // the turn ends here with the reason shown to the user.
        if postVerdict.preventContinuation {
            if let reason = postVerdict.continuationStopReason, !reason.isEmpty {
                showToast("Turn stopped by hook: \(reason)", style: .warning)
            }
            return
        }

        await self.continueOrStop(afterStep: currentStep, chatID: chatID, project: project)
    }

    /// Records a denied call (permission engine or `PermissionRequest` hook)
    /// and continues the loop.
    ///
    /// `continuesLoop: false` is the prevent-continuation shape: the refusal
    /// is still recorded, but the caller ends the turn instead, so no
    /// `continueOrStop` runs.
    private func recordDeniedCall(_ call: AppToolCall, extra: (calls: [AppToolCall], results: [AppToolResult]) = ([], []), reason: String, fullContent: String, reasoning: String, chatID: UUID, currentStep: Int, project: AppProject?, continuesLoop: Bool = true) async {
        let deniedResult = AppToolResult(
            callID: call.id,
            output: "Tool execution denied: \(reason)",
            isError: true,
            durationSeconds: 0.0
        )
        var deniedCall = call
        deniedCall.status = .denied
        mutateTurnMessages(for: chatID) {
            $0.append(AppChatMessage(
                role: .assistant,
                content: fullContent,
                reasoning: reasoning,
                stopReason: "tool_use",
                toolCalls: [deniedCall] + extra.calls,
                toolResults: [deniedResult] + extra.results
            ))
        }
        guard continuesLoop else { return }
        await self.continueOrStop(afterStep: currentStep, chatID: chatID, project: project)
    }

    /// The three names the `agent` tool answers to. One source of truth for
    /// the batch router and its tests; the registry's own `switch` spells
    /// the same three, and a drift test would catch the split if it grew.
    nonisolated static func isAgentFamilyToolName(_ name: String) -> Bool {
        switch name.lowercased() {
        case "agent", "subagent", "task": return true
        default: return false
        }
    }

    /// Routes one reply's extracted calls. **A WHOLE-AGENT BATCH RUNS
    /// TOGETHER; everything else keeps the one-call-per-turn rule** (state#37):
    /// mixing an `agent` call with a `write_file` still runs only the first
    /// and records the rest refused, because ordering a subagent against an
    /// ordinary tool is a dependency the model should spell out across turns.
    func handleExtractedToolCalls(_ calls: [AppToolCall], fullContent: String, reasoning: String, result: GenerationResult, currentStep: Int, chatID: UUID, project: AppProject?) async {
        guard let first = calls.first else { return }
        if calls.count > 1, calls.allSatisfy({ Self.isAgentFamilyToolName($0.name) }) {
            await runConcurrentAgentBatch(
                calls, fullContent: fullContent, reasoning: reasoning, result: result,
                currentStep: currentStep, chatID: chatID, project: project)
        } else {
            await handleExtractedToolCall(
                first, deferred: Array(calls.dropFirst()), fullContent: fullContent,
                reasoning: reasoning, result: result, currentStep: currentStep,
                chatID: chatID, project: project)
        }
    }

    /// Runs every call of an all-agent batch concurrently and records them
    /// all on ONE assistant message, then continues the loop once (the batch
    /// costs one of `maxAutonomousSteps`, not one per subagent).
    ///
    /// Gating is per call and reuses the single-call rules: PreToolUse
    /// hooks, then the permission engine. Every call is EVALUATED before
    /// anything runs, because the verdicts decide the shape:
    /// - all `.allow` -> concurrent execution (the ordinary batch);
    /// - any `.ask` and NOTHING denied or allowed -> the WHOLE batch parks
    ///   under one approval card (`pendingBatchCalls`), because under the
    ///   standard preset `agent` is category `.ask` and a fallback here
    ///   would make the parallel path unreachable under default
    ///   permissions; approving the card runs the batch, denying refuses
    ///   it;
    /// - `.ask` mixed with allow/deny -> the old single-call path (first
    ///   call, rest deferred), since the card machinery parks one shape at a
    ///   time and a mixed batch has no single honest card;
    /// - an engine deny records that one call as refused and lets the rest
    ///   run.
    /// The PreToolUse hooks for the first call run a second time on the
    /// fallback path; they are consultative, and that is the price of not
    /// duplicating the ladder.
    private func runConcurrentAgentBatch(_ calls: [AppToolCall], fullContent: String, reasoning: String, result: GenerationResult, currentStep: Int, chatID: UUID, project: AppProject?) async {
        var prepared: [AppToolCall] = []
        var asked: [(call: AppToolCall, assessment: ToolRiskAssessment)] = []
        // (call, reason, fromEngine) -- an engine deny dispatches
        // PermissionDenied to configured hooks; a hook deny does not, the
        // same split the single path makes.
        var denied: [(call: AppToolCall, reason: String, fromEngine: Bool)] = []
        var updatedCalls: [AppToolCall] = []

        for call in calls {
            let hookDecision = await evaluatePreToolUseHooks(
                toolName: call.name, toolArguments: call.arguments, chatID: chatID, project: project)
            if hookDecision.preventContinuation {
                let reason = hookDecision.continuationStopReason
                    ?? hookDecision.reason
                    ?? "A PreToolUse hook stopped the turn."
                await appendBatchMessage(
                    executed: [], executedResults: [],
                    denied: calls.map { ($0, reason, false) },
                    fullContent: fullContent, reasoning: reasoning, chatID: chatID)
                showToast("Turn stopped by hook: \(reason)", style: .warning)
                return
            }
            var updated = call
            if let newInput = hookDecision.updatedInput {
                for (key, value) in newInput { updated.arguments[key] = value }
            }
            updatedCalls.append(updated)
            if hookDecision.behavior == .deny {
                denied.append((updated, hookDecision.reason ?? "Blocked by PreToolUse hook", false))
                continue
            }
            let sessionApproved = await SessionApprovalStore.shared.isApproved(
                sessionID: chatID.uuidString, toolName: updated.name,
                command: updated.arguments["command"] ?? updated.arguments["cmd"])
            let decision = AppToolPermissionEngine.evaluate(
                call: updated, project: project, sessionApproved: sessionApproved,
                fallbackMode: activePermissionMode, globalServers: globalMcpServers)
            switch decision {
            case .deny(let reason):
                denied.append((updated, reason, true))
            case .ask(let assessment, _):
                asked.append((updated, assessment))
            case .allow:
                prepared.append(updated)
            }
        }

        // **AGENT MODE JUDGES THE ASKED SET BEFORE ANY CARD GOES UP**
        // (`swift/docs/SWIFT_AGENT_MODE.md`), the same routing the single
        // path takes so the two cannot drift. A classifier allow joins the
        // prepared set, a policy block joins the denied set (marked
        // `fromEngine: true` because that flag means "dispatch
        // PermissionDenied to hooks", and a policy refusal is one), and only
        // what the classifier could not decide -- or a hard gate -- remains
        // here to park. A mixed batch after routing falls through to the
        // single path below; its one remaining re-route re-classifies only
        // after unavailable verdicts, which the skip threshold bounds.
        var batchClassifierNotice: String?
        var routedAsked: [(call: AppToolCall, assessment: ToolRiskAssessment)] = []
        for entry in asked {
            let outcome = await resolveAskUnderAgentMode(
                entry.call, assessment: entry.assessment, chatID: chatID, project: project)
            switch outcome {
            case .run(let byClassifier):
                var approved = entry.call
                if byClassifier {
                    approved.autoApprovedBy = "classifier"
                }
                prepared.append(approved)
            case .denyWithReason(let reason):
                denied.append((entry.call, reason, true))
            case .parkCard(let notice):
                if batchClassifierNotice == nil {
                    batchClassifierNotice = notice
                }
                routedAsked.append(entry)
            }
        }
        asked = routedAsked

        // An engine deny is a `PermissionDenied` in Claude Code's contract,
        // so configured hooks hear about it -- at gate time, like the single
        // path does, not at execution time. A classifier policy block is the
        // same shape of verdict and rides the same dispatch.
        for entry in denied where entry.fromEngine {
            await dispatchPermissionDenied(
                toolName: entry.call.name, toolArguments: entry.call.arguments,
                reason: entry.reason, chatID: chatID, project: project)
        }

        // **THE WHOLE BATCH PARKS UNDER ONE CARD.** `pendingToolCall` stays
        // the call the card RENDERS (the one that asked); the batch is what
        // approval acts on.
        if let firstAsk = asked.first, prepared.isEmpty, denied.isEmpty {
            var pending = firstAsk.call
            pending.status = .pendingApproval
            pending.riskAssessment = firstAsk.assessment
            pendingToolCall = pending
            pendingToolCallChatID = chatID
            pendingToolCallStep = currentStep
            pendingToolCallProject = project
            pendingBatchCalls = updatedCalls
            // Batch asks that survive routing are classifier fallbacks or
            // hard gates; the notice (set by the routing below) says which.
            pendingToolCallClassifierNotice = batchClassifierNotice
            let parked = updatedCalls.map { call -> AppToolCall in
                var parked = call
                parked.status = .pendingApproval
                return parked
            }
            mutateTurnMessages(for: chatID) {
                $0.append(AppChatMessage(
                    role: .assistant,
                    content: fullContent,
                    reasoning: reasoning,
                    stopReason: "tool_use",
                    toolCalls: parked,
                    toolResults: []
                ))
            }
            return
        }

        if !asked.isEmpty {
            // Mixed batch: no single honest card. The historical shape.
            //
            // **THE PRIMARY CALL MUST COME FROM `asked`, NEVER FROM
            // `updatedCalls`/`calls` BY POSITION.** `handleExtractedToolCall`
            // re-evaluates whatever call it is given from scratch (hooks,
            // then the engine), so if the batch's first call in original
            // order happened to be one already resolved as `denied` above --
            // whose engine denial already dispatched `PermissionDenied` at
            // the loop a few lines up -- passing it here re-runs the same
            // evaluation, lands on the same `.deny`, and dispatches
            // `PermissionDenied` a SECOND time for one logical denial. Every
            // call in `asked` is, by construction, one the loop above did
            // NOT already resolve, so picking from there cannot re-trigger
            // anything already dispatched.
            let primary = asked[0].call
            await handleExtractedToolCall(
                primary,
                deferred: updatedCalls.filter { $0.id != primary.id },
                fullContent: fullContent,
                reasoning: reasoning, result: result, currentStep: currentStep,
                chatID: chatID, project: project)
            return
        }

        guard !prepared.isEmpty else {
            await appendBatchMessage(
                executed: [], executedResults: [],
                denied: denied.map { ($0.call, $0.reason, $0.fromEngine) },
                fullContent: fullContent, reasoning: reasoning, chatID: chatID)
            await continueOrStop(afterStep: currentStep, chatID: chatID, project: project)
            return
        }

        // **THE SUBAGENTS INTERLEAVE, THEY DO NOT RACE.** Each run's
        // `session.generate` queues on the one session, so N concurrent
        // batches alternate whole turns, correct by the session's own
        // contract. What overlaps for real is tool execution (file, shell)
        // in one run against generation in another.
        await runApprovedAgentBatch(
            prepared, currentStep: currentStep, chatID: chatID, project: project,
            extraDenied: denied.map { ($0.call, $0.reason, $0.fromEngine) },
            fullContent: fullContent, reasoning: reasoning)
    }

    /// Executes already-gated agent calls CONCURRENTLY and records every
    /// outcome. Shared by the batch router (the all-allow case, which
    /// appends the turn) and `approvePendingToolCall` (a parked batch the
    /// user approved, which UPDATES the parked message in place, one call
    /// id at a time -- `appendToolExecutionTurn`'s own rule, state#20).
    func runApprovedAgentBatch(
        _ calls: [AppToolCall], currentStep: Int, chatID: UUID, project: AppProject?,
        extraDenied: [(call: AppToolCall, reason: String, fromEngine: Bool)] = [],
        fullContent: String = "", reasoning: String = "",
        updatesParkedMessage: Bool = false
    ) async {
        var outcomesByID: [UUID: (call: AppToolCall, result: AppToolResult)] = [:]
        await withTaskGroup(of: (AppToolCall, AppToolResult).self) { group in
            for call in calls {
                var running = call
                running.status = .running
                group.addTask {
                    let result = await AppToolRegistry.execute(
                        call: running, in: project, chatID: chatID)
                    return (running, result)
                }
            }
            for await (call, result) in group {
                outcomesByID[call.id] = (call, result)
            }
        }
        let ordered = calls.compactMap { outcomesByID[$0.id] }

        var executedResults: [AppToolResult] = []
        var stopReason: String?
        for (var call, var executedResult) in ordered {
            call.status = executedResult.isError ? .failed : .completed
            let postVerdict = await dispatchPostToolUseVerdict(
                toolName: call.name, toolArguments: call.arguments,
                toolOutput: executedResult.output,
                toolDurationSeconds: executedResult.durationSeconds,
                isError: executedResult.isError, chatID: chatID, project: project)
            if let note = postVerdict.blockReason ?? postVerdict.feedbackMessage, !note.isEmpty {
                executedResult.output += "\n\n<hook_feedback>\n\(note)\n</hook_feedback>"
            }
            if let ctx = postVerdict.additionalContext, !ctx.isEmpty {
                executedResult.output += "\n\n<hook_context>\n\(ctx)\n</hook_context>"
            }
            if postVerdict.preventContinuation, stopReason == nil {
                stopReason = postVerdict.continuationStopReason
            }
            executedResults.append(executedResult)
            outcomesByID[call.id] = (call, executedResult)
        }

        if updatesParkedMessage {
            // Restore the prepared order for the in-place updates.
            for call in calls.compactMap({ outcomesByID[$0.id]?.call }) {
                let match = outcomesByID[call.id]
                appendToolExecutionTurn(
                    call: call, result: match?.result
                        ?? AppToolResult(callID: call.id, output: "", isError: true),
                    chatID: chatID)
            }
        } else {
            let executedCalls = calls.compactMap { outcomesByID[$0.id]?.call }
            await appendBatchMessage(
                executed: executedCalls, executedResults: executedResults,
                denied: extraDenied, fullContent: fullContent, reasoning: reasoning,
                chatID: chatID)
        }

        if let reason = stopReason {
            showToast("Turn stopped by hook: \(reason)", style: .warning)
            return
        }
        await continueOrStop(afterStep: currentStep, chatID: chatID, project: project)
    }

    /// Appends the ONE assistant message a batch produces, pairing every
    /// call with its result (`ToolGroupView` renders the multi-call row).
    private func appendBatchMessage(
        executed: [AppToolCall], executedResults: [AppToolResult],
        denied: [(call: AppToolCall, reason: String, fromEngine: Bool)],
        fullContent: String, reasoning: String, chatID: UUID
    ) async {
        let deniedCalls: [AppToolCall] = denied.map { entry in
            var call = entry.call
            call.status = .denied
            return call
        }
        let deniedResults = denied.map { entry in
            AppToolResult(
                callID: entry.call.id,
                output: "Tool execution denied: \(entry.reason)",
                isError: true,
                durationSeconds: 0.0)
        }
        mutateTurnMessages(for: chatID) {
            $0.append(AppChatMessage(
                role: .assistant,
                content: fullContent,
                reasoning: reasoning,
                stopReason: "tool_use",
                toolCalls: executed + deniedCalls,
                toolResults: executedResults + deniedResults
            ))
        }
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
