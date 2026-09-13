import Foundation
import TurboSpark

/// What a subagent is allowed to run, and what it is told when it is not.
///
/// **A SUBAGENT HAS NO APPROVAL UI, WHICH MAKES THIS THE WHOLE GATE.** The
/// main loop can raise a card and wait for a human; an isolated run cannot,
/// so `.ask` DENIES here rather than surfacing one -- the alternative is a
/// hidden prompt nobody answers, or worse, treating "would have asked" as
/// "may proceed" (state#18).
///
/// Split out of the loop for the reason the loop's own comment already gave
/// about `observation`: `run` needs a live model session, so a test over
/// `permissionRefusal` alone stays green with the check deleted from the
/// loop entirely, and the CALL SITE is what has to be reachable. It is a
/// file of its own now because it is what changes: state#68 (the hooks) and
/// state#74 (the message role) both landed here, on a path that had been
/// fixed separately three times.
extension SubagentRunner {
    /// Runs one proposed call and returns the observation to feed back.
    ///
    /// Split out of the loop so the GATE'S CALL SITE is testable and not just
    /// the gate: `run` needs a live model session, so a test over
    /// `permissionRefusal` alone stays green with the check deleted from the
    /// loop entirely, which is the exact defect this is guarding.
    ///
    /// **A SUBAGENT'S TOOL CALLS RUN THE LIFECYCLE HOOKS TOO** (state#68).
    /// This gated on `isToolAllowed`, `permissionRefusal` and the depth
    /// counter and then executed, while the main loop additionally runs
    /// `PreToolUse` and `PostToolUse` (`AppModel+AgentLoop.swift`). A deny
    /// hook -- which state#40 made fail CLOSED precisely so it can be relied
    /// on -- was therefore bypassed on the one path that runs unattended for
    /// `maxTurns` turns. `PermissionRequest` is deliberately NOT dispatched:
    /// it fires where the approval card would go up, and a subagent refuses
    /// `.ask` rather than surfacing one (state#18).
    ///
    /// Hook discovery is captured from this run's project for each dispatch;
    /// selection-global hook state is not authoritative for background work.
    static func observation(
        for call: AppToolCall, agent: AppAgentDefinition, project: AppProject?,
        chatID: UUID? = nil, depth: Int = 0, session: TurboSparkSession? = nil
    ) async -> ChatMessage {
        if !agent.isToolAllowed(call.name) {
            return errorObservation(
                "Tool '\(call.name)' is disallowed for agent profile '\(agent.name)'.")
        }
        // **THE NESTING BOUND IS ENFORCED WHERE THE CALL IS SEEN, NOT WHERE
        // THE RUN STARTS** (state#47). `run`'s own guard catches a run that
        // was started too deep; this catches the call that would start it,
        // and reports the reason to the model rather than letting it read a
        // failed subagent as a tool that broke.
        if AppModel.isAgentFamilyToolName(call.name) {
            guard depth < maxSubagentDepth else {
                return errorObservation(
                    "Refused: subagents may nest at most \(maxSubagentDepth) deep and this one "
                        + "is already at \(depth). Do the work yourself or report back.")
            }
        }

        let sessionID = chatID?.uuidString ?? "subagent"
        let projectDirectory = project?.rootDirectoryPath
        let hookDecision = await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: sessionID,
            toolName: call.name,
            toolArguments: call.arguments,
            workingDirectory: projectDirectory,
            projectBoundHookDirectory: projectDirectory)

        // `updatedInput` is applied BEFORE the permission gate, not after:
        // the rewritten command is what would run, so it is what has to be
        // evaluated. The main loop makes the same ordering choice.
        var call = call
        if let updated = hookDecision.updatedInput {
            for (key, value) in updated { call.arguments[key] = value }
        }
        if hookDecision.behavior == .deny {
            let reason = hookDecision.reason ?? "Blocked by PreToolUse hook"
            return errorObservation("Tool execution blocked by hook: \(reason)")
        }
        if hookDecision.behavior == .ask {
            let reason = hookDecision.reason ?? "a PreToolUse hook requested confirmation"
            return errorObservation(
                "Tool '\(call.name)' needs interactive approval (\(reason)), and a subagent runs "
                    + "with no approval UI. Ask the user to run this call in the main "
                    + "conversation.")
        }
        if let refusal = await permissionRefusal(
            for: call, project: project, session: session, chatID: chatID)
        {
            return errorObservation(refusal)
        }

        let toolResult = await AppToolRegistry.execute(
            call: call, in: project, chatID: chatID, subagentDepth: depth)

        var results = await AppHookExecutionEngine.shared.dispatch(
            event: .postToolUse,
            sessionID: sessionID,
            toolName: call.name,
            toolArguments: call.arguments,
            toolOutput: toolResult.output,
            toolDurationSeconds: toolResult.durationSeconds,
            isError: toolResult.isError,
            workingDirectory: projectDirectory,
            projectBoundHookDirectory: projectDirectory)
        if toolResult.isError {
            results += await AppHookExecutionEngine.shared.dispatch(
                event: .postToolUseFailure,
                sessionID: sessionID,
                toolName: call.name,
                toolArguments: call.arguments,
                toolOutput: toolResult.output,
                toolDurationSeconds: toolResult.durationSeconds,
                isError: true,
                workingDirectory: projectDirectory,
                projectBoundHookDirectory: projectDirectory)
        }
        let postVerdict = AppHookDecisionAggregator.aggregate(results, event: .postToolUse)
        var output = toolResult.output
        // Feedback, never a block: the tool already ran. Same folding the
        // main loop's `runApprovedCall` does.
        if let note = postVerdict.blockReason ?? postVerdict.feedbackMessage, !note.isEmpty {
            output += "\n\n<hook_feedback>\n\(note)\n</hook_feedback>"
        }
        if let ctx = postVerdict.additionalContext, !ctx.isEmpty {
            output += "\n\n<hook_context>\n\(ctx)\n</hook_context>"
        }

        let tag = toolResult.isError ? "tool_error" : "tool_response"
        return ChatMessage(role: .tool, content: "<\(tag)>\n\(output)\n</\(tag)>")
    }

    /// A refusal fed back to the subagent.
    ///
    /// **`.tool`, NOT `.system`** (state#74, which is state#32 on this path).
    /// The Gemma, ChatML and DeepSeek fallback renderers refuse a mid-history
    /// system message outright and `fit_window` prices a failing render at
    /// `u64::MAX`, so it drops turns until the render stops failing -- the run
    /// silently loses its own history rather than reporting anything.
    static func errorObservation(_ message: String) -> ChatMessage {
        ChatMessage(role: .tool, content: "<tool_error>\n\(message)\n</tool_error>")
    }

    /// The reason a subagent may not run `call`, or nil when it may.
    ///
    /// **A SUBAGENT IS NOT EXEMPT FROM THE PERMISSION GATE** (state#18). This loop went
    /// straight to `AppToolRegistry.execute` after checking only
    /// `agent.isToolAllowed`, which is a tool NAME list -- and the built-in
    /// `general-purpose` agent declares no `disallowedTools` at all. So under
    /// a project whose terminal permission is `.ask` or `.deny`, a subagent
    /// reached from the `agent` tool or from `/explore` ran `/bin/zsh -c`
    /// unprompted, up to `maxTurns` times, on the strength of ONE approval of
    /// the agent call itself (swift/CLAUDE.md Gotchas 11 and 29).
    ///
    /// An isolated run has no UI to prompt with, so `.ask` DENIES rather than
    /// surfacing a card: the alternative is a hidden prompt nobody answers, or
    /// worse, treating "would have asked" as "may proceed". The terminal
    /// allowlist runs on top of that, so an auto-mode project still only
    /// auto-runs what `isAutoApprovable` accepts.
    ///
    /// **AGENT MODE REPLACES THE ASK WITH A VERDICT** (`swift/docs/
    /// SWIFT_AGENT_MODE.md`). A subagent cannot raise the card Agent mode
    /// would show, but it CAN ask the classifier -- which is the one form of
    /// scrutiny that needs no UI. A classifier allow runs; a policy block
    /// refuses with the same message the main loop feeds its model; an
    /// unavailable verdict refuses exactly as before. The same routing rules
    /// apply (`AgentModeRouting`): hard gates and suspended or skipping
    /// sessions never reach the classifier.
    static func permissionRefusal(
        for call: AppToolCall, project: AppProject?,
        session: TurboSparkSession? = nil, chatID: UUID? = nil
    ) async -> String? {
        let askRefusal: (String) -> String = { reason in
            "Tool '\(call.name)' needs interactive approval (\(reason)), and a subagent "
                + "runs with no approval UI. Ask the user to run this call in the main "
                + "conversation, or widen the project's permissions."
        }
        switch AppToolPermissionEngine.evaluate(call: call, project: project, sessionApproved: false) {
        case .deny(let reason):
            return "Tool '\(call.name)' is denied by project permissions: \(reason)"
        case .ask(let assessment, let reason):
            // Rootless subagents resolve like the rootless chat default
            // (`.auto`), which is what the main loop's router falls back to
            // as well; only a project can select Agent mode.
            let mode = project?.permissions.mode ?? AppPermissionMode.auto
            guard mode == .agentAuto else { return askRefusal(reason) }
            let sessionID = chatID?.uuidString ?? "subagent"
            let decision = AgentModeRouting.preClassifierDecision(
                call: call, assessment: assessment,
                suspended: await AgentModeGate.shared.isSuspended(sessionID: sessionID),
                skipClassifier: await AgentModeGate.shared.shouldSkipClassifier(sessionID: sessionID))
            switch decision {
            case .manualCard:
                return askRefusal(reason)
            case .fastAllow:
                // The static ladder vouched; the positive allowlist gate
                // below reads the same allowlist and reaches the same nil.
                return nil
            case .classify:
                let hints = MacAppSettingsFileStore.load().agentModeHints
                let classifier = LocalModelToolClassifier(
                    session: session, hints: hints,
                    workspaceRoot: project?.rootDirectoryPath ?? "")
                let request = ClassifierRequest(
                    toolName: call.name, category: call.category,
                    projectedCall: ToolCallProjection.projectedCall(call),
                    // No parent transcript is reachable from here; the call
                    // is judged on policy, which is the conservative reading.
                    recentUserIntent: "")
                switch await classifier.classify(request) {
                case .allow:
                    await AgentModeGate.shared.recordAllow(sessionID: sessionID)
                    // **THE CLASSIFIER IS THE POSITIVE GATE IN THIS MODE.**
                    // Its allow is a context-aware verdict -- strictly more
                    // scrutiny than `isAutoApprovable`, which is why the
                    // terminal allowlist gate below is deliberately skipped
                    // on this return rather than re-refusing what the
                    // classifier just judged.
                    return nil
                case .block(let blockReason):
                    await AgentModeGate.shared.recordBlock(sessionID: sessionID)
                    return AgentModeRouting.blockMessage(reason: blockReason)
                case .unavailable:
                    await AgentModeGate.shared.recordUnavailable(sessionID: sessionID)
                    return askRefusal(reason)
                }
            }
        case .allow:
            break
        }

        // The positive gate from swift/CLAUDE.md Gotcha 29. It was written
        // because `permissive` returned `.allow` from `evaluate` before the
        // high-risk gate ran, so that mode alone would let a subagent run
        // `rm -rf ~` unattended; state#46 moved the mode below that gate, so
        // the engine no longer has the hole this was compensating for.
        //
        // **KEPT ANYWAY, AND NOT AS BELT-AND-BRACES.** A subagent cannot ask,
        // so `.ask` is a refusal here rather than a prompt, and the engine's
        // `.auto` and `.permissive` arms both return `.allow` for everything
        // the DENYLIST does not score `.high`. Gotcha 29 records 18 of 23
        // corpus strings surviving that denylist, so the positive allowlist
        // is what actually bounds an unattended shell -- a different question
        // from the one `evaluate` answers.
        if call.category == .terminal,
            let command = call.arguments["command"] ?? call.arguments["cmd"],
            !TerminalCommandClassifier.isAutoApprovable(command)
        {
            return "Command '\(command)' is not on the auto-approvable allowlist and a subagent "
                + "cannot ask. Run it in the main conversation instead."
        }
        return nil
    }
}
