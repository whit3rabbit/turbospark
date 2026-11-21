import Foundation

/// Edit-before-run on a pending shell card (`ToolCallCommandEdit`).
///
/// **AN EDIT RUNS NOTHING.** "Use edited command" supersedes the pending
/// call and evaluates a NEW call through the same ladder a model proposal
/// takes (PreToolUse hooks, the permission engine with the network policy,
/// the PermissionRequest hook, the Agent-mode router). A refusal anywhere is
/// recorded as usual. Anything else, an engine or hook `allow` included,
/// stops at a fresh card: the command is user-authored and the owner asked
/// that it get its own explicit approval, once only.
extension AppModel {
    /// Why the slot's pending shell card shows Edit disabled, or nil when
    /// Edit is enabled. Whether Edit shows at all is
    /// `showsPendingCommandEdit(for:)`.
    func pendingCommandEditUnavailableReason(for call: AppToolCall) -> ToolCallCommandEdit.Unavailable? {
        ToolCallCommandEdit.unavailableReason(
            batchCount: pendingBatchCalls?.count, hasValidationLeg: pendingValidationCall != nil)
    }

    /// Whether the card for `call` shows an Edit control at all (enabled or
    /// not): the slot's own pending shell call.
    func showsPendingCommandEdit(for call: AppToolCall) -> Bool {
        pendingToolCall?.id == call.id && ToolCallCommandEdit.isEditableTool(call)
    }

    /// Replaces the pending shell call with the user's edit and evaluates it
    /// from scratch. Returns false, changing nothing, when the edit is a
    /// no-op (empty or unchanged after trimming), the call is not the one in
    /// the slot, a turn is running, or editing is unavailable for it.
    @discardableResult
    public func submitEditedPendingToolCall(id: UUID, command text: String) -> Bool {
        // The approve path's guard (state#76): this installs its own
        // `toolExecutionTask` and raises `generating`.
        guard !generating, !submitting else { return false }
        guard let original = pendingToolCall, original.id == id,
            ToolCallCommandEdit.isEditableTool(original),
            pendingCommandEditUnavailableReason(for: original) == nil,
            let edited = ToolCallCommandEdit.editedCall(from: original, text: text)
        else { return false }

        let chatID = pendingToolCallChatID ?? selectedChatID
        let originStep = pendingToolCallStep
        // The LIVE project, as a parked card's replay uses: the edit is a new
        // decision point, so a policy tightened while the card waited applies.
        let captured = pendingToolCallProject ?? selectedProject
        let live = liveProject(matching: captured)
        clearPendingToolCall()

        var superseded = original
        superseded.status = .denied
        superseded.approvalRecord = ToolApprovalRecord(decision: .editedByUser, actor: .user)
        appendToolExecutionTurn(
            call: superseded,
            result: ToolCallCommandEdit.supersededResult(
                for: original, editedCommand: edited.shellCommand ?? ""),
            chatID: chatID)

        // Same lifecycle as approve: the evaluation awaits hooks, so Send
        // must not start a second turn beside it, and the shared tail lowers
        // the flags (an approval card parks the run rather than ending it).
        generating = true
        claimLane(for: chatID)
        isCancellationPending = false
        generationEpoch += 1
        let myEpoch = generationEpoch

        toolExecutionTask = Task {
            defer {
                self.finishTurnTail(myEpoch: myEpoch, turnChatID: chatID, session: self.session)
                self.toolExecutionTask = nil
            }
            if live.gone {
                await self.recordEditedCallDenied(
                    edited, after: original.id,
                    reason: "The project this call was proposed in no longer exists.",
                    chatID: chatID, step: originStep, project: nil)
                return
            }
            await self.evaluateEditedCall(
                edited, after: original.id, chatID: chatID, step: originStep,
                project: live.project ?? captured)
        }
        return true
    }

    /// The proposal ladder for an edited call, minus every path that runs
    /// it: each would-run outcome parks the card instead.
    private func evaluateEditedCall(
        _ edited: AppToolCall, after originalID: UUID, chatID: UUID, step: Int, project: AppProject?
    ) async {
        var call = edited
        call.riskAssessment = ToolRiskClassifier.assessRisk(
            name: call.name, arguments: call.arguments, projectURL: project?.rootDirectoryURL)

        let hook = await evaluatePreToolUseHooks(
            toolName: call.name, toolArguments: call.arguments, chatID: chatID, project: project)
        if let updated = hook.updatedInput {
            for (key, value) in updated { call.arguments[key] = value }
            // Re-rate, as the proposal path does: a rewrite must not be
            // judged by the rating of the text it replaced.
            call.riskAssessment = ToolRiskClassifier.assessRisk(
                name: call.name, arguments: call.arguments, projectURL: project?.rootDirectoryURL)
        }
        if hook.preventContinuation {
            let reason = hook.continuationStopReason ?? hook.reason ?? "A PreToolUse hook stopped the turn."
            await recordEditedCallDenied(
                call, after: originalID, reason: reason, chatID: chatID, step: step,
                project: project, continuesLoop: false)
            showToast(AppStrings.localized("Turn stopped by hook: \(reason)"), style: .warning)
            return
        }
        if hook.behavior == .deny {
            await recordEditedCallDenied(
                call, after: originalID, reason: hook.reason ?? "Blocked by PreToolUse hook",
                chatID: chatID, step: step, project: project)
            return
        }

        let sessionID = chatID.uuidString
        let sessionApproved = await SessionApprovalStore.shared.isApproved(
            sessionID: sessionID, toolName: call.name, command: call.shellCommand)
        let hostApproved = await SessionApprovalStore.shared.isHostApproved(
            sessionID: sessionID, host: AppNetworkPolicy.host(forWebCall: call))
        let decision = AppToolPermissionEngine.evaluate(
            call: call, project: project, sessionApproved: sessionApproved,
            fallbackMode: activePermissionMode, globalServers: globalMcpServers,
            browserContext: browserPermissionContext(for: call, project: project),
            networkHostApproved: hostApproved)

        let fallback = call.riskAssessment
            ?? ToolRiskClassifier.assessRisk(
                name: call.name, arguments: call.arguments, projectURL: project?.rootDirectoryURL)
        switch decision {
        case .deny(let reason):
            _ = await dispatchPermissionDenied(
                toolName: call.name, toolArguments: call.arguments, reason: reason,
                chatID: chatID, project: project)
            await recordEditedCallDenied(
                call, after: originalID, reason: reason, chatID: chatID, step: step, project: project)
            return
        case .allow:
            // **THE NO-AUTO-RUN GUARD.** The engine would run a model call
            // here (auto mode, a session grant). An edit still asks
            // (mutation-checked in ToolCallCommandEditTests).
            parkEditedCall(call, after: originalID, assessment: fallback, notice: nil,
                           chatID: chatID, step: step, project: project)
            return
        case .ask(let assessment, _):
            if hook.behavior == .ask {
                let withHook = ToolRiskAssessment(
                    level: assessment.level, category: assessment.category,
                    reasons: assessment.reasons + [
                        hook.reason ?? "A PreToolUse hook requested confirmation before this call runs."
                    ],
                    hardGated: assessment.hardGated)
                parkEditedCall(call, after: originalID, assessment: withHook, notice: nil,
                               chatID: chatID, step: step, project: project)
                return
            }
            let verdict = await evaluatePermissionRequest(
                toolName: call.name, toolArguments: call.arguments, chatID: chatID, project: project)
            if verdict.preventContinuation {
                let reason = verdict.continuationStopReason ?? "A PermissionRequest hook stopped the turn."
                await recordEditedCallDenied(
                    call, after: originalID, reason: reason, chatID: chatID, step: step,
                    project: project, continuesLoop: false)
                showToast(AppStrings.localized("Turn stopped by hook: \(reason)"), style: .warning)
                return
            }
            if verdict.permissionDecision == .deny {
                await recordEditedCallDenied(
                    call, after: originalID,
                    reason: verdict.permissionReason ?? "Denied by PermissionRequest hook",
                    chatID: chatID, step: step, project: project)
                return
            }
            if verdict.permissionDecision == .allow {
                // A hook allow would run a model call; an edit still asks.
                parkEditedCall(call, after: originalID, assessment: assessment, notice: nil,
                               chatID: chatID, step: step, project: project)
                return
            }
            switch await resolveAskUnderAgentMode(
                call, assessment: assessment, chatID: chatID, project: project)
            {
            case .denyWithReason(let reason):
                _ = await dispatchPermissionDenied(
                    toolName: call.name, toolArguments: call.arguments, reason: reason,
                    chatID: chatID, project: project)
                await recordEditedCallDenied(
                    call, after: originalID, reason: reason, chatID: chatID, step: step, project: project)
            case .run:
                // A fast path or classifier allow; an edit still asks.
                parkEditedCall(call, after: originalID, assessment: assessment, notice: nil,
                               chatID: chatID, step: step, project: project)
            case .parkCard(let notice):
                parkEditedCall(call, after: originalID, assessment: assessment, notice: notice,
                               chatID: chatID, step: step, project: project)
            }
        }
    }

    /// Puts the edited call in the slot and on the original's message, right
    /// after the superseded call it replaces.
    private func parkEditedCall(
        _ call: AppToolCall, after originalID: UUID, assessment: ToolRiskAssessment,
        notice: String?, chatID: UUID, step: Int, project: AppProject?
    ) {
        var pending = call
        pending.status = .pendingApproval
        pending.riskAssessment = assessment
        pendingToolCall = pending
        pendingToolCallChatID = chatID
        pendingToolCallStep = step
        pendingToolCallProject = project
        pendingValidationCall = nil
        pendingToolCallClassifierNotice = notice
        insertEditedCall(pending, after: originalID, chatID: chatID)
    }

    /// Records a refused edit (with both commands in the result) and goes on
    /// the way any refused proposal does.
    private func recordEditedCallDenied(
        _ call: AppToolCall, after originalID: UUID, reason: String, chatID: UUID, step: Int,
        project: AppProject?, continuesLoop: Bool = true
    ) async {
        var denied = call
        denied.status = .denied
        // The user expected a card; say the policy, not they, refused it.
        denied.approvalRecord = ToolApprovalRecord(decision: .denied, actor: .policy)
        insertEditedCall(denied, after: originalID, chatID: chatID)
        appendToolExecutionTurn(
            call: denied,
            result: ToolCallCommandEdit.annotate(
                AppToolResult(
                    callID: denied.id, output: "Tool execution denied: \(reason)",
                    isError: true, durationSeconds: 0.0),
                for: denied),
            chatID: chatID)
        guard continuesLoop else { return }
        await continueOrStop(afterStep: step, chatID: chatID, project: project)
    }

    /// Inserts `call` after `originalID` on the message that holds it, or
    /// updates it in place when already there. Falls back to a new row if
    /// the original vanished (the chat was rewound meanwhile), so a parked
    /// card is never invisible.
    private func insertEditedCall(_ call: AppToolCall, after originalID: UUID, chatID: UUID) {
        let messages = turnMessages(for: chatID)
        guard let index = messages.lastIndex(where: { $0.toolCalls.contains { $0.id == originalID } }) else {
            mutateTurnMessages(for: chatID) {
                $0.append(AppChatMessage(
                    role: .assistant, content: "", reasoning: "", stopReason: "tool_use",
                    toolCalls: [call], toolResults: []))
            }
            return
        }
        mutateTurnMessages(for: chatID) { messages in
            guard index < messages.count else { return }
            if let existing = messages[index].toolCalls.firstIndex(where: { $0.id == call.id }) {
                messages[index].toolCalls[existing] = call
            } else if let position = messages[index].toolCalls.firstIndex(where: { $0.id == originalID }) {
                messages[index].toolCalls.insert(call, at: position + 1)
            }
        }
    }
}
