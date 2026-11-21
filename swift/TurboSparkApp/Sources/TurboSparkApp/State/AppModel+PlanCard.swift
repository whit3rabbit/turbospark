import Foundation

extension AppModel {
    /// Whether the selected chat is idle enough for the plan card's buttons:
    /// nothing generating, no hook-awaiting submission, no open, and no
    /// approval card. The same busy terms as `canRunTerms`, minus input.
    var isIdleForPlanCard: Bool {
        !generating && !submitting && !opening && pendingToolCall == nil
    }

    /// The plan card's Start: an ordinary user reply, sent through the same
    /// submission path as the composer, so every tool approval and the
    /// permission mode apply exactly as if the user had typed it.
    ///
    /// A draft in the composer is never replaced: the reply is appended to it
    /// and left for the user to send, the same rule as `composeInChat`.
    func startPlanFromCard() {
        sendReplyFromCard(AppStrings.localized("Start on the plan."))
    }

    /// Sends a card's canned reply as if typed in the composer (shared with
    /// the failed-call recovery footer), under the draft rule above.
    func sendReplyFromCard(_ reply: String) {
        let hasDraft = !promptText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || !promptAttachments.isEmpty
        guard !hasDraft else {
            composeInChat(reply)
            return
        }
        writePromptTextDirectly(reply)
        // No model loaded (or something just became busy): leave the reply
        // drafted and focused rather than dropping it.
        guard canRun else {
            NotificationCenter.default.post(name: .focusPrompt, object: nil)
            return
        }
        run()
    }

    /// The plan card's Request changes: drafts a lead-in and focuses the
    /// composer. Nothing is sent, so the user's own words carry the change.
    func requestPlanChangesFromCard() {
        composeInChat(AppStrings.localized("Change the plan:") + " ")
    }

    // MARK: - Blocking gate

    /// One parked ExitPlanMode call. `toolCallID` keys the executor's
    /// continuation together with the chat id.
    public struct PendingPlanApproval: Identifiable, Equatable {
        public let id: UUID
        public let chatID: UUID?
        public let toolCallID: UUID?
        public let plan: String
    }

    /// The app-side `PlanModeExecutor.decisionWaiter`: publishes the plan,
    /// parks until Start, Request changes or a cancel, and returns the
    /// decision. Mirrors `waitForUserAnswers`, including holding the lane,
    /// so Stop, quit and the lane's chat-deletion guard behave the same.
    func waitForPlanDecision(
        chatID: UUID?, plan: String?, toolCallID: UUID?
    ) async -> PlanModeExecutor.Decision {
        let mine = PendingPlanApproval(
            id: UUID(), chatID: chatID, toolCallID: toolCallID, plan: plan ?? "")
        pendingPlanApproval = mine
        if let chatID { turnCoordinator.park(chatID, as: .awaitingQuestion) }
        if let chatID { raiseAttention(.needsPermission, chatID: chatID) }
        let decision = await withCheckedContinuation { continuation in
            PlanModeExecutor.waitForDecision(
                chatID: chatID, toolCallID: toolCallID, continuation: continuation)
        }
        // Clear only OUR card, the same rule as the question waiter.
        if pendingPlanApproval?.id == mine.id {
            pendingPlanApproval = nil
        }
        if let chatID, turnCoordinator.parked[chatID] == .awaitingQuestion {
            turnCoordinator.unpark(chatID)
        }
        return decision
    }

    /// Start on the waiting plan card: resumes ExitPlanMode as approved,
    /// which ends plan mode. No-op without a parked gate (double click).
    func approvePendingPlan() {
        guard let pending = pendingPlanApproval else { return }
        pendingPlanApproval = nil
        PlanModeExecutor.submitDecision(
            chatID: pending.chatID, toolCallID: pending.toolCallID, decision: .approved)
    }

    /// Request changes on the waiting plan card: resumes ExitPlanMode with
    /// the user's text and leaves plan mode on. Blank text is ignored so the
    /// card stays up rather than sending an empty request.
    func requestChangesToPendingPlan(_ text: String) {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, let pending = pendingPlanApproval else { return }
        pendingPlanApproval = nil
        PlanModeExecutor.submitDecision(
            chatID: pending.chatID, toolCallID: pending.toolCallID,
            decision: .changesRequested(trimmed))
    }
}
