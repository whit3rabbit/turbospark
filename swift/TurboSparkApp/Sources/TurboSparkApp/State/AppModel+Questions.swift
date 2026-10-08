import Foundation

/// The interactive AskUserQuestion flow (qwen-code parity): when the model
/// asks, the turn SUSPENDS inside the tool until the user picks. The
/// executor parks its continuation here; this file owns the published state
/// the transcript card reads and the submit/dismiss actions it calls.
extension AppModel {
    /// One parked question set. `toolCallID` is what the transcript card
    /// matches its interactive controls against, so only the asking call
    /// renders tappable options.
    public struct PendingUserQuestions: Identifiable, Equatable {
        public let id: UUID
        public let chatID: UUID?
        public let toolCallID: UUID?
        public let items: [UserQuestionItem]
    }

    /// The app-side `answerWaiter`: publishes the questions, parks until
    /// `submitUserQuestionAnswers` or `dismissUserQuestions` fires, and
    /// returns the tool-result text either way. Runs inside the tool's
    /// execution task, so it must hop to the main actor to publish.
    func waitForUserAnswers(
        chatID: UUID?, items: [UserQuestionItem], toolCallID: UUID?
    ) async -> String {
        let mine = PendingUserQuestions(
            id: UUID(), chatID: chatID, toolCallID: toolCallID, items: items)
        pendingUserQuestions = mine
        let answer = await withCheckedContinuation { continuation in
            AskUserQuestionExecutor.waitForAnswer(chatID: chatID, continuation: continuation)
        }
        // Clear only OUR card. A second AskUserQuestion for the same chat
        // retires this call's continuation (auto-dismissed) and publishes its
        // own card; this call resuming afterwards must not erase that newer
        // card, which would park the second waiter with no UI.
        if pendingUserQuestions?.id == mine.id {
            pendingUserQuestions = nil
        }
        return answer
    }

    /// Delivers the user's picks to the parked tool call. `answers` maps a
    /// question header to the chosen label(s); the card builds it from the
    /// rendered options.
    public func submitUserQuestionAnswers(_ answers: [String: String]) {
        guard let pending = pendingUserQuestions else { return }
        pendingUserQuestions = nil
        AskUserQuestionExecutor.submitAnswer(chatID: pending.chatID, answers: answers)
    }

    /// Dismisses a parked question set: the model is told the user declined
    /// to answer, which is a real answer it can act on, never a hang.
    public func dismissUserQuestions() {
        guard let pending = pendingUserQuestions else { return }
        pendingUserQuestions = nil
        AskUserQuestionExecutor.submitAnswer(
            chatID: pending.chatID, answers: [:], dismissed: true)
    }
}
