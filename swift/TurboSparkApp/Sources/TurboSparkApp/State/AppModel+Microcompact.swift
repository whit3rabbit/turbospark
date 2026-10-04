import Foundation
import TurboSpark

extension AppModel {
    var selectedCompactionBoundaryEvent: CompactionBoundaryEvent? {
        compactionBoundaryEvents[selectedChatID]
    }

    func clearCompactionBoundaryEvent(for chatID: UUID) {
        compactionBoundaryEvents.removeValue(forKey: chatID)
    }

    /// Projects eligible stale tool output before fitting the request window.
    /// The transient notice stays outside both the transcript and model history.
    func fitRequestHistoryAfterMicrocompact<Result>(
        chatID: UUID,
        history: AppChatHistoryProjection,
        countTokens: ([ChatMessage]) async throws -> Int,
        fitWindow: ([ChatMessage]) async throws -> Result
    ) async throws -> Result {
        clearCompactionBoundaryEvent(for: chatID)

        var requestMessages = history.messages
        if microcompactEnabled {
            let outcome = await AppChatMicrocompact.apply(
                history: history,
                keepRecent: compactionKeepRecentTurns,
                minimumSavingsTokens: microcompactMinimumSavingsTokens,
                countTokens: countTokens)
            requestMessages = outcome.history.messages
            if outcome.rowsCleared > 0 {
                compactionBoundaryEvents[chatID] = CompactionBoundaryEvent(
                    chatID: chatID,
                    rowsCleared: outcome.rowsCleared,
                    estimatedTokensSaved: outcome.estimatedTokensSaved)
            }
        }

        return try await fitWindow(requestMessages)
    }
}
