import Foundation
import TurboSpark

struct MicrocompactOutcome {
    var history: AppChatHistoryProjection
    var rowsCleared: Int
    var estimatedTokensSaved: Int
}

/// Request-only pruning for stale tool output. Transcript rows remain untouched.
enum AppChatMicrocompact {
    private static let placeholder = "[Tool output omitted from model context.]"

    static func apply(
        history: AppChatHistoryProjection,
        keepRecent: Int,
        minimumSavingsTokens: Int,
        countTokens: ([ChatMessage]) async throws -> Int
    ) async -> MicrocompactOutcome {
        let unchanged = MicrocompactOutcome(
            history: history,
            rowsCleared: 0,
            estimatedTokensSaved: 0)

        guard history.messages.count == history.sourceRowIndexByMessage.count,
              history.sourceTranscriptRowCount >= 0 else {
            return unchanged
        }

        for rowIndex in history.sourceRowIndexByMessage.compactMap({ $0 }) {
            guard rowIndex >= 0, rowIndex < history.sourceTranscriptRowCount else {
                return unchanged
            }
        }

        let retainedRows = AppChatCompaction.clampKeepRecent(keepRecent)
        let cutoff = max(0, history.sourceTranscriptRowCount - retainedRows)
        var projectedMessages = history.messages
        var changedSourceRows = Set<Int>()

        for messageIndex in history.messages.indices {
            guard history.messages[messageIndex].role == .tool,
                  let sourceRowIndex = history.sourceRowIndexByMessage[messageIndex],
                  sourceRowIndex < cutoff,
                  let compactedContent = compactedToolContent(history.messages[messageIndex])
            else {
                continue
            }

            projectedMessages[messageIndex].content = compactedContent
            changedSourceRows.insert(sourceRowIndex)
        }

        guard !changedSourceRows.isEmpty else { return unchanged }

        let originalTokenCount: Int
        let projectedTokenCount: Int
        do {
            originalTokenCount = try await countTokens(history.messages)
            projectedTokenCount = try await countTokens(projectedMessages)
        } catch {
            return unchanged
        }

        guard originalTokenCount >= 0,
              projectedTokenCount >= 0,
              projectedTokenCount <= originalTokenCount else {
            return unchanged
        }

        let savings = originalTokenCount - projectedTokenCount
        guard savings > 0, savings >= max(0, minimumSavingsTokens) else {
            return unchanged
        }

        var projectedHistory = history
        projectedHistory.messages = projectedMessages
        return MicrocompactOutcome(
            history: projectedHistory,
            rowsCleared: changedSourceRows.count,
            estimatedTokensSaved: savings)
    }

    /// What an old tool result is replaced with, or nil to leave it.
    ///
    /// **A NATIVE RESULT IS SENT BARE.** The checkpoint's template wraps a
    /// tool message itself, so the app does not add `<tool_response>`; the
    /// tag test below would never match one, and old native results would
    /// stay in the prompt forever. What marks them is the call id they carry.
    /// The id, the tool name and the message's role are untouched either way:
    /// only the body is replaced, so the template can still pair the (now
    /// short) result with the call that produced it.
    private static func compactedToolContent(_ message: ChatMessage) -> String? {
        if message.toolCallId != nil {
            return message.content == placeholder ? nil : placeholder
        }
        let content = message.content
        for tag in ["tool_response", "tool_error"] {
            let opening = "<\(tag)>\n"
            let closing = "\n</\(tag)>"
            guard content.hasPrefix(opening), content.hasSuffix(closing) else { continue }

            let bodyStart = content.index(content.startIndex, offsetBy: opening.count)
            let bodyEnd = content.index(content.endIndex, offsetBy: -closing.count)
            let body = content[bodyStart..<bodyEnd]
            guard body != placeholder else { return nil }
            return opening + placeholder + closing
        }
        return nil
    }
}
