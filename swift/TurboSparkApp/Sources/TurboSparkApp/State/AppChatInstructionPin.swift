import Foundation
import TurboSpark

enum AppChatInstructionPin {
    struct CandidateRow: Equatable {
        var index: Int
        var messageID: UUID? = nil
        var text: String
    }

    struct PinningOutcome {
        var pinnedGroups: [String]
        var estimatedTokens: Int
    }

    struct CacheEntry {
        var boundary: Int
        var outcome: PinningOutcome
        var sessionIdentity: ObjectIdentifier? = nil
        var ceilingTokens: Int = 512
        var reasoning: GenerateOptions.Reasoning = .off
        var sourceRows: [CandidateRow] = []
    }

    struct FitCandidate {
        var retained: [ChatMessage]
        var measuredTokens: Int
        var removedTurnCount: Int
        var hasRoomForGeneration: Bool
    }

    struct FitOutcome {
        var retained: [ChatMessage]
        var measuredTokens: Int
        var removedTurnCount: Int
        var hasRoomForGeneration: Bool
    }

    static func canPin(_ message: AppChatMessage) -> Bool {
        guard message.role == .user,
            !message.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
            message.imagePaths.isEmpty
        else { return false }

        // Text attachments are flattened into the transcript with this block
        // marker, so exclude the whole row before its file contents can be pinned.
        return !message.content.contains("--- Attachment: ")
    }

    static func pinnedUserRows(
        _ messages: [AppChatMessage], before boundary: Int
    ) -> [CandidateRow] {
        messages.enumerated().compactMap { index, message in
            guard index < boundary, message.isStandingInstruction, canPin(message) else {
                return nil
            }
            return CandidateRow(index: index, messageID: message.id, text: message.content)
        }
    }

    static func select(
        userRows: [CandidateRow], boundary: Int,
        ceilingTokens: Int, countTokens: (String) async -> Int?
    ) async -> PinningOutcome {
        guard ceilingTokens > 0 else {
            return PinningOutcome(pinnedGroups: [], estimatedTokens: 0)
        }

        var selected: [String] = []
        var estimatedTokens = 0
        let candidates = userRows.filter { row in
            row.index < boundary
                && !row.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }.sorted { $0.index < $1.index }
        for row in candidates {
            let proposed = selected + [row.text]
            let rendered = injectionBlock(proposed).content
            guard let tokens = await countTokens(rendered),
                tokens >= 0, tokens <= ceilingTokens
            else {
                continue
            }
            selected = proposed
            estimatedTokens = tokens
        }
        return PinningOutcome(pinnedGroups: selected, estimatedTokens: estimatedTokens)
    }

    static func injectionBlock(_ groups: [String]) -> ChatMessage {
        let content = (["Preserved user instructions:"] + groups).joined(separator: "\n\n")
        return ChatMessage(role: .user, content: content)
    }

    static func cacheMatches(
        _ entry: CacheEntry, activeSessionIdentity: ObjectIdentifier?,
        ceilingTokens: Int, reasoning: GenerateOptions.Reasoning,
        sourceRows: [CandidateRow]
    ) -> Bool {
        guard entry.ceilingTokens == ceilingTokens,
            entry.reasoning == reasoning,
            entry.sourceRows == sourceRows
        else {
            return false
        }
        return entry.sessionIdentity == activeSessionIdentity
    }

    static func fitPreservingBlock(
        _ messages: [ChatMessage], injectedBlockIndex: Int?,
        pinnedBlock: ChatMessage?, maximumTokens: Int,
        countTokens: ([ChatMessage]) async -> Int?,
        fitWindow: ([ChatMessage], Int) async -> FitCandidate?
    ) async -> FitOutcome {
        guard maximumTokens > 0 else {
            return FitOutcome(
                retained: messages, measuredTokens: Int.max, removedTurnCount: 0,
                hasRoomForGeneration: false)
        }

        var fittingHistory = messages
        if let injectedBlockIndex {
            guard fittingHistory.indices.contains(injectedBlockIndex) else {
                return FitOutcome(
                    retained: messages, measuredTokens: Int.max, removedTurnCount: 0,
                    hasRoomForGeneration: false)
            }
            fittingHistory.remove(at: injectedBlockIndex)
        }

        guard let pinnedBlock else {
            guard let candidate = await fitWindow(fittingHistory, maximumTokens) else {
                return FitOutcome(
                    retained: fittingHistory, measuredTokens: Int.max, removedTurnCount: 0,
                    hasRoomForGeneration: false)
            }
            return FitOutcome(
                retained: candidate.retained,
                measuredTokens: candidate.measuredTokens,
                removedTurnCount: candidate.removedTurnCount,
                hasRoomForGeneration: candidate.hasRoomForGeneration)
        }

        var fullHistory = fittingHistory
        insertPinnedBlock(pinnedBlock, into: &fullHistory)
        var fitLimit = maximumTokens
        var lastOutcome = FitOutcome(
            retained: fullHistory, measuredTokens: Int.max, removedTurnCount: 0,
            hasRoomForGeneration: false)

        // Fitting without the pin keeps the window fitter from dropping it as
        // the oldest ordinary user message. Recount the rebuilt prompt and
        // tighten the history budget if framing costs are not additive.
        for _ in 0..<max(1, messages.count + 1) {
            guard let candidate = await fitWindow(fittingHistory, fitLimit) else {
                return lastOutcome
            }
            var retained = candidate.retained
            insertPinnedBlock(pinnedBlock, into: &retained)
            guard let measured = await countTokens(retained), measured >= 0 else {
                return FitOutcome(
                    retained: retained, measuredTokens: Int.max,
                    removedTurnCount: candidate.removedTurnCount,
                    hasRoomForGeneration: false)
            }

            let fits = measured < maximumTokens
            lastOutcome = FitOutcome(
                retained: retained,
                measuredTokens: measured,
                removedTurnCount: candidate.removedTurnCount,
                hasRoomForGeneration: fits)
            if fits { return lastOutcome }
            if !candidate.hasRoomForGeneration { return lastOutcome }

            guard candidate.measuredTokens >= 0 else { return lastOutcome }
            let pinnedOverhead = measured - candidate.measuredTokens
            guard pinnedOverhead > 0 else { return lastOutcome }
            let nextFitLimit = maximumTokens - pinnedOverhead
            guard nextFitLimit > 0, nextFitLimit < fitLimit else { return lastOutcome }
            fitLimit = nextFitLimit
        }
        return lastOutcome
    }

    private static func insertPinnedBlock(
        _ pinnedBlock: ChatMessage, into messages: inout [ChatMessage]
    ) {
        let index = messages.firstIndex { $0.role != .system } ?? messages.count
        messages.insert(pinnedBlock, at: index)
    }
}
