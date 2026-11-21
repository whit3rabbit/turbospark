import Foundation

/// What conversation recall did for one request, for the reply's "Context"
/// group in `SourcesPopoverView`: the passages syntext chose, and whether
/// the packet reached the prompt.
///
/// IN MEMORY ONLY. It is never stored on a message or written to disk, so it
/// vanishes on relaunch and old chats show nothing. It carries no passage
/// text, only source labels the transcript already shows, and it is never
/// recorded for a chat `conversationRecallEligible` refuses (Ghost chats,
/// protected profiles).
struct RecallInspection: Equatable, Sendable {
    /// One evidence block, as syntext reported it.
    struct Block: Equatable, Sendable, Identifiable {
        /// Citation id (`E1`, `E2`, ...), unique within a packet.
        let id: String
        let source: String
        let breadcrumb: String
        /// `prose`, `code`, `definition`, `data`, or `turn`.
        let kind: String
        /// Unnormalized: only the order within one packet means anything.
        let score: Double
        /// Parts were elided to fit the evidence budget.
        let condensed: Bool
        let tokens: Int
    }

    /// The packet syntext returned, before the planner decided its fate.
    struct Retrieval: Equatable, Sendable {
        var blocks: [Block]
        var candidatesConsidered: Int
        var budget: Int
        var evidenceTokens: Int
    }

    enum DropReason: Equatable, Sendable {
        /// The window had no headroom for evidence, even after a refit.
        case noRoom
        /// The store answered with no passage.
        case nothingFound
        /// The store was off, failing, or slower than the turn waits.
        case unavailable
        /// The prompt with evidence would not fit, or could not be counted.
        case overBudget
    }

    enum Outcome: Equatable, Sendable {
        case injected
        case dropped(DropReason)
    }

    enum Summary: Equatable {
        case counts(retrieved: Int, used: Int, trimmed: Int)
        case skipped(DropReason)
    }

    var retrieval: Retrieval?
    var outcome: Outcome

    /// The planner cannot tell an empty packet from a store that never
    /// answered (both reach it as nil text); the retrieval can.
    static func make(outcome: Outcome, retrieval: Retrieval?) -> RecallInspection {
        if outcome == .dropped(.nothingFound), retrieval == nil {
            return RecallInspection(retrieval: nil, outcome: .dropped(.unavailable))
        }
        return RecallInspection(retrieval: retrieval, outcome: outcome)
    }

    var summary: Summary {
        switch outcome {
        case .dropped(let reason):
            return .skipped(reason)
        case .injected:
            let blocks = retrieval?.blocks ?? []
            // The receipt's candidate count can be missing (zero) on an
            // older store; never claim fewer retrieved than were used.
            let retrieved = max(retrieval?.candidatesConsidered ?? 0, blocks.count)
            return .counts(
                retrieved: retrieved, used: blocks.count,
                trimmed: blocks.filter(\.condensed).count)
        }
    }

    /// Rows to show: only a packet that reached the prompt, so no row ever
    /// says Included for evidence the model never saw.
    var rankedBlocks: [(rank: Int, block: Block)] {
        guard outcome == .injected, let blocks = retrieval?.blocks else { return [] }
        return Self.rank(blocks)
    }

    /// 1-based rank by score, highest first. Ties keep prompt order; a
    /// non-finite score ranks last rather than poisoning the sort.
    static func rank(_ blocks: [Block]) -> [(rank: Int, block: Block)] {
        let ordered = blocks.enumerated().sorted { lhs, rhs in
            let l = lhs.element.score.isFinite ? lhs.element.score : -.infinity
            let r = rhs.element.score.isFinite ? rhs.element.score : -.infinity
            return l == r ? lhs.offset < rhs.offset : l > r
        }
        return ordered.enumerated().map { (rank: $0.offset + 1, block: $0.element.element) }
    }
}

/// The latest inspection per turn, keyed by the person's message id. Bounded
/// so a long session does not grow it without limit; oldest entries leave
/// first. Never persisted.
struct RecallInspectionLog: Equatable {
    private struct Entry: Equatable {
        var inspection: RecallInspection
        var chatID: UUID
    }

    let capacity: Int
    private var entries: [UUID: Entry] = [:]
    private var order: [UUID] = []

    init(capacity: Int = 50) {
        self.capacity = max(1, capacity)
    }

    subscript(messageID: UUID) -> RecallInspection? { entries[messageID]?.inspection }

    mutating func record(_ inspection: RecallInspection, messageID: UUID, chatID: UUID) {
        order.removeAll { $0 == messageID }
        order.append(messageID)
        entries[messageID] = Entry(inspection: inspection, chatID: chatID)
        while order.count > capacity {
            entries.removeValue(forKey: order.removeFirst())
        }
    }

    mutating func removeAll(chatID: UUID) {
        order.removeAll { entries[$0]?.chatID == chatID }
        entries = entries.filter { $0.value.chatID != chatID }
    }

    mutating func removeAll() {
        entries.removeAll()
        order.removeAll()
    }
}
