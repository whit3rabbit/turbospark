import Foundation

/// Turn grouping for the qwen-code transcript collapse: a finished turn can
/// fold down to its prompt and its final answer, hiding the intermediate
/// tool rows and partial assistant messages behind a "N steps" toggle.
///
/// Pure grouping so the rules are testable without a view:
/// - a user message OPENS a turn and is its anchor;
/// - every following non-user message belongs to that turn until the next
///   user message;
/// - leading non-user messages (a compaction summary, a synthetic row) form
///   a headless turn that cannot collapse, because there is no prompt to
///   show for it.
enum TurnCollapseModel {
    struct Turn: Equatable {
        /// The user message id the turn opens with; nil for a headless turn.
        let anchorID: UUID?
        /// Every message id in the turn, transcript order.
        var messageIDs: [UUID]

        var isCollapsible: Bool { anchorID != nil }
    }

    static func turns(in messages: [AppChatMessage]) -> [Turn] {
        var result: [Turn] = []
        for message in messages {
            if message.role == .user {
                // A user message always OPENS a turn, even mid-run.
                result.append(Turn(anchorID: message.id, messageIDs: [message.id]))
            } else if var last = result.popLast() {
                last.messageIDs.append(message.id)
                result.append(last)
            } else {
                result.append(Turn(anchorID: nil, messageIDs: [message.id]))
            }
        }
        return result
    }

    /// The ids a COLLAPSED turn renders: the prompt, the final answer, and
    /// nothing between. The final answer is the last assistant message
    /// carrying content or tool calls; when a turn somehow ends without one
    /// (a user-only turn) only the prompt shows. The prompt and the answer
    /// are the same ids an expanded turn renders first and last, so a
    /// collapse can never reorder or drop a boundary row.
    static func visibleIDs(
        collapsedFor turn: Turn,
        messagesByID: [UUID: AppChatMessage]
    ) -> [UUID] {
        guard turn.isCollapsible, turn.messageIDs.count > 1 else { return turn.messageIDs }
        let answer = turn.messageIDs.last { id in
            guard let message = messagesByID[id] else { return false }
            return message.role == .assistant
                && (!message.content.isEmpty || !message.toolCalls.isEmpty)
        }
        guard let answer else { return [turn.messageIDs.first].compactMap { $0 } }
        return [turn.messageIDs.first!, answer].removingDuplicates()
    }

    /// How many rows the collapse hides, what the toggle's label counts.
    static func hiddenCount(
        for turn: Turn,
        messagesByID: [UUID: AppChatMessage]
    ) -> Int {
        turn.messageIDs.count - visibleIDs(collapsedFor: turn, messagesByID: messagesByID).count
    }
}

private extension Array where Element == UUID {
    func removingDuplicates() -> [UUID] {
        var seen = Set<UUID>()
        return filter { seen.insert($0).inserted }
    }
}
