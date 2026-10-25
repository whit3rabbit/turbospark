import Foundation
import TurboSpark

/// The pure half of conversation recall (syntext's context engine, see
/// `ChatRecallStore`): which transcript rows are journaled, how the journal
/// is reconciled with the transcript, which rows the fitted prompt still
/// carries, and where recalled evidence goes. No store and no session, so
/// all of it is testable offline.
enum ChatRecallPlanner {
    enum Role: String, Equatable, Sendable {
        case user, assistant
    }

    /// One journaled turn and the transcript row it came from.
    struct Turn: Equatable, Sendable {
        let rowIndex: Int
        let role: Role
        let text: String
    }

    enum Reconcile: Equatable {
        case none
        case append([Turn])
        case rebuild
    }

    /// A turn's text is capped before it is recorded. Syntext does not index
    /// a turn over 8 MiB or one with a NUL early on, so such a turn would be
    /// journaled, vanish from `history()` after a relaunch, and force a
    /// rebuild on every launch. The cap also bounds pasted attachments.
    static let maxTurnBytes = 32 * 1024

    /// Evidence never takes more than this, and never more than an eighth
    /// of the prompt budget.
    static let maxEvidenceTokens = 1_024
    /// Below this there is no room for a useful passage.
    static let minEvidenceTokens = 96

    /// Rows recall records: what the person wrote and the model's prose
    /// answers. Synthetic user rows (hook reasons, guardrail nudges, goal
    /// check-ins) can carry injected text and are not the person's words;
    /// tool turns stay with the tool-observation store; a guardrail retry is
    /// a discarded attempt.
    static func recordable(_ rows: [AppChatMessage]) -> [Turn] {
        rows.enumerated().compactMap { index, row in
            let role: Role
            switch row.role {
            case .user:
                guard !row.isSynthetic else { return nil }
                role = .user
            case .assistant:
                guard row.toolCalls.isEmpty, row.toolResults.isEmpty,
                      row.stopReason != "guardrail_retry"
                else { return nil }
                role = .assistant
            default:
                return nil
            }
            let text = normalize(row.content)
            guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return nil }
            return Turn(rowIndex: index, role: role, text: text)
        }
    }

    /// The exact text recorded for a row, and compared against `history()`.
    static func normalize(_ text: String) -> String {
        var cleaned = text.replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\u{0}", with: "")
        if cleaned.utf8.count > maxTurnBytes {
            // A cut inside a multi-byte character decodes to U+FFFD, which is
            // stable: the same input always normalizes to the same string.
            cleaned = String(decoding: cleaned.utf8.prefix(maxTurnBytes), as: UTF8.self)
        }
        return cleaned
    }

    /// How to bring the journal (`recorded`, oldest first, as `history()`
    /// returns it) in line with the transcript. Any divergence, including a
    /// shorter transcript after a rewind, is a rebuild: forget, then record
    /// every turn again.
    static func reconcilePlan(
        recorded: [(role: String, text: String)], recordable: [Turn]
    ) -> Reconcile {
        guard recorded.count <= recordable.count else { return .rebuild }
        for (index, entry) in recorded.enumerated() {
            let expected = recordable[index]
            guard entry.role == expected.role.rawValue, entry.text == expected.text else {
                return .rebuild
            }
        }
        return recorded.count == recordable.count
            ? .none : .append(Array(recordable[recorded.count...]))
    }

    /// Transcript index of the oldest row the fitted prompt still carries,
    /// or nil when it carries none.
    ///
    /// The window fitter removes MESSAGES, oldest first after a leading
    /// system or developer message, and often removes the compaction
    /// summary or a tool result before any turn; the preserved-instructions
    /// block is fitted outside the history and re-inserted. So the kept
    /// history is a suffix of the projection with the pin taken out, and its
    /// length is what the retained array holds besides the leading
    /// instruction and the pin.
    static func earliestRetainedRow(
        projectionMessages: [ChatMessage],
        sourceRows: [Int?],
        pinIndex: Int?,
        boundary: Int,
        retained: [ChatMessage]
    ) -> Int? {
        guard projectionMessages.count == sourceRows.count else { return nil }
        var messages = projectionMessages
        var rows = sourceRows
        var retainedPin = false
        if let pinIndex, messages.indices.contains(pinIndex) {
            let pin = messages.remove(at: pinIndex)
            rows.remove(at: pinIndex)
            retainedPin = retained.contains(pin)
        }
        let lead = messages.first.map { $0.role == .system || $0.role == .developer } == true ? 1 : 0
        let keptSuffix = retained.count - lead - (retainedPin ? 1 : 0)
        guard keptSuffix > 0 else { return nil }
        let firstKept = max(lead, messages.count - keptSuffix)
        guard let rebased = rows[firstKept...].lazy.compactMap({ $0 }).first else { return nil }
        return max(boundary, 0) + rebased
    }

    /// Recorded turns the prompt carries verbatim. Recall leaves them out.
    static func keptTurns(_ turns: [Turn], earliestRetainedRow: Int?) -> Int {
        guard let earliestRetainedRow else { return 0 }
        return turns.filter { $0.rowIndex >= earliestRetainedRow }.count
    }

    /// The model-bound copy of the person's newest message with evidence in
    /// front of it. The layout is syntext's (docs/CONTEXT.md): the policy,
    /// the trust-labelled evidence blocks, then the question. It is sent to
    /// the model and never stored, so it is not localized.
    static func inject(evidence: String, policy: String, into message: ChatMessage) -> ChatMessage {
        var injected = message
        injected.content = policy + "\n\n" + evidence + "\n\nQuestion: " + message.content
        return injected
    }

    /// What a fit produced, in the shape recall works with.
    struct Fit {
        var retained: [ChatMessage]
        var measuredTokens: Int
        var removedTurnCount: Int
        var hasRoomForGeneration: Bool
    }

    /// The facts about this turn that recall needs, taken from the projection
    /// the prompt was fitted from.
    struct Inputs {
        /// The projection's messages after microcompaction (same count).
        var projectionMessages: [ChatMessage]
        var sourceRows: [Int?]
        var pinIndex: Int?
        var boundary: Int
        var turns: [Turn]
        /// The projection's copy of the person's newest message: the message
        /// evidence is attached to, found again in the fitted prompt by value.
        var question: ChatMessage?
        var promptBudget: Int
    }

    /// Adds recalled evidence to a fitted prompt, or returns it unchanged.
    ///
    /// Unchanged, byte for byte, whenever every recorded turn is already in
    /// the prompt (no evidence is even requested), when the person's message
    /// was itself dropped, when there is no room, or when the result would
    /// not fit. The window is never exceeded: the final prompt is counted.
    static func applyConversationRecall(
        fitted: Fit,
        inputs: Inputs,
        refit: (_ maxTokens: Int) async -> Fit?,
        countTokens: ([ChatMessage]) async -> Int?,
        evidence: (_ keptTurns: Int, _ budget: Int) async -> String?,
        policy: String
    ) async -> Fit {
        guard let question = inputs.question, !inputs.turns.isEmpty else { return fitted }

        func outside(_ fit: Fit) -> (kept: Int, outside: Bool) {
            let earliest = earliestRetainedRow(
                projectionMessages: inputs.projectionMessages, sourceRows: inputs.sourceRows,
                pinIndex: inputs.pinIndex, boundary: inputs.boundary, retained: fit.retained)
            let kept = keptTurns(inputs.turns, earliestRetainedRow: earliest)
            return (kept, kept < inputs.turns.count)
        }

        guard outside(fitted).outside, fitted.retained.contains(question) else { return fitted }

        let reserve = min(maxEvidenceTokens, inputs.promptBudget / 8)
        var base = fitted
        if inputs.promptBudget - 1 - fitted.measuredTokens < reserve {
            guard let tighter = await refit(inputs.promptBudget - reserve),
                  tighter.hasRoomForGeneration, tighter.retained.contains(question)
            else { return fitted }
            base = tighter
        }

        // The policy rides in the same message; leave it room in the budget.
        let policyOverhead = policy.utf8.count / 3 + 16
        let headroom = inputs.promptBudget - 1 - base.measuredTokens - policyOverhead
        let budget = min(reserve, headroom)
        guard budget >= minEvidenceTokens else { return fitted }

        let kept = outside(base).kept
        guard let text = await evidence(kept, budget),
              !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
              let target = base.retained.lastIndex(of: question)
        else { return fitted }

        var retained = base.retained
        retained[target] = inject(evidence: text, policy: policy, into: question)
        guard let measured = await countTokens(retained), measured < inputs.promptBudget else {
            return fitted
        }
        return Fit(
            retained: retained, measuredTokens: measured,
            removedTurnCount: base.removedTurnCount, hasRoomForGeneration: true)
    }
}
