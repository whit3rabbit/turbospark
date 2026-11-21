import Foundation

/// Measures how long the model reasoned in one turn, from the arrival of the
/// first reasoning chunk to the last one.
///
/// Kept on `AppModel` and stamped onto the turn's committed assistant row
/// (`AppChatMessage.reasoningSeconds`) when the row is written, so the
/// "Thought for N" label survives relaunch. The span excludes prefill on
/// purpose: it is the time spent thinking, not the time to first token.
struct ReasoningClock: Equatable {
    /// The chat whose turn is being timed. A row in any other chat is never
    /// stamped with this measurement.
    var chatID: UUID?
    private(set) var firstAt: Date?
    private(set) var lastAt: Date?

    init(chatID: UUID? = nil) {
        self.chatID = chatID
    }

    mutating func note(at date: Date) {
        if firstAt == nil { firstAt = date }
        lastAt = date
    }

    /// Nil until a reasoning chunk arrived: no measurement, no duration.
    var seconds: Double? {
        guard let firstAt, let lastAt else { return nil }
        return max(0, lastAt.timeIntervalSince(firstAt))
    }

    /// Writes the measurement onto the newest assistant row of `chatID` that
    /// carries reasoning, has no duration yet, and was created after this
    /// turn's reasoning began. Consumes the measurement on success so it is
    /// written at most once.
    ///
    /// The `createdAt` floor is what keeps an older reply (or a variant
    /// swapped back in with its own timestamp) from inheriting this turn's
    /// time. When no row qualifies, nothing is written and the row shows no
    /// duration rather than a guess.
    @discardableResult
    mutating func stamp(_ messages: inout [AppChatMessage], chatID: UUID) -> Bool {
        guard self.chatID == chatID, let firstAt, let seconds else { return false }
        guard let index = messages.lastIndex(where: {
            $0.role == .assistant && !$0.reasoning.isEmpty
                && $0.reasoningSeconds == nil && $0.createdAt >= firstAt
        }) else { return false }
        messages[index].reasoningSeconds = seconds
        self = ReasoningClock(chatID: chatID)
        return true
    }
}

/// The duration in "Thought for N" as the activity line shows it.
enum ReasoningDurationText {
    /// Nil when there is no usable measurement. Units are verbatim, like the
    /// live elapsed readout (`ChatActivityStatus.elapsedText`).
    static func text(seconds: Double?) -> String? {
        guard let seconds, seconds.isFinite, seconds >= 0 else { return nil }
        if seconds < 1 { return "<1s" }
        return ChatActivityStatus.elapsedText(seconds: Int(seconds.rounded()))
    }
}

/// The text the folded live preview lays out.
enum ReasoningPreview {
    /// Characters kept for the 72 pt fold. Several times what the fold can
    /// show at any width, and small enough that re-laying it out on every
    /// streamed token stays cheap however long the trace grows.
    static let defaultLimit = 600

    /// The end of `text`, at most `limit` characters, starting at a line or
    /// word boundary so the first visible line is not a word fragment.
    static func tail(of text: String, limit: Int = defaultLimit) -> String {
        guard text.count > limit else { return text }
        let window = text.suffix(limit)
        // Drop the partial first word or line the cut landed in.
        guard let boundary = window.firstIndex(where: { $0.isWhitespace }) else {
            return String(window)
        }
        let rest = window[boundary...].drop(while: { $0.isWhitespace })
        return rest.isEmpty ? String(window) : String(rest)
    }
}
