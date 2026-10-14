import Foundation
import TurboSpark

/// Conversation recall: finished chat turns recorded into a context store so
/// older turns come back as cited evidence once compaction or the window fit
/// drops them (`swift/docs/SWIFT_CONTEXT_ENGINE.md`).
///
/// **THE TRANSCRIPT IS THE SOURCE OF TRUTH; THE STORE IS A PROJECTION OF IT.**
/// Edit, retry, variant switching and message deletion all rewrite rows in
/// place, and hooking each of them would leave the store wrong the first time
/// a new rewrite path is added. Instead every reconcile compares what the
/// store holds with what the transcript says now: a recorded prefix gets the
/// missing rows appended, anything else is forgotten and recorded again.
/// These are pure functions so that rule is testable without the engine.
enum ContextRecall {
    /// Longest text recorded for one row. Recall wants the gist of an old
    /// turn, and the journal refuses records over 16 MiB.
    static let maxRowCharacters = 16_000

    /// One recorded turn.
    struct Row: Equatable, Sendable {
        enum Role: String, Sendable { case user, assistant }
        let role: Role
        let text: String
    }

    /// What reconciling a conversation has to do.
    enum Plan: Equatable, Sendable {
        case upToDate
        /// Record these after what is already recorded.
        case append([Row])
        /// Forget the conversation, then record these.
        case rebuild([Row])
    }

    /// The rows of `messages` that recall records: what the user typed and
    /// the assistant's prose. Tool traffic, synthetic rows (hook and
    /// guardrail text, which can carry injected content) and empty rows are
    /// left out. Rows are the ACTIVE version of each message.
    ///
    /// With `beforeLastQuestion`, the newest user row and everything after it
    /// are left out too: that is the question being asked now and the turn
    /// still in progress, which the prompt already holds verbatim.
    static func rows(from messages: [AppChatMessage], beforeLastQuestion: Bool) -> [Row] {
        var source = messages[...]
        if beforeLastQuestion {
            guard let last = messages.lastIndex(where: { isRecordedUserRow($0) }) else { return [] }
            source = messages[..<last]
        }
        return source.compactMap { message in
            if isRecordedUserRow(message) {
                return Row(role: .user, text: capped(message.content))
            }
            if message.role == .assistant, !trimmed(message.content).isEmpty {
                return Row(role: .assistant, text: capped(message.content))
            }
            return nil
        }
    }

    static func isRecordedUserRow(_ message: AppChatMessage) -> Bool {
        message.role == .user && !message.isSynthetic && !trimmed(message.content).isEmpty
    }

    /// Compares what the store holds (`recorded`, oldest first) with the
    /// rows the transcript has now.
    static func plan(recorded: [Row], desired: [Row]) -> Plan {
        if recorded == desired { return .upToDate }
        if recorded.count < desired.count, Array(desired.prefix(recorded.count)) == recorded {
            return .append(Array(desired.dropFirst(recorded.count)))
        }
        return .rebuild(desired)
    }

    /// How many of the newest recorded rows the fitted prompt already holds
    /// verbatim, so retrieval does not repeat them as evidence.
    ///
    /// Compaction and the window fit only ever drop the OLDEST rows, so the
    /// rows still in the prompt are a suffix of `rows`; this counts that
    /// suffix by text, since prompt assembly appends to user rows (system
    /// reminders, evidence) and to assistant rows (tool invocations). An
    /// undercount only repeats a turn as evidence; an overcount would hide a
    /// dropped turn from recall, so the walk stops at the first row it
    /// cannot find.
    static func keptRows(_ rows: [Row], in prompt: [ChatMessage]) -> Int {
        var kept = 0
        for row in rows.reversed() {
            let role: ChatMessage.Role = row.role == .user ? .user : .assistant
            guard prompt.contains(where: { $0.role == role && $0.content.contains(row.text) })
            else { break }
            kept += 1
        }
        return kept
    }

    private static func capped(_ text: String) -> String {
        text.count > maxRowCharacters ? String(text.prefix(maxRowCharacters)) : text
    }

    private static func trimmed(_ text: String) -> String {
        text.trimmingCharacters(in: .whitespacesAndNewlines)
    }
}
