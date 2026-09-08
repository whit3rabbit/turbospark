import Foundation
import TurboSpark

/// The goal evaluator: the prompt a side model query judges a goal
/// condition with, and the parsing of its answer.
///
/// Claude Code reference: `src/utils/hooks/execPromptHook.ts` -- a
/// prompt-based Stop hook is judged by a small model that answers one JSON
/// object, and the goal is exactly such a hook with a third verdict
/// ("impossible") on top. The judge here never runs tools and never sees
/// files; it can only weigh what the conversation already shows, which is
/// the spec's own restriction.
///
/// Deviation from CC, recorded in `swift/docs/SWIFT_GOALS.md`: CC judges on
/// a Haiku-class small model; this app has one loaded session, so the
/// evaluator rides the main model in the gap between turns, with the same
/// boring settings compaction's summarizer uses.
enum GoalEvaluator {
    /// The judge's two-message prompt: the schema instruction, then the
    /// condition over the transcript it applies to.
    static func judgeMessages(
        condition: String, transcript: String
    ) -> [ChatMessage] {
        [
            ChatMessage(
                role: .system,
                content: """
                You are evaluating whether a conversation has satisfied a stated goal. \
                Respond ONLY with a JSON object and nothing else: {"ok": true} if the goal \
                is fully met; {"ok": false, "reason": "..."} if it is not yet met, with a \
                short reason saying what still stands between the work and the goal; \
                {"impossible": true, "reason": "..."} if the goal cannot be satisfied. \
                Judge only what the conversation shows: you cannot run tools, read files, \
                or ask questions. A goal that is merely PARTIALLY done is not met.
                """),
            ChatMessage(
                role: .user,
                content: """
                Goal: \(condition)

                Conversation so far:

                <transcript>
                \(transcript)
                </transcript>

                Has the goal been met? Answer with the JSON object only.
                """),
        ]
    }

    /// Parses the judge's reply, tolerating prose around the object: the
    /// first `{` through the LAST `}`, so a reason string containing a
    /// brace cannot cut the object short. Nil means "no verdict" --
    /// malformed JSON, no JSON at all -- which the caller treats as a
    /// transient failure, never as met and never as a reason to clear the
    /// goal.
    static func parseVerdict(_ text: String) -> GoalVerdict? {
        guard let start = text.firstIndex(of: "{"),
            let end = text.lastIndex(of: "}"),
            start < end,
            let data = String(text[start...end]).data(using: .utf8),
            let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        let reason = (object["reason"] as? String).flatMap {
            let trimmed = $0.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
        if (object["impossible"] as? Bool) == true {
            return .impossible(reason: reason ?? "the evaluator judged the goal unsatisfiable")
        }
        switch object["ok"] as? Bool {
        case true:
            return .met(reason: reason)
        case false:
            return .notMet(reason: reason ?? "the evaluator did not say why it is not met")
        case nil:
            return nil
        }
    }

    /// The assistant-facing status row appended to the transcript for one
    /// verdict. Short by design: every row here rides the context window
    /// for the rest of the goal's life.
    static func statusRow(for verdict: GoalVerdict, iterations: Int) -> String {
        switch verdict {
        case .met(let reason):
            var line = "Goal met after \(max(iterations, 0)) iteration\(iterations == 1 ? "" : "s")."
            if let reason, !reason.isEmpty { line += " \(reason)" }
            return line
        case .notMet(let reason):
            return "Goal check (iteration \(iterations + 1)): not yet met. \(reason)"
        case .impossible(let reason):
            return "Goal stopped: the evaluator judged it impossible. \(reason)"
        }
    }

    /// The status row for a stall pause.
    static let stallRow =
        "Goal paused: several consecutive replies produced no tool calls, so the loop "
        + "stopped itself rather than spin. The goal stays set; evaluation resumes after "
        + "your next message."
}
