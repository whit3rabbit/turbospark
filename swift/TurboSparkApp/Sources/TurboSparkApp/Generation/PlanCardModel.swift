import Foundation

/// One numbered step of a submitted plan.
struct PlanCardStep: Equatable, Sendable {
    let number: Int
    /// The item's first line with emphasis markers removed (a row title).
    let title: String
    /// Every following line up to the next step, de-indented, as markdown.
    let detail: String
}

/// A plan split into an intro and numbered steps. Every non-blank line of the
/// source lands in `intro` or in a step, so drawing the outline loses no text.
struct PlanOutline: Equatable, Sendable {
    let intro: String
    let steps: [PlanCardStep]
}

/// Parses ExitPlanMode markdown into a step outline, or nil when unsure.
///
/// Nil is the common, safe answer: the card then shows the plan as the
/// markdown block it always was. The parser only commits when the plan is a
/// top-level ordered list numbered 1, 2, 3 with at least two items, because a
/// weak local model's prose with a stray "3." in it must not be re-cut into
/// steps that read differently from what it wrote.
enum PlanOutlineParser {
    static func parse(_ text: String) -> PlanOutline? {
        var introLines: [String] = []
        var steps: [(number: Int, title: String, detail: [String])] = []
        var inFence = false

        for line in text.components(separatedBy: .newlines) {
            let trimmed = line.trimmingCharacters(in: .whitespaces)
            if trimmed.hasPrefix("```") || trimmed.hasPrefix("~~~") {
                inFence.toggle()
            } else if !inFence, let item = topLevelItem(line) {
                // Numbering that skips or restarts is not a list we trust.
                guard item.number == steps.count + 1 else { return nil }
                steps.append((item.number, item.title, []))
                continue
            }
            if steps.isEmpty {
                introLines.append(line)
            } else {
                steps[steps.count - 1].detail.append(line)
            }
        }

        guard steps.count >= 2 else { return nil }
        return PlanOutline(
            intro: introLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines),
            steps: steps.map {
                PlanCardStep(number: $0.number, title: $0.title, detail: dedent($0.detail))
            })
    }

    /// `1. Title` or `1) Title` at column 0. Indented items belong to the
    /// step above them (a sub-list), so they are left in its detail.
    private static func topLevelItem(_ line: String) -> (number: Int, title: String)? {
        let digits = line.prefix(while: \.isNumber)
        guard !digits.isEmpty, digits.count <= 3, let number = Int(digits) else { return nil }
        var rest = line.dropFirst(digits.count)
        guard let marker = rest.first, marker == "." || marker == ")" else { return nil }
        rest = rest.dropFirst()
        guard let space = rest.first, space == " " || space == "\t" else { return nil }
        let title = stripEmphasis(rest.trimmingCharacters(in: .whitespaces))
        guard !title.isEmpty else { return nil }
        return (number, title)
    }

    /// Bold and code markers read as noise in a one-line row title.
    private static func stripEmphasis(_ text: String) -> String {
        text.replacingOccurrences(of: "**", with: "")
            .replacingOccurrences(of: "__", with: "")
            .replacingOccurrences(of: "`", with: "")
            .trimmingCharacters(in: .whitespaces)
    }

    /// Removes the common leading indent so a continuation or sub-list
    /// renders as ordinary markdown instead of a code block.
    private static func dedent(_ lines: [String]) -> String {
        let indents = lines
            .filter { !$0.trimmingCharacters(in: .whitespaces).isEmpty }
            .map { $0.prefix(while: { $0 == " " || $0 == "\t" }).count }
        let common = indents.min() ?? 0
        return lines
            .map { $0.trimmingCharacters(in: .whitespaces).isEmpty ? "" : String($0.dropFirst(common)) }
            .joined(separator: "\n")
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }
}

/// When the plan card offers Start and Request changes.
///
/// These buttons only send or draft an ordinary user message through the
/// existing composer path. They are the fallback for a plan the blocking
/// gate did not decide (a stopped turn, a reloaded chat, a host with no
/// gate), so they are honest only after the agent stopped on the plan: the
/// turn is idle, the plan call succeeded, nothing has happened after it (no
/// user reply, no further tool call), and the gate is neither waiting on
/// this call (its own buttons are up) nor already decided it. Otherwise a
/// "Start" would race an agent that already started, or ask twice.
enum PlanCardActions {
    static func isOffered(
        callID: UUID, in messages: [AppChatMessage], isIdle: Bool, awaitingCallID: UUID? = nil
    ) -> Bool {
        guard isIdle, awaitingCallID != callID,
            let index = messages.firstIndex(where: { $0.toolCalls.contains { $0.id == callID } })
        else { return false }
        let message = messages[index]
        guard let result = message.toolResults.first(where: { $0.callID == callID }), !result.isError
        else { return false }
        if isDecidedInGate(result.output) { return false }
        // A later call in the same batch means the agent kept acting.
        if message.toolCalls.last?.id != callID { return false }
        return !messages[(index + 1)...].contains { $0.role == .user || !$0.toolCalls.isEmpty }
    }

    /// Whether a plan call's result says the gate already took the user's
    /// Start or Request changes. A cancelled gate is not a decision.
    static func isDecidedInGate(_ output: String) -> Bool {
        output.hasPrefix(PlanModeExecutor.approvedResult)
            || output.hasPrefix(PlanModeExecutor.changesRequestedPrefix)
    }
}
