import Foundation
import TurboSpark

/// The SKILL.state arm of the agent loop: a bounded prompt carrying the current
/// execution state instead of the whole transcript. Opt-in per project, default
/// off, so the append-only path stays byte-identical when it is not used.
///
/// Measured justification and its caveats: `docs/SKILL_STATE.md`.
extension AppModel {
    /// Whether this turn should build a bounded-state prompt.
    var skillStateEnabled: Bool {
        selectedProject?.skillStateEnabled ?? false
    }

    /// The procedural specification P: fixed for the life of a project, so it
    /// is the only part of the prompt a prefix cache could ever reuse.
    func skillStateProtocol() -> String {
        """
        ## Execution state

        You are running with a bounded execution state instead of a full \
        transcript. You will NOT be shown earlier steps. The state below is \
        everything you carry forward, so anything you will need later must be \
        written into it NOW, when you first observe it.

        After your reasoning, and alongside any tool call, emit a state patch:

        <state_patch>
        {"goal": "...", "facts": ["..."]}
        </state_patch>

        The patch is an RFC 7386 JSON Merge Patch over this schema:

        \(AppSkillStateSchema.promptDescription)

        Patch rules:
        - Include ONLY fields that change. If nothing changed, emit no patch.
        - A list or a map you include REPLACES the stored one, so restate its \
        whole contents, including the entries you are keeping.
        - To drop a field entirely, map it to null.
        - Record a finding the step it appears. There is no later chance.
        """
    }

    /// The bounded prompt: [system + P][the task][current state][latest observation].
    ///
    /// Compare `AppModel+Generation.swift`'s append-only assembly, which walks
    /// every message and every tool result. This one is O(1) in step count,
    /// which is the whole point.
    func buildSkillStateHistory(chatIndex: Int) -> [ChatMessage] {
        var history: [ChatMessage] = []

        var systemContent = buildSystemPrompt(for: selectedProject)
        systemContent = systemContent.isEmpty
            ? skillStateProtocol()
            : "\(systemContent)\n\n\(skillStateProtocol())"
        history.append(ChatMessage(role: .system, content: systemContent))

        // The task, which is the FIRST user turn rather than the last: later
        // user turns in an agent run are guardrail nudges, and the original
        // request is what the whole run is for.
        if let task = chats[chatIndex].messages.first(where: { $0.role == .user && !$0.content.isEmpty }) {
            history.append(ChatMessage(role: .user, content: task.content))
        }

        let state = chats[chatIndex].skillState ?? AppSkillState()
        var latest = "CURRENT STATE:\n\(state.rendered)"

        if let observation = latestObservation(chatIndex: chatIndex) {
            latest += "\n\nLATEST RESULT:\n\(observation)"
        }
        latest += "\n\nContinue. Emit a state patch for anything you learned, "
            + "then either call one tool or give your final answer."
        history.append(ChatMessage(role: .user, content: latest))

        return history
    }

    /// The single most recent tool result, or the last guardrail nudge. This is
    /// O_t: only the newest observation reaches the model, because everything
    /// worth keeping from older ones is supposed to be in the state by now.
    private func latestObservation(chatIndex: Int) -> String? {
        for message in chats[chatIndex].messages.reversed() {
            if let result = message.toolResults.last {
                let tag = result.isError ? "tool_error" : "tool_response"
                let call = message.toolCalls.last.map { "\($0.name)\n" } ?? ""
                return "<\(tag)>\n\(call)\(result.output)\n</\(tag)>"
            }
            // A nudge is a user turn that is not the original task.
            if message.role == .user,
                message.id != chats[chatIndex].messages.first(where: { $0.role == .user })?.id {
                return message.content
            }
        }
        return nil
    }

    /// Parses, validates and merges a patch out of a finished turn.
    ///
    /// Returns the text with the patch block removed, so the user never sees
    /// the bookkeeping and the downstream tool-call parser is unaffected.
    /// An invalid patch is DROPPED rather than applied: a bad merge silently
    /// corrupts everything after it, where a dropped one loses a single step's
    /// bookkeeping and shows up in the badge.
    @discardableResult
    func applySkillStatePatch(from text: String, chatIndex: Int) -> String {
        guard skillStateEnabled else { return text }
        let stripped = AppSkillStatePatch.taggedBody(in: text) != nil
            ? removeStatePatchBlock(from: text)
            : text

        guard let patch = AppSkillStatePatch.extract(from: text) else {
            return stripped
        }
        let errors = AppSkillStatePatch.validate(patch)
        guard errors.isEmpty else {
            skillStateLastError = errors.joined(separator: "; ")
            return stripped
        }
        skillStateLastError = nil
        var state = chats[chatIndex].skillState ?? AppSkillState()
        state.apply(patch: patch)
        chats[chatIndex].skillState = state
        return stripped
    }

    func removeStatePatchBlock(from text: String) -> String {
        var out = text
        while let open = out.range(of: "<state_patch>"),
            let close = out.range(of: "</state_patch>", range: open.upperBound..<out.endIndex) {
            out.replaceSubrange(open.lowerBound..<close.upperBound, with: "")
        }
        return out.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Clears the carried state. Called when a chat starts a fresh task, since
    /// the state describes one run rather than one conversation.
    public func resetSkillState() {
        guard let idx = selectedChatIndex else { return }
        chats[idx].skillState = nil
        skillStateLastError = nil
        persistChats()
    }
}
