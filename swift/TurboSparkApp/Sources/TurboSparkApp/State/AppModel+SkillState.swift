import Foundation
import TurboSpark

/// The SKILL.state arm of the agent loop: a bounded prompt carrying the current
/// execution state instead of the whole transcript. Opt-in per project, default
/// off, so the append-only path stays byte-identical when it is not used.
///
/// Measured justification and its caveats: `docs/SKILL_STATE.md`.
extension AppModel {
    /// Whether the SELECTED project runs in bounded-state mode.
    ///
    /// The UI's read. A generation turn asks its own project instead
    /// (`turnProject(chatID:)`, state#30): the selection can move while a
    /// call sits at an approval card, and switching prompt SHAPE mid-run
    /// discards the carried state the run depends on.
    var skillStateEnabled: Bool {
        selectedProject?.skillStateEnabled ?? false
    }

    /// The procedural specification P: fixed for the life of a project, so it
    /// is the only part of the prompt a prefix cache could ever reuse.
    ///
    /// **THE PATCH RULES DESCRIBE WHAT `AppSkillState.apply` DOES, AND ONE OF
    /// THEM DID NOT** (state#91). It said a list or a map you include
    /// REPLACES the stored one; `apply` merges an object entry by entry (RFC
    /// 7386, which is what the sentence above it already claims the patch
    /// is). A model following the instruction restated the whole map every
    /// time, which is harmless -- but a model that could never DELETE an
    /// entry, because the prompt offered no way to, watched `files` grow to
    /// `maxRenderedBytes` and then had every later patch rejected for a limit
    /// the instructions had told it could not be avoided. The fix is in the
    /// PROMPT: the merge semantics are the correct ones and are what the
    /// validator, the renderer and the size check are all written against.
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
        - A LIST you include REPLACES the stored one, so restate its whole \
        contents, including the entries you are keeping.
        - A MAP you include MERGES entry by entry: the entries you name are \
        added or overwritten and every other entry is kept. Set an entry to \
        null to delete it.
        - To drop a field entirely, map it to null.
        - Record a finding the step it appears. There is no later chance.
        """
    }

    /// The bounded prompt: [system + P][the task][current state][latest observation].
    ///
    /// Compare `AppModel+Generation.swift`'s append-only assembly, which walks
    /// every message and every tool result. This one is O(1) in step count,
    /// which is the whole point.
    func buildSkillStateHistory(chatIndex: Int, project: AppProject?) -> [ChatMessage] {
        var history: [ChatMessage] = []

        var systemContent = buildSystemPrompt(for: project)
        systemContent = systemContent.isEmpty
            ? skillStateProtocol()
            : "\(systemContent)\n\n\(skillStateProtocol())"
        history.append(ChatMessage(role: .system, content: systemContent))

        // The task (see `taskMessage`), **AND ITS IMAGES** (state#58). The bounded prompt rebuilds the task
        // from scratch on every step, so an `imagePaths` dropped here is a
        // picture sent on step one of the append-only path and on NO step of
        // this one -- the model answers about an image it was never shown,
        // which is `AppChatMessage.imagePaths`'s own documented hazard
        // arriving in the other prompt shape.
        if let task = Self.taskMessage(in: chats[chatIndex]) {
            history.append(
                ChatMessage(
                    role: .user,
                    content: task.content,
                    images: task.imagePaths.map(ChatImage.path)))
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
            // A nudge is a user turn that is not the original task -- and
            // "the task" means the same thing here as it does above
            // (state#58). These were two different predicates: the task was
            // the first user turn with NON-EMPTY content, the nudge test
            // compared against the first user turn of any kind. On a run
            // whose first user turn carries only an image, they name
            // different messages, so the real task was fed back as the
            // latest observation on every step.
            if message.role == .user,
                message.id != Self.taskMessage(in: chats[chatIndex])?.id {
                return message.content
            }
        }
        return nil
    }

    /// The ORIGINAL request, which is what the whole run is for.
    ///
    /// The FIRST user turn rather than the last: later user turns in an agent
    /// run are guardrail nudges. One definition, because two spellings of it
    /// disagree on a turn that carries only a picture (state#58).
    static func taskMessage(in chat: AppChat) -> AppChatMessage? {
        chat.messages.first {
            $0.role == .user && (!$0.content.isEmpty || !$0.imagePaths.isEmpty)
        }
    }

    /// Parses, validates and merges a patch out of a finished turn.
    ///
    /// Returns the text with the patch block removed, so the user never sees
    /// the bookkeeping and the downstream tool-call parser is unaffected.
    /// An invalid patch is DROPPED rather than applied: a bad merge silently
    /// corrupts everything after it, where a dropped one loses a single step's
    /// bookkeeping and shows up in the badge.
    @discardableResult
    func applySkillStatePatch(from text: String, chatIndex: Int, project: AppProject?) -> String {
        guard project?.skillStateEnabled ?? false else { return text }
        let stripped = AppSkillStatePatch.taggedBody(in: text) != nil
            ? removeStatePatchBlock(from: text)
            : text

        // **A PATCH THAT WOULD NOT PARSE IS AN ERROR, NOT AN ABSENCE**
        // (state#92). A turn carrying `<state_patch>` whose body is not JSON
        // fell into the no-patch arm below and CLEARED the badge, so the one
        // failure a user cannot see in the transcript -- the block is
        // stripped before display -- was also the one that reported nothing.
        if AppSkillStatePatch.taggedBody(in: text) != nil,
            AppSkillStatePatch.extract(from: text) == nil
        {
            skillStateLastError =
                "the <state_patch> block could not be parsed as JSON and was dropped."
            return stripped
        }
        guard let patch = AppSkillStatePatch.extract(from: text) else {
            // **A TURN WITH NO PATCH CLEARS THE ERROR** (state#58). This
            // returned before the `skillStateLastError = nil` below, so a
            // badge raised by one rejected patch stayed lit for the rest of
            // the run -- through every later turn that emitted nothing wrong,
            // reporting a failure that had already been recovered from.
            skillStateLastError = nil
            return stripped
        }
        let errors = AppSkillStatePatch.validate(patch)
        guard errors.isEmpty else {
            skillStateLastError = errors.joined(separator: "; ")
            return stripped
        }
        // **THE TOTAL IS CHECKED ON THE RESULT, NOT ON THE PATCH** (state#58).
        // Every individual patch can be within bounds while the merged state
        // grows without limit, which is the only growth curve that matters:
        // this is what the prompt carries on every step. Rejected like any
        // other invalid patch -- one step's bookkeeping lost, with a reason
        // the model can act on.
        var candidate = chats[chatIndex].skillState ?? AppSkillState()
        candidate.apply(patch: patch)
        let renderedBytes = candidate.rendered.utf8.count
        guard renderedBytes <= AppSkillStatePatch.maxRenderedBytes else {
            skillStateLastError =
                "applying this patch would take the state to \(renderedBytes) bytes; the limit is "
                + "\(AppSkillStatePatch.maxRenderedBytes). Summarize what you are carrying."
            return stripped
        }
        skillStateLastError = nil
        chats[chatIndex].skillState = candidate
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
