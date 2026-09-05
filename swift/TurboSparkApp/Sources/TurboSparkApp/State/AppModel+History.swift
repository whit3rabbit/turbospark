import Foundation
import TurboSpark

/// The prompt a turn is built from, and the estimate the composer shows for
/// it.
///
/// The BOUNDED prompt is `AppModel+SkillState`'s. This is the append-only
/// arm -- the default, and the one that grows with every turn and every tool
/// result until a 4,096-context install stops the run outright between step
/// 30 and 35 (`docs/SKILL_STATE.md`).
extension AppModel {
    /// The append-only prompt: the system message, then every turn and every
    /// tool result in order.
    ///
    /// **A VALUE-RETURNING FUNCTION RATHER THAN INLINE ASSEMBLY**, for
    /// `swift/CLAUDE.md` Gotcha 26's reason: `executeGenerationTurn` returns
    /// at its `session` guard, so nothing about the prompt it builds could be
    /// asserted without a Metal device and a 13 GB install -- and both of the
    /// defects this function carries (state#31, state#32) are silent, so
    /// there was nothing to notice either.
    func buildAppendOnlyHistory(chatIndex: Int, project: AppProject?) -> [ChatMessage] {
        var history: [ChatMessage] = []
        let systemContent = buildSystemPrompt(
            for: project,
            userPrompt: resolvedUserSystemPrompt(chatIndex: chatIndex))
        if !systemContent.isEmpty {
            history.append(ChatMessage(role: .system, content: systemContent))
        }

        for msg in chats[chatIndex].messages {
            // **AN IMAGE-ONLY TURN HAS NO TEXT AND IS STILL A TURN.** This
            // guard predates images and would drop one entirely, leaving the
            // model to answer a question whose picture was never sent -- the
            // emptiness-guard failure in its usual shape.
            //
            // **AND A TOOL TURN IS ONE TOO** (state#31). A call the guardrail
            // engine RESCUED out of raw text has its prose sanitized away, so
            // a reply that was nothing but the call arrives here with empty
            // content and a non-empty `toolResults` -- and this guard skipped
            // the whole message before the result loop below ever ran. The
            // model then saw no record of having called anything, re-issued
            // the same call, and burned the step cap doing it.
            let carriesToolTurn = !msg.toolResults.isEmpty || !msg.toolCalls.isEmpty
            guard !msg.content.isEmpty || !msg.imagePaths.isEmpty || carriesToolTurn else {
                continue
            }
            if !msg.content.isEmpty || !msg.imagePaths.isEmpty {
                // **AN UNFINISHED TURN IS MARKED AS ONE** (state#65).
                // `finishCancelled` persists whatever the turn produced
                // before it stopped, with `stopReason` recording WHY -- and
                // nothing read it, so a reply truncated by an engine error or
                // by the user pressing Stop was replayed on the next step as
                // a completed assistant turn. The model then built on half a
                // sentence as though it had meant to stop there.
                let content = Self.truncationNote(for: msg).map { "\(msg.content)\n\n\($0)" }
                    ?? msg.content
                history.append(
                    ChatMessage(
                        role: msg.role,
                        content: content,
                        images: msg.imagePaths.map(ChatImage.path)))
            }
            // **A TOOL RESULT GOES BACK AS `.tool`, NOT AS `.system`**
            // (state#32). `ChatMessage.Role.tool` exists and the FFI maps it.
            // A mid-history `.system` message is REFUSED outright by the
            // Gemma, ChatML and DeepSeek fallback renderers ("system message
            // must be first") and by several real Jinja templates that
            // require alternating roles -- and `fit_window` prices a failing
            // render at `u64::MAX`, so it drops turns until the history stops
            // failing rather than reporting anything. The run silently loses
            // its own history, or errors at step 2 with a message naming
            // neither cause.
            for res in msg.toolResults {
                let tag = res.isError ? "tool_error" : "tool_response"
                history.append(
                    ChatMessage(role: .tool, content: "<\(tag)>\n\(res.output)\n</\(tag)>"))
            }
        }
        return history
    }

    /// The note appended to a turn that did not finish, or nil for one that
    /// did (state#65).
    ///
    /// The reasons are the two `finishCancelled` writes plus the guardrail
    /// retry, which is the same shape: text the model did not choose to end.
    /// A pure static so it can be asserted without a session.
    static func truncationNote(for message: AppChatMessage) -> String? {
        guard message.role == .assistant, let reason = message.stopReason else { return nil }
        switch reason {
        case "cancelled":
            return "[This reply was stopped by the user before it finished.]"
        case "error":
            return "[This reply was cut off by an engine error before it finished.]"
        case "context_overflow":
            return "[This reply was cut off: the conversation no longer fits the context window.]"
        default:
            return nil
        }
    }

    public func updateTokenEstimate() {
        guard let session else {
            estimatedPromptTokens = 0
            return
        }
        // **THIS ESTIMATE IS A FLOOR ON A TURN CARRYING IMAGES, NOT A
        // COUNT.** `countTokens` renders the template and encodes it, and the
        // template emits ONE marker per image whatever its size -- the
        // expansion to that page's merged-token count happens later, in the
        // engine's splice, which needs the preprocessed grid. So an image
        // turn is undercounted by roughly a page's worth of positions. The
        // images are passed anyway so the estimate tracks what is actually
        // sent rather than a different conversation.
        // **THE SYSTEM MESSAGE COUNTS, AND IT DID NOT USED TO.** This estimate
        // omitted it entirely, which was a small undercount while the prompt
        // was project-derived and Chat mode had none at all. A user-authored
        // default applies to EVERY chat, so leaving it out would under-report
        // the composer's context fill by the whole prompt on every turn --
        // and the meter exists to tell a user how close to the window they
        // are. Assembled the same way `buildAppendOnlyHistory` does, so the
        // two cannot describe different conversations.
        var history: [ChatMessage] = []
        let systemContent = buildSystemPrompt(
            for: turnProject(chatID: selectedChatID),
            // BY CHAT rather than by index: `selectedChat` falls back to the
            // transient draft, which is not in `chats` and has no index.
            userPrompt: resolvedUserSystemPrompt(chat: selectedChat))
        if !systemContent.isEmpty {
            history.append(ChatMessage(role: .system, content: systemContent))
        }
        history += selectedChat.messages.compactMap { msg -> ChatMessage? in
            guard !msg.content.isEmpty || !msg.imagePaths.isEmpty else { return nil }
            return ChatMessage(
                role: msg.role,
                content: msg.content,
                images: msg.imagePaths.map(ChatImage.path))
        }
        if !promptText.isEmpty {
            history.append(ChatMessage(role: .user, content: promptText))
        }

        // **DEBOUNCED, AND SKIPPED WHILE GENERATING.** `promptText`'s setter
        // calls this on every keystroke, and `countTokens` dispatches onto
        // the session's SERIAL queue -- which a running generation holds for
        // the whole turn. `tokenEstimateTask?.cancel()` cannot recall work
        // already queued behind it, so typing during a turn enqueued one full
        // template render per character to run after the turn finished.
        // Cancelling here is what keeps the queue empty in the first place.
        tokenEstimateTask?.cancel()
        guard !generating else { return }
        tokenEstimateTask = Task {
            try? await Task.sleep(nanoseconds: 250_000_000)
            guard !Task.isCancelled else { return }
            if let count = try? await session.countTokens(history, reasoning: self.reasoning) {
                if !Task.isCancelled {
                    self.estimatedPromptTokens = count
                }
            }
        }
    }
}
