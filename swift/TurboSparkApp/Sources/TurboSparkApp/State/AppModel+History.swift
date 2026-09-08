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
    /// The ONE place the compaction summary enters a prompt: immediately
    /// after the system message, or at the front when there is none. Both
    /// builders below call this, so the prompt and the token estimate cannot
    /// describe different summaries.
    func insertingSummaryInjection(
        _ history: [ChatMessage], summary: String?
    ) -> [ChatMessage] {
        guard let summary, !summary.isEmpty else { return history }
        let injection = AppChatCompaction.injectionMessage(summary)
        let at = history.firstIndex { $0.role != .system } ?? history.count
        return history[..<at] + [injection] + history[at...]
    }

    func buildAppendOnlyHistory(chatIndex: Int, project: AppProject?) -> [ChatMessage] {
        var history: [ChatMessage] = []
        let systemContent = buildSystemPrompt(
            for: project,
            userPrompt: resolvedUserSystemPrompt(chatIndex: chatIndex))
        if !systemContent.isEmpty {
            history.append(ChatMessage(role: .system, content: systemContent))
        }

        // Vault-aware: a ghost chat's decrypted transcript is what the model
        // gets, exactly as a normal chat's row is.
        let chatID = chats[chatIndex].id
        let compaction = compactionState(chatID: chatID)
        for (rowIndex, msg) in turnMessages(for: chatID).enumerated() {
            // Rows the summary replaces never reach the prompt. They stay in
            // the transcript and on disk; only this assembly skips them.
            if rowIndex < compaction.boundary { continue }
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
        // **SYSTEM REMINDERS ARE ASSEMBLY-TIME, NEVER STORED** (see
        // `SystemReminders`). Appended to the model-bound copy of the last
        // user turn, which is where the current step's prompt ends anyway.
        // The persisted transcript row keeps exactly what the user sent, so
        // a reminder that qualified on step 3 leaves no residue when it no
        // longer qualifies on step 4.
        if let reminder = SystemReminders.reminder(
            todos: chats[chatIndex].todos,
            messages: turnMessages(for: chatID),
            planModeActive: PlanModeExecutor.isPlanModeActive(for: chatID),
            goal: activeGoals[chatID]) {
            if let lastUser = history.lastIndex(where: { $0.role == .user }) {
                history[lastUser].content += "\n\n\(reminder)"
            }
        }
        return insertingSummaryInjection(history, summary: compaction.summary)
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

    /// The next turn's prompt as the exact estimator and the context
    /// breakdown both see it: the flat message list the template render
    /// counts, plus the same content in labeled pieces for per-section
    /// counting, plus the one thing neither count can see (attachments,
    /// priced at the characters-per-token approximation).
    ///
    /// ONE builder, two consumers, for the same reason the estimate and
    /// `buildAppendOnlyHistory` were already kept on one path: the meter and
    /// the breakdown cannot be allowed to describe different conversations.
    func buildEstimateParts() -> (
        history: [ChatMessage], pieces: [ContextUsagePiece], attachmentTokens: Int
    ) {
        var history: [ChatMessage] = []
        var pieces: [ContextUsagePiece] = []
        let sections = buildSystemPromptSections(
            for: turnProject(chatID: selectedChatID),
            // BY CHAT rather than by index: `selectedChat` falls back to the
            // transient draft, which is not in `chats` and has no index.
            userPrompt: resolvedUserSystemPrompt(chat: selectedChat))
        let systemContent = sections.map(\.content).joined(separator: "\n\n")
        if !systemContent.isEmpty {
            history.append(ChatMessage(role: .system, content: systemContent))
        }
        // **THE SYSTEM MESSAGE COUNTS, AND IT DID NOT USED TO.** This estimate
        // omitted it entirely, which was a small undercount while the prompt
        // was project-derived and Chat mode had none at all. A user-authored
        // default applies to EVERY chat, so leaving it out would under-report
        // the composer's context fill by the whole prompt on every turn --
        // and the meter exists to tell a user how close to the window they
        // are. Assembled the same way `buildAppendOnlyHistory` does, so the
        // two cannot describe different conversations.
        //
        // The breakdown slices the system message back into labeled groups;
        // the exact count always sees the joined form above.
        func joined(_ kinds: [AppModel.SystemPromptSection]) -> String {
            sections.filter { kinds.contains($0.section) }.map(\.content).joined(separator: "\n\n")
        }
        let systemPiece = joined([.userPrompt, .agentPrompt, .workspace, .projectRules])
        if !systemPiece.isEmpty {
            pieces.append(ContextUsagePiece(kind: .system, label: "System prompt", content: systemPiece))
        }
        let memoryPiece = joined([.memory])
        if !memoryPiece.isEmpty {
            pieces.append(ContextUsagePiece(kind: .memory, label: "Memory", content: memoryPiece))
        }
        let toolsPiece = joined([.tools])
        if !toolsPiece.isEmpty {
            pieces.append(ContextUsagePiece(kind: .tools, label: "Tools & skills", content: toolsPiece))
        }
        let mcpPiece = joined([.mcpServers])
        if !mcpPiece.isEmpty {
            pieces.append(ContextUsagePiece(kind: .tools, label: "MCP servers", content: mcpPiece))
        }

        // Same boundary skip and same injection the prompt builder applies
        // (both through `insertingSummaryInjection`), so the meter prices
        // the prompt that would actually be sent rather than the
        // uncompacted one.
        //
        // **THE IMAGE PATHS ARE PASSED EVEN THOUGH THE COUNT IS A FLOOR.**
        // A render emits ONE marker per image whatever its size; the
        // expansion to that page's merged-token count happens later, in the
        // engine's splice. The paths ride along anyway so the estimate
        // tracks the conversation that is actually sent.
        let compaction = compactionState(chatID: selectedChatID)
        var conversationParts: [String] = []
        for (rowIndex, msg) in selectedTurnMessages.enumerated() {
            if rowIndex < compaction.boundary { continue }
            guard !msg.content.isEmpty || !msg.imagePaths.isEmpty else { continue }
            history.append(
                ChatMessage(
                    role: msg.role,
                    content: msg.content,
                    images: msg.imagePaths.map(ChatImage.path)))
            if !msg.content.isEmpty {
                conversationParts.append(msg.content)
            }
        }
        let conversationPiece = conversationParts.joined(separator: "\n\n")
        if !conversationPiece.isEmpty {
            pieces.append(
                ContextUsagePiece(kind: .conversation, label: "Conversation", content: conversationPiece))
        }
        history = insertingSummaryInjection(history, summary: compaction.summary)
        if let summary = compaction.summary, !summary.isEmpty {
            pieces.append(
                ContextUsagePiece(kind: .summary, label: "Compaction summary", content: summary))
        }
        if !promptText.isEmpty {
            history.append(ChatMessage(role: .user, content: promptText))
            pieces.append(ContextUsagePiece(kind: .draft, label: "Draft", content: promptText))
        }

        // Attachments ride in no ChatMessage (one template marker per image
        // is all a render emits), so they are approximated at the same
        // characters-per-token the headline has always used for them.
        let attachmentCharacters = promptAttachments.reduce(0) { $0 + $1.characterCount }
        return (history, pieces, attachmentCharacters / 4)
    }

    public func updateTokenEstimate() {
        guard let session else {
            estimatedPromptTokens = 0
            contextUsageSummary = nil
            return
        }
        let parts = buildEstimateParts()

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
            var exact: Int?
            if let count = try? await session.countTokens(parts.history, reasoning: self.reasoning) {
                if !Task.isCancelled {
                    self.estimatedPromptTokens = count
                }
                exact = count
            }
            // The breakdown rides the SAME task and snapshot as the exact
            // count, so the ring, the status bar and the popover can never
            // be looking at different conversations.
            await self.countAndStoreContextUsage(
                pieces: parts.pieces,
                attachmentTokens: parts.attachmentTokens,
                exactTokens: exact,
                session: session)
        }
    }
}
