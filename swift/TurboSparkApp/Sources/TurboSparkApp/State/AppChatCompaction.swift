import Foundation
import TurboSpark

/// Context compaction: summarizing a conversation's older turns so the
/// append-only prompt keeps fitting the window.
///
/// Before this existed, a history that outgrew the window ended the turn
/// with an error (`executeGenerationTurn`'s no-room guard), and the
/// `fitWindow` fallback silently DROPPED older turns to fit -- a lossy
/// truncation with no summary. Compaction replaces both for the append-only
/// path: the boundary rows are summarized by the model itself, the summary
/// rides in `contextSummary` (a field that sat unwritten until now), and
/// history assembly skips everything at or before `compactedMessageCount`.
///
/// The rows themselves are never deleted: the transcript, the archive and a
/// ghost vault keep the full conversation, so Clear and the transcript view
/// see exactly what they always did. Only the PROMPT disagrees, by exactly
/// the summarized prefix.
///
/// The bounded SKILL.state path (`buildSkillStateHistory`) is O(1) in step
/// count by design and never compacts; subagent histories are short-lived
/// and also never do.
///
/// Mirrors Claude Code's `services/compact/` in shape -- a threshold, a
/// structured summary prompt, and re-injection ahead of the retained turns
/// -- at the scale this app actually runs at.
enum AppChatCompaction {
    /// Auto-compaction fires when the measured prompt reaches four fifths of
    /// the usable window (`measured * 5 >= usable * 4`). Claude Code's
    /// threshold is a flat 13k buffer on a ~200k context; proportionally
    /// that is far too small a fraction to matter at the 4k-8k windows this
    /// engine ships, so the trigger is a fraction of the USABLE window
    /// (context minus the reply reservation).
    static let triggerNumerator = 4
    static let triggerDenominator = 5

    /// Bounds on how many trailing message rows stay verbatim after a
    /// compaction. A settings file can hold any integer; the arithmetic
    /// downstream assumes at least one recent row survives.
    static let keepRecentRange = 1...8
    static let defaultKeepRecent = 2

    /// Character bounds on the transcript handed to the summarizer. The
    /// summarizer runs on the SAME session and window that just overflowed,
    /// so an uncapped transcript of a huge history can fail the very render
    /// the compaction exists to avoid. Chars, not tokens: this is a coarse
    /// guard well before the tokenizer's exact count.
    static let transcriptCharCap = 100_000
    static let transcriptHeadChars = 60_000
    static let transcriptTailChars = 40_000

    /// Whether a prompt of `measuredTokens` should compact before generating.
    ///
    /// Integer cross-multiplication rather than a float comparison, and a
    /// `usable <= 0` refusal: a window smaller than the reply reservation
    /// cannot be solved by summarizing, and claiming otherwise would loop.
    static func shouldAutoCompact(
        measuredTokens: Int, maxContext: UInt32, reservedForNew: UInt32
    ) -> Bool {
        let usable = Int(maxContext) - Int(reservedForNew)
        guard usable > 0, measuredTokens > 0 else { return false }
        return measuredTokens * triggerDenominator >= usable * triggerNumerator
    }

    /// Clamps a persisted keep-recent count into the legal range.
    static func clampKeepRecent(_ value: Int) -> Int {
        min(max(value, keepRecentRange.lowerBound), keepRecentRange.upperBound)
    }

    /// The row index a compaction summarizes up to, or nil when there is
    /// nothing worth summarizing: fewer rows than the keep-recent count
    /// means the whole conversation IS the recent tail.
    ///
    /// Slicing by MESSAGE ROW is what keeps tool pairing safe: a row carries
    /// its own calls and results together (`AppChatMessage.toolCalls` and
    /// `toolResults`), so a boundary can never split a call from its answer.
    static func newBoundary(messageCount: Int, keepRecent: Int) -> Int? {
        let cutoff = messageCount - clampKeepRecent(keepRecent)
        guard cutoff > 0 else { return nil }
        return cutoff
    }

    /// Renders rows to the plain-text transcript the summarizer reads.
    ///
    /// Roles are named rather than implied, tool calls are recorded even when
    /// the row's prose is empty (a rescued call has no prose -- state#31),
    /// and an image row is named rather than silently absent, since the
    /// summarizer sees text only.
    static func renderTranscript(_ rows: [AppChatMessage]) -> String {
        var lines: [String] = []
        for row in rows {
            let speaker: String
            switch row.role {
            case .user: speaker = "USER"
            case .assistant: speaker = "ASSISTANT"
            case .system: speaker = "SYSTEM"
            case .tool: speaker = "TOOL"
            case .developer: speaker = "DEVELOPER"
            @unknown default: speaker = "MESSAGE"
            }
            if !row.content.isEmpty {
                lines.append("\(speaker): \(row.content)")
            }
            let namesByID = Dictionary(
                row.toolCalls.map { ($0.id, $0.name) }, uniquingKeysWith: { first, _ in first })
            for call in row.toolCalls {
                lines.append("\(speaker) called tool \(call.name).")
            }
            for result in row.toolResults {
                let name = namesByID[result.callID] ?? "tool"
                let verdict = result.isError ? "failed" : "returned"
                lines.append("[\(name) \(verdict): \(result.output)]")
            }
            for path in row.imagePaths {
                let name = (path as NSString).lastPathComponent
                lines.append("[\(speaker) attached an image: \(name)]")
            }
        }
        let joined = lines.joined(separator: "\n")
        guard joined.count > transcriptCharCap else { return joined }
        let head = String(joined.prefix(transcriptHeadChars))
        let tail = String(joined.suffix(transcriptTailChars))
        return head + "\n[... middle of the transcript omitted ...]\n" + tail
    }

    /// The two-message prompt the summarizer generates from.
    static func summarizerMessages(
        transcript: String, priorSummary: String?, focus: String?
    ) -> [ChatMessage] {
        var instructions = """
        Summarize the conversation above so it can continue in a smaller \
        context window. Write a dense, factual summary for whoever continues \
        the work, using exactly these sections, each a short paragraph or \
        tight bullet list; omit a section only when the conversation gives \
        it nothing:

        Primary request and intent:
        Key technical concepts and decisions:
        Files, paths and code touched:
        Errors and fixes:
        Pending tasks and open questions:
        Current state of the work:
        Next step:

        Record concrete details - names, paths, commands, numbers - rather \
        than impressions, and add nothing the conversation does not contain.
        """
        if let focus, !focus.isEmpty {
            instructions += "\n\nThe user asked to pay particular attention to: \(focus)"
        }
        if let priorSummary, !priorSummary.isEmpty {
            instructions +=
                "\n\nA summary of even earlier turns follows. Fold it in rather than "
                + "repeating it verbatim:\n\n<previous_summary>\n\(priorSummary)\n</previous_summary>"
        }
        let user =
            "\(instructions)\n\nConversation to summarize:\n\n<transcript>\n\(transcript)\n</transcript>"
        return [
            ChatMessage(
                role: .system,
                content: "You compress conversation transcripts into faithful, structured summaries."),
            ChatMessage(role: .user, content: user),
        ]
    }

    /// The summary as it enters a rebuilt prompt: a user-role message placed
    /// immediately after the system message.
    ///
    /// User-role deliberately: a mid-history `.system` message is refused
    /// outright by the fallback renderers and by real templates that require
    /// alternating roles (state#32), and the summary must not be dropped the
    /// way a refused message would be.
    static func injectionMessage(_ summary: String) -> ChatMessage {
        ChatMessage(
            role: .user,
            content: """
            <context_summary>
            The earlier part of this conversation was summarized to fit the \
            context window. Treat the summary below as authoritative for \
            everything before the next message; those earlier turns are no \
            longer available verbatim.

            \(summary)
            </context_summary>
            """)
    }
}

/// The turn-integrated half of compaction: reading and writing the
/// vault-aware state, running the summarizer, and the two entry points
/// (auto, from the turn flow; manual, from the `/compact` command).
extension AppModel {
    /// What one compaction attempt concluded.
    enum CompactionOutcome {
        case compacted(rowsSummarized: Int)
        case nothingToCompact
        /// The user stopped the summarizer. Not a failure: both callers go
        /// quiet, and the turn flow's own `checkCancellation` still ends the
        /// cancelled turn.
        case cancelled
        case failed(String)
    }

    /// A chat's compaction state, decrypting from the vault when ghost.
    /// What history assembly and the transcript divider both read.
    func compactionState(chatID: UUID) -> (summary: String?, boundary: Int) {
        guard let index = chats.firstIndex(where: { $0.id == chatID }) else { return (nil, 0) }
        if chats[index].isGhost {
            let payload = ghostVault.payload(for: chatID)
            return (payload.contextSummary, payload.compactedMessageCount)
        }
        return (chats[index].contextSummary, chats[index].compactedMessageCount)
    }

    /// The selected chat's compaction state, for the transcript divider.
    public var selectedCompactionState: (summary: String?, boundary: Int) {
        compactionState(chatID: selectedChatID)
    }

    /// Writes compaction state through the same primitive as every other
    /// transcript mutation, so ghost chats re-seal and ordinary ones persist.
    /// Internal rather than private: `clampStoredCompaction` (message
    /// editing) writes through the same ghost-aware primitive rather than
    /// restating its two arms.
    func setStoredCompaction(chatIndex: Int, summary: String?, boundary: Int) {
        let chatID = chats[chatIndex].id
        if chats[chatIndex].isGhost {
            mutateGhostPayload(for: chatID) {
                $0.contextSummary = summary
                $0.compactedMessageCount = boundary
            }
        } else {
            chats[chatIndex].contextSummary = summary
            chats[chatIndex].compactedMessageCount = boundary
            chats[chatIndex].updatedAt = Date()
            persistChats()
        }
    }

    /// Summarizes everything between the current boundary and the recent
    /// tail, advances the boundary, and rewrites the summary.
    ///
    /// Runs on the session's serial queue like any generation, so it must
    /// never be called while a turn is generating -- the two callers respect
    /// that (inside the turn's own task, before `generate`; and from the
    /// `/compact` command, which refuses while `generating`).
    func performCompaction(
        chatID: UUID, project: AppProject?, focus: String?, trigger: String, keepRecent: Int
    ) async -> CompactionOutcome {
        guard let session else { return .failed("no model is loaded") }
        guard let chatIndex = chats.firstIndex(where: { $0.id == chatID }) else {
            return .failed("the conversation no longer exists")
        }
        let (priorSummary, boundary) = compactionState(chatID: chatID)
        let allMessages = turnMessages(for: chatID)
        guard let newBoundary = AppChatCompaction.newBoundary(
            messageCount: allMessages.count, keepRecent: keepRecent),
            newBoundary > boundary
        else { return .nothingToCompact }
        let rows = Array(allMessages[boundary..<newBoundary])

        // Informational and content-free (ids only), so a ghost chat's
        // privacy rule is untouched: the payload carries the trigger, never
        // transcript text.
        _ = await dispatchLifecycleHook(
            event: .preCompact, chatID: chatID, project: project, source: trigger)

        let transcript = AppChatCompaction.renderTranscript(rows)
        let messages = AppChatCompaction.summarizerMessages(
            transcript: transcript, priorSummary: priorSummary, focus: focus)

        var options = GenerateOptions()
        // A summary wants the boring setting, not the user's creative one,
        // and no reasoning budget: every reasoning token is one taken from
        // the summary itself.
        options.reasoning = .off
        options.temperature = 0.2
        options.maxNewTokens = 700

        phase = .prefill
        // Every exit past this point -- cancel, error, empty summary, success
        // -- must lower the flag, so a defer rather than an assignment per exit.
        isCompacting = true
        defer { isCompacting = false }
        var summaryText = ""
        do {
            for try await event in session.generate(messages, options: options) {
                try Task.checkCancellation()
                switch event {
                case .content(let chunk):
                    summaryText += chunk
                case .prefill(let done, let total):
                    phase = .prefill
                    livePrefillDone = done
                    livePrefillTotal = total
                case .reasoning, .toolCall, .stopped, .finished:
                    break
                }
            }
        } catch is CancellationError {
            return .cancelled
        } catch {
            return .failed(error.localizedDescription)
        }
        let summary = summaryText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !summary.isEmpty else {
            return .failed("the model returned an empty summary")
        }

        setStoredCompaction(chatIndex: chatIndex, summary: summary, boundary: newBoundary)
        updateTokenEstimate()
        showToast(
            "Compacted \(rows.count) earlier message\(rows.count == 1 ? "" : "s") into a summary.",
            style: .success)
        return .compacted(rowsSummarized: rows.count)
    }

    /// The automatic entry point, called inside `executeGenerationTurn`'s
    /// task before the window fit. Returns true when the chat compacted and
    /// the caller must REBUILD its history from the new boundary.
    ///
    /// Every failure is fail-open: a skipped estimate, a rejected threshold
    /// or a failed summarizer all fall through to the window fit the turn
    /// would have run anyway, which is today's behaviour with its own error
    /// message. Compaction must never be a new way for a turn to die.
    func runAutoCompactionIfNeeded(
        chatID: UUID, project: AppProject?, rawHistory: [ChatMessage],
        maxContext: UInt32, reservedForNew: UInt32, reasoning: GenerateOptions.Reasoning
    ) async -> Bool {
        guard autoCompactEnabled, let session else { return false }
        guard let measured = try? await session.countTokens(rawHistory, reasoning: reasoning)
        else { return false }
        guard AppChatCompaction.shouldAutoCompact(
            measuredTokens: measured, maxContext: maxContext, reservedForNew: reservedForNew)
        else { return false }
        switch await performCompaction(
            chatID: chatID, project: project, focus: nil, trigger: "auto",
            keepRecent: compactionKeepRecentTurns)
        {
        case .compacted:
            return true
        case .nothingToCompact, .cancelled:
            return false
        case .failed(let message):
            showToast("Auto-compaction failed: \(message)", style: .warning)
            return false
        }
    }

    /// The manual `/compact [focus]` entry point. A meta-command, not a
    /// prompt: it runs in place of a submission, before any
    /// `UserPromptSubmit` hook, and clears the draft like one.
    func handleCompactCommand(_ text: String) {
        let focus = text.dropFirst("/compact".count)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        guard session != nil else {
            error = "Load a model before compacting."
            return
        }
        guard !generating, !submitting, pendingToolCall == nil else { return }
        let chatID = selectedChatID
        promptText = ""
        submitting = true
        // Held on `submissionTask`, the same slot Stop reaches (state#33), so
        // a summarizer that will not finish is cancellable.
        submissionTask = Task {
            defer {
                self.submitting = false
                self.submissionTask = nil
                if !self.generating { self.isCancellationPending = false }
            }
            switch await self.performCompaction(
                chatID: chatID, project: self.turnProject(chatID: chatID),
                focus: focus.isEmpty ? nil : focus, trigger: "manual",
                keepRecent: self.compactionKeepRecentTurns)
            {
            case .compacted, .cancelled:
                break
            case .nothingToCompact:
                self.showToast(
                    "Nothing to compact yet: the recent conversation is the whole of it.",
                    style: .info)
            case .failed(let message):
                self.error = "Compaction failed: \(message)"
            }
        }
    }
}
