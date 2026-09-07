import Foundation
import TurboSpark

/// Message edit, regenerate and branch: transcript surgery plus a re-run.
///
/// The prompt is rebuilt from the transcript on every turn
/// (`buildAppendOnlyHistory`), so none of the three operations touches the
/// engine or the session -- they reshape `turnMessages` and call
/// `executeGenerationTurn(step: 0, chatID:)` again. Every mutation goes
/// through `mutateTurnMessages`, which is what makes all three work inside
/// a ghost chat without a separate path: the vault is the transcript there.
///
/// The pure cores below (`retryApplied`, `editApplied`, `branchedChat`,
/// `applyVariantStep`) are static and session-free so they can be asserted
/// without a Metal device -- the same reason `buildAppendOnlyHistory` is a
/// value-returning function (`swift/CLAUDE.md` Gotcha 26).
extension AppModel {
    // MARK: - Variant math

    /// Every version of a message, oldest first: the inactive alternates
    /// plus the active fields themselves, ordered by `createdAt`. Each write
    /// stamps a fresh `createdAt`, so the order is stable and the active
    /// version is found by timestamp rather than by id -- the row's id
    /// NEVER changes across a swap, because it is the transcript's ForEach
    /// identity.
    static func sortedVersions(of message: AppChatMessage) -> [AppChatMessage] {
        (message.alternates + [message]).sorted { $0.createdAt < $1.createdAt }
    }

    /// The 1-based position of the active version among all versions, and
    /// the total. A message with no alternates is "1 of 1" and the switcher
    /// hides on that.
    public func variantPosition(of message: AppChatMessage) -> (position: Int, count: Int) {
        let versions = Self.sortedVersions(of: message)
        let current = Self.activeVersionIndex(of: message, in: versions)
        return (current + 1, versions.count)
    }

    /// Index of the message's active fields inside `versions`, matching on
    /// timestamp and content; a miss reads as newest, which is where every
    /// write leaves the row.
    static func activeVersionIndex(
        of message: AppChatMessage, in versions: [AppChatMessage]
    ) -> Int {
        versions.firstIndex {
            $0.createdAt == message.createdAt && $0.content == message.content
        } ?? versions.count - 1
    }

    /// Steps a message to a neighbouring version, in place. The previously
    /// active fields go back into `alternates` and the activated one takes
    /// over the struct's own fields; the row's `id` is untouched. No-op at
    /// either end of the range.
    static func applyVariantStep(_ message: inout AppChatMessage, delta: Int) {
        guard delta != 0 else { return }
        var versions = Self.sortedVersions(of: message)
        guard versions.count > 1 else { return }
        let current = Self.activeVersionIndex(of: message, in: versions)
        let target = current + delta
        guard versions.indices.contains(target) else { return }
        let activated = versions.remove(at: target)
        message.content = activated.content
        message.reasoning = activated.reasoning
        message.stopReason = activated.stopReason
        message.toolCalls = activated.toolCalls
        message.toolResults = activated.toolResults
        message.imagePaths = activated.imagePaths
        message.createdAt = activated.createdAt
        message.alternates = versions.map { version in
            var flat = version
            flat.alternates = []
            return flat
        }
    }

    /// Switches a transcript row to a neighbouring version.
    ///
    /// Stepping a USER prompt also steps the response immediately after it
    /// by the same delta, clamped to that response's own range: an edit
    /// creates the prompt variant and its response variant together, so
    /// navigating between prompts must carry their answers along or the
    /// transcript would pair a prompt with a reply written for a different
    /// one.
    public func stepVariant(of messageID: UUID, delta: Int, chatID: UUID? = nil) {
        let targetID = chatID ?? selectedChatID
        guard delta != 0, turnLifecycleIdle else { return }
        mutateTurnMessages(for: targetID) { messages in
            guard let index = messages.firstIndex(where: { $0.id == messageID }) else { return }
            Self.applyVariantStep(&messages[index], delta: delta)
            if messages[index].role == .user,
                index + 1 < messages.count,
                messages[index + 1].role == .assistant {
                Self.applyVariantStep(&messages[index + 1], delta: delta)
            }
        }
        updateTokenEstimate()
    }

    // MARK: - What the operations guard on

    /// Every editing operation rides on top of a quiet turn lifecycle; the
    /// same trio `createChat` and `selectChat` refuse on.
    var turnLifecycleIdle: Bool {
        !generating && !submitting && pendingToolCall == nil
    }

    /// Retry, edit and branch re-run the turn, so there must be a session
    /// to run it on. Separate from `turnLifecycleIdle` because the
    /// affordances hide on it too.
    func editingAllowed(chatID: UUID) -> Bool {
        guard session != nil, turnLifecycleIdle else { return false }
        // A bounded-state project's prompt comes from `skillState`, not from
        // transcript rows; truncating rows there desyncs bookkeeping the
        // prompt no longer reads. Out of scope rather than subtly wrong.
        return !(turnProject(chatID: chatID)?.skillStateEnabled ?? false)
    }

    /// The index of the last REAL user prompt: a `.user` row carrying no
    /// tool results and no memory wrap. Tool-result turns and `#`
    /// quick-saves are `.user` rows too, and neither is a prompt a response
    /// can be regenerated against.
    static func lastPromptAnchorIndex(in messages: [AppChatMessage]) -> Int? {
        messages.lastIndex {
            $0.role == .user
                && $0.toolResults.isEmpty
                && UserMemoryInputMessage.parse($0.content) == nil
        }
    }

    /// Whether the response chain after a prompt is one plain prose reply --
    /// the only shape Retry v1 handles. A chain with tool calls or results
    /// stays out: restoring it as a single alternate would have to carry
    /// every row, and re-rolling an agent's work is a different feature.
    static func isSingleProseResponse(_ chain: [AppChatMessage]) -> Bool {
        chain.count == 1 && chain[0].role == .assistant
            && chain[0].toolCalls.isEmpty && chain[0].toolResults.isEmpty
    }

    // MARK: - Retry

    /// Whether the given assistant message is the last response and Retry
    /// applies to it. What the hover bar reads.
    public func canRetry(response message: AppChatMessage) -> Bool {
        guard editingAllowed(chatID: selectedChatID) else { return false }
        let messages = selectedTurnMessages
        guard let anchorIndex = Self.lastPromptAnchorIndex(in: messages),
            anchorIndex + 1 < messages.count
        else { return false }
        return messages[anchorIndex + 1].id == message.id
            && Self.isSingleProseResponse(Array(messages[(anchorIndex + 1)...]))
    }

    /// Pure: what Retry does to a transcript. Returns the transcript
    /// truncated back to the prompt and the replaced response as the variant
    /// to seed onto the regenerated one, or nil when there is nothing to
    /// retry.
    static func retryApplied(to messages: [AppChatMessage]) -> (
        messages: [AppChatMessage], variants: [AppChatMessage]
    )? {
        guard let anchorIndex = lastPromptAnchorIndex(in: messages),
            anchorIndex + 1 < messages.count,
            isSingleProseResponse(Array(messages[(anchorIndex + 1)...]))
        else { return nil }
        var oldResponse = messages[anchorIndex + 1]
        oldResponse.alternates = []
        let truncated = Array(messages[...anchorIndex])
        return (truncated, [oldResponse])
    }

    /// Regenerates the selected chat's last response in place. The replaced
    /// reply becomes a variant of the new one.
    @discardableResult
    public func regenerateResponse(chatID: UUID? = nil) -> Bool {
        let targetID = chatID ?? selectedChatID
        guard editingAllowed(chatID: targetID) else { return false }
        guard let applied = Self.retryApplied(to: turnMessages(for: targetID)) else {
            return false
        }
        pendingResponseVariants[targetID] = applied.variants
        mutateTurnMessages(for: targetID) { $0 = applied.messages }
        clampStoredCompaction(chatID: targetID, toRow: applied.messages.count - 1)
        updateTokenEstimate()
        executeGenerationTurn(step: 0, chatID: targetID)
        return true
    }

    // MARK: - Edit in place

    /// Whether a row may open the in-place editor: the last real prompt of
    /// an edit-allowed chat.
    public func canEditInPlace(_ message: AppChatMessage) -> Bool {
        guard message.role == .user, editingAllowed(chatID: selectedChatID) else {
            return false
        }
        let messages = selectedTurnMessages
        guard let anchorIndex = Self.lastPromptAnchorIndex(in: messages) else {
            return false
        }
        return messages[anchorIndex].id == message.id
    }

    /// Opens the in-place editor for a row.
    @discardableResult
    public func beginEdit(messageID: UUID) -> Bool {
        guard let message = selectedTurnMessages.first(where: { $0.id == messageID }),
            canEditInPlace(message)
        else { return false }
        editingMessageID = messageID
        return true
    }

    /// Closes the in-place editor without changing anything.
    public func cancelEdit() {
        editingMessageID = nil
    }

    /// Pure: what an in-place prompt edit does to a transcript. The old
    /// prompt becomes the edited row's alternate with its original
    /// timestamp; the edited row takes a fresh one, so it sorts newest. The
    /// response chain after the prompt is removed, and a single prose
    /// response among it comes back as the variant to seed onto the reply
    /// the re-run will write.
    static func editApplied(
        to messages: [AppChatMessage], anchorIndex: Int, newText: String, now: Date
    ) -> (messages: [AppChatMessage], responseVariants: [AppChatMessage]) {
        var oldVersion = messages[anchorIndex]
        oldVersion.alternates = []
        var edited = oldVersion
        edited.content = newText
        edited.createdAt = now
        edited.alternates = [oldVersion]

        var responseVariants: [AppChatMessage] = []
        if anchorIndex + 1 < messages.count {
            let chain = Array(messages[(anchorIndex + 1)...])
            if isSingleProseResponse(chain) {
                var oldResponse = chain[0]
                oldResponse.alternates = []
                responseVariants = [oldResponse]
            }
        }

        var result = Array(messages[...anchorIndex])
        result[result.count - 1] = edited
        return (result, responseVariants)
    }

    /// Commits an in-place edit of the last prompt and re-runs the turn.
    @discardableResult
    public func commitEdit(messageID: UUID, newText: String, chatID: UUID? = nil) -> Bool {
        let targetID = chatID ?? selectedChatID
        let trimmed = newText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        guard editingAllowed(chatID: targetID) else { return false }
        let messages = turnMessages(for: targetID)
        guard let anchorIndex = Self.lastPromptAnchorIndex(in: messages),
            messages[anchorIndex].id == messageID
        else { return false }
        let applied = Self.editApplied(
            to: messages, anchorIndex: anchorIndex, newText: trimmed, now: Date())
        pendingResponseVariants[targetID] = applied.responseVariants
        mutateTurnMessages(for: targetID) { $0 = applied.messages }
        clampStoredCompaction(chatID: targetID, toRow: anchorIndex)
        editingMessageID = nil
        updateTokenEstimate()
        executeGenerationTurn(step: 0, chatID: targetID)
        return true
    }

    // MARK: - Branch

    /// Whether an earlier prompt may open the branch editor: any real user
    /// prompt outside Ghost Mode. Branching COPIES transcript rows into a
    /// new persisted chat, and a ghost chat's contents must never reach
    /// disk, so Ghost Mode refuses the operation outright rather than
    /// laundering the vault through a normal row.
    public func canBranch(_ message: AppChatMessage) -> Bool {
        guard message.role == .user else { return false }
        guard let chat = chats.first(where: { $0.id == selectedChatID }), !chat.isGhost else {
            return false
        }
        guard UserMemoryInputMessage.parse(message.content) == nil else { return false }
        return editingAllowed(chatID: selectedChatID)
    }

    /// Pure: the branch chat a fork produces -- transcript prefix up to and
    /// including the edited prompt, the prompt replaced, the compaction
    /// summary carried over with its boundary clamped inside the prefix,
    /// and the source row untouched. Todos and artifacts stay behind: they
    /// describe the ORIGINAL run's work, and a branch is a conversation
    /// fork, not a task fork.
    static func branchedChat(
        from source: AppChat, messageIndex: Int, messages: [AppChatMessage],
        newText: String, summary: String?, boundary: Int, now: Date
    ) -> AppChat {
        var branch = AppChat(projectID: source.projectID)
        branch.title =
            source.title.isEmpty || source.title == "New Chat"
            ? String(newText.prefix(40)).replacingOccurrences(of: "\n", with: " ")
            : source.title + " (branch)"
        branch.systemPrompt = source.systemPrompt
        var prefix = Array(messages[...messageIndex])
        prefix[messageIndex].content = newText
        prefix[messageIndex].createdAt = now
        prefix[messageIndex].alternates = []
        branch.messages = prefix
        branch.contextSummary = summary
        branch.compactedMessageCount = min(boundary, messageIndex)
        return branch
    }

    /// Forks the conversation at a user prompt into a new chat with the
    /// edited text in place, selects the branch and re-runs from there.
    /// The original conversation is left exactly as it was.
    @discardableResult
    public func branchFrom(messageID: UUID, editedText: String, chatID: UUID? = nil) -> UUID? {
        let sourceID = chatID ?? selectedChatID
        let trimmed = editedText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        guard editingAllowed(chatID: sourceID) else { return nil }
        guard let sourceIndex = chats.firstIndex(where: { $0.id == sourceID }),
            !chats[sourceIndex].isGhost
        else { return nil }
        let source = chats[sourceIndex]
        let messages = turnMessages(for: sourceID)
        guard let messageIndex = messages.firstIndex(where: { $0.id == messageID }),
            messages[messageIndex].role == .user,
            UserMemoryInputMessage.parse(messages[messageIndex].content) == nil
        else { return nil }
        let compaction = compactionState(chatID: sourceID)
        let branch = Self.branchedChat(
            from: source, messageIndex: messageIndex, messages: messages,
            newText: trimmed, summary: compaction.summary, boundary: compaction.boundary,
            now: Date())
        chats.insert(branch, at: 0)
        selectedChatID = branch.id
        persistChats()
        updateTokenEstimate()
        let branchID = branch.id
        let branchProject = projects.first { $0.id == branch.projectID }
        Task {
            _ = await self.dispatchLifecycleHook(
                event: .sessionStart, chatID: branchID, project: branchProject,
                source: "clear")
        }
        executeGenerationTurn(step: 0, chatID: branchID)
        return branchID
    }

    // MARK: - Compaction clamp

    /// Brings a chat's compaction boundary down to `row` when the summary
    /// covers it. Assembly skips rows below the boundary, so a summary
    /// reaching past the prompt a retry or edit is about to re-run would
    /// send the model a turn with no prompt in it. Rows [0, row) keep their
    /// summary; the row itself goes back live.
    func clampStoredCompaction(chatID: UUID, toRow row: Int) {
        let state = compactionState(chatID: chatID)
        guard state.boundary > row else { return }
        guard let chatIndex = chats.firstIndex(where: { $0.id == chatID }) else { return }
        setStoredCompaction(
            chatIndex: chatIndex, summary: state.summary, boundary: row)
    }
}
