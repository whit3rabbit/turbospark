import Foundation

extension AppModel {
    @discardableResult
    public func createChat(projectID: UUID? = nil) -> UUID {
        // `pendingToolCall == nil` for `selectChat`'s reason (state#49):
        // `generating` is false while a call waits, so this was legal, and
        // starting a new chat there strands the approval card on a
        // conversation the user can no longer see.
        guard !generating, pendingToolCall == nil else { return selectedChatID }
        activeSection = .chat
        let assignedProjectID = projectID ?? selectedProjectID
        // If current chat is already empty and matches the target project, reset and stay on it
        if selectedChat.messages.isEmpty && selectedChat.draft.isEmpty && selectedChat.draftAttachments.isEmpty && selectedChat.projectID == assignedProjectID {
            outputText = ""
            outputReasoningText = ""
            outputPromptText = ""
            updateTokenEstimate()
            return selectedChatID
        }
        let chat = AppChat(projectID: assignedProjectID)
        chats.insert(chat, at: 0)
        selectedChatID = chat.id
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
        persistChats()
        updateTokenEstimate()
        let createdID = chat.id
        let createdProject = projects.first { $0.id == assignedProjectID }
        Task {
            _ = await self.dispatchLifecycleHook(
                event: .sessionStart, chatID: createdID, project: createdProject, source: "clear")
        }
        return chat.id
    }

    public func selectChat(id: UUID) {
        // A pending tool call already captures its own chat ID at proposal
        // time (state#9), so approving/denying it lands in the right place
        // regardless; this guard is the UX half -- switching away mid
        // approval, with `generating` already false, read as an ordinary
        // chat switch and made it easy to lose track of which chat is
        // waiting on a decision.
        guard !generating, pendingToolCall == nil else { return }
        activeSection = .chat
        guard id != selectedChatID else { return }
        selectedChatID = id
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
        persistChats()
        updateTokenEstimate()
    }

    public func renameChat(id: UUID, title: String) {
        guard let index = chats.firstIndex(where: { $0.id == id }) else { return }
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        chats[index].title = String(trimmed.prefix(80))
        chats[index].updatedAt = Date()
        persistChats()
    }

    public func deleteChat(id: UUID) {
        guard !generating, let index = chats.firstIndex(where: { $0.id == id }) else { return }
        // Captured before the removal below: `SessionEnd` describes the chat
        // that is going away, and after `chats.remove` there is no row left
        // to resolve its project from (state#67).
        let deletedProject = project(forChat: id)
        // A pending approval on THIS chat has no chat left to land its
        // decision in once it's gone -- Approve/Deny would silently no-op
        // against a stale ID (state#9's own reasoning, applied to deletion
        // rather than to a chat switch).
        if pendingToolCallChatID == id {
            // Through the helper, not three of its four fields open-coded
            // (state#49): `pendingToolCallProject` was left behind, which is
            // exactly the stale value the helper exists to prevent.
            clearPendingToolCall()
        }
        chats.remove(at: index)
        if chats.isEmpty {
            selectedChatID = UUID()
        } else if selectedChatID == id {
            selectedChatID = chats[min(index, chats.count - 1)].id
        }
        outputText = ""
        outputReasoningText = ""
        persistChats()
        updateTokenEstimate()
        Task {
            // The grants belonged to a conversation that no longer exists,
            // and the ids are UUIDs, so nothing would ever collect them
            // (state#49).
            await SessionApprovalStore.shared.clear(sessionID: id.uuidString)
            _ = await self.dispatchSessionEnd(
                reason: "clear", chatID: id, project: deletedProject)
        }
    }

    /// Clears the conversation.
    ///
    /// **FOUR THINGS OUTLIVED IT** (state#49), each of which describes the run
    /// that was just erased: the SKILL.state bookkeeping, a call still
    /// awaiting approval, the checklist, and the session's "always allow"
    /// grants. `resetSkillState()` and `SessionApprovalStore.clear` both
    /// existed for this and neither had a caller -- so a cleared conversation
    /// came back with a tool pre-approved on the strength of a call the user
    /// could no longer read.
    public func clearOutput() {
        guard !generating else { return }
        let clearedChatID = selectedChatID
        let clearedProject = project(forChat: clearedChatID)
        if let index = selectedChatIndex {
            chats[index].messages.removeAll()
            chats[index].contextSummary = nil
            chats[index].skillState = nil
            chats[index].todos = []
            chats[index].updatedAt = Date()
        }
        skillStateLastError = nil
        if pendingToolCallChatID == clearedChatID {
            clearPendingToolCall()
        }
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
        diagnostics = nil
        error = nil
        persistChats()
        updateTokenEstimate()
        Task {
            await SessionApprovalStore.shared.clear(sessionID: clearedChatID.uuidString)
            _ = await self.dispatchSessionEnd(
                reason: "clear", chatID: clearedChatID, project: clearedProject)
        }
    }

    public func addPromptAttachment(_ attachment: AppPromptAttachment, toChatID chatID: UUID? = nil) {
        guard !generating else {
            error = "Cannot attach files while generating."
            return
        }
        let targetID = chatID ?? selectedChatID
        if let index = chats.firstIndex(where: { $0.id == targetID }) {
            chats[index].draftAttachments.append(attachment)
            chats[index].updatedAt = Date()
            persistChats()
        } else {
            var chat = AppChat(id: targetID)
            chat.draftAttachments.append(attachment)
            chats.insert(chat, at: 0)
            selectedChatID = targetID
            persistChats()
        }
        updateTokenEstimate()
    }

    public func removePromptAttachment(id: UUID) {
        guard !generating, let index = selectedChatIndex else { return }
        chats[index].draftAttachments.removeAll { $0.id == id }
        chats[index].updatedAt = Date()
        persistChats()
        updateTokenEstimate()
    }

    public func clearPromptAttachments() {
        guard !generating, let index = selectedChatIndex else { return }
        chats[index].draftAttachments.removeAll()
        chats[index].updatedAt = Date()
        persistChats()
        updateTokenEstimate()
    }

    /// Returns chats filtered by the active project, ordered by most recently updated.
    public var orderedChats: [AppChat] {
        filteredChats.sorted { $0.updatedAt > $1.updatedAt }
    }

    /// Selects the previous chat in the sorted conversation list.
    public func selectPreviousChat() {
        guard !generating else { return }
        let list = orderedChats
        guard !list.isEmpty else { return }
        guard let currentIndex = list.firstIndex(where: { $0.id == selectedChatID }) else {
            if let first = list.first { selectChat(id: first.id) }
            return
        }
        if currentIndex > 0 {
            selectChat(id: list[currentIndex - 1].id)
        }
    }

    /// Selects the next chat in the sorted conversation list.
    public func selectNextChat() {
        guard !generating else { return }
        let list = orderedChats
        guard !list.isEmpty else { return }
        guard let currentIndex = list.firstIndex(where: { $0.id == selectedChatID }) else {
            if let first = list.first { selectChat(id: first.id) }
            return
        }
        if currentIndex < list.count - 1 {
            selectChat(id: list[currentIndex + 1].id)
        }
    }
}

