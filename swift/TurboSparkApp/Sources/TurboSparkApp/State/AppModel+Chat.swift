import Foundation

extension AppModel {
    @discardableResult
    public func createChat(projectID: UUID? = nil) -> UUID {
        guard !generating else { return selectedChatID }
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
        return chat.id
    }

    public func selectChat(id: UUID) {
        guard !generating else { return }
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
    }

    public func clearOutput() {
        guard !generating else { return }
        if let index = selectedChatIndex {
            chats[index].messages.removeAll()
            chats[index].contextSummary = nil
            chats[index].updatedAt = Date()
        }
        outputText = ""
        outputReasoningText = ""
        outputPromptText = ""
        diagnostics = nil
        error = nil
        persistChats()
        updateTokenEstimate()
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

