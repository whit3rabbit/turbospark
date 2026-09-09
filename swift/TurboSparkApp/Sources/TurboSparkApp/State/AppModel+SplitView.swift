import Foundation

/// The qwen-code split-view parity: up to three additional chats rendered
/// beside the main conversation as live follow panes. A pane is READ-ONLY by
/// design -- this app's composer, streaming state and approval cards are
/// bound to the one selected chat, and a second live composer would need a
/// second submission pipeline to be honest. The pane's header promotes a
/// chat to the main position, which is how a pane becomes editable.
extension AppModel {
    /// How many secondary panes fit beside the main conversation before the
    /// window stops being readable.
    @MainActor public static let maxSplitPanes = 3
    static let splitPaneDefaultsKey = "TurboSpark.splitChatIDs"

    /// The resolved panes: only chats that still exist, ghost chats
    /// excluded. A ghost's transcript is vault-scoped and its row is empty,
    /// so a ghost pane would render a chat the user cannot see anywhere
    /// else either.
    public var splitPaneChats: [AppChat] {
        splitChatIDs.compactMap { id in
            chats.first { $0.id == id && !$0.isGhost }
        }
    }

    /// Whether the chat is already on screen as a pane. The sidebar menu
    /// reads this to offer Open instead of Add.
    public func isSplitPane(chatID: UUID) -> Bool {
        splitChatIDs.contains(chatID)
    }

    /// Whether anything is visibly working in a chat: the main
    /// conversation's own turn, or a background agent parked on it. A pane's
    /// status dot reads this; "running" is the honest word for both.
    public func isChatRunning(chatID: UUID) -> Bool {
        if chatID == selectedChatID && isRunning { return true }
        return backgroundAgentRuns.values.contains {
            $0.chatID == chatID && $0.status == "running"
        }
    }

    /// Adds a pane, dropping the oldest past the cap. A chat already in a
    /// pane is a no-op; the selected chat cannot become a pane of itself.
    public func addSplitPane(chatID: UUID) {
        guard chatID != selectedChatID else { return }
        guard let chat = chats.first(where: { $0.id == chatID }), !chat.isGhost else { return }
        guard !splitChatIDs.contains(chatID) else { return }
        var next = splitChatIDs
        next.append(chatID)
        while next.count > Self.maxSplitPanes {
            next.removeFirst()
        }
        splitChatIDs = next
        persistSplitPaneIDs()
    }

    /// Adds the pane or removes it, for the sidebar menu's checkmark item.
    public func toggleSplitPane(chatID: UUID) {
        if let index = splitChatIDs.firstIndex(of: chatID) {
            splitChatIDs.remove(at: index)
            persistSplitPaneIDs()
        } else {
            addSplitPane(chatID: chatID)
        }
    }

    /// Closes one pane.
    public func closeSplitPane(chatID: UUID) {
        splitChatIDs.removeAll { $0 == chatID }
        persistSplitPaneIDs()
    }

    /// Selects the pane's chat as the main conversation and drops it from
    /// the pane list, swapping the main chat INTO the pane it came from so
    /// nothing leaves the screen.
    public func promoteSplitPane(chatID: UUID) {
        guard chatID != selectedChatID else { return }
        guard chats.first(where: { $0.id == chatID }) != nil else { return }
        guard !generating, !submitting, pendingToolCall == nil else {
            showToast("Stop the current turn before switching chats.", style: .warning)
            return
        }
        let previous = selectedChatID
        selectChat(id: chatID)
        if let index = splitChatIDs.firstIndex(of: chatID) {
            splitChatIDs[index] = previous
            persistSplitPaneIDs()
        }
    }

    private func persistSplitPaneIDs() {
        UserDefaults.standard.set(
            splitChatIDs.map(\.uuidString), forKey: Self.splitPaneDefaultsKey)
    }

    static func loadSplitPaneIDs() -> [UUID] {
        guard let strings = UserDefaults.standard.stringArray(forKey: splitPaneDefaultsKey) else {
            return []
        }
        return strings.compactMap { UUID(uuidString: $0) }
    }
}
