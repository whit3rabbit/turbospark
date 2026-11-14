import Foundation

/// ZP-02: the per-chat read marker. Unread is derived (`AppChat.hasUnreadReply`
/// and not selected); the only state is `AppChat.lastReadAt`, advanced when
/// the user selects a chat and when they leave it, so a reply that landed
/// while they were looking counts as seen. The store writes only when a
/// marker actually moves.
extension AppModel {
    /// The slot the sidebar draws for `chat`.
    func chatRowSlot(for chat: AppChat) -> ChatRowSlot {
        ChatRowSlot.resolve(
            turnState: chatTurnState(for: chat.id),
            isSelected: chat.id == selectedChatID,
            hasUnreadReply: chat.hasUnreadReply)
    }

    /// Advances the marker of `id` to now when it holds a reply it has not
    /// recorded as seen. Returns whether anything changed.
    @discardableResult
    func markChatRead(id: UUID, now: Date = Date()) -> Bool {
        guard let index = chats.firstIndex(where: { $0.id == id }),
            chats[index].hasUnreadReply
        else { return false }
        chats[index].lastReadAt = now
        persistChatsDebounced()
        return true
    }

    /// Called from `selectedChatID.didSet`: the chat being left and the chat
    /// being opened are both "seen" at this moment.
    func selectionChangedForReadState(from old: UUID) {
        guard old != selectedChatID else { return }
        markChatRead(id: old)
        markChatRead(id: selectedChatID)
    }
}
