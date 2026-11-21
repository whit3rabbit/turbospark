import Foundation

/// One "pick up where you left off" row on the empty chat screen.
struct EmptyChatResumeItem: Equatable, Identifiable {
    /// Why the chat counts as unfinished, highest priority first.
    enum Reason: Equatable {
        /// The last reply did not finish (engine error, full context, a limit).
        case stopped(MessageStopNotice)
        /// A `/goal` is still set, so the agent was not done.
        case goal
        /// Text or attachments were typed and never sent.
        case draft
    }

    let chatID: UUID
    let title: String
    let reason: Reason
    var id: UUID { chatID }
}

/// What the empty chat screen offers besides the composer: unfinished chats
/// and, inside a project, that project's recent chats.
///
/// Pure so the selection is testable without views. Bounded on purpose: it
/// reads only fields already on each `AppChat` row (the last message, the
/// draft, the goal), never walks a transcript, because the hero re-renders on
/// every composer keystroke.
enum EmptyChatHome {
    static let resumeLimit = 3
    static let projectRecentLimit = 3

    /// The unfinished reason for `chat`, or nil when it needs nothing.
    ///
    /// Only the LAST message is read: an earlier failed reply that the user
    /// already retried past is history. A user stop is excluded because the
    /// user chose it (`MessageStopNotice.Severity.neutral`).
    static func resumeReason(for chat: AppChat) -> EmptyChatResumeItem.Reason? {
        if let last = chat.messages.last,
            let notice = MessageStopNotice.resolve(last),
            notice != .stoppedByUser
        {
            return .stopped(notice)
        }
        if chat.goal != nil { return .goal }
        // `contains` short-circuits, so a long draft costs one scan to its
        // first visible character, not a full trim per render.
        if chat.draft.contains(where: { !$0.isWhitespace }) || !chat.draftAttachments.isEmpty {
            return .draft
        }
        return nil
    }

    /// Up to `limit` unfinished chats, most recently updated first.
    ///
    /// `chats` should be `AppModel.filteredChats`; ghost and archived rows
    /// are dropped here again so a caller passing the raw list cannot leak a
    /// ghost chat onto this screen. An empty ghost chat shows nothing: Ghost
    /// Mode's own screen must not list the user's saved chats. Busy chats
    /// (generating, waiting on an approval) belong to the sidebar inbox.
    static func resumeItems(
        chats: [AppChat],
        selectedChatID: UUID,
        selectedIsGhost: Bool,
        isBusy: (UUID) -> Bool,
        limit: Int = resumeLimit
    ) -> [EmptyChatResumeItem] {
        guard !selectedIsGhost, limit > 0 else { return [] }
        let candidates: [(chat: AppChat, reason: EmptyChatResumeItem.Reason)] = chats.compactMap { chat in
            guard isListable(chat, selectedChatID: selectedChatID), !isBusy(chat.id),
                let reason = resumeReason(for: chat)
            else { return nil }
            return (chat, reason)
        }
        return candidates
            .sorted { $0.chat.updatedAt > $1.chat.updatedAt }
            .prefix(limit)
            .map { EmptyChatResumeItem(chatID: $0.chat.id, title: $0.chat.title, reason: $0.reason) }
    }

    /// Up to `limit` recent chats in `projectID`, in sidebar order, skipping
    /// empty chats and the rows already shown under Resume.
    static func projectRecents(
        chats: [AppChat],
        projectID: UUID?,
        selectedChatID: UUID,
        selectedIsGhost: Bool,
        excluding: Set<UUID>,
        limit: Int = projectRecentLimit
    ) -> [AppChat] {
        guard let projectID, !selectedIsGhost, limit > 0 else { return [] }
        let inProject = chats.filter { chat in
            chat.projectID == projectID
                && isListable(chat, selectedChatID: selectedChatID)
                && !excluding.contains(chat.id)
                && !chat.messages.isEmpty
        }
        return Array(AppChat.sortedForSidebar(inProject).prefix(limit))
    }

    private static func isListable(_ chat: AppChat, selectedChatID: UUID) -> Bool {
        !chat.isGhost && !chat.isArchived && chat.id != selectedChatID
    }
}
