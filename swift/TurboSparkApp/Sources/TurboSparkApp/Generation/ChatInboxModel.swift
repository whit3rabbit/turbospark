import Foundation

/// OC-03: how the chat sidebar groups its rows. `chats` is the time-bucket
/// list; `inbox` groups by what a chat needs from the user. The choice is
/// persisted (`MacAppSettings.chatSidebarMode`); `chats` is the default.
enum ChatSidebarMode: String, CaseIterable, Identifiable {
    case chats
    case inbox

    var id: String { rawValue }

    /// Tolerant decode: anything unknown falls back to the default list.
    init(storedValue: String) {
        self = ChatSidebarMode(rawValue: storedValue) ?? .chats
    }
}

/// The inbox sections a chat can sit in, in priority order. Archived chats
/// are not a case here: the archived flag wins over every turn state, and the
/// sidebar's existing Archived disclosure (Restore, Delete) renders them in
/// both modes.
enum ChatInboxSection: Int, CaseIterable {
    /// A tool approval or a question is waiting on the user.
    case needsYou
    /// A turn is generating, or input is parked for the lane.
    case working
    /// A reply landed that the user has not opened yet.
    case finished

    /// Catalog key of the section header.
    var title: String {
        switch self {
        case .needsYou: return "Needs you"
        case .working: return "Working"
        case .finished: return "Finished"
        }
    }

    /// The one section a chat belongs to, or nil when it needs nothing.
    ///
    /// The order mirrors `ChatRowSlot` (pending, then running, then unread),
    /// so the row's glyph and its section never disagree. Differences from
    /// the slot, both deliberate: a parked prompt counts as Needs you even on
    /// the selected chat (it is still waiting, and the row must not vanish
    /// while the card is on screen), and a queued chat counts as Working.
    ///
    /// `isRetained` keeps a chat the user just opened out of Finished's
    /// vanishing act: opening it marks it read, and a row that leaves the
    /// list under the pointer is the thing M9 of the review ruled out.
    static func resolve(
        turnState: ChatTurnState, hasUnreadReply: Bool, isSelected: Bool, isRetained: Bool
    ) -> ChatInboxSection? {
        switch turnState {
        case .awaitingTool, .awaitingQuestion: return .needsYou
        case .generating, .queued: return .working
        case .idle: break
        }
        if isSelected {
            return isRetained ? .finished : nil
        }
        return hasUnreadReply ? .finished : nil
    }
}

/// The inbox grouping, derived once per body from the already scoped and
/// sorted list. Pure so section membership is testable without views. It
/// shows only chats that need attention; everything else stays in the
/// Chats list.
struct ChatInboxLayout {
    struct Group: Identifiable {
        let section: ChatInboxSection
        let chats: [AppChat]
        var id: ChatInboxSection { section }
    }

    /// Non-empty sections only, in priority order.
    let groups: [Group]
    let isSearching: Bool

    var isEmpty: Bool { groups.isEmpty }

    /// `chats` must come from `AppModel.filteredChats` (project scope, no
    /// ghost chats, no archived chats) in `AppChat.sortedForSidebar` order;
    /// that order holds inside each section. A non-blank `searchText` filters
    /// by the same rule as the Chats list.
    static func make(
        chats: [AppChat],
        searchText: String,
        selectedID: UUID,
        retainedID: UUID?,
        turnState: (UUID) -> ChatTurnState
    ) -> ChatInboxLayout {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        var bySection: [ChatInboxSection: [AppChat]] = [:]
        for chat in chats {
            if !query.isEmpty, !ChatSidebarLayout.matches(chat, query: query) { continue }
            let section = ChatInboxSection.resolve(
                turnState: turnState(chat.id),
                hasUnreadReply: chat.hasUnreadReply,
                isSelected: chat.id == selectedID,
                isRetained: chat.id == retainedID)
            if let section { bySection[section, default: []].append(chat) }
        }
        let groups = ChatInboxSection.allCases.compactMap { section in
            bySection[section].map { Group(section: section, chats: $0) }
        }
        return ChatInboxLayout(groups: groups, isSearching: !query.isEmpty)
    }
}
