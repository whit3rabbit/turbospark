import Foundation

/// The persisted shape (OC-04): chat ids in strip order and nothing else.
///
/// No draft text and no transcript. A draft tab is an ordinary chat row whose
/// text already lives on `AppChat.draft` and is archived with the chats, so a
/// second copy here would be a second source of truth. The selection is not
/// stored either: `AppModel.selectedChatID` is owned by the chat archive and
/// the store mirrors it after restore.
public struct SessionTabArchive: Codable, Equatable, Sendable {
    public var version: Int
    public var tabs: [UUID]

    public static let currentVersion = 1

    public init(version: Int = SessionTabArchive.currentVersion, tabs: [UUID] = []) {
        self.version = version
        self.tabs = tabs
    }
}

/// A tab that was closed: the chat and the strip position it left.
public struct ClosedSessionTab: Equatable, Sendable {
    public var chatID: UUID
    public var index: Int
}

/// The session tab list as a pure value (OC-04), so every rule is testable
/// without a model.
///
/// **A TAB IS A CHAT ID.** A draft tab is a chat with no messages (it may
/// carry unsent text on the chat row); sending in it adds messages to the same
/// chat, so "promotion" needs no swap and cannot flash an empty state.
///
/// **DORMANT BY DEFAULT.** Below two tabs the list is empty and the strip is
/// not drawn, exactly as ZP-21's `WorkspaceTabStore`. The closed stack
/// survives dormancy but never the process (in-memory, bounded).
///
/// **SELECTION FOLLOWS THE APP.** `AppModel.selectedChatID` is the truth;
/// `navigate(to:)` mirrors it. A chat with no tab retargets the selected tab
/// in place (a browser tab navigating) rather than growing the list on every
/// sidebar click.
public struct SessionTabStore: Equatable, Sendable {
    /// Upper bound on open tabs: strip width, not memory.
    public static let maxTabs = 12
    /// Upper bound on the Recently Closed stack (draft Open Question, lean 20).
    public static let closedLimit = 20

    public private(set) var tabs: [UUID] = []
    /// The tab whose chat is selected. Nil when the selected chat has no tab.
    public private(set) var selectedChatID: UUID?
    /// Most recent last.
    public private(set) var closed: [ClosedSessionTab] = []

    public init() {}

    /// Restores from an archive, repairing duplicates and an over-long list.
    /// The closed stack is not part of the archive.
    public init(archive: SessionTabArchive) {
        var seen = Set<UUID>()
        var kept: [UUID] = []
        for id in archive.tabs where seen.insert(id).inserted {
            kept.append(id)
            if kept.count == Self.maxTabs { break }
        }
        if kept.count >= 2 { tabs = kept }
    }

    public var archive: SessionTabArchive { SessionTabArchive(tabs: tabs) }

    public var isActive: Bool { tabs.count >= 2 }

    public func contains(_ chatID: UUID) -> Bool { tabs.contains(chatID) }

    // MARK: - Opening

    public enum OpenResult: Equatable, Sendable {
        case opened
        case atCapacity
    }

    /// Opens `chatID` as a tab at `index` (default: the end) and selects it.
    /// An open chat just takes the selection: one tab per chat. `current` is
    /// the chat on screen now; the first call seeds it so a new tab always
    /// yields two.
    @discardableResult
    public mutating func open(_ chatID: UUID, current: UUID?, at index: Int? = nil) -> OpenResult {
        if contains(chatID) {
            selectedChatID = chatID
            return .opened
        }
        var next = tabs
        if next.isEmpty, let current, current != chatID { next.append(current) }
        guard next.count < Self.maxTabs else { return .atCapacity }
        let position = min(max(index ?? next.count, 0), next.count)
        next.insert(chatID, at: position)
        guard next.count >= 2 else { return .opened }
        tabs = next
        selectedChatID = chatID
        return .opened
    }

    // MARK: - Following the app's selection

    /// Mirrors a change of the selected chat. `chatID` is nil for a chat that
    /// may not be a tab (ghost, not yet a row). A no-op while dormant.
    public mutating func navigate(to chatID: UUID?) {
        guard !tabs.isEmpty else { return }
        guard let chatID else {
            selectedChatID = nil
            return
        }
        if contains(chatID) {
            selectedChatID = chatID
            return
        }
        if let selected = selectedChatID, let index = tabs.firstIndex(of: selected) {
            tabs[index] = chatID
            selectedChatID = chatID
        } else {
            selectedChatID = nil
        }
    }

    // MARK: - Closing

    /// The tab to activate when `chatID`'s tab closes while selected: right
    /// neighbor, else left. Nil for a background tab or no neighbor.
    public func neighborAfterClosing(_ chatID: UUID) -> UUID? {
        guard selectedChatID == chatID, let index = tabs.firstIndex(of: chatID) else { return nil }
        if index + 1 < tabs.count { return tabs[index + 1] }
        if index > 0 { return tabs[index - 1] }
        return nil
    }

    /// Removes a tab. With `remember`, it joins the closed stack (bounded;
    /// the oldest entry falls off). Falling below two tabs empties the list.
    public mutating func close(_ chatID: UUID, remember: Bool) {
        guard let index = tabs.firstIndex(of: chatID) else { return }
        tabs.remove(at: index)
        if remember {
            closed.removeAll { $0.chatID == chatID }
            closed.append(ClosedSessionTab(chatID: chatID, index: index))
            if closed.count > Self.closedLimit {
                closed.removeFirst(closed.count - Self.closedLimit)
            }
        }
        if selectedChatID == chatID { selectedChatID = nil }
        if tabs.count < 2 {
            tabs = []
            selectedChatID = nil
        }
    }

    /// Removes and returns the closed entry for `chatID`, or the most recent
    /// entry that `isValid` accepts when nil. Entries `isValid` rejects (a
    /// deleted chat) are dropped on the way so they do not linger.
    public mutating func takeClosed(
        _ chatID: UUID? = nil, isValid: (UUID) -> Bool
    ) -> ClosedSessionTab? {
        if let chatID {
            guard let index = closed.lastIndex(where: { $0.chatID == chatID }) else { return nil }
            let entry = closed.remove(at: index)
            return isValid(entry.chatID) ? entry : nil
        }
        while let entry = closed.popLast() {
            if isValid(entry.chatID) { return entry }
        }
        return nil
    }

    /// Drops tabs and closed entries whose chat is no longer valid (deleted,
    /// archived, ghost).
    public mutating func prune(isValid: (UUID) -> Bool) {
        closed.removeAll { !isValid($0.chatID) }
        let kept = tabs.filter(isValid)
        guard kept.count != tabs.count else { return }
        tabs = kept
        if let selected = selectedChatID, !kept.contains(selected) { selectedChatID = nil }
        if tabs.count < 2 {
            tabs = []
            selectedChatID = nil
        }
    }

    // MARK: - Cycling

    /// The chat `delta` tabs from the selected one, wrapping. With no
    /// selected tab, forward starts at the first and backward at the last.
    public func chatID(cycling delta: Int) -> UUID? {
        guard isActive, delta != 0 else { return nil }
        let count = tabs.count
        guard let selected = selectedChatID, let index = tabs.firstIndex(of: selected) else {
            return delta > 0 ? tabs.first : tabs.last
        }
        return tabs[((index + delta) % count + count) % count]
    }
}

/// Disk persistence for the tab list: one small file under the profile's
/// storage root through `AppJSONStore`, so an unreadable file is quarantined
/// (`corrupt-...`) instead of overwritten and the app starts with no tabs.
public enum SessionTabFileStore {
    static let fileName = "session_tabs.json"

    private static var defaultURL: URL { AppStorageRoot.file(fileName) }

    public static func load(from url: URL? = nil) -> SessionTabArchive {
        AppJSONStore.load(SessionTabArchive.self, from: url ?? defaultURL, label: "Session tabs")
            ?? SessionTabArchive()
    }

    @discardableResult
    public static func save(_ archive: SessionTabArchive, to url: URL? = nil) -> Bool {
        AppJSONStore.save(archive, to: url ?? defaultURL, label: "Session tabs")
    }
}
