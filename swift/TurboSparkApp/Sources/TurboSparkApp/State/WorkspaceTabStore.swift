import Foundation

/// One open workspace tab (ZP-21): a reference to an existing project and the
/// chat the user last had open in it.
///
/// A tab holds NO conversation state. Drafts, queued prompts, attachments and
/// read markers already live on `AppChat`, so they survive a tab switch by
/// construction, and a second per-tab copy would be a second source of truth.
/// The id is the project id: a project has at most one tab.
public struct WorkspaceTab: Codable, Equatable, Identifiable, Sendable {
    public var projectID: UUID
    /// The chat to reopen when the tab is activated. A hint: it is checked
    /// against the live chat list before use, and a stale one falls back to
    /// the project's own default selection.
    public var lastChatID: UUID?

    public var id: UUID { projectID }

    public init(projectID: UUID, lastChatID: UUID? = nil) {
        self.projectID = projectID
        self.lastChatID = lastChatID
    }
}

/// The persisted shape. Only project and chat ids: no draft text, no scroll
/// offsets, no model state, so the file holds nothing sensitive and is safe
/// to quarantine and rebuild.
public struct WorkspaceTabArchive: Codable, Equatable, Sendable {
    public var version: Int
    public var tabs: [WorkspaceTab]
    public var selectedProjectID: UUID?

    public static let currentVersion = 1

    public init(
        version: Int = WorkspaceTabArchive.currentVersion,
        tabs: [WorkspaceTab] = [], selectedProjectID: UUID? = nil
    ) {
        self.version = version
        self.tabs = tabs
        self.selectedProjectID = selectedProjectID
    }
}

/// The tab list as a pure value, so every rule is testable without a model.
///
/// **DORMANT BY DEFAULT.** With fewer than two tabs the list is empty and the
/// strip is not drawn: the app behaves exactly as before the feature. The
/// first "Open in New Tab" seeds the list with the project already selected,
/// and closing down to one tab empties it again.
///
/// **SELECTION FOLLOWS THE PROJECT, NOT THE OTHER WAY.** `AppModel` owns
/// `selectedProjectID`. Every change to it, from a sidebar click or a tab
/// click alike, is mirrored here through `navigate(to:)`, so the strip cannot
/// disagree with the sidebar. A project with no tab retargets the selected
/// tab (like navigating inside a browser tab) rather than growing the list
/// on every sidebar click.
public struct WorkspaceTabStore: Equatable, Sendable {
    /// Upper bound on open tabs. A tab is two ids, so this is about strip
    /// width, not memory: eight fits the narrowest main window at the
    /// default text size.
    public static let maxTabs = 8

    public private(set) var tabs: [WorkspaceTab] = []
    /// The project whose tab is selected. Nil is "All chats" (no project).
    public private(set) var selectedProjectID: UUID?

    public init() {}

    /// Restores from an archive, repairing what a hand edit or an older
    /// build could have left: duplicates, an over-long list, a selection
    /// that names no tab, a list too short to be worth showing.
    public init(archive: WorkspaceTabArchive) {
        var seen = Set<UUID>()
        var kept: [WorkspaceTab] = []
        for tab in archive.tabs where seen.insert(tab.projectID).inserted {
            kept.append(tab)
            if kept.count == Self.maxTabs { break }
        }
        guard kept.count >= 2 else { return }
        tabs = kept
        if let selected = archive.selectedProjectID, seen.contains(selected),
            kept.contains(where: { $0.projectID == selected })
        {
            selectedProjectID = selected
        }
    }

    public var archive: WorkspaceTabArchive {
        WorkspaceTabArchive(tabs: tabs, selectedProjectID: selectedProjectID)
    }

    /// Whether the strip is shown and the tab commands apply.
    public var isActive: Bool { tabs.count >= 2 }

    public func contains(_ projectID: UUID) -> Bool {
        tabs.contains { $0.projectID == projectID }
    }

    public func tab(for projectID: UUID) -> WorkspaceTab? {
        tabs.first { $0.projectID == projectID }
    }

    // MARK: - Opening

    public enum OpenResult: Equatable, Sendable {
        /// The tab exists and is selected (new, or already open).
        case opened
        /// The list is full; nothing changed.
        case atCapacity
    }

    /// Opens `projectID` as a tab and selects it. An already-open project
    /// just takes the selection (never a duplicate). `current` is the
    /// project the user is in now: the first call seeds it as the first tab
    /// so "Open in New Tab" always yields two.
    @discardableResult
    public mutating func open(
        _ projectID: UUID, current: UUID?, currentChatID: UUID? = nil
    ) -> OpenResult {
        if contains(projectID) {
            selectedProjectID = projectID
            return .opened
        }
        var next = tabs
        if next.isEmpty, let current, current != projectID {
            next.append(WorkspaceTab(projectID: current, lastChatID: currentChatID))
        }
        guard next.count < Self.maxTabs else { return .atCapacity }
        next.append(WorkspaceTab(projectID: projectID))
        // Nothing to switch between (no current project to seed): the caller
        // still moves to the project, but no one-tab list is kept.
        guard next.count >= 2 else { return .opened }
        tabs = next
        selectedProjectID = projectID
        return .opened
    }

    // MARK: - Following the project selection

    /// Mirrors a change of the app's selected project. A no-op while dormant.
    public mutating func navigate(to projectID: UUID?) {
        guard !tabs.isEmpty else { return }
        guard let projectID else {
            selectedProjectID = nil
            return
        }
        if contains(projectID) {
            selectedProjectID = projectID
            return
        }
        // No tab for it: the selected tab changes project in place; with no
        // selected tab (the user was on All chats) the project takes the
        // last slot's place only when the list is full, else it is appended.
        if let selected = selectedProjectID,
            let index = tabs.firstIndex(where: { $0.projectID == selected })
        {
            tabs[index] = WorkspaceTab(projectID: projectID)
        } else if tabs.count >= Self.maxTabs {
            tabs[tabs.count - 1] = WorkspaceTab(projectID: projectID)
        } else {
            tabs.append(WorkspaceTab(projectID: projectID))
        }
        selectedProjectID = projectID
    }

    /// Records the chat open in `projectID`'s tab. No tab, no effect.
    public mutating func noteChat(_ chatID: UUID, in projectID: UUID) {
        guard let index = tabs.firstIndex(where: { $0.projectID == projectID }),
            tabs[index].lastChatID != chatID
        else { return }
        tabs[index].lastChatID = chatID
    }

    // MARK: - Closing

    /// The project to activate when `projectID`'s tab closes while selected:
    /// the tab to its right, else the one to its left. Nil when there is no
    /// such neighbor or the tab is not the selected one.
    public func neighborAfterClosing(_ projectID: UUID) -> UUID? {
        guard selectedProjectID == projectID,
            let index = tabs.firstIndex(where: { $0.projectID == projectID })
        else { return nil }
        if index + 1 < tabs.count { return tabs[index + 1].projectID }
        if index > 0 { return tabs[index - 1].projectID }
        return nil
    }

    /// Removes a tab. Falling below two tabs empties the list (dormant), and
    /// the project selection is untouched: closing a tab never changes what
    /// the app is showing, activating a neighbor first does.
    public mutating func remove(_ projectID: UUID) {
        tabs.removeAll { $0.projectID == projectID }
        if selectedProjectID == projectID { selectedProjectID = nil }
        if tabs.count < 2 {
            tabs = []
            selectedProjectID = nil
        }
    }

    /// Drops tabs whose project no longer exists.
    public mutating func prune(validProjectIDs: Set<UUID>) {
        let kept = tabs.filter { validProjectIDs.contains($0.projectID) }
        guard kept.count != tabs.count else { return }
        tabs = kept
        if let selected = selectedProjectID, !validProjectIDs.contains(selected) {
            selectedProjectID = nil
        }
        if tabs.count < 2 {
            tabs = []
            selectedProjectID = nil
        }
    }

    // MARK: - Cycling

    /// The project `delta` tabs from the selected one, wrapping. With no
    /// selected tab, forward starts at the first and backward at the last.
    public func projectID(cycling delta: Int) -> UUID? {
        guard isActive, delta != 0 else { return nil }
        let count = tabs.count
        guard let selected = selectedProjectID,
            let index = tabs.firstIndex(where: { $0.projectID == selected })
        else {
            return delta > 0 ? tabs.first?.projectID : tabs.last?.projectID
        }
        let target = ((index + delta) % count + count) % count
        return tabs[target].projectID
    }
}

/// What a tab's badge shows: the strongest ZP-02 state among the project's
/// chats. Same priority as the row slot (pending, running, unread).
public enum WorkspaceTabBadge: Equatable, Sendable {
    case none
    case unread
    case running
    case pending

    var rank: Int {
        switch self {
        case .none: return 0
        case .unread: return 1
        case .running: return 2
        case .pending: return 3
        }
    }

    /// Folds row slots into the tab's one badge.
    static func resolve(slots: [ChatRowSlot]) -> WorkspaceTabBadge {
        var best = WorkspaceTabBadge.none
        for slot in slots {
            let badge: WorkspaceTabBadge
            switch slot {
            case .pending: badge = .pending
            case .running: badge = .running
            case .unread: badge = .unread
            case .normal: badge = .none
            }
            if badge.rank > best.rank { best = badge }
        }
        return best
    }
}

/// Disk persistence for the tab list. One small file under the profile's
/// storage root, through `AppJSONStore`, so an unreadable file is
/// quarantined (`corrupt-...`) instead of overwritten, and the app starts
/// with no tabs (the dormant default).
public enum WorkspaceTabFileStore {
    static let fileName = "workspace_tabs.json"

    private static var defaultURL: URL { AppStorageRoot.file(fileName) }

    public static func load(from url: URL? = nil) -> WorkspaceTabArchive {
        AppJSONStore.load(
            WorkspaceTabArchive.self, from: url ?? defaultURL, label: "Workspace tabs")
            ?? WorkspaceTabArchive()
    }

    @discardableResult
    public static func save(_ archive: WorkspaceTabArchive, to url: URL? = nil) -> Bool {
        AppJSONStore.save(archive, to: url ?? defaultURL, label: "Workspace tabs")
    }
}
