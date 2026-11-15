import Foundation

/// One stop in the back/forward history: a rail section and, for the Chat
/// section only, the chat that was open.
///
/// Settings is not part of it. The Settings window is its own scene, not a
/// section, and in-pane state (settings tabs, audio pages) is deliberately out
/// of scope (OC-05), so there is nothing to collapse and no settings tab id.
struct NavigationEntry: Equatable, Sendable {
    let section: AppModel.AppNavigationSection
    /// Non-nil only when `section == .chat`; other sections have no chat.
    let chatID: UUID?

    init(section: AppModel.AppNavigationSection, chatID: UUID?) {
        self.section = section
        self.chatID = section == .chat ? chatID : nil
    }
}

/// A bounded linear navigation history with a cursor, as a browser has.
///
/// Pure value: no UI and no model access, so the stack rules are tested
/// headlessly. `AppModel` owns the one instance, records into it, and applies
/// the entries it walks to.
struct NavigationHistory: Equatable, Sendable {
    /// Copied from OpenCode's titlebar history; nothing here measured a better
    /// number. Oldest entries are trimmed first.
    static let capacity = 100

    private(set) var entries: [NavigationEntry] = []
    /// Index of the current entry, or -1 while empty.
    private(set) var cursor = -1

    var current: NavigationEntry? { entries.indices.contains(cursor) ? entries[cursor] : nil }

    /// Records a new stop. A repeat of the current entry is ignored (so a
    /// replay or an unrelated write cannot grow or truncate the stack), and a
    /// new stop reached after going back discards the forward entries.
    mutating func record(_ entry: NavigationEntry) {
        if entry == current { return }
        if cursor < entries.count - 1 {
            entries.removeSubrange((cursor + 1)...)
        }
        entries.append(entry)
        cursor = entries.count - 1
        if entries.count > Self.capacity {
            let overflow = entries.count - Self.capacity
            entries.removeFirst(overflow)
            cursor -= overflow
        }
    }

    /// Records the starting point without moving anything, only while empty.
    mutating func seedIfEmpty(_ entry: NavigationEntry) {
        guard entries.isEmpty else { return }
        entries = [entry]
        cursor = 0
    }

    /// The nearest earlier entry `isValid` accepts (a deleted chat is skipped),
    /// as its index.
    func backIndex(isValid: (NavigationEntry) -> Bool) -> Int? {
        guard cursor > 0 else { return nil }
        return (0..<cursor).reversed().first { isValid(entries[$0]) }
    }

    /// The nearest later entry `isValid` accepts, as its index.
    func forwardIndex(isValid: (NavigationEntry) -> Bool) -> Int? {
        guard cursor >= 0, cursor < entries.count - 1 else { return nil }
        return ((cursor + 1)..<entries.count).first { isValid(entries[$0]) }
    }

    /// Moves the cursor onto an index returned by `backIndex`/`forwardIndex`.
    mutating func move(to index: Int) {
        guard entries.indices.contains(index) else { return }
        cursor = index
    }
}
