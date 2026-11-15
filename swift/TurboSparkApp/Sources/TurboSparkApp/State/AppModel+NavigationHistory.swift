import Foundation

/// Back/forward over section and chat switches (OC-05).
///
/// The history is owned here, not by `RootView`, because the menu items, their
/// enabled state and the roughly two dozen writers of `activeSection` all sit
/// outside any one view. `activeSection` and `selectedChatID` report their
/// changes through `navigationStateWillChange`, and one deferred flush records
/// the settled state, so `selectChat` (which writes both) is one stop.
extension AppModel {
    private var currentNavigationEntry: NavigationEntry {
        NavigationEntry(section: activeSection, chatID: selectedChatID)
    }

    /// A deleted chat is not a place to go back to. Ghost chats are rows in
    /// `chats` while they exist, so the same test covers them. The unsaved
    /// draft chat (a fresh launch with no chats) is not in `chats` yet but is a
    /// real place to return to.
    private func isNavigationEntryValid(_ entry: NavigationEntry) -> Bool {
        guard let id = entry.chatID else { return true }
        return id == activeDraftChat.id || chats.contains { $0.id == id }
    }

    var canGoBackInHistory: Bool {
        !isRunning && navigationHistory.backIndex(isValid: isNavigationEntryValid) != nil
    }

    var canGoForwardInHistory: Bool {
        !isRunning && navigationHistory.forwardIndex(isValid: isNavigationEntryValid) != nil
    }

    /// Called from the `didSet` of `activeSection` and `selectedChatID` with the
    /// entry as it was BEFORE the write, so the first change can seed the
    /// stack with where the user started.
    func navigationStateWillChange(from old: NavigationEntry) {
        navigationHistory.seedIfEmpty(old)
        guard !navigationFlushScheduled else { return }
        navigationFlushScheduled = true
        Task { @MainActor [weak self] in self?.flushNavigationRecord() }
    }

    /// Records the settled state. Tests call this directly instead of waiting
    /// for the scheduled task.
    func flushNavigationRecord() {
        navigationFlushScheduled = false
        navigationHistory.record(currentNavigationEntry)
    }

    @discardableResult
    func goBackInHistory() -> Bool {
        flushNavigationRecord()
        guard !isRunning, let index = navigationHistory.backIndex(isValid: isNavigationEntryValid) else { return false }
        return applyNavigationEntry(at: index)
    }

    @discardableResult
    func goForwardInHistory() -> Bool {
        flushNavigationRecord()
        guard !isRunning, let index = navigationHistory.forwardIndex(isValid: isNavigationEntryValid) else { return false }
        return applyNavigationEntry(at: index)
    }

    /// Navigates to an entry and moves the cursor only if the app actually
    /// arrived: `selectChat` can refuse (lane verdict), and a cursor moved
    /// without arriving would make the next record look like new navigation
    /// and wipe the forward stack.
    private func applyNavigationEntry(at index: Int) -> Bool {
        let target = navigationHistory.entries[index]
        if target.section == .chat, let id = target.chatID {
            selectChat(id: id)
        } else {
            activeSection = target.section
        }
        guard currentNavigationEntry == target else { return false }
        navigationHistory.move(to: index)
        return true
    }
}
