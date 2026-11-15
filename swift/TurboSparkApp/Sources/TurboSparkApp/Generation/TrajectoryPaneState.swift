import Foundation

/// What the trajectory pane shows and does (ZP-03). Every read of the log
/// runs off the main thread; the pane holds a bounded window (the last 500
/// entries, plus whatever "load earlier" pulled in) and never the whole file.
@MainActor
final class TrajectoryPaneState: ObservableObject {
    @Published private(set) var records: [RolloutRecord] = []
    @Published private(set) var startOffset: UInt64 = 0
    @Published private(set) var unreadableCount = 0
    @Published private(set) var isLoading = false
    /// Non-nil while a whole-log search result is on screen.
    @Published private(set) var searchResults: [RolloutRecord]?
    @Published private(set) var searchHitLimit = false
    @Published private(set) var logBytes: UInt64 = 0

    let chatID: UUID
    private let store: RolloutStore

    init(chatID: UUID, store: RolloutStore = .shared) {
        self.chatID = chatID
        self.store = store
    }

    var hasEarlier: Bool { startOffset > 0 }

    /// Reloads the tail. A snapshot, not a live view: the pane never polls.
    func reload() async {
        isLoading = true
        let store = store, chatID = chatID
        let (page, bytes) = await Task.detached(priority: .userInitiated) {
            (store.readTail(chatID: chatID), store.byteCount(chatID: chatID))
        }.value
        records = page.records
        startOffset = page.startOffset
        unreadableCount = page.unreadableCount
        logBytes = bytes
        searchResults = nil
        isLoading = false
    }

    func loadEarlier() async {
        guard hasEarlier, !isLoading else { return }
        isLoading = true
        let store = store, chatID = chatID, offset = startOffset
        let page = await Task.detached(priority: .userInitiated) {
            store.readBefore(chatID: chatID, offset: offset)
        }.value
        records = page.records + records
        startOffset = page.startOffset
        unreadableCount += page.skippedLines
        isLoading = false
    }

    func searchAll(_ query: String) async {
        isLoading = true
        let store = store, chatID = chatID
        let result = await Task.detached(priority: .userInitiated) {
            store.search(chatID: chatID, query: query)
        }.value
        searchResults = result.records
        searchHitLimit = result.hitLimit
        isLoading = false
    }

    func clearSearch() { searchResults = nil }

    func deleteLog() async {
        let store = store, chatID = chatID
        store.delete(chatID: chatID)
        await Task.detached { store.flush() }.value
        await reload()
    }

    /// Returns the exported entry count, or throws.
    func export(to url: URL) async throws -> Int {
        let store = store, chatID = chatID
        return try await Task.detached(priority: .userInitiated) {
            try store.exportRedacted(chatID: chatID, to: url)
        }.value
    }
}
