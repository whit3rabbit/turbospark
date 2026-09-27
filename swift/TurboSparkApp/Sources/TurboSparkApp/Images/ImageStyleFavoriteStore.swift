import Foundation

/// Starred image style IDs.
///
/// **UNDER `AppStorageRoot`, NOT `UserDefaults.standard`**, for the same two
/// reasons `DisabledItemStore` documents: the test suite must not write the
/// developer's real preferences, and `UserDefaults` follows the bundle
/// identity, which changes between `swift run` and the shipped app.
enum ImageStyleFavoriteStore {
    struct Archive: Codable {
        var styleIDs: [String] = []
    }

    private static var fileURL: URL { AppStorageRoot.file("image_style_favorites.json") }

    /// Guards `cache` (state#69): the settings-gated pickers read this on the
    /// main actor while nothing here is actor-isolated.
    private static let lock = NSLock()
    private static nonisolated(unsafe) var cache: Archive?

    /// Requires `lock` to be held.
    private static func loadLocked() -> Archive {
        if let cache { return cache }
        let archive =
            AppJSONStore.load(Archive.self, from: fileURL, label: "image style favorites")
            ?? Archive()
        Self.cache = archive
        return archive
    }

    static func ids() -> Set<String> {
        lock.lock()
        defer { lock.unlock() }
        return Set(loadLocked().styleIDs)
    }

    static func setIDs(_ ids: Set<String>) {
        lock.lock()
        defer { lock.unlock() }
        var archive = loadLocked()
        archive.styleIDs = ids.sorted()
        cache = archive
        AppJSONStore.save(archive, to: fileURL, label: "Image style favorites")
    }

    /// Drops the in-memory copy. For tests that write the file directly.
    static func invalidateCache() {
        lock.lock()
        cache = nil
        lock.unlock()
    }
}

/// Main-actor read/write facade over `ImageStyleFavoriteStore` so the pickers
/// refresh when a star is toggled.
@MainActor
final class ImageStyleFavorites: ObservableObject {
    static let shared = ImageStyleFavorites()

    @Published private(set) var ids: Set<String>

    /// Internal so tests can build isolated instances around a saved and
    /// restored store state instead of sharing `.shared`.
    init() {
        ids = ImageStyleFavoriteStore.ids()
    }

    func contains(_ id: String) -> Bool {
        ids.contains(id)
    }

    func toggle(_ id: String) {
        if ids.contains(id) {
            ids.remove(id)
        } else {
            ids.insert(id)
        }
        ImageStyleFavoriteStore.setIDs(ids)
    }

    /// The starred styles in catalog order.
    var styles: [AppImageStyle] {
        AppImageStyleCatalog.all.filter { ids.contains($0.id) }
    }
}
