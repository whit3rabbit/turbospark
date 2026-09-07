import Foundation

/// One referable path under a project root, as the `@` autocomplete offers it.
struct ProjectFileEntry: Equatable, Sendable {
    /// Path relative to the project root, `/`-separated, no leading `./`.
    let relativePath: String
    let isDirectory: Bool
}

/// A bounded, cached listing of a project root's files and folders for the
/// `@` autocomplete.
///
/// Cached per root path with a freshness window rather than invalidated by
/// hand: a project switch changes the key, so the next trigger rebuilds with
/// no invalidation hook to forget, and edits under an open project surface
/// when the window lapses. The walk is bounded like the attachment
/// importer's (same skip list, depth cap, symlink containment) but lists
/// DIRECTORIES too and does not extension-filter -- any file can be
/// referenced; whether its content can be extracted is decided at attach.
@MainActor
final class ProjectFileIndex {
    static let shared = ProjectFileIndex()

    /// How long a root's listing is trusted. Deliberately short: a rebuild
    /// is one bounded directory walk, and offering a file that is gone reads
    /// as a broken feature in a way a briefly stale popup never does.
    nonisolated static let freshnessInterval: TimeInterval = 20

    nonisolated static let maxEntries = 2000
    nonisolated static let maxDepth = 8

    private var cache: [String: (date: Date, entries: [ProjectFileEntry])] = [:]
    private var inFlight: [String: Task<[ProjectFileEntry], Never>] = [:]

    func cachedEntries(for root: URL) -> [ProjectFileEntry]? {
        cache[PathContainment.canonical(root).path]?.entries
    }

    /// The listing for `root`, fresh or rebuilt. Concurrent callers share
    /// one walk.
    func entries(for root: URL) async -> [ProjectFileEntry] {
        let key = PathContainment.canonical(root).path
        if let cached = cache[key], Date().timeIntervalSince(cached.date) < Self.freshnessInterval {
            return cached.entries
        }
        if let running = inFlight[key] {
            return await running.value
        }
        let rootURL = URL(fileURLWithPath: key)
        let task = Task.detached(priority: .userInitiated) {
            Self.scan(root: rootURL, maxEntries: Self.maxEntries, maxDepth: Self.maxDepth)
        }
        inFlight[key] = task
        let entries = await task.value
        inFlight[key] = nil
        cache[key] = (Date(), entries)
        return entries
    }

    func invalidate() {
        cache.removeAll()
    }

    // MARK: - Walk

    /// Hidden files ARE offered (`.env`, `.github` are exactly the kind of
    /// thing a mention is for); the skip list is what keeps the walk sane.
    nonisolated static func scan(root: URL, maxEntries: Int, maxDepth: Int) -> [ProjectFileEntry] {
        let fileManager = FileManager.default
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: root.path, isDirectory: &isDir), isDir.boolValue else {
            return []
        }

        let canonicalRoot = PathContainment.canonical(root)
        let rootDepth = canonicalRoot.pathComponents.count
        guard let enumerator = fileManager.enumerator(
            at: root,
            includingPropertiesForKeys: [.isDirectoryKey, .isSymbolicLinkKey],
            options: [.skipsPackageDescendants]
        ) else {
            return []
        }

        var results: [ProjectFileEntry] = []
        var visitedCanonicalPaths: Set<String> = [canonicalRoot.path]
        while let itemURL = enumerator.nextObject() as? URL {
            let canonical = PathContainment.canonical(itemURL)
            guard PathContainment.isContained(canonical, in: canonicalRoot) else {
                // Escaped the root via symlink: neither offer nor descend.
                enumerator.skipDescendants()
                continue
            }
            guard !visitedCanonicalPaths.contains(canonical.path) else {
                if (try? itemURL.resourceValues(forKeys: [.isDirectoryKey]))?.isDirectory == true {
                    enumerator.skipDescendants()
                }
                continue
            }
            visitedCanonicalPaths.insert(canonical.path)

            let values = try? itemURL.resourceValues(forKeys: [.isDirectoryKey, .isSymbolicLinkKey])
            let isSymlink = values?.isSymbolicLink ?? false
            let isDirectory = values?.isDirectory ?? itemURL.hasDirectoryPath

            if isDirectory {
                // Symlinked directories are never descended into.
                if isSymlink {
                    enumerator.skipDescendants()
                    continue
                }
                let depth = canonical.pathComponents.count - rootDepth
                if depth > maxDepth
                    || FileSystemScanRules.skipDirectoryNames.contains(itemURL.lastPathComponent)
                {
                    enumerator.skipDescendants()
                    continue
                }
            } else if FileSystemScanRules.skipFileNames.contains(itemURL.lastPathComponent) {
                continue
            }

            guard canonical.path.hasPrefix(canonicalRoot.path + "/") else { continue }
            results.append(
                ProjectFileEntry(
                    relativePath: String(canonical.path.dropFirst(canonicalRoot.path.count + 1)),
                    isDirectory: isDirectory))
            if results.count >= maxEntries {
                break
            }
        }
        return results
    }
}
