import Foundation
import Syntext

/// Actor managing the background index and providing an LLM Grep tool.
public actor SyntextCodeSearchTool {
    private var index: SyntextIndex?
    private var indexLoadTask: Task<SyntextIndex, Error>?
    private var indexGeneration: UInt64 = 0
    private var isDeleting = false
    public let repoRoot: URL
    public let indexDir: URL

    public init(repoRoot: URL, indexDir: URL? = nil) {
        self.repoRoot = repoRoot.standardizedFileURL.resolvingSymlinksInPath()
        if let indexDir = indexDir {
            self.indexDir = indexDir.standardizedFileURL.resolvingSymlinksInPath()
        } else {
            let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
                ?? URL(fileURLWithPath: NSTemporaryDirectory())
            let rawPath = repoRoot.path
            let repoId = Data(rawPath.utf8).base64EncodedString()
                .replacingOccurrences(of: "/", with: "_")
                .replacingOccurrences(of: "+", with: "-")
                .replacingOccurrences(of: "=", with: "")
            self.indexDir = caches.appendingPathComponent("com.turbospark.app/syntext/\(repoId).syntext")
        }
    }

    /// Whether an index is currently loaded or exists on disk.
    public var isIndexed: Bool {
        if index != nil { return true }
        return FileManager.default.fileExists(atPath: indexDir.path)
    }

    /// Whether this actor currently retains an open mmap-backed index handle.
    var isLoaded: Bool { index != nil }

    /// Initializes or opens the index asynchronously without rebuilding if present.
    public func ensureIndex() async throws {
        if index != nil { return }
        _ = try await loadIndex(forceRebuild: false)
    }

    /// Coalesces concurrent opens/builds so project selection and the first
    /// search cannot launch multiple high-memory index builds for one root.
    private func loadIndex(forceRebuild: Bool) async throws -> SyntextIndex {
        guard !isDeleting else { throw CancellationError() }
        if !forceRebuild, let index { return index }
        if let indexLoadTask {
            return try await indexLoadTask.value
        }

        try FileManager.default.createDirectory(
            at: indexDir.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // A rebuild needs the old handle released first: it owns a shared
        // directory lock and would otherwise conflict with the builder's
        // exclusive lock.
        if forceRebuild { index = nil }
        indexGeneration &+= 1
        let generation = indexGeneration
        let indexDirPath = indexDir.path
        let repoRootPath = repoRoot.path
        let task = Task.detached(priority: .utility) {
            if !forceRebuild {
                do {
                    return try SyntextIndex(indexDir: indexDirPath, repoRoot: repoRootPath)
                } catch let SyntextError.indexError(code, _) where code == 2 {
                    // A missing index is the only open failure that should
                    // become a build. Corruption and permission errors remain
                    // visible to the caller.
                }
            }
            return try SyntextIndex.build(indexDir: indexDirPath, repoRoot: repoRootPath)
        }
        indexLoadTask = task

        do {
            let loaded = try await task.value
            guard generation == indexGeneration else { throw CancellationError() }
            index = loaded
            indexLoadTask = nil
            return loaded
        } catch {
            if generation == indexGeneration { indexLoadTask = nil }
            throw error
        }
    }

    /// Forces a fresh build of the index.
    @discardableResult
    public func buildIndex() async throws -> SyntextStats {
        let newIndex = try await loadIndex(forceRebuild: true)
        return try newIndex.stats()
    }

    /// Returns index statistics if available.
    public func stats() async throws -> SyntextStats {
        if let index { return try index.stats() }
        if let indexLoadTask { return try await indexLoadTask.value.stats() }

        // Settings can inspect an inactive project's on-disk index without
        // making that mmap-backed handle resident for the rest of the app run.
        let indexDirPath = indexDir.path
        let repoRootPath = repoRoot.path
        return try await Task.detached(priority: .utility) {
            let temporary = try SyntextIndex(indexDir: indexDirPath, repoRoot: repoRootPath)
            return try temporary.stats()
        }.value
    }

    /// LLM Tool function: `grep_search`
    public func grep(
        query: String,
        pathFilter: String? = nil,
        fileTypes: [String] = [],
        caseSensitive: Bool = false,
        literalSearch: Bool = false,
        wordMatch: Bool = false,
        contextLines: Int = 2,
        maxResults: Int = 50
    ) async throws -> String {
        try await ensureIndex()
        guard let index = self.index else {
            throw SyntextError.indexError(code: 2, message: "Index unavailable")
        }
        let options = SyntextSearchOptions(
            pathFilter: pathFilter,
            fileTypes: fileTypes,
            maxResults: maxResults,
            caseInsensitive: !caseSensitive,
            fixedStrings: literalSearch,
            wordRegexp: wordMatch,
            context: contextLines
        )

        let isGit = FileManager.default.fileExists(
            atPath: repoRoot.appendingPathComponent(".git").path
        )
        if isGit {
            if let fresh = try? await index.searchFreshAsync(query, options: options) {
                let formatted = fresh.matches.formattedGrepOutput(contextSeparator: "--")
                if formatted.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    return "No matches found for '\(query)'."
                }
                return formatted
            }
        }

        let result = try await index.grepAsync(query, options: options)
        if result.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return "No matches found for '\(query)'."
        }
        return result
    }

    private var quietCommitTask: Task<Void, Never>?
    private var pendingChangesCount: Int = 0

    /// Removes the on-disk index and resets in-memory handle.
    public func deleteIndex() async throws {
        isDeleting = true
        defer { isDeleting = false }
        let inFlightLoad = indexLoadTask
        indexGeneration &+= 1
        indexLoadTask?.cancel()
        indexLoadTask = nil
        quietCommitTask?.cancel()
        quietCommitTask = nil
        pendingChangesCount = 0
        self.index = nil
        // Rust index construction is synchronous behind the Swift task and
        // cannot be interrupted once entered. Wait for it to release file
        // handles before removing the directory, otherwise it can recreate a
        // cache after the user has chosen Remove Index.
        if let inFlightLoad { _ = try? await inFlightLoad.value }
        let fm = FileManager.default
        if fm.fileExists(atPath: indexDir.path) {
            try fm.removeItem(at: indexDir)
        }
    }

    /// Drops the live handle while preserving the reusable on-disk index.
    /// Only the active project stays mmap-backed in memory.
    public func unload() {
        commitChangesQuietly()
        indexGeneration &+= 1
        indexLoadTask?.cancel()
        indexLoadTask = nil
        quietCommitTask?.cancel()
        quietCommitTask = nil
        pendingChangesCount = 0
        index = nil
    }

    /// Update index when a file is modified in the editor/workspace.
    /// Buffers the change and debounces a batch commit until the workspace is quiet (1.5s).
    public func fileDidChange(at fileURL: URL) throws {
        guard let index = self.index else { return }
        let standardized = fileURL.standardizedFileURL.resolvingSymlinksInPath()
        try index.notifyChange(standardized.path)
        pendingChangesCount += 1

        quietCommitTask?.cancel()
        quietCommitTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: 1_500_000_000)
            guard !Task.isCancelled else { return }
            await self?.commitChangesQuietly()
        }
    }

    /// Internal commit executed after quiet debounce interval expires.
    private func commitChangesQuietly() {
        guard let index = self.index, pendingChangesCount > 0 else { return }
        try? index.commitBatch()
        pendingChangesCount = 0
    }

    /// Commit buffered file changes immediately.
    public func commitChanges() throws {
        quietCommitTask?.cancel()
        quietCommitTask = nil
        guard let index = self.index else { return }
        try index.commitBatch()
        pendingChangesCount = 0
    }

    /// Flush pending in-app file notifications when the repository is quiet.
    /// Git repositories already run bounded freshness detection immediately
    /// before each indexed search, so doing it again at every turn boundary
    /// only adds process and CPU churn.
    public func syncQuietly() async {
        commitChangesQuietly()
    }
}

/// Shared manager holding per-project SyntextCodeSearchTool actors.
public actor SyntextIndexManager {
    public static let shared = SyntextIndexManager()

    private var tools: [String: SyntextCodeSearchTool] = [:]
    private var activeKey: String?

    private init() {}

    private func key(for rootURL: URL) -> String {
        rootURL.standardizedFileURL.resolvingSymlinksInPath().path
    }

    /// Obtains the tool for a project and unloads any prior project's live
    /// handle. On-disk indexes remain reusable, but resident index memory is
    /// bounded to one active project.
    public func tool(for rootURL: URL) async -> SyntextCodeSearchTool {
        let k = key(for: rootURL)
        if activeKey == k, let existing = tools[k] { return existing }

        let selected = tools[k] ?? SyntextCodeSearchTool(repoRoot: rootURL)
        let inactive = tools.filter { $0.key != k }.map(\.value)
        tools = [k: selected]
        activeKey = k
        for tool in inactive { await tool.unload() }
        return selected
    }

    /// Checks if an on-disk index exists for the given workspace.
    public func isIndexed(for rootURL: URL) async -> Bool {
        let k = key(for: rootURL)
        let t = activeKey == k ? tools[k] : SyntextCodeSearchTool(repoRoot: rootURL)
        guard let t else { return false }
        return await t.isIndexed
    }

    /// Builds or rebuilds the index asynchronously for the workspace.
    @discardableResult
    public func buildIndex(for rootURL: URL) async throws -> SyntextStats {
        let t = await tool(for: rootURL)
        return try await t.buildIndex()
    }

    /// Deletes the on-disk index directory and resets memory handle for the workspace.
    public func deleteIndex(for rootURL: URL) async throws {
        let k = key(for: rootURL)
        let t = activeKey == k ? tools[k] : SyntextCodeSearchTool(repoRoot: rootURL)
        guard let t else { return }
        try await t.deleteIndex()
        if activeKey == k {
            tools.removeValue(forKey: k)
            activeKey = nil
        }
    }

    /// Returns statistics for the workspace index.
    public func stats(for rootURL: URL) async throws -> SyntextStats {
        let k = key(for: rootURL)
        let t = activeKey == k ? tools[k] : SyntextCodeSearchTool(repoRoot: rootURL)
        guard let t else {
            throw SyntextError.indexError(code: 2, message: "Index unavailable")
        }
        return try await t.stats()
    }

    /// Notifies the tool of a file modification within the workspace.
    public func notifyChange(at fileURL: URL, rootURL: URL) async {
        let k = key(for: rootURL)
        guard activeKey == k, let t = tools[k] else { return }
        try? await t.fileDidChange(at: fileURL)
    }

    /// Bounded background sync for an active project when repo is quiet.
    public func syncQuietly(for rootURL: URL) async {
        let k = key(for: rootURL)
        guard activeKey == k, let t = tools[k] else { return }
        await t.syncQuietly()
    }

    /// Ensures the active project is indexed in the background without blocking the UI.
    public func ensureActiveProjectIndexed(for rootURL: URL) async {
        let t = await tool(for: rootURL)
        try? await t.ensureIndex()
    }

    /// Unloads every live handle when indexing is disabled or no indexed
    /// project is selected. Disk caches are preserved for fast reopening.
    public func deactivateAll() async {
        let inactive = Array(tools.values)
        tools.removeAll()
        activeKey = nil
        for tool in inactive { await tool.unload() }
    }
}
