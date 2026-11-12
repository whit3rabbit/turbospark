import CryptoKit
import Foundation
import Syntext

/// Thrown by `grep` while a project's index is still opening or building in
/// the background. The registry answers it with the unindexed scan, so the
/// first search after opting in never waits on a build that the user's Stop
/// cannot interrupt (the Rust build is synchronous once entered).
public struct SyntextIndexNotReady: Error {}

/// Actor managing the background index and providing an LLM Grep tool.
public actor SyntextCodeSearchTool {
    private var index: SyntextIndex?
    private var indexLoadTask: Task<SyntextIndex, Error>?
    private var indexGeneration: UInt64 = 0
    private var isDeleting = false
    /// Only the manager's active project keeps its mmap-backed handle. A load
    /// that finishes after its project was switched away still answers its
    /// caller, but is not kept resident. Direct construction (tests, one-off
    /// tools) is active.
    private var isActive: Bool
    /// When the last open or build failed. A failure is not retried on every
    /// search: each retry of a failing build is a full repository walk.
    private var lastLoadFailure: Date?
    /// Changes that arrived with no handle loaded (a build in flight, or the
    /// project inactive). They are replayed into the next handle, because
    /// `notifyChange` is the only way a non-git directory learns of an edit.
    private var pendingChangedPaths: Set<String> = []
    private var pendingDeletedPaths: Set<String> = []
    /// Too many buffered paths to replay: the next load rebuilds instead.
    private var pendingOverflowed = false
    private var quietCommitTask: Task<Void, Never>?
    private var pendingChangesCount: Int = 0
    public let repoRoot: URL
    public let indexDir: URL

    static let failureCooldown: TimeInterval = 60
    /// `SYNTEXT_ERR_OVERLAY_FULL` in syntext.h: a commit the overlay has no
    /// room for. Only a rebuild clears it.
    static let overlayFullCode: UInt32 = 9
    static let maxBufferedPaths = 512
    /// Output bounds. The unindexed scan already caps matches at 500; the
    /// indexed path had no cap at all (the library allows 1,000,000).
    static let maxResultsCap = 500
    static let maxContextLines = 10

    public init(repoRoot: URL, indexDir: URL? = nil) {
        self.init(repoRoot: repoRoot, indexDir: indexDir, active: true)
    }

    init(repoRoot: URL, indexDir: URL? = nil, active: Bool) {
        self.repoRoot = repoRoot.standardizedFileURL.resolvingSymlinksInPath()
        self.indexDir = (indexDir ?? Self.defaultIndexDirectory(for: repoRoot))
            .standardizedFileURL.resolvingSymlinksInPath()
        self.isActive = active
    }

    /// `<Caches>/TurboSpark/syntext/<digest>.syntext`, a fixed-length digest
    /// of the standardized root. The old name was base64 of the raw path: it
    /// passed APFS's 255-byte name limit for roots longer than about 185
    /// bytes (so every build failed), changed with how the path was spelled,
    /// and decoded back to the project path. Tests get the redirected storage
    /// root, never the real Caches folder.
    static func defaultIndexDirectory(for repoRoot: URL) -> URL {
        let root = repoRoot.standardizedFileURL.resolvingSymlinksInPath().path
        let digest = SHA256.hash(data: Data(root.utf8))
        let name = digest.prefix(16).map { String(format: "%02x", $0) }.joined()
        return cacheRoot.appendingPathComponent("\(name).syntext", isDirectory: true)
    }

    static var cacheRoot: URL {
        if AppStorageRoot.isRunningTests {
            return AppStorageRoot.machineRoot
                .appendingPathComponent("Caches/syntext", isDirectory: true)
        }
        let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
            ?? URL(fileURLWithPath: NSTemporaryDirectory())
        return caches.appendingPathComponent("TurboSpark/syntext", isDirectory: true)
    }

    /// The folder the old naming used. Nothing maps to its entries any
    /// more, so it is only disk to give back.
    static func removeLegacyCache() {
        guard !AppStorageRoot.isRunningTests,
              let caches = FileManager.default.urls(for: .cachesDirectory, in: .userDomainMask).first
        else { return }
        try? FileManager.default.removeItem(
            at: caches.appendingPathComponent("com.turbospark.app/syntext", isDirectory: true))
    }

    /// Whether an index is currently loaded or exists on disk.
    public var isIndexed: Bool {
        if index != nil { return true }
        return FileManager.default.fileExists(atPath: indexDir.path)
    }

    /// Whether this actor currently retains an open mmap-backed index handle.
    var isLoaded: Bool { index != nil }

    /// Whether an open or build is running for this root.
    var isLoading: Bool { indexLoadTask != nil }

    /// Makes this root the one that may keep a handle, or releases it. An
    /// in-flight open or build is NOT cancelled on release: switching to
    /// another project and back joins it instead of starting a second
    /// high-memory build of the same root.
    func setActive(_ active: Bool) {
        isActive = active
        if !active { unload() }
    }

    /// Initializes or opens the index, waiting for a build if one is needed.
    public func ensureIndex() async throws {
        if index != nil { return }
        _ = try await loadIndex(forceRebuild: false)
    }

    /// Coalesces concurrent opens/builds so project selection and the first
    /// search cannot launch multiple high-memory index builds for one root.
    @discardableResult
    private func loadIndex(forceRebuild: Bool) async throws -> SyntextIndex {
        guard !isDeleting else { throw CancellationError() }
        if !forceRebuild, let index { return index }
        if let inFlight = indexLoadTask {
            // Whichever awaiter resumes first installs the result. Only the
            // starter used to, so a caller joining a build could return, and
            // its next search still found no handle and a load "in flight".
            let generation = indexGeneration
            let result = await inFlight.result
            if generation == indexGeneration {
                switch result {
                case .success(let loaded): adopt(loaded)
                case .failure: if indexLoadTask == inFlight { indexLoadTask = nil }
                }
            }
            if !forceRebuild { return try result.get() }
            // A rebuild never runs beside an open or build of the same root:
            // the builder's exclusive lock would conflict with it.
            return try await loadIndex(forceRebuild: true)
        }

        try FileManager.default.createDirectory(
            at: indexDir.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )

        // A rebuild needs the old handle released first: it owns a shared
        // directory lock and would otherwise conflict with the builder's
        // exclusive lock.
        let rebuild = forceRebuild || pendingOverflowed
        if rebuild {
            index = nil
            pendingOverflowed = false
            pendingChangedPaths.removeAll()
            pendingDeletedPaths.removeAll()
        }
        indexGeneration &+= 1
        let generation = indexGeneration
        let indexDirPath = indexDir.path
        let repoRootPath = repoRoot.path
        let task = Task.detached(priority: .utility) { () throws -> SyntextIndex in
            if !rebuild {
                do {
                    return try Self.retryingLockConflicts {
                        try SyntextIndex(indexDir: indexDirPath, repoRoot: repoRootPath)
                    }
                } catch let SyntextError.indexError(code, _) where code == 2 {
                    // A missing index is the only open failure that should
                    // become a build. Corruption and permission errors remain
                    // visible to the caller.
                }
            }
            return try await SyntextBuildQueue.shared.run {
                try Self.retryingLockConflicts {
                    try SyntextIndex.build(indexDir: indexDirPath, repoRoot: repoRootPath)
                }
            }
        }
        indexLoadTask = task

        do {
            let loaded = try await task.value
            guard generation == indexGeneration else { throw CancellationError() }
            adopt(loaded)
            return loaded
        } catch {
            if generation == indexGeneration {
                indexLoadTask = nil
                if !(error is CancellationError) { lastLoadFailure = Date() }
            }
            throw error
        }
    }

    /// Installs a finished open or build. Idempotent: the starter and any
    /// caller that joined the same load may both get here.
    private func adopt(_ loaded: SyntextIndex) {
        indexLoadTask = nil
        lastLoadFailure = nil
        guard isActive, index !== loaded else { return }
        index = loaded
        replayPending(into: loaded)
    }

    /// Lock errors can be transient without real contention (`ENOLCK` under
    /// process churn, or a descriptor inherited across a `fork` before
    /// `exec`; the app spawns `git`). Bounded: three tries.
    nonisolated static func retryingLockConflicts<T>(_ body: () throws -> T) throws -> T {
        var delay: useconds_t = 100_000
        for attempt in 1...3 {
            do {
                return try body()
            } catch let error as SyntextError where error.isRetryableLockConflict && attempt < 3 {
                usleep(delay)
                delay *= 2
            }
        }
        return try body()
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

    /// A handle a search can use now, or `SyntextIndexNotReady`. Opening an
    /// existing index is a memory map and is awaited; building one is
    /// started in the background and this search falls back.
    private func searchableIndex() async throws -> SyntextIndex {
        if let index { return index }
        if indexLoadTask != nil { throw SyntextIndexNotReady() }
        if let lastLoadFailure, Date().timeIntervalSince(lastLoadFailure) < Self.failureCooldown {
            throw SyntextIndexNotReady()
        }
        if !pendingOverflowed, FileManager.default.fileExists(atPath: indexDir.path) {
            return try await loadIndex(forceRebuild: false)
        }
        Task { try? await self.loadIndex(forceRebuild: false) }
        throw SyntextIndexNotReady()
    }

    /// LLM Tool function: `grep_search`.
    ///
    /// - Parameter scope: the project-relative directory prefix (`src/`) or
    ///   file the caller asked for. It is enforced here, not only handed to
    ///   Syntext: the library takes one filter, so an explicit `pathFilter`
    ///   used to replace the scope and search the whole repository, and its
    ///   directory filter matches the name at any depth.
    public func grep(
        query: String,
        pathFilter: String? = nil,
        scope: String? = nil,
        fileTypes: [String] = [],
        caseSensitive: Bool = false,
        literalSearch: Bool = false,
        wordMatch: Bool = false,
        contextLines: Int = 2,
        maxResults: Int = 50
    ) async throws -> String {
        let index = try await searchableIndex()
        // A write notified just before this search must be visible to it;
        // the debounced commit would otherwise hold it back for 1.5 s.
        if pendingChangesCount > 0, self.index === index {
            commitPending(index)
            // An overlay-full commit dropped this handle and started a
            // rebuild. The old handle cannot see the pending change; the
            // unindexed scan can.
            guard self.index === index else { throw SyntextIndexNotReady() }
        }

        let limit = max(1, min(maxResults, Self.maxResultsCap))
        // Over-fetch: matches the policy hides still count against Syntext's
        // own limit.
        let fetchLimit = min(max(limit * 4, 200), 2_000)
        let options = SyntextSearchOptions(
            pathFilter: pathFilter ?? scope,
            fileTypes: fileTypes,
            maxResults: fetchLimit,
            caseInsensitive: !caseSensitive,
            deterministic: true,
            fixedStrings: literalSearch,
            wordRegexp: wordMatch,
            context: max(0, min(contextLines, Self.maxContextLines))
        )

        var matches: [SyntextSearchMatch]
        var staleNote: String?
        if FileManager.default.fileExists(atPath: repoRoot.appendingPathComponent(".git").path) {
            do {
                let fresh = try await index.searchFreshAsync(query, options: options)
                matches = fresh.matches
                staleNote = Self.staleNote(for: fresh.outcome)
            } catch {
                matches = try await index.searchAsync(query, options: options)
            }
        } else {
            matches = try await index.searchAsync(query, options: options)
        }

        let visible = matches.filter {
            GrepPathPolicy.isInside($0.path, scope: scope)
                && GrepPathPolicy.allows($0.path, scope: scope)
        }
        let shown = visible.prefix(limit)
        var output = shown.isEmpty
            ? "No matches found for '\(query)'."
            : shown.formattedGrepOutput(contextSeparator: "--")
        if !shown.isEmpty, visible.count > shown.count || matches.count >= fetchLimit {
            output += "\n[Showing the first \(shown.count) matches. Narrow the pattern or path for the rest.]"
        }
        if let staleNote { output += "\n\(staleNote)" }
        return output
    }

    /// A bounded git update that could not apply its delta leaves the index
    /// behind the working tree. The search still answers, so say so.
    static func staleNote(for outcome: SyntextUpdateOutcome) -> String? {
        switch outcome {
        case .budgetExceeded, .tooManyFiles, .overlayFull:
            return "[The code index is behind recent changes, so these results may be stale. "
                + "Reindex the project in Project Settings for exact results.]"
        default:
            return nil
        }
    }

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
        pendingChangedPaths.removeAll()
        pendingDeletedPaths.removeAll()
        pendingOverflowed = false
        lastLoadFailure = nil
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
        quietCommitTask?.cancel()
        quietCommitTask = nil
        pendingChangesCount = 0
        index = nil
    }

    /// Update index when a file is modified in the editor/workspace.
    /// Buffers the change and debounces a batch commit until the workspace is quiet (1.5s).
    public func fileDidChange(at fileURL: URL) throws {
        let path = fileURL.standardizedFileURL.resolvingSymlinksInPath().path
        guard let index else {
            buffer(path, deleted: false)
            return
        }
        try index.notifyChange(path)
        scheduleQuietCommit()
    }

    /// Update index when a workspace file is deleted.
    public func fileDidDelete(at fileURL: URL) throws {
        let path = fileURL.standardizedFileURL.resolvingSymlinksInPath().path
        guard let index else {
            buffer(path, deleted: true)
            return
        }
        try index.notifyDelete(path)
        scheduleQuietCommit()
    }

    /// Keeps a change for the next handle. Only worth it when there will be
    /// one: an index on disk or a load in flight. A never-indexed project
    /// gets every file from its first build's walk.
    private func buffer(_ path: String, deleted: Bool) {
        guard !pendingOverflowed,
              indexLoadTask != nil || FileManager.default.fileExists(atPath: indexDir.path)
        else { return }
        if deleted {
            pendingChangedPaths.remove(path)
            pendingDeletedPaths.insert(path)
        } else {
            pendingDeletedPaths.remove(path)
            pendingChangedPaths.insert(path)
        }
        if pendingChangedPaths.count + pendingDeletedPaths.count > Self.maxBufferedPaths {
            pendingOverflowed = true
            pendingChangedPaths.removeAll()
            pendingDeletedPaths.removeAll()
        }
    }

    private func replayPending(into index: SyntextIndex) {
        guard !pendingChangedPaths.isEmpty || !pendingDeletedPaths.isEmpty else { return }
        for path in pendingChangedPaths { try? index.notifyChange(path) }
        for path in pendingDeletedPaths { try? index.notifyDelete(path) }
        pendingChangedPaths.removeAll()
        pendingDeletedPaths.removeAll()
        commitPending(index)
    }

    /// Commits buffered notifications. **AN OVERLAY-FULL COMMIT IS NOT A
    /// SKIPPABLE ERROR.** Syntext refuses a commit once the overlay outgrows
    /// the base index, and every later commit fails the same way. Swallowing
    /// it (as `try?` did) left each file created after the build invisible to
    /// search until a manual Reindex. Edits to already-indexed files still
    /// showed, since syntext re-reads changed files at query time, which is
    /// why nothing looked wrong. Rebuild in the background instead; searches
    /// fall back to the unindexed scan until it lands.
    private func commitPending(_ index: SyntextIndex) {
        quietCommitTask?.cancel()
        quietCommitTask = nil
        pendingChangesCount = 0
        do {
            try index.commitBatch()
        } catch SyntextError.indexError(let code, _) where code == Self.overlayFullCode {
            guard self.index === index else { return }
            self.index = nil
            pendingOverflowed = true
            Task { try? await self.loadIndex(forceRebuild: false) }
        } catch {
            // Other commit failures leave the index as it was. The next
            // successful commit or rebuild picks the change up.
        }
    }

    private func scheduleQuietCommit() {
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
        commitPending(index)
    }

    /// Commit buffered file changes immediately.
    public func commitChanges() {
        guard let index = self.index else { return }
        commitPending(index)
    }

    /// Flush pending in-app file notifications when the repository is quiet.
    /// Git repositories already run bounded freshness detection immediately
    /// before each indexed search, so doing it again at every turn boundary
    /// only adds process and CPU churn.
    public func syncQuietly() async {
        commitChangesQuietly()
    }
}

/// One Syntext build at a time across every project. A build can peak
/// around 1.5 GB; two roots, or one root switched away from and back to
/// mid-build, used to run theirs side by side.
actor SyntextBuildQueue {
    static let shared = SyntextBuildQueue()
    private var tail: Task<Void, Never>?

    func run<T: Sendable>(_ work: @escaping @Sendable () throws -> T) async throws -> T {
        let previous = tail
        let next = Task.detached(priority: .utility) { () throws -> T in
            await previous?.value
            return try work()
        }
        tail = Task { _ = try? await next.value }
        return try await next.value
    }
}

/// Shared manager holding per-project SyntextCodeSearchTool actors.
public actor SyntextIndexManager {
    public static let shared = SyntextIndexManager()

    /// One instance per root for the life of the process. Inactive roots
    /// hold no handle, but keep their in-flight build and buffered changes;
    /// throwing them away is what let a project switch start a second build.
    private var tools: [String: SyntextCodeSearchTool] = [:]
    private var activeKey: String?
    private var lastActivationToken: UInt64 = 0
    private var removedLegacyCache = false

    private init() {}

    private func key(for rootURL: URL) -> String {
        rootURL.standardizedFileURL.resolvingSymlinksInPath().path
    }

    private func instance(for rootURL: URL) -> SyntextCodeSearchTool {
        let k = key(for: rootURL)
        if let existing = tools[k] { return existing }
        if !removedLegacyCache {
            removedLegacyCache = true
            Task.detached(priority: .background) { SyntextCodeSearchTool.removeLegacyCache() }
        }
        let created = SyntextCodeSearchTool(repoRoot: rootURL, active: false)
        tools[k] = created
        return created
    }

    /// Obtains the tool for a project and makes it the active one, releasing
    /// any prior project's live handle. On-disk indexes remain reusable, but
    /// resident index memory is bounded to one active project.
    public func tool(for rootURL: URL) async -> SyntextCodeSearchTool {
        let k = key(for: rootURL)
        let selected = instance(for: rootURL)
        guard activeKey != k else { return selected }
        let previous = activeKey.flatMap { tools[$0] }
        activeKey = k
        await selected.setActive(true)
        await previous?.setActive(false)
        return selected
    }

    /// Checks if an on-disk index exists for the given workspace.
    public func isIndexed(for rootURL: URL) async -> Bool {
        await instance(for: rootURL).isIndexed
    }

    /// Builds or rebuilds the index for the workspace. Does not change which
    /// project is active: Reindex in one project's settings used to take the
    /// slot from the selected project and drop its handle.
    @discardableResult
    public func buildIndex(for rootURL: URL) async throws -> SyntextStats {
        try await instance(for: rootURL).buildIndex()
    }

    /// Deletes the on-disk index directory and resets memory handle for the workspace.
    public func deleteIndex(for rootURL: URL) async throws {
        try await instance(for: rootURL).deleteIndex()
    }

    /// Returns statistics for the workspace index.
    public func stats(for rootURL: URL) async throws -> SyntextStats {
        try await instance(for: rootURL).stats()
    }

    /// Notifies the tool of a file modification within the workspace.
    public func notifyChange(at fileURL: URL, rootURL: URL) async {
        try? await instance(for: rootURL).fileDidChange(at: fileURL)
    }

    /// Notifies the tool of a file deletion within the workspace.
    public func notifyDelete(at fileURL: URL, rootURL: URL) async {
        try? await instance(for: rootURL).fileDidDelete(at: fileURL)
    }

    /// Bounded background sync for an active project when repo is quiet.
    public func syncQuietly(for rootURL: URL) async {
        guard let t = tools[key(for: rootURL)] else { return }
        await t.syncQuietly()
    }

    /// Ensures the active project is indexed in the background without blocking the UI.
    public func ensureActiveProjectIndexed(for rootURL: URL) async {
        let t = await tool(for: rootURL)
        try? await t.ensureIndex()
    }

    /// Applies one activation decision from the app. Callers start these from
    /// unordered tasks, so a token older than the newest one seen is dropped:
    /// rapid project switches used to land in whatever order the tasks ran.
    public func activate(_ rootURL: URL?, token: UInt64) async {
        guard token > lastActivationToken else { return }
        lastActivationToken = token
        if let rootURL {
            await ensureActiveProjectIndexed(for: rootURL)
        } else {
            await deactivateAll()
        }
    }

    /// Unloads every live handle when indexing is disabled or no indexed
    /// project is selected. Disk caches are preserved for fast reopening.
    public func deactivateAll() async {
        activeKey = nil
        for tool in tools.values { await tool.setActive(false) }
    }
}
