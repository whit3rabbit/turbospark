import Foundation
import Syntext
import SyntextContext
import TurboSpark

/// Which context store a chat's turns go to, and under which memory scope.
///
/// A chat in a project whose context engine is on records into that
/// project's store under `project:<id>`, so a project's chats can recall each
/// other. Every other chat records into the shared chats store under its own
/// `app:chat-<id>` scope, so chats never see each other's turns.
struct ContextChatLocation: Hashable, Sendable {
    enum Store: Hashable, Sendable {
        case project(UUID)
        case chats
    }

    let store: Store
    let chatID: UUID

    var session: String { chatID.uuidString.lowercased() }

    var memoryMode: SyntextMemoryMode {
        switch store {
        case .project(let id): return .project(id.uuidString.lowercased())
        case .chats: return .app("chat-\(session)")
        }
    }

    /// The scope label `memoryMode` writes under, for tool scopes.
    var memoryScope: String {
        switch store {
        case .project(let id): return "project:\(id.uuidString.lowercased())"
        case .chats: return "app:chat-\(session)"
        }
    }
}

/// A project store's state, for the settings sheet.
public enum ContextEngineSyncState: Equatable, Sendable {
    case idle
    case syncing(observed: Int, total: Int)
    case ready(documents: Int, passages: UInt64, bytes: UInt64)
    case failed(String)
}

/// Owns every open context store (`swift/docs/SWIFT_CONTEXT_ENGINE.md`).
///
/// **ONE HANDLE PER STORE DIRECTORY, IN THIS PROCESS AS ACROSS PROCESSES.** A
/// store takes an exclusive writer lock at open, so a second handle on the
/// same directory fails with `isContextStoreLocked` even from this app. Every
/// open goes through `openStore`, and a store is closed only once nothing
/// still syncing holds it.
///
/// Every store is DERIVED data: project files are re-read by a sync and chat
/// turns are recorded again from the transcript (`ContextRecall`). So stores
/// are opened with `sync: false`, and a store that fails validation is
/// deleted and rebuilt rather than reported.
public actor ContextEngineManager {
    public static let shared = ContextEngineManager()

    /// Key prefix of a project's files in its store (`project:<path>`).
    static let projectLabel = "project"

    struct OpenStore {
        let store: SyntextContextStore
        var engine: SyntextContextEngine?
        var engineModelPath: String?
    }

    struct ActiveProject: Equatable {
        let id: UUID
        let root: URL
    }

    var stores: [ContextChatLocation.Store: OpenStore] = [:]
    var conversations: [ContextChatLocation: SyntextConversation] = [:]
    var activeProject: ActiveProject?
    var syncTask: Task<Void, Never>?
    var syncGeneration: UInt64 = 0
    var stateObserver: (@Sendable (UUID, ContextEngineSyncState) -> Void)?
    var recallTasks: [ContextChatLocation: Task<Void, Error>] = [:]
    /// The current user turn's tool options per chat: ids continue after the
    /// turn's evidence and searches skip what the model has already seen.
    var toolOptions: [UUID: SyntextToolOptions] = [:]
    /// Debounced forced compactions after chats are forgotten, per store.
    var compactionTasks: [ContextChatLocation.Store: Task<Void, Never>] = [:]
    /// The model whose tokenizer counts for the engines.
    var modelPath: String?
    var tokenizer: (path: String, tokenizer: TurboSparkTokenizer)?

    private init() {}

    // MARK: - Locations

    static var rootDirectory: URL { AppStorageRoot.subdirectory("ContextEngine") }

    static func directory(for store: ContextChatLocation.Store) -> URL {
        switch store {
        case .project(let id):
            return rootDirectory.appendingPathComponent("projects", isDirectory: true)
                .appendingPathComponent(id.uuidString.lowercased(), isDirectory: true)
        case .chats:
            return rootDirectory.appendingPathComponent("chats", isDirectory: true)
        }
    }

    /// The project's files as a source: everything syntext's walk keeps
    /// (`.gitignore` respected), minus the git directory itself.
    static func projectSource(root: URL) -> SyntextDirectorySource {
        SyntextDirectorySource(label: projectLabel, root: root.path, exclude: [".git"])
    }

    static func standardized(_ url: URL) -> URL {
        url.standardizedFileURL.resolvingSymlinksInPath()
    }

    /// Owner-only, and out of Time Machine: the chats store holds chat text
    /// in plaintext, and everything here can be rebuilt.
    static func prepareRoot() throws {
        var root = rootDirectory
        let fm = FileManager.default
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        try fm.setAttributes([.posixPermissions: 0o700], ofItemAtPath: root.path)
        var values = URLResourceValues()
        values.isExcludedFromBackup = true
        try root.setResourceValues(values)
    }

    // MARK: - Stores and engines

    public func setObserver(_ observer: @escaping @Sendable (UUID, ContextEngineSyncState) -> Void) {
        stateObserver = observer
    }

    /// The model whose tokenizer the engines count with; `nil` uses the
    /// engine's conservative estimate. Engines are rebuilt lazily.
    public func setModelPath(_ path: String?) {
        modelPath = (path?.isEmpty ?? true) ? nil : path
    }

    func openStore(_ key: ContextChatLocation.Store) async throws -> OpenStore {
        if let open = stores[key] { return open }
        try Self.prepareRoot()
        let directory = Self.directory(for: key)
        let store = try await Self.open(directory: directory)
        // Another call may have opened it while this one waited.
        if let open = stores[key] { return open }
        let open = OpenStore(store: store, engine: nil, engineModelPath: nil)
        stores[key] = open
        return open
    }

    /// Opens a store, retrying a lock a closing handle still holds, and
    /// rebuilding one that fails validation.
    private static func open(directory: URL) async throws -> SyntextContextStore {
        try await Task.detached(priority: .utility) {
            let options = SyntextContextStoreOptions(sync: false)
            var attempt = 0
            while true {
                do {
                    return try SyntextContextStore(directory: directory.path, options: options)
                } catch let error as SyntextError where error.isContextStoreLocked && attempt < 5 {
                    attempt += 1
                    try await Task.sleep(nanoseconds: 100_000_000)
                } catch let error as SyntextError where Self.isUnusable(error) {
                    try? FileManager.default.removeItem(at: directory)
                    return try SyntextContextStore(directory: directory.path, options: options)
                }
            }
        }.value
    }

    /// A store that failed validation (`SYNTEXT_CONTEXT_ERR_CORRUPT`, 301)
    /// or was written by another format version
    /// (`SYNTEXT_CONTEXT_ERR_UNSUPPORTED_VERSION`, 304): derived data, so it
    /// is rebuilt. The codes are stable and append-only (`syntext_context.h`).
    static func isUnusable(_ error: SyntextError) -> Bool {
        guard case let .indexError(code, _) = error else { return false }
        return code == 301 || code == 304
    }

    func engine(for key: ContextChatLocation.Store) async throws -> SyntextContextEngine {
        var open = try await openStore(key)
        if let engine = open.engine, open.engineModelPath == modelPath { return engine }
        let counter = tokenCounter()
        let modelName = modelPath.map { URL(fileURLWithPath: $0).lastPathComponent }
        let engine = try SyntextContextEngine(
            store: open.store,
            // Never touch the system message: project turns carry the
            // policy in a constant section (`AppModel+ContextEvidence`), so
            // the system message stays the same and its KV prefix is reused.
            options: .init(
                model: modelName, tokenizer: counter == nil ? nil : "turbospark",
                appendPolicy: false),
            tokenCounter: counter)
        open.engine = engine
        open.engineModelPath = modelPath
        stores[key] = open
        return engine
    }

    private func tokenCounter() -> (@Sendable (String) -> Int)? {
        guard let modelPath else { return nil }
        let opened: TurboSparkTokenizer
        if let cached = tokenizer, cached.path == modelPath {
            opened = cached.tokenizer
        } else {
            guard let fresh = try? TurboSparkTokenizer(modelPath: modelPath) else { return nil }
            tokenizer = (modelPath, fresh)
            opened = fresh
        }
        // A counting failure must not fail a turn: over-count instead, the
        // safe direction for a budget.
        return { text in (try? opened.count(text)) ?? max(1, text.utf8.count / 2) }
    }

    /// Drops the handles of `key`. Callers cancel and await any sync of it
    /// first, so the writer lock is released when the last reference goes.
    func closeStore(_ key: ContextChatLocation.Store) {
        stores.removeValue(forKey: key)
        conversations = conversations.filter { $0.key.store != key }
        recallTasks = recallTasks.filter { $0.key.store != key }
    }

    // MARK: - Project lifecycle

    /// Makes `id` the one project whose store is open and kept in sync, or
    /// closes the previous one when `id` is nil. With `resync`, re-activating
    /// the active project runs a sync, which is cheap when nothing changed
    /// and picks up edits made outside the app; a turn passes `false` so it
    /// only makes sure the project is active.
    public func activateProject(id: UUID?, root: URL?, resync: Bool = true) async {
        let target = id.flatMap { id in root.map { ActiveProject(id: id, root: Self.standardized($0)) } }
        if let target, target == activeProject {
            if resync, syncTask == nil { await startSync() }
            return
        }
        let previous = activeProject
        activeProject = target
        await cancelSync()
        if let previous, previous.id != activeProject?.id {
            closeStore(.project(previous.id))
        }
        if activeProject != nil, syncTask == nil { await startSync() }
    }

    func cancelSync() async {
        guard let task = syncTask else { return }
        syncTask = nil
        task.cancel()
        await task.value
    }

    private func startSync() async {
        guard let active = activeProject else { return }
        syncGeneration &+= 1
        let generation = syncGeneration
        let observer = stateObserver
        let open: OpenStore
        do {
            open = try await openStore(.project(active.id))
        } catch {
            observer?(active.id, .failed(String(describing: error)))
            return
        }
        // The open awaited; a newer activation owns the sync now.
        guard generation == syncGeneration, activeProject == active, syncTask == nil else { return }
        let store = open.store
        let source = Self.projectSource(root: active.root)
        observer?(active.id, .syncing(observed: 0, total: 0))
        syncTask = Task.detached(priority: .utility) {
            let state: ContextEngineSyncState
            do {
                try store.sync(source) { progress in
                    observer?(active.id, .syncing(observed: progress.observed, total: progress.total))
                    return !Task.isCancelled
                }
                _ = try? store.compact()
                let stats = try store.stats()
                state = .ready(documents: stats.documents, passages: stats.passages, bytes: stats.bytes)
            } catch let error as SyntextError where error.isCancelled {
                state = .idle
            } catch {
                state = .failed(String(describing: error))
            }
            observer?(active.id, state)
            await self.syncFinished(generation: generation)
        }
    }

    private func syncFinished(generation: UInt64) {
        if generation == syncGeneration { syncTask = nil }
    }

    /// Files the app itself just wrote or deleted under the active project.
    /// Only those paths are re-read; the next full sync reconciles the rest.
    public func filesChanged(_ urls: [URL], root: URL) {
        guard let active = activeProject, active.root == Self.standardized(root),
            let open = stores[.project(active.id)]
        else { return }
        let prefix = active.root.path.hasSuffix("/") ? active.root.path : active.root.path + "/"
        let paths = urls.map { Self.standardized($0).path }.filter { $0.hasPrefix(prefix) }
        guard !paths.isEmpty else { return }
        let store = open.store
        let source = Self.projectSource(root: active.root)
        Task.detached(priority: .utility) {
            _ = try? store.sync(paths: paths, in: source)
        }
    }

    /// Merges small segments and drops dead documents where due, off the
    /// request path.
    public func compactWhenDue() {
        let open = stores.values.map(\.store)
        guard !open.isEmpty else { return }
        Task.detached(priority: .background) {
            for store in open { _ = try? store.compact() }
        }
    }

    /// Statistics of a project's store, opening it when it is not the
    /// active one. `nil` when the project has no store yet.
    public func stats(projectID: UUID) async -> SyntextStoreStats? {
        let key = ContextChatLocation.Store.project(projectID)
        if let open = stores[key] { return try? open.store.stats() }
        let directory = Self.directory(for: key)
        guard FileManager.default.fileExists(atPath: directory.path) else { return nil }
        guard let open = try? await openStore(key) else { return nil }
        let stats = try? open.store.stats()
        if activeProject?.id != projectID { closeStore(key) }
        return stats
    }

    /// Re-reads the whole project: the store is deleted and synced again.
    public func rebuildProject(id: UUID, root: URL) async {
        await removeProjectStore(id: id)
        await activateProject(id: id, root: root)
    }

    /// Deletes a project's store: its files' index and its chats' recall.
    public func removeProjectStore(id: UUID) async {
        if activeProject?.id == id {
            activeProject = nil
            await cancelSync()
        }
        closeStore(.project(id))
        toolOptions = [:]
        try? FileManager.default.removeItem(at: Self.directory(for: .project(id)))
        stateObserver?(id, .idle)
    }

    /// Deletes every store. Recall data leaves the disk at once; project
    /// stores are rebuilt by the next activation.
    public func removeAllStores() async {
        let active = activeProject
        activeProject = nil
        await cancelSync()
        let keys = Array(stores.keys)
        for key in keys { closeStore(key) }
        toolOptions = [:]
        try? FileManager.default.removeItem(at: Self.rootDirectory)
        if let active { stateObserver?(active.id, .idle) }
    }
}
