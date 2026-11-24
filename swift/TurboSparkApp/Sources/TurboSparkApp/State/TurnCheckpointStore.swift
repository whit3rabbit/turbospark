import Foundation

/// Per-turn file snapshots on disk:
/// `<store root>/checkpoints/<chat id>/<turn id>/manifest.json` plus
/// `blobs/<n>.bin`, one blob per captured file.
///
/// Under `AppStorageRoot`, so it follows the active profile,
/// `TURBOSPARK_STATE_DIR` and the per-process test root like every other
/// store. Never used for Ghost chats or a protected profile: the caller
/// (`AppModel.checkpointTurnID`) refuses those before any byte is copied,
/// because blobs here are plaintext copies of project files.
///
/// **CAPTURE NEVER BLOCKS THE TOOL.** Every failure becomes a
/// `notCaptured` entry (the checkpoint shows "incomplete") or a stderr line;
/// nothing here throws back into `AppToolRegistry.execute`.
final class TurnCheckpointStore: @unchecked Sendable {
    static let shared = TurnCheckpointStore()

    let rootURL: URL
    private let fileManager: FileManager
    /// One lock for capture, restore and listing. Batched tool calls run
    /// concurrently, and two writers of one manifest would lose an entry,
    /// which is a path whose pre-turn bytes then get captured late.
    private let lock = NSLock()
    /// Manifests this process wrote, so the first-capture rule does not
    /// depend on a save that failed.
    private var cache: [String: TurnCheckpointManifest] = [:]
    /// Turns whose manifest was corrupt. Nothing more is captured for them:
    /// a fresh manifest would record the turn's own bytes as "before".
    private var poisoned: Set<String> = []

    init(rootURL: URL? = nil, fileManager: FileManager = .default) {
        self.rootURL = rootURL ?? AppStorageRoot.subdirectory("checkpoints")
        self.fileManager = fileManager
    }

    private static func key(_ chatID: UUID, _ turnID: UUID) -> String {
        "\(chatID.uuidString)/\(turnID.uuidString)"
    }

    private func chatDirectory(_ chatID: UUID) -> URL {
        rootURL.appendingPathComponent(chatID.uuidString, isDirectory: true)
    }

    private func turnDirectory(_ chatID: UUID, _ turnID: UUID) -> URL {
        chatDirectory(chatID).appendingPathComponent(turnID.uuidString, isDirectory: true)
    }

    private func manifestURL(_ chatID: UUID, _ turnID: UUID) -> URL {
        turnDirectory(chatID, turnID).appendingPathComponent("manifest.json")
    }

    private func blobURL(_ chatID: UUID, _ turnID: UUID, _ name: String) -> URL {
        turnDirectory(chatID, turnID).appendingPathComponent("blobs", isDirectory: true)
            .appendingPathComponent(name)
    }

    private static let encoder: JSONEncoder = {
        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return encoder
    }()

    private static let decoder: JSONDecoder = {
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        return decoder
    }()

    // MARK: Reading

    /// The manifest, or nil when there is none. A corrupt one is quarantined
    /// by `AppJSONStore` (renamed, never overwritten) and reads as none.
    /// Caller holds the lock.
    private func loadLocked(_ chatID: UUID, _ turnID: UUID) -> TurnCheckpointManifest? {
        let key = Self.key(chatID, turnID)
        if let cached = cache[key] { return cached }
        if poisoned.contains(key) { return nil }
        let url = manifestURL(chatID, turnID)
        let directoryExists = fileManager.fileExists(atPath: turnDirectory(chatID, turnID).path)
        guard let manifest = AppJSONStore.load(
            TurnCheckpointManifest.self, from: url, label: "Checkpoint", decoder: Self.decoder),
            manifest.chatID == chatID, manifest.turnID == turnID
        else {
            // A turn folder with no readable manifest is a quarantined or
            // half-written checkpoint. Never start it over.
            if directoryExists { poisoned.insert(key) }
            return nil
        }
        cache[key] = manifest
        return manifest
    }

    func manifest(chatID: UUID, turnID: UUID) -> TurnCheckpointManifest? {
        lock.withLock { loadLocked(chatID, turnID) }
    }

    /// Every readable checkpoint of a chat, oldest first.
    func manifests(chatID: UUID) -> [TurnCheckpointManifest] {
        lock.withLock { manifestsLocked(chatID) }
    }

    private func manifestsLocked(_ chatID: UUID) -> [TurnCheckpointManifest] {
        let names = (try? fileManager.contentsOfDirectory(atPath: chatDirectory(chatID).path)) ?? []
        return names.compactMap(UUID.init(uuidString:))
            .compactMap { loadLocked(chatID, $0) }
            .sorted { $0.createdAt < $1.createdAt }
    }

    // MARK: Snapshot reads (artifact history)

    /// The entry's snapshot file, or nil when its name is not a plain file
    /// name. The manifest is our own file, but a name like `../x` must never
    /// turn a read into one outside the turn's blobs folder.
    private func snapshotURL(chatID: UUID, turnID: UUID, entry: TurnCheckpointEntry) -> URL? {
        guard entry.state == .captured, let name = entry.blobName, !name.isEmpty,
              !name.contains("/"), name != ".", name != ".."
        else { return nil }
        return blobURL(chatID, turnID, name)
    }

    /// Whether a captured entry's snapshot file is still on disk. Existence
    /// only, so listing versions never reads up to 5 MB per turn.
    func snapshotExists(chatID: UUID, turnID: UUID, entry: TurnCheckpointEntry) -> Bool {
        guard let url = snapshotURL(chatID: chatID, turnID: turnID, entry: entry) else { return false }
        return fileManager.fileExists(atPath: url.path)
    }

    /// The bytes a captured entry holds, checked against the manifest's size
    /// and sha256 exactly as Restore checks them, or nil. Reads take no lock:
    /// blobs are written atomically and never rewritten, and a pruned one
    /// reads as nil.
    func snapshotData(chatID: UUID, turnID: UUID, entry: TurnCheckpointEntry) -> Data? {
        guard let url = snapshotURL(chatID: chatID, turnID: turnID, entry: entry),
              let data = try? Data(contentsOf: url),
              data.count == entry.byteCount, ToolObservationStore.hash(data) == entry.sha256
        else { return nil }
        return data
    }

    // MARK: Capture

    /// Records each named path the turn has not recorded yet, before the
    /// tool writes it. Returns the manifest when it changed.
    @discardableResult
    func capture(
        paths: [String], rootURL: URL, chatID: UUID, turnID: UUID, now: Date = Date()
    ) -> TurnCheckpointManifest? {
        let targets = paths.compactMap { TurnCheckpointPolicy.containedTarget($0, root: rootURL) }
        guard !targets.isEmpty else { return nil }
        return lock.withLock {
            let key = Self.key(chatID, turnID)
            var manifest: TurnCheckpointManifest
            var isNew = false
            if let existing = loadLocked(chatID, turnID) {
                manifest = existing
            } else {
                guard !poisoned.contains(key) else { return nil }
                manifest = TurnCheckpointManifest(
                    chatID: chatID, turnID: turnID, createdAt: now,
                    rootPath: PathContainment.canonical(rootURL).path)
                isNew = true
            }
            // A checkpoint is for one root. A call against another root (the
            // chat's project changed mid-turn) is not mixed in.
            guard manifest.rootPath == PathContainment.canonical(rootURL).path else { return nil }

            var changed = false
            for target in targets {
                let decision = TurnCheckpointPolicy.decide(
                    relativePath: target.relativePath, recorded: manifest.paths,
                    isSensitive: ToolRiskClassifier.isSensitivePath(target.relativePath)
                        || ToolRiskClassifier.isSensitivePath(target.url.path),
                    kind: Self.kind(of: target.url, fileManager: fileManager),
                    capturedBytes: manifest.capturedBytes)
                switch decision {
                case .alreadyRecorded:
                    continue
                case .recordAbsent:
                    manifest.entries.append(TurnCheckpointEntry(relativePath: target.relativePath, state: .absent))
                case .skip(let reason):
                    manifest.entries.append(TurnCheckpointEntry(
                        relativePath: target.relativePath, state: .notCaptured, skipReason: reason))
                case .capture:
                    manifest.entries.append(copyBlob(
                        target: target, index: manifest.entries.count, chatID: chatID, turnID: turnID))
                }
                changed = true
            }
            guard changed else { return nil }
            if !AppJSONStore.save(manifest, to: manifestURL(chatID, turnID), label: "Checkpoint", encoder: Self.encoder) {
                FileHandle.standardError.write(
                    "TurboSpark: checkpoint manifest for turn \(turnID) could not be saved\n".data(using: .utf8)!)
            }
            cache[key] = manifest
            if isNew { pruneLocked(chatID: chatID, keeping: turnID) }
            return manifest
        }
    }

    private func copyBlob(
        target: (url: URL, relativePath: String), index: Int, chatID: UUID, turnID: UUID
    ) -> TurnCheckpointEntry {
        let name = "\(index).bin"
        do {
            let data = try Data(contentsOf: target.url)
            let destination = blobURL(chatID, turnID, name)
            try fileManager.createDirectory(
                at: destination.deletingLastPathComponent(), withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
            try data.write(to: destination, options: .atomic)
            try? fileManager.setAttributes([.posixPermissions: 0o600], ofItemAtPath: destination.path)
            let permissions = (try? fileManager.attributesOfItem(atPath: target.url.path))?[.posixPermissions] as? Int
            return TurnCheckpointEntry(
                relativePath: target.relativePath, state: .captured, blobName: name,
                byteCount: data.count, sha256: ToolObservationStore.hash(data),
                posixPermissions: permissions)
        } catch {
            FileHandle.standardError.write(
                "TurboSpark: checkpoint could not copy a file: \(error.localizedDescription)\n".data(using: .utf8)!)
            return TurnCheckpointEntry(relativePath: target.relativePath, state: .notCaptured, skipReason: .readFailed)
        }
    }

    /// `lstat`, not `stat`: `containedTarget` already resolved symlinks, so a
    /// link here is one the walk could not follow, and it is not captured.
    static func kind(of url: URL, fileManager: FileManager = .default) -> TurnCheckpointPolicy.FileKind {
        var info = stat()
        guard lstat(url.path, &info) == 0 else { return .absent }
        guard (info.st_mode & S_IFMT) == S_IFREG else { return .other }
        return .regular(byteCount: Int(info.st_size))
    }

    // MARK: Retention and deletion

    private func pruneLocked(chatID: UUID, keeping current: UUID) {
        let names = (try? fileManager.contentsOfDirectory(atPath: chatDirectory(chatID).path)) ?? []
        let turns: [(turnID: UUID, createdAt: Date)] = names.compactMap(UUID.init(uuidString:)).map { turnID in
            if let manifest = loadLocked(chatID, turnID) { return (turnID, manifest.createdAt) }
            // An unreadable one sorts by its folder date, so it ages out too.
            let attributes = try? fileManager.attributesOfItem(atPath: turnDirectory(chatID, turnID).path)
            return (turnID, (attributes?[.creationDate] as? Date) ?? .distantPast)
        }
        for turnID in TurnCheckpointPolicy.turnsToPrune(turns) where turnID != current {
            removeTurnLocked(chatID: chatID, turnID: turnID)
        }
    }

    private func removeTurnLocked(chatID: UUID, turnID: UUID) {
        let key = Self.key(chatID, turnID)
        cache[key] = nil
        poisoned.remove(key)
        try? fileManager.removeItem(at: turnDirectory(chatID, turnID))
    }

    /// Removes every checkpoint of a deleted chat.
    func delete(chatID: UUID) {
        lock.withLock {
            let prefix = chatID.uuidString + "/"
            cache = cache.filter { !$0.key.hasPrefix(prefix) }
            poisoned = poisoned.filter { !$0.hasPrefix(prefix) }
            try? fileManager.removeItem(at: chatDirectory(chatID))
        }
    }

    // MARK: Restore

    /// The plan for one checkpoint against the disk now, or nil when there is
    /// no readable checkpoint.
    func restorePlan(chatID: UUID, turnID: UUID) -> TurnCheckpointRestorePlan? {
        lock.withLock { restorePlanLocked(chatID: chatID, turnID: turnID)?.plan }
    }

    private func restorePlanLocked(
        chatID: UUID, turnID: UUID
    ) -> (manifest: TurnCheckpointManifest, plan: TurnCheckpointRestorePlan)? {
        guard let manifest = loadLocked(chatID, turnID) else { return nil }
        let later = manifestsLocked(chatID)
            .filter { $0.turnID != turnID && $0.createdAt > manifest.createdAt }
            .reduce(into: Set<String>()) { $0.formUnion($1.paths) }
        let root = URL(fileURLWithPath: manifest.rootPath, isDirectory: true)
        let plan = TurnCheckpointRestorePlan.make(
            manifest: manifest,
            current: { path in
                // The path must still resolve to the same place inside the
                // root; a symlink swapped in since capture is refused.
                guard let target = TurnCheckpointPolicy.containedTarget(path, root: root),
                      target.relativePath == path
                else { return nil }
                return Self.kind(of: target.url, fileManager: self.fileManager)
            },
            laterTurnPaths: later)
        return (manifest, plan)
    }

    struct RestoreOutcome: Equatable, Sendable {
        var restored: [String] = []
        var trashed: [String] = []
        var failed: [String] = []
        var skipped: [String] = []
    }

    /// Carries out the plan. `trash` moves a file to the Trash; it is a seam
    /// so tests never touch the user's real Trash. Files the turn created are
    /// never hard-deleted.
    func restore(
        chatID: UUID, turnID: UUID, now: Date = Date(),
        trash: (URL) throws -> Void = { try FileManager.default.trashItem(at: $0, resultingItemURL: nil) }
    ) -> RestoreOutcome? {
        lock.withLock {
            guard let planned = restorePlanLocked(chatID: chatID, turnID: turnID) else { return nil }
            let manifest = planned.manifest
            let plan = planned.plan
            let root = URL(fileURLWithPath: manifest.rootPath, isDirectory: true)
            let entries = Dictionary(manifest.entries.map { ($0.relativePath, $0) }, uniquingKeysWith: { first, _ in first })
            var outcome = RestoreOutcome()
            for item in plan.items {
                // Re-resolved at write time, not trusted from the plan.
                guard let target = TurnCheckpointPolicy.containedTarget(item.relativePath, root: root),
                      target.relativePath == item.relativePath,
                      !ToolRiskClassifier.isSensitivePath(target.url.path)
                else {
                    outcome.skipped.append(item.relativePath)
                    continue
                }
                switch item.action {
                case .cannotRestore:
                    outcome.skipped.append(item.relativePath)
                case .trash:
                    do {
                        try trash(target.url)
                        outcome.trashed.append(item.relativePath)
                    } catch {
                        outcome.failed.append(item.relativePath)
                    }
                case .overwrite, .recreate:
                    guard let entry = entries[item.relativePath], let blob = entry.blobName,
                          let data = try? Data(contentsOf: blobURL(chatID, turnID, blob)),
                          data.count == entry.byteCount, ToolObservationStore.hash(data) == entry.sha256
                    else {
                        outcome.failed.append(item.relativePath)
                        continue
                    }
                    do {
                        try fileManager.createDirectory(
                            at: target.url.deletingLastPathComponent(), withIntermediateDirectories: true)
                        try data.write(to: target.url, options: .atomic)
                        if let permissions = entry.posixPermissions {
                            try? fileManager.setAttributes([.posixPermissions: permissions], ofItemAtPath: target.url.path)
                        }
                        outcome.restored.append(item.relativePath)
                    } catch {
                        outcome.failed.append(item.relativePath)
                    }
                }
            }
            var marked = manifest
            marked.restoredAt = now
            if AppJSONStore.save(marked, to: manifestURL(chatID, turnID), label: "Checkpoint", encoder: Self.encoder) {
                cache[Self.key(chatID, turnID)] = marked
            }
            return outcome
        }
    }
}

extension AppToolRegistry {
    /// Installed by `AppModel`: snapshots the files a file-changing call
    /// names, before its handler runs. Awaited, so the copy is on disk before
    /// the first byte is written; it never changes the call or its result.
    static var turnCheckpointCapturer: (@Sendable (AppToolCall, URL, UUID) async -> Void)?
}
