import Foundation

/// One file as it was before a turn first changed it.
///
/// Recorded at the FIRST file-changing call of a turn that names the path,
/// and never updated afterwards: by the second call the bytes on disk are the
/// turn's own work, so a later capture would make Restore a no-op.
public struct TurnCheckpointEntry: Codable, Equatable, Sendable {
    public enum State: String, Codable, Sendable {
        /// The bytes were copied into the store.
        case captured
        /// The file did not exist, so restoring means the turn's file goes to the Trash.
        case absent
        /// The file was seen but its bytes were not kept (`skipReason` says why).
        case notCaptured
    }

    public enum SkipReason: String, Codable, Sendable {
        case tooLarge
        case turnBudget
        case sensitive
        case notRegularFile
        case readFailed
        case outsideProject
    }

    /// Relative to the canonical project root, symlinks resolved.
    public var relativePath: String
    public var state: State
    public var blobName: String?
    public var byteCount: Int?
    public var sha256: String?
    public var posixPermissions: Int?
    public var skipReason: SkipReason?

    public init(
        relativePath: String, state: State, blobName: String? = nil, byteCount: Int? = nil,
        sha256: String? = nil, posixPermissions: Int? = nil, skipReason: SkipReason? = nil
    ) {
        self.relativePath = relativePath
        self.state = state
        self.blobName = blobName
        self.byteCount = byteCount
        self.sha256 = sha256
        self.posixPermissions = posixPermissions
        self.skipReason = skipReason
    }
}

/// The per-turn manifest, `checkpoints/<chat>/<turn>/manifest.json`.
/// `turnID` is the id of the user message that started the turn.
public struct TurnCheckpointManifest: Codable, Equatable, Sendable {
    public static let currentVersion = 1

    public var version: Int
    public var chatID: UUID
    public var turnID: UUID
    public var createdAt: Date
    /// The canonical project root at capture time. Restore never writes
    /// outside it.
    public var rootPath: String
    public var entries: [TurnCheckpointEntry]
    public var restoredAt: Date?

    public init(chatID: UUID, turnID: UUID, createdAt: Date, rootPath: String, entries: [TurnCheckpointEntry] = []) {
        self.version = Self.currentVersion
        self.chatID = chatID
        self.turnID = turnID
        self.createdAt = createdAt
        self.rootPath = rootPath
        self.entries = entries
        self.restoredAt = nil
    }

    public var isIncomplete: Bool { entries.contains { $0.state == .notCaptured } }

    public var capturedBytes: Int {
        entries.reduce(0) { $0 + ($1.state == .captured ? ($1.byteCount ?? 0) : 0) }
    }

    public var paths: Set<String> { Set(entries.map(\.relativePath)) }
}

/// The pure rules: which calls change files, which paths may be captured,
/// the caps, the restore plan, and retention. Disk access lives in
/// `TurnCheckpointStore`; everything here takes its facts as arguments.
enum TurnCheckpointPolicy {
    static let perFileByteLimit = 5 * 1_024 * 1_024
    static let perTurnByteLimit = 50 * 1_024 * 1_024
    static let retainedTurnsPerChat = 20

    /// The tools whose handler writes a file the call names. Built from the
    /// registry's own alias tables so a new write alias is covered without
    /// a second list to forget. Shell commands are not here: a command's
    /// effects cannot be known before it runs, and the confirmation says so.
    static let fileChangingToolNames: Set<String> = AppToolRegistry.writeFileAliases
        .union(AppToolRegistry.editFileAliases)
        .union([
            "apply_patch", "applypatch", "multiedit", "multi_edit", "document_write",
            "notebookedit", "notebook_edit",
        ])

    static func isFileChangingTool(_ name: String) -> Bool {
        fileChangingToolNames.contains(name.lowercased())
    }

    /// The project-relative paths a file-changing call names, read with the
    /// same resolvers its handler uses. A call whose arguments do not parse
    /// yields nothing: its handler will refuse it before writing.
    static func targetPaths(for call: AppToolCall) -> [String] {
        let name = call.name.lowercased()
        if AppToolRegistry.writeFileAliases.contains(name) || AppToolRegistry.editFileAliases.contains(name) {
            let resolved = ToolArgumentResolver.fileArguments(call.arguments)
            guard resolved.conflict == nil, let path = resolved.path else { return [] }
            return [path]
        }
        switch name {
        case "apply_patch", "applypatch":
            return ToolActionFusion.patchEntries(call).map(\.path)
        case "multiedit", "multi_edit":
            let edits = (try? MultiEditExecutor.parseEdits(from: call.arguments)) ?? []
            var seen = Set<String>()
            return edits.map(\.filePath).filter { seen.insert($0).inserted }
        case "document_write":
            return [call.arguments["path"] ?? call.arguments["file_path"] ?? call.arguments["filePath"]]
                .compactMap { $0 }
        case "notebookedit", "notebook_edit":
            return [call.arguments["notebook_path"] ?? call.arguments["notebookPath"]
                ?? call.arguments["path"] ?? call.arguments["file_path"]].compactMap { $0 }
        default:
            return []
        }
    }

    /// The canonical file url and its path relative to the canonical root,
    /// or nil when the path is absolute, home-relative, the root itself, or
    /// resolves (through any symlink, dangling ones included) outside the
    /// root. The handlers refuse every one of those, so there is nothing a
    /// checkpoint could need to restore.
    static func containedTarget(_ requested: String, root: URL) -> (url: URL, relativePath: String)? {
        let cleaned = requested.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !cleaned.isEmpty, !cleaned.hasPrefix("/"), !cleaned.hasPrefix("~") else { return nil }
        let canonicalRoot = PathContainment.canonical(root)
        // `resolvedForContainment` walks a path that does not exist yet, so
        // `link/new.txt` with `link -> /outside` is judged by its real place.
        let candidate = AppToolSandbox.resolvedForContainment(canonicalRoot.appendingPathComponent(cleaned))
        guard let resolved = PathContainment.resolvedIfContained(candidate, in: canonicalRoot) else { return nil }
        let rootPath = canonicalRoot.path
        guard resolved.path != rootPath else { return nil }
        // Contained and not the root, so the path starts with root + "/".
        let prefixLength = rootPath.hasSuffix("/") ? rootPath.count : rootPath.count + 1
        return (resolved, String(resolved.path.dropFirst(prefixLength)))
    }

    /// What is on disk at a path, as far as the rules care.
    enum FileKind: Equatable, Sendable {
        case absent
        case regular(byteCount: Int)
        case other
    }

    enum CaptureDecision: Equatable {
        /// This turn already recorded the path; its first state stands.
        case alreadyRecorded
        case capture
        case recordAbsent
        case skip(TurnCheckpointEntry.SkipReason)
    }

    /// The first-capture rule and the caps. A skipped path still counts as
    /// recorded, so a later call in the same turn cannot capture the turn's
    /// own bytes and present them as the pre-turn state.
    static func decide(
        relativePath: String, recorded: Set<String>, isSensitive: Bool, kind: FileKind,
        capturedBytes: Int,
        perFileLimit: Int = perFileByteLimit, perTurnLimit: Int = perTurnByteLimit
    ) -> CaptureDecision {
        if recorded.contains(relativePath) { return .alreadyRecorded }
        if isSensitive { return .skip(.sensitive) }
        switch kind {
        case .absent:
            return .recordAbsent
        case .other:
            return .skip(.notRegularFile)
        case .regular(let byteCount):
            if byteCount > perFileLimit { return .skip(.tooLarge) }
            if capturedBytes + byteCount > perTurnLimit { return .skip(.turnBudget) }
            return .capture
        }
    }

    /// The checkpoints to delete so only the newest `keep` remain, oldest first.
    static func turnsToPrune(_ turns: [(turnID: UUID, createdAt: Date)], keep: Int = retainedTurnsPerChat) -> [UUID] {
        guard turns.count > keep else { return [] }
        let newestFirst = turns.sorted { $0.createdAt > $1.createdAt }
        return newestFirst.dropFirst(max(0, keep)).sorted { $0.createdAt < $1.createdAt }.map(\.turnID)
    }
}

/// What Restore will do, built from a manifest and the disk as it is now.
struct TurnCheckpointRestorePlan: Equatable, Sendable {
    enum Action: Equatable, Sendable {
        /// The file exists and gets the snapshot bytes.
        case overwrite
        /// The file is gone and is written again from the snapshot.
        case recreate
        /// The turn created the file; it goes to the Trash.
        case trash
        /// Nothing can be done for this path.
        case cannotRestore(TurnCheckpointEntry.SkipReason)
    }

    struct Item: Equatable, Sendable {
        var relativePath: String
        var action: Action
        /// A later turn also changed this path, so restoring discards that
        /// turn's work on it too.
        var changedInLaterTurn: Bool
    }

    var items: [Item]

    var actionable: [Item] {
        items.filter { if case .cannotRestore = $0.action { return false } else { return true } }
    }

    var unrestorable: [Item] {
        items.filter { if case .cannotRestore = $0.action { return true } else { return false } }
    }

    /// `current` maps a relative path to what is there now, or nil when the
    /// path no longer resolves inside the root (a symlink swapped in since).
    /// A file the turn created that is already gone needs nothing and is left
    /// out of the plan.
    static func make(
        manifest: TurnCheckpointManifest, current: (String) -> TurnCheckpointPolicy.FileKind?,
        laterTurnPaths: Set<String>
    ) -> TurnCheckpointRestorePlan {
        var items: [Item] = []
        for entry in manifest.entries {
            let later = laterTurnPaths.contains(entry.relativePath)
            func add(_ action: Action) {
                items.append(Item(relativePath: entry.relativePath, action: action, changedInLaterTurn: later))
            }
            if entry.state == .notCaptured {
                add(.cannotRestore(entry.skipReason ?? .readFailed))
                continue
            }
            guard let kind = current(entry.relativePath) else {
                add(.cannotRestore(.outsideProject))
                continue
            }
            switch (entry.state, kind) {
            case (.captured, .absent): add(.recreate)
            case (.captured, .regular): add(.overwrite)
            case (.absent, .regular): add(.trash)
            case (.absent, .absent): continue
            case (_, .other): add(.cannotRestore(.notRegularFile))
            case (.notCaptured, _): continue
            }
        }
        return TurnCheckpointRestorePlan(items: items)
    }
}
