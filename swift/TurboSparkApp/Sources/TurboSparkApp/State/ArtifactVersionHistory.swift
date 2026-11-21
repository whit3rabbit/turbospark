import Foundation

/// One row of the artifact panel's History menu: the file on disk now, or the
/// file as it was before a turn first changed it.
///
/// Read-only over `TurnCheckpointStore`. Nothing here stores bytes: an
/// earlier version exists only while its turn's checkpoint does, so history
/// ages out with checkpoint retention and goes when the chat is deleted.
struct ArtifactVersion: Identifiable, Equatable, Sendable {
    enum Source: Equatable, Sendable {
        case current
        /// The pre-turn bytes of `relativePath` in that turn's checkpoint.
        case beforeTurn(turnID: UUID, relativePath: String)
    }

    enum Unavailable: Equatable, Sendable {
        /// The checkpoint saw the file but did not keep its bytes.
        case notCaptured(TurnCheckpointEntry.SkipReason)
        /// The manifest says captured, but the snapshot file is gone.
        case snapshotMissing
        /// The current file is no longer at its written path.
        case fileMissing
    }

    var source: Source
    /// When the turn's checkpoint was taken; nil for the current file.
    var date: Date?
    /// 1-based position of the turn's prompt among the chat's prompts, or
    /// nil when that message is no longer in the chat.
    var turnNumber: Int?
    var byteCount: Int?
    var sha256: String?
    var unavailable: Unavailable?

    var id: String {
        switch source {
        case .current: return "current"
        case .beforeTurn(let turnID, _): return turnID.uuidString
        }
    }

    var isCurrent: Bool { source == .current }
    var turnID: UUID? {
        if case .beforeTurn(let turnID, _) = source { return turnID }
        return nil
    }
}

enum ArtifactVersionHistory {
    /// What is on disk for the artifact now. `sha256` is nil when the file is
    /// too large to hash cheaply; that only turns off dedupe against it.
    struct CurrentFile: Equatable, Sendable {
        var exists: Bool
        var byteCount: Int?
        var sha256: String?
    }

    /// The path of `artifactPath` relative to a checkpoint's root, by the
    /// store's own containment rule (`TurnCheckpointPolicy.containedTarget`),
    /// or nil when the file is not inside that root. The round trip through
    /// `containedTarget` means a path the store would refuse to capture or
    /// restore is never matched here either.
    static func relativePath(ofArtifactPath artifactPath: String, inRoot rootPath: String) -> String? {
        let root = PathContainment.canonical(URL(fileURLWithPath: rootPath, isDirectory: true))
        let file = AppToolSandbox.resolvedForContainment(URL(fileURLWithPath: artifactPath))
        guard let contained = PathContainment.resolvedIfContained(file, in: root) else { return nil }
        let base = root.path.hasSuffix("/") ? root.path : root.path + "/"
        guard contained.path.hasPrefix(base) else { return nil }
        let relative = String(contained.path.dropFirst(base.count))
        guard let target = TurnCheckpointPolicy.containedTarget(relative, root: root),
              target.relativePath == relative
        else { return nil }
        return relative
    }

    /// The version list, newest first: Current, then one row per checkpoint
    /// that recorded the path. A turn that CREATED the file (`absent`) has no
    /// earlier content and adds no row. A row whose content equals the row
    /// just above it (same sha256) is dropped, so a turn that rewrote the
    /// file unchanged, or a restore, does not show two identical versions.
    ///
    /// - Parameters:
    ///   - relativePath: the artifact's path inside a manifest's root, or nil
    ///     when it is not inside it (a checkpoint of another project folder).
    ///   - snapshotExists: whether a captured entry's snapshot file is on disk.
    static func versions(
        manifests: [TurnCheckpointManifest],
        relativePath: (TurnCheckpointManifest) -> String?,
        current: CurrentFile,
        turnNumbers: [UUID: Int],
        snapshotExists: (TurnCheckpointManifest, TurnCheckpointEntry) -> Bool
    ) -> [ArtifactVersion] {
        var rows = [ArtifactVersion(
            source: .current, date: nil, turnNumber: nil,
            byteCount: current.exists ? current.byteCount : nil,
            sha256: current.exists ? current.sha256 : nil,
            unavailable: current.exists ? nil : .fileMissing)]

        // Manifest dates are stored to the second, so two turns in one second
        // tie; the prompt order breaks it before the arbitrary id does.
        let newestFirst = manifests.sorted {
            if $0.createdAt != $1.createdAt { return $0.createdAt > $1.createdAt }
            let left = turnNumbers[$0.turnID] ?? 0
            let right = turnNumbers[$1.turnID] ?? 0
            if left != right { return left > right }
            return $0.turnID.uuidString > $1.turnID.uuidString
        }
        for manifest in newestFirst {
            guard let path = relativePath(manifest),
                  let entry = manifest.entries.first(where: { $0.relativePath == path })
            else { continue }
            var row = ArtifactVersion(
                source: .beforeTurn(turnID: manifest.turnID, relativePath: path),
                date: manifest.createdAt, turnNumber: turnNumbers[manifest.turnID],
                byteCount: entry.byteCount, sha256: entry.sha256, unavailable: nil)
            switch entry.state {
            case .absent:
                continue
            case .notCaptured:
                row.byteCount = nil
                row.sha256 = nil
                row.unavailable = .notCaptured(entry.skipReason ?? .readFailed)
            case .captured:
                if entry.blobName == nil || entry.sha256 == nil || !snapshotExists(manifest, entry) {
                    row.unavailable = .snapshotMissing
                }
            }
            if row.unavailable == nil, let sha = row.sha256, let above = rows.last,
               above.unavailable == nil, above.sha256 == sha {
                continue
            }
            rows.append(row)
        }
        return rows
    }

    /// The turn number of each prompt: the chat's user messages, in order.
    static func turnNumbers(in messages: [AppChatMessage]) -> [UUID: Int] {
        var numbers: [UUID: Int] = [:]
        var count = 0
        for message in messages where message.role == .user {
            count += 1
            numbers[message.id] = count
        }
        return numbers
    }
}

/// Copy for the History menu and banner, kept out of the view so it can be
/// tested and so every string passes through the catalog.
enum ArtifactVersionText {
    static func title(_ version: ArtifactVersion) -> String {
        if version.isCurrent { return AppStrings.localized("Current version") }
        if let number = version.turnNumber {
            return AppStrings.localized("Before turn \(number)")
        }
        return AppStrings.localized("Before a removed turn")
    }

    static func reason(_ unavailable: ArtifactVersion.Unavailable) -> String {
        switch unavailable {
        case .fileMissing: return AppStrings.localized("The file is no longer at its written path.")
        case .snapshotMissing: return AppStrings.localized("Snapshot missing")
        case .notCaptured(let reason):
            switch reason {
            case .tooLarge: return AppStrings.localized("Not captured: file too large")
            case .turnBudget: return AppStrings.localized("Not captured: turn size limit")
            case .sensitive: return AppStrings.localized("Not captured: sensitive file")
            case .notRegularFile: return AppStrings.localized("Not captured: not a regular file")
            case .readFailed: return AppStrings.localized("Not captured: could not be read")
            case .outsideProject: return AppStrings.localized("Not captured: outside the project")
            }
        }
    }

    /// One menu row: title, then time and size, or the reason it cannot be
    /// shown. The separator is not translated: it is punctuation only.
    static func menuRow(_ version: ArtifactVersion, now: Date = Date()) -> String {
        var parts = [title(version)]
        if let date = version.date {
            parts.append(date.formatted(date: Calendar.current.isDate(date, inSameDayAs: now) ? .omitted : .abbreviated,
                                        time: .shortened))
        }
        if let unavailable = version.unavailable {
            parts.append(reason(unavailable))
        } else if let bytes = version.byteCount {
            parts.append(ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file))
        }
        return parts.joined(separator: " \u{00B7} ")
    }
}

// MARK: Disk side

extension ArtifactVersionHistory {
    /// Bytes above this are not read as text for display. Snapshots are
    /// capped at `TurnCheckpointPolicy.perFileByteLimit`, so this also bounds
    /// hashing the current file for dedupe.
    static let displayByteLimit = TurnCheckpointPolicy.perFileByteLimit

    /// Builds the list from the chat's checkpoints and the file on disk. Off
    /// the main actor: it takes the store lock and may hash a few MB.
    static func load(
        chatID: UUID, artifactPath: String, turnNumbers: [UUID: Int],
        store: TurnCheckpointStore = .shared, fileManager: FileManager = .default
    ) -> [ArtifactVersion] {
        let manifests = store.manifests(chatID: chatID)
        guard !manifests.isEmpty else { return [] }
        var relativeByRoot: [String: String?] = [:]
        return versions(
            manifests: manifests,
            relativePath: { manifest in
                if let cached = relativeByRoot[manifest.rootPath] { return cached }
                let relative = relativePath(ofArtifactPath: artifactPath, inRoot: manifest.rootPath)
                relativeByRoot[manifest.rootPath] = relative
                return relative
            },
            current: currentFile(atPath: artifactPath, fileManager: fileManager),
            turnNumbers: turnNumbers,
            snapshotExists: { manifest, entry in
                store.snapshotExists(chatID: manifest.chatID, turnID: manifest.turnID, entry: entry)
            })
    }

    static func currentFile(atPath path: String, fileManager: FileManager = .default) -> CurrentFile {
        let url = URL(fileURLWithPath: path)
        guard case .regular(let byteCount) = TurnCheckpointStore.kind(of: PathContainment.canonical(url), fileManager: fileManager)
        else { return CurrentFile(exists: fileManager.fileExists(atPath: path), byteCount: nil, sha256: nil) }
        let sha = byteCount <= displayByteLimit ? (try? Data(contentsOf: url)).map(ToolObservationStore.hash) : nil
        return CurrentFile(exists: true, byteCount: byteCount, sha256: sha)
    }

    /// The earlier version's text, verified against its manifest, or nil
    /// when it is gone, fails the check, or is not UTF-8.
    static func text(
        of version: ArtifactVersion, chatID: UUID, store: TurnCheckpointStore = .shared
    ) -> String? {
        guard case .beforeTurn(let turnID, let relativePath) = version.source,
              let manifest = store.manifest(chatID: chatID, turnID: turnID),
              let entry = manifest.entries.first(where: { $0.relativePath == relativePath }),
              (entry.byteCount ?? Int.max) <= displayByteLimit,
              let data = store.snapshotData(chatID: chatID, turnID: turnID, entry: entry)
        else { return nil }
        return String(data: data, encoding: .utf8)
    }
}
