import CryptoKit
import Foundation
import SQLCipher

public final class ProfileRepository: @unchecked Sendable {
    public static let shared = ProfileRepository()
    static let legacyAssetsRecordKey = "migration:legacy-assets"

    private let store: ProfileVaultStore
    private let encoder = JSONEncoder()
    private let decoder = JSONDecoder()

    init(store: ProfileVaultStore = .shared) {
        self.store = store
    }

    public var isAvailable: Bool { store.session != nil }

    public func load<T: Decodable>(
        _ type: T.Type,
        key: String,
        legacyURL: URL? = nil
    ) throws -> T? {
        let database = try database()
        if let data = try database.loadRecord(key: key) {
            do {
                return try decoder.decode(type, from: data)
            } catch {
                // The caller falls back to an empty default and its next save
                // replaces this record, so keep the raw payload recoverable.
                try? quarantineRecord(key: key, payload: data, in: database)
                throw error
            }
        }
        guard let legacyURL,
              FileManager.default.fileExists(atPath: legacyURL.path)
        else { return nil }
        let data = try Data(contentsOf: legacyURL)
        let value = try decoder.decode(type, from: data)
        try database.saveRecord(key: key, payload: data)
        return value
    }

    /// Keeps a copy of a record its owner could not read, as `load` does,
    /// for stores that read raw records themselves. The original is left in
    /// place: its asset references may still protect files it owns.
    func quarantineRawRecord(key: String, payload: Data) throws {
        try quarantineRecord(key: key, payload: payload, in: database())
    }

    /// Copies an undecodable record to `quarantine:<key>:<unix time>`. Skips
    /// the copy when an identical payload is already quarantined so a record
    /// that stays broken across launches does not grow the table.
    func quarantineRecord(key: String, payload: Data, in database: ProfileDatabase) throws {
        let prefix = "quarantine:\(key):"
        if try database.loadRecords(prefix: prefix).contains(where: { $0.1 == payload }) {
            return
        }
        try database.saveRecord(
            key: prefix + String(Int(Date().timeIntervalSince1970)), payload: payload)
    }

    public func save<T: Encodable>(_ value: T, key: String) throws {
        try database().saveRecord(key: key, payload: encoder.encode(value))
    }

    public func loadChatArchive(legacyURL: URL? = nil) throws -> AppChatArchive? {
        let database = try database()
        if let archive = try database.loadChatArchive() { return archive }
        guard let legacyURL,
              FileManager.default.fileExists(atPath: legacyURL.path)
        else { return nil }
        let data = try Data(contentsOf: legacyURL)
        let archive = try decoder.decode(AppChatArchive.self, from: data)
        _ = try database.saveChatArchive(archive)
        return archive
    }

    public func saveChatArchive(_ archive: AppChatArchive) throws {
        let unreachable = try database().saveChatArchive(archive)
        ManagedAssetStore(vault: store).garbageCollect(ids: unreachable)
    }

    public func loadProjectArchive(legacyURL: URL? = nil) throws -> AppProjectArchive? {
        let database = try database()
        if let archive = try database.loadProjectArchive() { return archive }
        guard let legacyURL, FileManager.default.fileExists(atPath: legacyURL.path) else {
            return nil
        }
        let archive = try decoder.decode(
            AppProjectArchive.self, from: Data(contentsOf: legacyURL))
        try database.saveProjectArchive(archive)
        return archive
    }

    public func saveProjectArchive(_ archive: AppProjectArchive) throws {
        try database().saveProjectArchive(archive)
    }

    public func checkpoint() throws { try database().checkpoint() }

    func rawRecord(key: String) throws -> Data? {
        try database().loadRecord(key: key)
    }

    func saveRawRecord(_ data: Data, key: String) throws {
        try database().saveRecord(key: key, payload: data)
    }

    func deleteRecord(key: String) throws {
        try database().deleteRecord(key: key)
    }

    func rawRecords(prefix: String) throws -> [(String, Data)] {
        try database().loadRecords(prefix: prefix)
    }

    func loadMemoryLedger() throws -> Data? {
        try database().loadMemoryLedger()
    }

    func saveMemoryLedger(_ payload: Data, projections: [String: Data]) throws {
        try database().saveMemoryLedger(payload, projections: projections)
    }

    func saveMemoryEmbeddings(_ rows: [MemoryClaimEmbedding]) throws {
        try database().saveMemoryEmbeddings(rows)
    }

    func loadMemoryEmbeddings(model: String) throws -> [MemoryClaimEmbedding] {
        try database().loadMemoryEmbeddings(model: model)
    }

    func clearMemoryEmbeddings() throws {
        try database().clearMemoryEmbeddings()
    }

    func memoryClaimSearchIDs(query: String, limit: Int) throws -> [UUID] {
        try database().memoryClaimSearchIDs(query: query, limit: limit)
    }

    static let protectedFileNames: Set<String> = [
            "appearance.json",
            "chats_archive.json",
            "cron_jobs.json",
            "disabled_items.json",
            "excluded_scan_paths.json",
            "global_mcp_servers.json",
            "granted_folders.json",
            "input_history.json",
            "mcp_marketplaces.json",
            "model_organization.json",
            "projects_archive.json",
            "settings.json",
        ]

    static func protectedRecordKey(for url: URL) -> String? {
        guard protectedFileNames.contains(url.lastPathComponent) else { return nil }
        let root = AppStorageRoot.directory.standardizedFileURL.path
        let candidate = url.standardizedFileURL.path
        guard candidate == root || candidate.hasPrefix(root + "/") else { return nil }
        let relative = String(candidate.dropFirst(root.count)).trimmingCharacters(in: CharacterSet(charactersIn: "/"))
        return "json:\(relative)"
    }

    /// True when the migration wrote anything the open-time check has not
    /// already covered, or is about to delete plaintext sources.
    static func needsPostMigrationIntegrityCheck(
        importedFiles: Int, memoryRoots: Int, observationRoots: Int
    ) -> Bool {
        importedFiles > 0 || memoryRoots > 0 || observationRoots > 0
    }

    func migrateLegacyPrivateFiles() throws {
        let root = store.rootURL.deletingLastPathComponent()
        let names = Self.protectedFileNames.sorted()
        guard let session = store.session else { throw ProfileVaultStore.VaultError.locked }
        let recoveryKey = ProfileVaultCrypto.deriveKey(
            masterKey: session.masterKey, purpose: "legacy-recovery")
        var filesToDelete: [URL] = []
        for name in names {
            let source = root.appendingPathComponent(name)
            guard FileManager.default.fileExists(atPath: source.path) else { continue }
            // **BEST EFFORT PER FILE.** One truncated legacy file (a restored
            // v1 backup is copied without validating its JSON) used to throw
            // out of every launch and unlock, locking the whole profile on a
            // file the error never named. A file that fails stays on disk,
            // is not deleted, and is reported; the vault still opens.
            do {
                if let key = Self.protectedRecordKey(for: source) {
                    if name == "chats_archive.json" {
                        _ = try loadChatArchive(legacyURL: source)
                    } else if name == "projects_archive.json" {
                        _ = try loadProjectArchive(legacyURL: source)
                    } else if try database().loadRecord(key: key) == nil {
                        try database().saveRecord(key: key, payload: Data(contentsOf: source))
                    }
                }
                let data = try Data(contentsOf: source)
                let sealed = try AES.GCM.seal(
                    data,
                    using: SymmetricKey(data: recoveryKey),
                    authenticating: Data(name.utf8))
                guard let combined = sealed.combined else {
                    throw ProfileVaultCrypto.CryptoError.malformedEnvelope
                }
                let destination = store.recoveryURL.appendingPathComponent("\(name).legacy.enc")
                try combined.write(to: destination, options: .atomic)
                try FileManager.default.setAttributes(
                    [.posixPermissions: 0o600], ofItemAtPath: destination.path)
                let verify = try AES.GCM.open(
                    AES.GCM.SealedBox(combined: Data(contentsOf: destination)),
                    using: SymmetricKey(data: recoveryKey),
                    authenticating: Data(name.utf8))
                guard verify == data else { throw ProfileVaultStore.VaultError.integrityCheckFailed }
                filesToDelete.append(source)
            } catch {
                NSLog(
                    "Legacy migration skipped %@ (left in place): %@",
                    name, String(describing: error))
                continue
            }
        }
        let memoryRoots = try migrateLegacyMemory()
        let observationRoots = try migrateLegacyToolObservations()
        // The open already ran a full integrity check. Repeating it here
        // re-verified every page on each launch and unlock even when nothing
        // was imported; only gate deletion of legacy sources on it.
        if Self.needsPostMigrationIntegrityCheck(
            importedFiles: filesToDelete.count, memoryRoots: memoryRoots.count,
            observationRoots: observationRoots.count)
        {
            guard try database().integrityCheck() else {
                throw ProfileVaultStore.VaultError.integrityCheckFailed
            }
        }
        try database().checkpoint()
        // Deletion is the final phase. If any import, authentication, or
        // integrity check above fails, every plaintext source remains.
        for source in filesToDelete {
            try FileManager.default.removeItem(at: source)
        }
        // The profile and projects memory roots overlap, so de-duplicate.
        let migratedRoots = Array(Set(memoryRoots + observationRoots))
        for root in migratedRoots.sorted(by: { $0.path.count > $1.path.count }) {
            if migratedRoots.contains(where: { root.path.hasPrefix($0.path + "/") }) { continue }
            try FileManager.default.removeItem(at: root)
        }
    }

    /// Imports every regular file under `root` as a raw record and returns what
    /// is safe to delete afterwards. The whole root is returned only when every
    /// entry on disk (including hidden files and symlinks, which the import
    /// skips) was imported and read back; otherwise only the imported files are
    /// returned so unimported data is never removed.
    private func importLegacyTree(
        root: URL, skipsHiddenFiles: Bool, keyFor: (String) -> String
    ) throws -> [URL] {
        let manager = FileManager.default
        guard manager.fileExists(atPath: root.path) else { return [] }
        // Path-based enumeration yields paths relative to `root`. URL
        // enumeration can return /private/var for a /var root, and slicing
        // that by the root's string length corrupts the record keys.
        guard let enumerator = manager.enumerator(atPath: root.path) else {
            throw ProfileVaultStore.VaultError.integrityCheckFailed
        }
        var everything = 0
        var imported: [URL] = []
        // `while let` over the enumerator only (no trailing boolean condition),
        // so a directory entry is skipped instead of ending the walk.
        while let relative = enumerator.nextObject() as? String {
            let file = root.appendingPathComponent(relative)
            let values = try file.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
            let isLink = (try? file.resourceValues(forKeys: [.isSymbolicLinkKey]))?.isSymbolicLink == true
            if values.isDirectory == true && !isLink { continue }
            // Everything that is not a plain directory must be accounted for
            // before the root may be deleted.
            everything += 1
            let hidden = relative.split(separator: "/").contains { $0.hasPrefix(".") }
            guard values.isRegularFile == true, !isLink, !(skipsHiddenFiles && hidden) else {
                continue
            }
            let key = keyFor(relative)
            let data = try Data(contentsOf: file)
            try saveRawRecord(data, key: key)
            guard try rawRecord(key: key) == data else {
                throw ProfileVaultStore.VaultError.integrityCheckFailed
            }
            imported.append(file)
        }
        return imported.count == everything ? [root] : imported
    }

    private func migrateLegacyMemory() throws -> [URL] {
        let roots: [(url: URL, prefix: String)] = [
            (ProfileMemoryStore.shared.directory, "profile"),
            (MemoryStore.defaultBase().appendingPathComponent("projects", isDirectory: true),
             "projects"),
        ]
        var removable: [URL] = []
        for (root, prefix) in roots {
            removable += try importLegacyTree(root: root, skipsHiddenFiles: false) {
                "memory:file:\(prefix)/\($0)"
            }
        }
        return removable
    }

    private func migrateLegacyToolObservations() throws -> [URL] {
        let root = store.rootURL.deletingLastPathComponent()
            .appendingPathComponent("tool-observations", isDirectory: true)
        return try importLegacyTree(root: root, skipsHiddenFiles: true) {
            ToolObservationStore.recordKey(relativePath: $0)
        }
    }

    private func database() throws -> ProfileDatabase {
        guard let session = store.session else { throw ProfileVaultStore.VaultError.locked }
        return session.database
    }
}

private extension NSRecursiveLock {
    func withLock<T>(_ body: () throws -> T) rethrows -> T {
        lock()
        defer { unlock() }
        return try body()
    }
}

extension ProfileRepository {
    func saveAudioRecord(key: String, payload: Data, references: [String]) throws {
        try database().saveAudioRecord(key: key, payload: payload, references: references)
    }

    func collectUnusedAudioAssets() throws {
        let unreachable = try database().collectUnusedAudioAssets()
        ManagedAssetStore(vault: store).garbageCollect(ids: unreachable)
    }

    func deleteAudioRecord(key: String) throws {
        let unreachable = try database().deleteAudioRecord(key: key)
        ManagedAssetStore(vault: store).garbageCollect(ids: unreachable)
    }

    /// The audio search index, or nil when its optional schema did not open
    /// on this profile (search then falls back to matching titles in memory).
    func audioIndex() throws -> ProfileDatabase? {
        let database = try database()
        return database.audioIndexAvailable ? database : nil
    }
}
