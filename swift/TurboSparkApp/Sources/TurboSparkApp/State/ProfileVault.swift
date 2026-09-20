import CryptoKit
import Foundation
import LocalAuthentication

public struct ProfileSecurityManifest: Codable, Equatable, Sendable {
    public enum ProtectionMode: String, Codable, Sendable {
        case local
        case passphrase
    }

    public struct KDF: Codable, Equatable, Sendable {
        public var algorithm: String
        public var rounds: UInt32
        public var salt: Data
    }

    public static let currentFormatVersion = 1

    public var formatVersion: Int
    public var profileID: String
    public var publicLabel: String
    public var protectionMode: ProtectionMode
    public var kdf: KDF?
    public var localMasterKey: Data?
    public var wrappedMasterKey: Data?
    public var quickUnlockEnabled: Bool
    public var createdAt: Date
    public var updatedAt: Date
}

public final class ProfileVaultSession: @unchecked Sendable {
    public let profileID: String
    var masterKey: Data
    let database: ProfileDatabase

    fileprivate init(profileID: String, masterKey: Data, database: ProfileDatabase) {
        self.profileID = profileID
        self.masterKey = masterKey
        self.database = database
    }

    fileprivate func close() {
        database.close()
        masterKey.wipe()
    }
}

final class ProfileVaultStore: @unchecked Sendable {
    static let shared = ProfileVaultStore()

    enum VaultError: Error, LocalizedError, Equatable {
        case locked
        case malformedManifest
        case unsupportedFormat(Int)
        case missingKey
        case integrityCheckFailed
        case writeFailed(String)

        var errorDescription: String? {
            switch self {
            case .locked: return "This profile is locked."
            case .malformedManifest: return "The profile security manifest is malformed."
            case .unsupportedFormat(let version):
                return "This profile uses unsupported vault format \(version)."
            case .missingKey: return "The profile master key is missing."
            case .integrityCheckFailed: return "The encrypted profile database failed its integrity check."
            case .writeFailed(let message): return "Profile security settings could not be saved: \(message)"
            }
        }
    }

    private let lock = NSRecursiveLock()
    private let rootProvider: @Sendable () -> URL
    private let profileIDProvider: @Sendable () -> String
    private let keychain: any ProfileVaultKeychainProtocol
    private let migrateLegacyData: Bool
    private var prepared = false
    private var manifestStorage: ProfileSecurityManifest?
    private var sessionStorage: ProfileVaultSession?

    init(
        rootProvider: @escaping @Sendable () -> URL = {
            AppStorageRoot.directory.appendingPathComponent("private-vault", isDirectory: true)
        },
        profileIDProvider: @escaping @Sendable () -> String = { UserProfileStore.active.id },
        keychain: any ProfileVaultKeychainProtocol = ProfileVaultKeychain(),
        migrateLegacyData: Bool = true
    ) {
        self.rootProvider = rootProvider
        self.profileIDProvider = profileIDProvider
        self.keychain = keychain
        self.migrateLegacyData = migrateLegacyData
    }

    var rootURL: URL { rootProvider() }
    var databaseURL: URL { rootURL.appendingPathComponent("profile.sqlite3") }
    var assetsURL: URL { rootURL.appendingPathComponent("assets", isDirectory: true) }
    var recoveryURL: URL { rootURL.appendingPathComponent("recovery", isDirectory: true) }
    private var manifestURL: URL { rootURL.appendingPathComponent("security.json") }

    var manifest: ProfileSecurityManifest? {
        lock.withLock { manifestStorage }
    }

    var session: ProfileVaultSession? {
        lock.withLock {
            if !prepared { try? prepareLocked() }
            return sessionStorage
        }
    }

    var isProtected: Bool {
        lock.withLock {
            if !prepared { try? prepareLocked() }
            return manifestStorage?.protectionMode == .passphrase
        }
    }

    var isUnlocked: Bool { session != nil }
    var canUseQuickUnlock: Bool { keychain.canUseSystemAuthentication() }

    @discardableResult
    func prepareForLaunch() throws -> ProfileVaultSession? {
        try lock.withLock {
            try prepareLocked()
            return sessionStorage
        }
    }

    @discardableResult
    func unlock(passphrase: String) throws -> ProfileVaultSession {
        try lock.withLock {
            try prepareLocked()
            guard let manifest = manifestStorage else { throw VaultError.malformedManifest }
            guard manifest.protectionMode == .passphrase,
                  let kdf = manifest.kdf,
                  let envelope = manifest.wrappedMasterKey
            else { throw VaultError.missingKey }
            let wrappingKey = try ProfileVaultCrypto.derivePassphraseKey(
                passphrase: passphrase, salt: kdf.salt, rounds: kdf.rounds)
            let masterKey = try ProfileVaultCrypto.unwrapMasterKey(envelope, with: wrappingKey)
            let session = try openLocked(masterKey: masterKey)
            if migrateLegacyData {
                try ProfileRepository(store: self).migrateLegacyPrivateFiles()
            }
            return session
        }
    }

    @discardableResult
    func unlockWithSystemAuthentication(context: LAContext) throws -> ProfileVaultSession {
        try lock.withLock {
            try prepareLocked()
            guard let manifest = manifestStorage,
                  manifest.protectionMode == .passphrase,
                  manifest.quickUnlockEnabled
            else { throw VaultError.locked }
            let masterKey = try keychain.load(profileID: manifest.profileID, context: context)
            return try openLocked(masterKey: masterKey)
        }
    }

    func protect(passphrase: String, enableQuickUnlock: Bool) throws {
        try lock.withLock {
            try prepareLocked()
            guard let session = sessionStorage else { throw VaultError.locked }
            let salt = try ProfileVaultCrypto.randomBytes(count: ProfileVaultCrypto.saltByteCount)
            let wrappingKey = try ProfileVaultCrypto.derivePassphraseKey(
                passphrase: passphrase, salt: salt)
            let envelope = try ProfileVaultCrypto.wrapMasterKey(
                session.masterKey, with: wrappingKey)

            if migrateLegacyData {
                try ProfileRepository(store: self).migrateLegacyPrivateFiles()
            }

            var manifest = manifestStorage ?? newLocalManifest(masterKey: session.masterKey)
            manifest.protectionMode = .passphrase
            manifest.kdf = .init(
                algorithm: "PBKDF2-HMAC-SHA256",
                rounds: ProfileVaultCrypto.pbkdf2Rounds,
                salt: salt)
            manifest.localMasterKey = nil
            manifest.wrappedMasterKey = envelope
            manifest.quickUnlockEnabled = false
            manifest.updatedAt = Date()

            try writeManifestLocked(manifest)
            if enableQuickUnlock {
                // The recovery passphrase is authoritative. Optional system
                // authentication can be unavailable under an ad-hoc
                // signature, so its failure must not roll back protection.
                do {
                    try keychain.save(masterKey: session.masterKey, profileID: manifest.profileID)
                    manifest.quickUnlockEnabled = true
                    manifest.updatedAt = Date()
                    try writeManifestLocked(manifest)
                } catch {
                    keychain.delete(profileID: manifest.profileID)
                }
            }
        }
    }

    func changePassphrase(current: String, replacement: String) throws {
        _ = try unlock(passphrase: current)
        try lock.withLock {
            guard let session = sessionStorage, var manifest = manifestStorage else {
                throw VaultError.locked
            }
            let salt = try ProfileVaultCrypto.randomBytes(count: ProfileVaultCrypto.saltByteCount)
            let wrappingKey = try ProfileVaultCrypto.derivePassphraseKey(
                passphrase: replacement, salt: salt)
            manifest.kdf = .init(
                algorithm: "PBKDF2-HMAC-SHA256",
                rounds: ProfileVaultCrypto.pbkdf2Rounds,
                salt: salt)
            manifest.wrappedMasterKey = try ProfileVaultCrypto.wrapMasterKey(
                session.masterKey, with: wrappingKey)
            manifest.updatedAt = Date()
            try writeManifestLocked(manifest)
        }
    }

    func setQuickUnlock(enabled: Bool) throws {
        try lock.withLock {
            guard let session = sessionStorage, var manifest = manifestStorage,
                  manifest.protectionMode == .passphrase
            else { throw VaultError.locked }
            if enabled {
                try keychain.save(masterKey: session.masterKey, profileID: manifest.profileID)
            } else {
                keychain.delete(profileID: manifest.profileID)
            }
            manifest.quickUnlockEnabled = enabled
            manifest.updatedAt = Date()
            try writeManifestLocked(manifest)
        }
    }

    func disableProtection(passphrase: String) throws {
        try lock.withLock {
            guard let session = sessionStorage, var manifest = manifestStorage else {
                throw VaultError.locked
            }
            guard let kdf = manifest.kdf, let envelope = manifest.wrappedMasterKey else {
                throw VaultError.missingKey
            }
            let wrappingKey = try ProfileVaultCrypto.derivePassphraseKey(
                passphrase: passphrase, salt: kdf.salt, rounds: kdf.rounds)
            let verified = try ProfileVaultCrypto.unwrapMasterKey(envelope, with: wrappingKey)
            guard verified == session.masterKey else {
                throw ProfileVaultCrypto.CryptoError.authenticationFailed
            }
            keychain.delete(profileID: manifest.profileID)
            manifest.protectionMode = .local
            manifest.kdf = nil
            manifest.localMasterKey = session.masterKey
            manifest.wrappedMasterKey = nil
            manifest.quickUnlockEnabled = false
            manifest.updatedAt = Date()
            try writeManifestLocked(manifest)
        }
    }

    func lockVault() {
        lock.withLock {
            sessionStorage?.close()
            sessionStorage = nil
        }
    }

    func resetForTests() {
        lock.withLock {
            sessionStorage?.close()
            sessionStorage = nil
            manifestStorage = nil
            prepared = false
        }
    }

    private func prepareLocked() throws {
        guard !prepared else { return }
        ManagedAssetStore.purgeDecryptedCache(profileID: profileIDProvider())
        try createPrivateDirectory(rootURL)
        try createPrivateDirectory(assetsURL)
        try createPrivateDirectory(recoveryURL)

        if FileManager.default.fileExists(atPath: manifestURL.path) {
            let data = try Data(contentsOf: manifestURL)
            guard let manifest = try? JSONDecoder().decode(ProfileSecurityManifest.self, from: data)
            else { throw VaultError.malformedManifest }
            guard manifest.formatVersion == ProfileSecurityManifest.currentFormatVersion else {
                throw VaultError.unsupportedFormat(manifest.formatVersion)
            }
            manifestStorage = manifest
        } else {
            let masterKey = try ProfileVaultCrypto.randomBytes(
                count: ProfileVaultCrypto.masterKeyByteCount)
            let manifest = newLocalManifest(masterKey: masterKey)
            try writeManifestLocked(manifest)
        }
        prepared = true

        if manifestStorage?.protectionMode == .local {
            guard let key = manifestStorage?.localMasterKey else { throw VaultError.missingKey }
            _ = try openLocked(masterKey: key)
            if migrateLegacyData {
                try ProfileRepository(store: self).migrateLegacyPrivateFiles()
            }
        }
    }

    private func openLocked(masterKey: Data) throws -> ProfileVaultSession {
        if let existing = sessionStorage { return existing }
        let databaseKey = ProfileVaultCrypto.deriveKey(masterKey: masterKey, purpose: "database")
        let database = try ProfileDatabase(url: databaseURL, key: databaseKey)
        guard try database.integrityCheck() else {
            database.close()
            throw VaultError.integrityCheckFailed
        }
        let session = ProfileVaultSession(
            profileID: profileIDProvider(), masterKey: masterKey, database: database)
        sessionStorage = session
        return session
    }

    private func newLocalManifest(masterKey: Data) -> ProfileSecurityManifest {
        let now = Date()
        return ProfileSecurityManifest(
            formatVersion: ProfileSecurityManifest.currentFormatVersion,
            profileID: profileIDProvider(),
            publicLabel: "Protected Profile",
            protectionMode: .local,
            kdf: nil,
            localMasterKey: masterKey,
            wrappedMasterKey: nil,
            quickUnlockEnabled: false,
            createdAt: now,
            updatedAt: now)
    }

    private func writeManifestLocked(_ manifest: ProfileSecurityManifest) throws {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: manifestURL.path)
            manifestStorage = manifest
        } catch {
            throw VaultError.writeFailed(error.localizedDescription)
        }
    }

    private func createPrivateDirectory(_ url: URL) throws {
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: url.path)
    }
}

public final class ProfileRepository: @unchecked Sendable {
    public static let shared = ProfileRepository()

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
            return try decoder.decode(type, from: data)
        }
        guard let legacyURL,
              FileManager.default.fileExists(atPath: legacyURL.path)
        else { return nil }
        let data = try Data(contentsOf: legacyURL)
        let value = try decoder.decode(type, from: data)
        try database.saveRecord(key: key, payload: data)
        return value
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
        }
        let memoryRoots = try migrateLegacyMemory()
        let observationRoots = try migrateLegacyToolObservations()
        guard try database().integrityCheck() else {
            throw ProfileVaultStore.VaultError.integrityCheckFailed
        }
        try database().checkpoint()
        // Deletion is the final phase. If any import, authentication, or
        // integrity check above fails, every plaintext source remains.
        for source in filesToDelete {
            try FileManager.default.removeItem(at: source)
        }
        let migratedRoots = memoryRoots + observationRoots
        for root in migratedRoots.sorted(by: { $0.path.count > $1.path.count }) {
            if migratedRoots.contains(where: { root.path.hasPrefix($0.path + "/") }) { continue }
            try FileManager.default.removeItem(at: root)
        }
    }

    private func migrateLegacyMemory() throws -> [URL] {
        let manager = FileManager.default
        let roots: [(url: URL, prefix: String)] = [
            (ProfileMemoryStore.shared.directory, "profile"),
            (MemoryStore.defaultBase().appendingPathComponent("projects", isDirectory: true),
             "projects"),
        ]
        var migratedRoots: [URL] = []
        for (root, prefix) in roots {
            guard manager.fileExists(atPath: root.path) else { continue }
            let enumerator = manager.enumerator(
                at: root, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsPackageDescendants])
            while let file = enumerator?.nextObject() as? URL,
                  (try file.resourceValues(forKeys: [.isRegularFileKey])).isRegularFile == true {
                let relative = String(file.path.dropFirst(root.path.count))
                    .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
                let key = "memory:file:\(prefix)/\(relative)"
                let data = try Data(contentsOf: file)
                try saveRawRecord(data, key: key)
                guard try rawRecord(key: key) == data else {
                    throw ProfileVaultStore.VaultError.integrityCheckFailed
                }
            }
            migratedRoots.append(root)
        }
        return migratedRoots
    }

    private func migrateLegacyToolObservations() throws -> [URL] {
        let manager = FileManager.default
        let root = store.rootURL.deletingLastPathComponent()
            .appendingPathComponent("tool-observations", isDirectory: true)
        guard manager.fileExists(atPath: root.path) else { return [] }
        let enumerator = manager.enumerator(
            at: root, includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants])
        while let file = enumerator?.nextObject() as? URL,
              (try file.resourceValues(forKeys: [.isRegularFileKey])).isRegularFile == true {
            let relative = String(file.path.dropFirst(root.path.count))
                .trimmingCharacters(in: CharacterSet(charactersIn: "/"))
            let key = ToolObservationStore.recordKey(relativePath: relative)
            let data = try Data(contentsOf: file)
            try saveRawRecord(data, key: key)
            guard try rawRecord(key: key) == data else {
                throw ProfileVaultStore.VaultError.integrityCheckFailed
            }
        }
        return [root]
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
