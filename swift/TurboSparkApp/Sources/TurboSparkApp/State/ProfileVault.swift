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

    /// Version 1 is the original layout. Version 2 only appears while a key
    /// rotation is pending (see `wrappedPreviousKey`); once the rotation
    /// completes the manifest is written back as version 1, so a crash-free
    /// profile stays readable by older builds.
    public static let baseFormatVersion = 1
    public static let currentFormatVersion = 2

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
    /// The previous master key wrapped under the same passphrase key, kept
    /// only while a rotation to `wrappedMasterKey` is unfinished so an
    /// interrupted rotation can resume.
    public var wrappedPreviousKey: Data?
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

    fileprivate func replaceMasterKey(_ replacement: Data) {
        masterKey.wipe()
        masterKey = replacement
    }
}

/// Admits key-using operations (managed asset reads and writes) and lets the
/// vault lock or a key rotation drain them first.
///
/// Lock order is always gate, then the store lock: an operation takes the
/// gate shared and only then snapshots the key under the store lock, while
/// lock/rotation take the gate exclusive before touching the store lock.
/// Nothing may enter the gate while holding the store lock from another
/// thread, or it would deadlock against an exclusive holder.
final class ProfileVaultOperationGate: @unchecked Sendable {
    private let condition = NSCondition()
    private var sharedCount = 0
    private var exclusiveActive = false
    private var exclusivePending = 0
    private var cancelPending = 0
    private var exclusiveOwner: ObjectIdentifier?
    private var exclusiveDepth = 0

    /// Per-thread shared depth so an operation that calls another gated
    /// method (a consumer closure that touches assets) cannot deadlock
    /// against a pending exclusive request.
    private var depthKey: String { "turbospark.vault.gate.\(ObjectIdentifier(self).hashValue)" }

    private var currentThreadID: ObjectIdentifier { ObjectIdentifier(Thread.current) }

    private var threadDepth: Int {
        get { Thread.current.threadDictionary[depthKey] as? Int ?? 0 }
        set {
            if newValue == 0 {
                Thread.current.threadDictionary.removeObject(forKey: depthKey)
            } else {
                Thread.current.threadDictionary[depthKey] = newValue
            }
        }
    }

    /// True while a lock is waiting for in-flight operations to finish.
    /// Long operations poll this between chunks and stop early.
    var isCancelRequested: Bool {
        condition.lock()
        defer { condition.unlock() }
        return cancelPending > 0
    }

    func enterShared() throws {
        if threadDepth > 0 {
            threadDepth += 1
            return
        }
        condition.lock()
        defer { condition.unlock() }
        if exclusiveActive && exclusiveOwner == currentThreadID {
            // The exclusive holder may use gated helpers on its own thread.
            sharedCount += 1
            threadDepth = 1
            return
        }
        while true {
            // A lock in progress closes the vault: fail now instead of
            // queueing behind it and then finding no session.
            if cancelPending > 0 { throw ProfileVaultStore.VaultError.locked }
            if !exclusiveActive && exclusivePending == 0 { break }
            condition.wait()
        }
        sharedCount += 1
        threadDepth = 1
    }

    func leaveShared() {
        let depth = threadDepth
        threadDepth = max(0, depth - 1)
        guard depth == 1 else { return }
        condition.lock()
        sharedCount -= 1
        condition.broadcast()
        condition.unlock()
    }

    func beginExclusive(cancelInFlight: Bool) {
        condition.lock()
        defer { condition.unlock() }
        if exclusiveActive && exclusiveOwner == currentThreadID {
            exclusiveDepth += 1
            return
        }
        exclusivePending += 1
        if cancelInFlight { cancelPending += 1 }
        // The caller's own shared entries (it is draining itself) would never
        // finish, so only other threads' operations are waited for.
        while sharedCount > (threadDepth > 0 ? 1 : 0) || exclusiveActive { condition.wait() }
        exclusivePending -= 1
        if cancelInFlight { cancelPending -= 1 }
        exclusiveActive = true
        exclusiveOwner = currentThreadID
        exclusiveDepth = 1
    }

    func endExclusive() {
        condition.lock()
        defer { condition.unlock() }
        exclusiveDepth -= 1
        guard exclusiveDepth == 0 else { return }
        exclusiveActive = false
        exclusiveOwner = nil
        condition.broadcast()
    }
}

/// A key snapshot taken under the vault lock for one asset operation. The
/// gate guarantees the vault cannot lock or rotate while it is alive, so the
/// copy is never a wiped or stale key.
final class ProfileVaultOperation {
    let profileID: String
    let masterKey: Data
    let database: ProfileDatabase
    private let gate: ProfileVaultOperationGate

    fileprivate init(
        profileID: String, masterKey: Data, database: ProfileDatabase,
        gate: ProfileVaultOperationGate
    ) {
        self.profileID = profileID
        self.masterKey = masterKey
        self.database = database
        self.gate = gate
    }

    var isCancelled: Bool { gate.isCancelRequested }

    /// Long loops call this between chunks so a lock request does not wait
    /// for a multi-gigabyte import to finish.
    func checkNotCancelled() throws {
        if isCancelled { throw ProfileVaultStore.VaultError.locked }
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
        case concurrentChange

        var errorDescription: String? {
            switch self {
            case .locked: return "This profile is locked."
            case .malformedManifest: return "The profile security manifest is malformed."
            case .unsupportedFormat(let version):
                return "This profile uses unsupported vault format \(version)."
            case .missingKey: return "The profile master key is missing."
            case .integrityCheckFailed: return "The encrypted profile database failed its integrity check."
            case .writeFailed(let message): return "Profile security settings could not be saved: \(message)"
            case .concurrentChange:
                return "The profile security settings changed while this was in progress. Try again."
            }
        }
    }

    /// Points where key rotation can be interrupted. Tests throw from here to
    /// simulate a crash; production leaves the hook nil.
    enum RotationStage: Equatable {
        case manifestWritten
        case databaseRekeyed
        case file(Int)
        case beforeFinalManifest
    }

    static let previousKeyContext = "TurboSpark profile previous key v1"

    private let lock = NSRecursiveLock()
    let gate = ProfileVaultOperationGate()
    private let rootProvider: @Sendable () -> URL
    private let profileIDProvider: @Sendable () -> String
    private let keychain: any ProfileVaultKeychainProtocol
    private let migrateLegacyData: Bool
    private var prepared = false
    private var manifestStorage: ProfileSecurityManifest?
    private var sessionStorage: ProfileVaultSession?

    /// Passphrase key derivation, injectable so tests can observe that it
    /// never runs under the vault lock.
    var passphraseKeyDeriver: @Sendable (String, Data, UInt32) throws -> Data = {
        try ProfileVaultCrypto.derivePassphraseKey(passphrase: $0, salt: $1, rounds: $2)
    }
    var rotationHook: (@Sendable (RotationStage) throws -> Void)?

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

    /// Runs `body` with a key snapshot that the vault cannot lock or rotate
    /// out from under. Throws `.locked` when no session is open or a lock is
    /// in progress.
    func withOperation<T>(_ body: (ProfileVaultOperation) throws -> T) throws -> T {
        try gate.enterShared()
        defer { gate.leaveShared() }
        let operation: ProfileVaultOperation = try lock.withLock {
            if !prepared { try? prepareLocked() }
            guard let session = sessionStorage else { throw VaultError.locked }
            return ProfileVaultOperation(
                profileID: session.profileID, masterKey: session.masterKey,
                database: session.database, gate: gate)
        }
        return try body(operation)
    }

    @discardableResult
    func prepareForLaunch() throws -> ProfileVaultSession? {
        try lock.withLock {
            try prepareLocked()
            return sessionStorage
        }
    }

    @discardableResult
    func unlock(passphrase: String) throws -> ProfileVaultSession {
        let manifest = try preparedManifest()
        guard manifest.protectionMode == .passphrase,
              let kdf = manifest.kdf,
              let envelope = manifest.wrappedMasterKey
        else { throw VaultError.missingKey }
        // PBKDF2 takes a few hundred milliseconds; running it under the
        // store lock froze every reader of `manifest` (the SwiftUI body).
        let wrappingKey = try passphraseKeyDeriver(passphrase, kdf.salt, kdf.rounds)
        let masterKey = try ProfileVaultCrypto.unwrapMasterKey(envelope, with: wrappingKey)
        let previousKey = try manifest.wrappedPreviousKey.map {
            try ProfileVaultCrypto.unwrapMasterKey(
                $0, with: wrappingKey, context: Self.previousKeyContext)
        }
        return try installSession(masterKey: masterKey, previousKey: previousKey)
    }

    @discardableResult
    func unlockWithSystemAuthentication(context: LAContext) throws -> ProfileVaultSession {
        let manifest = try preparedManifest()
        // An interrupted rotation needs the passphrase: the Keychain copy
        // may still hold the key the data is being moved away from.
        guard manifest.protectionMode == .passphrase,
              manifest.quickUnlockEnabled,
              manifest.wrappedPreviousKey == nil
        else { throw VaultError.locked }
        // The system authentication prompt can wait on the person for as
        // long as they like; never hold the store lock across it.
        let masterKey = try keychain.load(profileID: manifest.profileID, context: context)
        return try installSession(masterKey: masterKey, previousKey: nil)
    }

    func protect(passphrase: String, enableQuickUnlock: Bool) throws {
        try lock.withLock {
            if !prepared { try prepareLocked() }
            guard sessionStorage != nil else { throw VaultError.locked }
        }
        let salt = try ProfileVaultCrypto.randomBytes(count: ProfileVaultCrypto.saltByteCount)
        let wrappingKey = try passphraseKeyDeriver(
            passphrase, salt, ProfileVaultCrypto.pbkdf2Rounds)
        // Rotation rewrites the database and every asset, so it must not
        // overlap a running import or a lock.
        gate.beginExclusive(cancelInFlight: false)
        defer { gate.endExclusive() }
        try lock.withLock {
            guard let session = sessionStorage else { throw VaultError.locked }
            if migrateLegacyData {
                try ProfileRepository(store: self).migrateLegacyPrivateFiles()
            }

            var manifest = manifestStorage ?? newLocalManifest(masterKey: session.masterKey)
            let keyWasPlaintext = manifest.protectionMode == .local || manifest.localMasterKey != nil
            manifest.protectionMode = .passphrase
            manifest.kdf = .init(
                algorithm: "PBKDF2-HMAC-SHA256",
                rounds: ProfileVaultCrypto.pbkdf2Rounds,
                salt: salt)
            manifest.quickUnlockEnabled = false
            manifest.updatedAt = Date()
            // A stale Keychain item would keep a copy of the pre-rotation key.
            keychain.delete(profileID: manifest.profileID)

            if keyWasPlaintext {
                try rotateToWrappedKeyLocked(
                    session: session, manifest: manifest, wrappingKey: wrappingKey)
            } else {
                manifest.localMasterKey = nil
                manifest.wrappedMasterKey = try ProfileVaultCrypto.wrapMasterKey(
                    session.masterKey, with: wrappingKey)
                try writeManifestLocked(manifest)
            }

            guard enableQuickUnlock, var current = manifestStorage else { return }
            // The recovery passphrase is authoritative. Optional system
            // authentication can be unavailable under an ad-hoc
            // signature, so its failure must not roll back protection.
            do {
                try keychain.save(masterKey: session.masterKey, profileID: current.profileID)
                current.quickUnlockEnabled = true
                current.updatedAt = Date()
                try writeManifestLocked(current)
            } catch {
                keychain.delete(profileID: current.profileID)
            }
        }
    }

    func changePassphrase(current: String, replacement: String) throws {
        _ = try unlock(passphrase: current)
        let before = try lock.withLock { () -> ProfileSecurityManifest in
            guard sessionStorage != nil, let manifest = manifestStorage else {
                throw VaultError.locked
            }
            return manifest
        }
        let salt = try ProfileVaultCrypto.randomBytes(count: ProfileVaultCrypto.saltByteCount)
        let wrappingKey = try passphraseKeyDeriver(
            replacement, salt, ProfileVaultCrypto.pbkdf2Rounds)
        try lock.withLock {
            guard let session = sessionStorage, var manifest = manifestStorage else {
                throw VaultError.locked
            }
            guard manifest.updatedAt == before.updatedAt,
                  manifest.wrappedMasterKey == before.wrappedMasterKey
            else { throw VaultError.concurrentChange }
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
        let before = try lock.withLock { () -> ProfileSecurityManifest in
            guard sessionStorage != nil, let manifest = manifestStorage else {
                throw VaultError.locked
            }
            return manifest
        }
        guard let kdf = before.kdf, let envelope = before.wrappedMasterKey else {
            throw VaultError.missingKey
        }
        let wrappingKey = try passphraseKeyDeriver(passphrase, kdf.salt, kdf.rounds)
        var verified = try ProfileVaultCrypto.unwrapMasterKey(envelope, with: wrappingKey)
        defer { verified.wipe() }
        try lock.withLock {
            guard let session = sessionStorage, var manifest = manifestStorage else {
                throw VaultError.locked
            }
            guard manifest.updatedAt == before.updatedAt else { throw VaultError.concurrentChange }
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

    /// Closes the session. In-flight asset operations are asked to stop and
    /// awaited first, so the key is never wiped under an import that would
    /// then encrypt chunks with an empty key.
    func lockVault() {
        gate.beginExclusive(cancelInFlight: true)
        defer { gate.endExclusive() }
        lock.withLock {
            sessionStorage?.close()
            sessionStorage = nil
        }
    }

    func resetForTests() {
        gate.beginExclusive(cancelInFlight: true)
        defer { gate.endExclusive() }
        lock.withLock {
            sessionStorage?.close()
            sessionStorage = nil
            manifestStorage = nil
            prepared = false
        }
    }

    private func preparedManifest() throws -> ProfileSecurityManifest {
        try lock.withLock {
            try prepareLocked()
            guard let manifest = manifestStorage else { throw VaultError.malformedManifest }
            return manifest
        }
    }

    /// Installs the session once the key is known. Everything slow (PBKDF2,
    /// the Keychain prompt) already happened without the store lock; only
    /// the database open and integrity check remain under it.
    private func installSession(
        masterKey: Data, previousKey: Data?
    ) throws -> ProfileVaultSession {
        if previousKey != nil {
            gate.beginExclusive(cancelInFlight: false)
        }
        defer { if previousKey != nil { gate.endExclusive() } }
        return try lock.withLock {
            let session: ProfileVaultSession
            if let previousKey {
                session = try resumeRotationLocked(newKey: masterKey, previousKey: previousKey)
            } else {
                session = try openLocked(masterKey: masterKey)
            }
            if migrateLegacyData {
                try ProfileRepository(store: self).migrateLegacyPrivateFiles()
            }
            return session
        }
    }

    // MARK: Key rotation

    private func databaseKey(for masterKey: Data) -> Data {
        ProfileVaultCrypto.deriveKey(masterKey: masterKey, purpose: "database")
    }

    /// Moves a profile whose master key sat in plaintext (security.json) to a
    /// fresh key that exists only wrapped under the passphrase.
    ///
    /// Crash safety: the manifest carrying BOTH wrapped keys is durable before
    /// any data changes, and the plaintext key is dropped from it in the same
    /// write. Every later step is idempotent and `resumeRotationLocked`
    /// replays them from the passphrase alone, so a crash at any point leaves
    /// a profile that opens after the next unlock.
    private func rotateToWrappedKeyLocked(
        session: ProfileVaultSession,
        manifest base: ProfileSecurityManifest,
        wrappingKey: Data
    ) throws {
        var oldKey = session.masterKey
        var newKey = try ProfileVaultCrypto.randomBytes(count: ProfileVaultCrypto.masterKeyByteCount)
        defer {
            oldKey.wipe()
            newKey.wipe()
        }
        var pending = base
        pending.localMasterKey = nil
        pending.wrappedMasterKey = try ProfileVaultCrypto.wrapMasterKey(newKey, with: wrappingKey)
        pending.wrappedPreviousKey = try ProfileVaultCrypto.wrapMasterKey(
            oldKey, with: wrappingKey, context: Self.previousKeyContext)
        try writeManifestLocked(pending)

        do {
            try rotationHook?(.manifestWritten)
            try session.database.rekey(to: databaseKey(for: newKey))
            try rotationHook?(.databaseRekeyed)
            try rewrapFilesLocked(from: oldKey, to: newKey)
            try rotationHook?(.beforeFinalManifest)
            var final = pending
            final.wrappedPreviousKey = nil
            try writeManifestLocked(final)
        } catch {
            // The database handle may now be under either key. Close it so
            // the next unlock reopens it through the recovery path, which
            // accepts both.
            sessionStorage?.close()
            sessionStorage = nil
            throw error
        }
        session.replaceMasterKey(newKey)
    }

    /// Finishes a rotation recorded in the manifest. Opens the database with
    /// whichever key it is currently under, then replays the idempotent steps.
    private func resumeRotationLocked(
        newKey: Data, previousKey: Data
    ) throws -> ProfileVaultSession {
        if let existing = sessionStorage { return existing }
        var oldKey = previousKey
        defer { oldKey.wipe() }
        let database: ProfileDatabase
        var needsRekey = false
        do {
            database = try ProfileDatabase(url: databaseURL, key: databaseKey(for: newKey))
        } catch {
            database = try ProfileDatabase(url: databaseURL, key: databaseKey(for: previousKey))
            needsRekey = true
        }
        do {
            guard try database.integrityCheck() else { throw VaultError.integrityCheckFailed }
            if needsRekey { try database.rekey(to: databaseKey(for: newKey)) }
            try rotationHook?(.databaseRekeyed)
            try rewrapFilesLocked(from: previousKey, to: newKey)
            try rotationHook?(.beforeFinalManifest)
            guard var manifest = manifestStorage else { throw VaultError.malformedManifest }
            manifest.wrappedPreviousKey = nil
            manifest.updatedAt = Date()
            try writeManifestLocked(manifest)
        } catch {
            database.close()
            throw error
        }
        let session = ProfileVaultSession(
            profileID: profileIDProvider(), masterKey: newKey, database: database)
        sessionStorage = session
        return session
    }

    /// Re-encrypts managed assets and legacy recovery copies from `oldKey` to
    /// `newKey`. Each file is replaced atomically and files already under the
    /// new key are skipped, so this can be replayed after a crash.
    private func rewrapFilesLocked(from oldKey: Data, to newKey: Data) throws {
        var index = 0
        try ManagedAssetStore(vault: self).rewrapAssets(from: oldKey, to: newKey) {
            try rotationHook?(.file(index))
            index += 1
        }
        try rewrapRecoveryFiles(from: oldKey, to: newKey)
        ManagedAssetStore.purgeDecryptedCache(profileID: profileIDProvider())
    }

    private func rewrapRecoveryFiles(from oldKey: Data, to newKey: Data) throws {
        let names = (try? FileManager.default.contentsOfDirectory(
            atPath: recoveryURL.path)) ?? []
        let suffix = ".legacy.enc"
        let oldSymmetric = SymmetricKey(
            data: ProfileVaultCrypto.deriveKey(masterKey: oldKey, purpose: "legacy-recovery"))
        let newSymmetric = SymmetricKey(
            data: ProfileVaultCrypto.deriveKey(masterKey: newKey, purpose: "legacy-recovery"))
        for fileName in names.sorted() where fileName.hasSuffix(suffix) {
            let logicalName = Data(fileName.dropLast(suffix.count).utf8)
            let url = recoveryURL.appendingPathComponent(fileName)
            let data = try Data(contentsOf: url)
            let box = try AES.GCM.SealedBox(combined: data)
            if (try? AES.GCM.open(box, using: newSymmetric, authenticating: logicalName)) != nil {
                continue
            }
            guard let plain = try? AES.GCM.open(
                box, using: oldSymmetric, authenticating: logicalName)
            else {
                NSLog("Recovery copy %@ opens under neither key; left unchanged", fileName)
                continue
            }
            let sealed = try AES.GCM.seal(plain, using: newSymmetric, authenticating: logicalName)
            guard let combined = sealed.combined else {
                throw ProfileVaultCrypto.CryptoError.malformedEnvelope
            }
            try combined.write(to: url, options: .atomic)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: url.path)
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
            // Version 1 predates key rotation (no wrappedPreviousKey) and is
            // still read as is; version 2 only appears while a rotation is
            // pending, so older builds refuse it rather than misread it.
            guard manifest.formatVersion == ProfileSecurityManifest.baseFormatVersion
                || manifest.formatVersion == ProfileSecurityManifest.currentFormatVersion
            else {
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
            formatVersion: ProfileSecurityManifest.baseFormatVersion,
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
        var manifest = manifest
        manifest.formatVersion = manifest.wrappedPreviousKey == nil
            ? ProfileSecurityManifest.baseFormatVersion
            : ProfileSecurityManifest.currentFormatVersion
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        do {
            try encoder.encode(manifest).write(to: manifestURL, options: .atomic)
            try FileManager.default.setAttributes(
                [.posixPermissions: 0o600], ofItemAtPath: manifestURL.path)
            // The rotation protocol relies on this write being on disk before
            // any data is re-keyed.
            if let handle = try? FileHandle(forWritingTo: manifestURL) {
                try? handle.synchronize()
                try? handle.close()
            }
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
