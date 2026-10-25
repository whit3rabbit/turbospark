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
    /// Bumped by every lock. An unlock snapshots it together with the
    /// manifest and `installSession` refuses to open a session unless it
    /// still matches, so an authentication that was still running (PBKDF2,
    /// the Keychain prompt) when `lockVault` returned stays stale.
    private var lockGeneration = 0

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
        let (manifest, generation) = try preparedManifestAndGeneration()
        guard manifest.protectionMode == .passphrase,
              let kdf = manifest.kdf,
              let envelope = manifest.wrappedMasterKey
        else { throw VaultError.missingKey }
        // PBKDF2 takes a few hundred milliseconds; running it under the
        // store lock froze every reader of `manifest` (the SwiftUI body).
        var wrappingKey = try passphraseKeyDeriver(passphrase, kdf.salt, kdf.rounds)
        defer { wrappingKey.wipe() }
        var masterKey = try ProfileVaultCrypto.unwrapMasterKey(envelope, with: wrappingKey)
        var previousKey = try manifest.wrappedPreviousKey.map {
            try ProfileVaultCrypto.unwrapMasterKey(
                $0, with: wrappingKey, context: Self.previousKeyContext)
        }
        do {
            return try installSession(
                masterKey: masterKey, previousKey: previousKey, generation: generation)
        } catch {
            // No session was built, so nothing else holds these copies. A
            // stale unlock (the vault locked while we derived) lands here.
            masterKey.wipe()
            previousKey?.wipe()
            throw error
        }
    }

    @discardableResult
    func unlockWithSystemAuthentication(context: LAContext) throws -> ProfileVaultSession {
        let (manifest, generation) = try preparedManifestAndGeneration()
        // An interrupted rotation needs the passphrase: the Keychain copy
        // may still hold the key the data is being moved away from.
        guard manifest.protectionMode == .passphrase,
              manifest.quickUnlockEnabled,
              manifest.wrappedPreviousKey == nil
        else { throw VaultError.locked }
        // The system authentication prompt can wait on the person for as
        // long as they like; never hold the store lock across it.
        var masterKey = try keychain.load(profileID: manifest.profileID, context: context)
        do {
            return try installSession(
                masterKey: masterKey, previousKey: nil, generation: generation)
        } catch {
            masterKey.wipe()
            throw error
        }
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
            // Invalidate every unlock that captured the older generation:
            // its authentication may still be running outside the gate and
            // must not install a session after this lock has returned.
            lockGeneration &+= 1
            sessionStorage?.close()
            sessionStorage = nil
        }
    }

    func resetForTests() {
        gate.beginExclusive(cancelInFlight: true)
        defer { gate.endExclusive() }
        lock.withLock {
            lockGeneration &+= 1
            sessionStorage?.close()
            sessionStorage = nil
            manifestStorage = nil
            prepared = false
        }
    }

    /// Snapshots the manifest and the lock generation under one acquisition
    /// of the store lock. The generation captured here is what
    /// `installSession` later requires to still hold.
    private func preparedManifestAndGeneration()
        throws -> (manifest: ProfileSecurityManifest, generation: Int)
    {
        try lock.withLock {
            () -> (manifest: ProfileSecurityManifest, generation: Int) in
            try prepareLocked()
            guard let manifest = manifestStorage else { throw VaultError.malformedManifest }
            return (manifest, lockGeneration)
        }
    }

    /// Installs the session once the key is known. Everything slow (PBKDF2,
    /// the Keychain prompt) already happened without the store lock; only
    /// the database open and integrity check remain under it.
    ///
    /// `generation` must still be the current lock generation. Authentication
    /// runs outside both the operation gate and the store lock, so
    /// `lockVault` can complete in between; if it did, this unlock is stale
    /// and must not reopen the vault. Callers wipe their key buffers when
    /// this throws.
    private func installSession(
        masterKey: Data, previousKey: Data?, generation: Int
    ) throws -> ProfileVaultSession {
        if previousKey != nil {
            gate.beginExclusive(cancelInFlight: false)
        }
        defer { if previousKey != nil { gate.endExclusive() } }
        return try lock.withLock {
            guard lockGeneration == generation else { throw VaultError.locked }
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

