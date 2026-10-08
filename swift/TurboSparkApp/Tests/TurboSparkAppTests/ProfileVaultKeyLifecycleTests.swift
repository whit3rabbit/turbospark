import CryptoKit
import Foundation
import LocalAuthentication
@testable import TurboSparkApp
import XCTest

/// Key rotation on enabling protection, the operation gate that keeps asset
/// work and vault locking from racing, and the consistent export snapshot.
/// Everything is driven through injected fakes (key deriver, rotation hook,
/// chunk observer); nothing sleeps.
private final class RecordingKeychain: ProfileVaultKeychainProtocol, @unchecked Sendable {
    var stored: [String: Data] = [:]
    var onLoad: (() -> Void)?

    func save(masterKey: Data, profileID: String) throws { stored[profileID] = masterKey }

    func load(profileID: String, context _: LAContext) throws -> Data {
        onLoad?()
        guard let key = stored[profileID] else { throw NSError(domain: "test", code: 1) }
        return key
    }

    func delete(profileID: String) { stored.removeValue(forKey: profileID) }
    func canUseSystemAuthentication() -> Bool { true }
}

final class ProfileVaultKeyLifecycleTests: XCTestCase {
    private var root: URL!
    private let passphrase = "correct horse battery"

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProfileVaultKeyLifecycle-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    // MARK: Fixtures

    /// One PBKDF2 round keeps the suite fast; the lock and rotation logic is
    /// independent of the round count.
    private static let fastDeriver: @Sendable (String, Data, UInt32) throws -> Data = { passphrase, salt, _ in
        try ProfileVaultCrypto.derivePassphraseKey(passphrase: passphrase, salt: salt, rounds: 1)
    }

    private func makeStore(
        _ id: String,
        keychain: (any ProfileVaultKeychainProtocol)? = nil,
        fast: Bool = true
    ) -> ProfileVaultStore {
        let vault = root.appendingPathComponent(id, isDirectory: true)
        let store = ProfileVaultStore(
            rootProvider: { vault },
            profileIDProvider: { id },
            keychain: keychain ?? RecordingKeychain(),
            migrateLegacyData: false)
        if fast { store.passphraseKeyDeriver = Self.fastDeriver }
        return store
    }

    private func manifestURL(_ id: String) -> URL {
        root.appendingPathComponent("\(id)/security.json")
    }

    private func diskManifest(_ id: String) throws -> ProfileSecurityManifest {
        try JSONDecoder().decode(
            ProfileSecurityManifest.self, from: Data(contentsOf: manifestURL(id)))
    }

    private func recoveryURL(_ id: String, _ name: String) -> URL {
        root.appendingPathComponent("\(id)/recovery/\(name).legacy.enc")
    }

    private struct Seeded {
        var oldKey: Data
        var assets: [(descriptor: ManagedAssetDescriptor, bytes: Data)]
    }

    /// A local-mode profile with a record, assets (one multi-chunk), and a
    /// legacy recovery copy sealed under the plaintext key.
    private func seedLocalProfile(_ id: String, store: ProfileVaultStore) throws -> Seeded {
        _ = try store.prepareForLaunch()
        let oldKey = try XCTUnwrap(store.manifest?.localMasterKey)
        try ProfileRepository(store: store).save("private value", key: "value")
        let assetStore = ManagedAssetStore(vault: store)
        var assets: [(ManagedAssetDescriptor, Data)] = []
        for (index, size) in [100, 1_048_576 + 7, 5].enumerated() {
            let bytes = Data((0..<size).map { UInt8(truncatingIfNeeded: $0 &+ index) })
            let descriptor = try assetStore.store(
                data: bytes, fileName: "asset\(index).bin", mimeType: "application/octet-stream")
            assets.append((descriptor, bytes))
        }
        let recoveryKey = SymmetricKey(data: ProfileVaultCrypto.deriveKey(
            masterKey: oldKey, purpose: "legacy-recovery"))
        let sealed = try AES.GCM.seal(
            Data("legacy settings".utf8), using: recoveryKey,
            authenticating: Data("settings.json".utf8))
        try XCTUnwrap(sealed.combined).write(to: recoveryURL(id, "settings.json"))
        return Seeded(oldKey: oldKey, assets: assets)
    }

    private func assertFullyReadable(
        _ store: ProfileVaultStore, seeded: Seeded, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        XCTAssertEqual(
            try ProfileRepository(store: store).load(String.self, key: "value"), "private value",
            file: file, line: line)
        let assetStore = ManagedAssetStore(vault: store)
        for (descriptor, bytes) in seeded.assets {
            var opened = Data()
            try assetStore.streamDecrypted(reference: descriptor.storedReference) {
                opened.append($0)
            }
            XCTAssertEqual(opened, bytes, file: file, line: line)
        }
    }

    /// The old key must no longer open anything: that is the point of rotating.
    private func assertOldKeyIsUseless(
        _ id: String, seeded: Seeded, store: ProfileVaultStore,
        file: StaticString = #filePath, line: UInt = #line
    ) throws {
        XCTAssertThrowsError(try ProfileDatabase(
            url: root.appendingPathComponent("\(id)/profile.sqlite3"),
            key: ProfileVaultCrypto.deriveKey(masterKey: seeded.oldKey, purpose: "database")),
            file: file, line: line)
        let assetStore = ManagedAssetStore(vault: store)
        for (descriptor, _) in seeded.assets {
            let url = root.appendingPathComponent(
                "\(id)/assets/\(descriptor.id.prefix(2))/\(descriptor.id).tsasset")
            XCTAssertThrowsError(try assetStore.authenticate(
                fileAt: url, id: descriptor.id, masterKey: seeded.oldKey), file: file, line: line)
        }
        let box = try AES.GCM.SealedBox(combined: Data(contentsOf: recoveryURL(id, "settings.json")))
        let oldRecovery = SymmetricKey(data: ProfileVaultCrypto.deriveKey(
            masterKey: seeded.oldKey, purpose: "legacy-recovery"))
        XCTAssertThrowsError(try AES.GCM.open(
            box, using: oldRecovery, authenticating: Data("settings.json".utf8)),
            file: file, line: line)
    }

    private func assertPlaintextKeyGone(
        _ id: String, oldKey: Data, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        let text = try String(contentsOf: manifestURL(id), encoding: .utf8)
        XCTAssertFalse(text.contains(oldKey.base64EncodedString()), file: file, line: line)
        XCTAssertNil(try diskManifest(id).localMasterKey, file: file, line: line)
    }

    // MARK: Item 1: rotation on enabling protection

    func testEnablingProtectionRotatesAwayThePlaintextKey() throws {
        let store = makeStore("rotate")
        let seeded = try seedLocalProfile("rotate", store: store)
        XCTAssertEqual(try diskManifest("rotate").localMasterKey, seeded.oldKey)

        try store.protect(passphrase: passphrase, enableQuickUnlock: false)

        try assertPlaintextKeyGone("rotate", oldKey: seeded.oldKey)
        let manifest = try diskManifest("rotate")
        XCTAssertNil(manifest.wrappedPreviousKey)
        XCTAssertEqual(manifest.formatVersion, ProfileSecurityManifest.baseFormatVersion)
        XCTAssertNotEqual(try XCTUnwrap(store.session).masterKey, seeded.oldKey)
        try assertFullyReadable(store, seeded: seeded)

        store.lockVault()
        let reopened = makeStore("rotate")
        _ = try reopened.prepareForLaunch()
        _ = try reopened.unlock(passphrase: passphrase)
        try assertFullyReadable(reopened, seeded: seeded)
        try assertOldKeyIsUseless("rotate", seeded: seeded, store: reopened)
        // Dedup after rotation still works with the new key.
        let again = try ManagedAssetStore(vault: reopened).store(
            data: seeded.assets[0].bytes, fileName: "again.bin")
        XCTAssertNotNil(try ManagedAssetStore(vault: reopened).descriptor(
            for: again.storedReference))
    }

    func testProtectingKeepsAnExistingKeychainItemFromHoldingTheOldKey() throws {
        let keychain = RecordingKeychain()
        let store = makeStore("keychain", keychain: keychain)
        let seeded = try seedLocalProfile("keychain", store: store)
        keychain.stored["keychain"] = seeded.oldKey
        try store.protect(passphrase: passphrase, enableQuickUnlock: true)
        XCTAssertNotEqual(keychain.stored["keychain"], seeded.oldKey)
        XCTAssertEqual(keychain.stored["keychain"], try XCTUnwrap(store.session).masterKey)
        store.lockVault()
        _ = try store.unlockWithSystemAuthentication(context: LAContext())
        try assertFullyReadable(store, seeded: seeded)
    }

    func testInterruptedRotationLeavesAProfileThatOpensAtEveryStage() throws {
        let stages: [(String, ProfileVaultStore.RotationStage)] = [
            ("manifest", .manifestWritten),
            ("rekeyed", .databaseRekeyed),
            ("file0", .file(0)),
            ("file2", .file(2)),
            ("final", .beforeFinalManifest),
        ]
        struct Crash: Error {}
        for (name, stage) in stages {
            let id = "crash-\(name)"
            let store = makeStore(id)
            let seeded = try seedLocalProfile(id, store: store)
            store.rotationHook = { if $0 == stage { throw Crash() } }

            XCTAssertThrowsError(
                try store.protect(passphrase: passphrase, enableQuickUnlock: false), id)
            // The failed protect closed the session: nothing readable without
            // the passphrase, and the plaintext key is already gone.
            XCTAssertFalse(store.isUnlocked, id)
            try assertPlaintextKeyGone(id, oldKey: seeded.oldKey)
            let pending = try diskManifest(id)
            XCTAssertNotNil(pending.wrappedPreviousKey, id)
            XCTAssertEqual(pending.formatVersion, ProfileSecurityManifest.currentFormatVersion, id)

            // "Relaunch": a fresh store object over the same directory.
            let relaunched = makeStore(id)
            XCTAssertNil(try relaunched.prepareForLaunch(), id)
            XCTAssertThrowsError(try relaunched.unlock(passphrase: "wrong passphrase!!"), id)
            _ = try relaunched.unlock(passphrase: passphrase)
            try assertFullyReadable(relaunched, seeded: seeded)
            try assertOldKeyIsUseless(id, seeded: seeded, store: relaunched)
            let finished = try diskManifest(id)
            XCTAssertNil(finished.wrappedPreviousKey, id)
            XCTAssertEqual(finished.formatVersion, ProfileSecurityManifest.baseFormatVersion, id)

            relaunched.lockVault()
            let third = makeStore(id)
            _ = try third.prepareForLaunch()
            _ = try third.unlock(passphrase: passphrase)
            try assertFullyReadable(third, seeded: seeded)
        }
    }

    func testResumeThatFailsAgainStaysResumable() throws {
        struct Crash: Error {}
        let store = makeStore("twice")
        let seeded = try seedLocalProfile("twice", store: store)
        store.rotationHook = { if $0 == .file(1) { throw Crash() } }
        XCTAssertThrowsError(try store.protect(passphrase: passphrase, enableQuickUnlock: false))

        let second = makeStore("twice")
        second.rotationHook = { if $0 == .beforeFinalManifest { throw Crash() } }
        XCTAssertNil(try second.prepareForLaunch())
        XCTAssertThrowsError(try second.unlock(passphrase: passphrase))
        XCTAssertFalse(second.isUnlocked)
        XCTAssertNotNil(try diskManifest("twice").wrappedPreviousKey)

        let third = makeStore("twice")
        _ = try third.prepareForLaunch()
        _ = try third.unlock(passphrase: passphrase)
        try assertFullyReadable(third, seeded: seeded)
        try assertOldKeyIsUseless("twice", seeded: seeded, store: third)
    }

    func testQuickUnlockIsRefusedWhileARotationIsPending() throws {
        struct Crash: Error {}
        let keychain = RecordingKeychain()
        let store = makeStore("pending-quick", keychain: keychain)
        let seeded = try seedLocalProfile("pending-quick", store: store)
        store.rotationHook = { if $0 == .databaseRekeyed { throw Crash() } }
        XCTAssertThrowsError(try store.protect(passphrase: passphrase, enableQuickUnlock: true))
        // Even a stale Keychain copy of the old key cannot open the vault.
        keychain.stored["pending-quick"] = seeded.oldKey
        let relaunched = makeStore("pending-quick", keychain: keychain)
        _ = try relaunched.prepareForLaunch()
        XCTAssertThrowsError(try relaunched.unlockWithSystemAuthentication(context: LAContext()))
        XCTAssertFalse(relaunched.isUnlocked)
    }

    // MARK: Compatibility with the previous format

    private func writeVersionOneManifest(
        id: String, mode: String, extra: String, key: Data? = nil
    ) throws {
        let directory = root.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let keyField = key.map { "\"localMasterKey\": \"\($0.base64EncodedString())\"," } ?? ""
        // Exactly the keys the previous build wrote: no wrappedPreviousKey.
        let json = """
        {
          "createdAt": 700000000,
          \(keyField)
          \(extra)
          "formatVersion": 1,
          "profileID": "\(id)",
          "protectionMode": "\(mode)",
          "publicLabel": "Protected Profile",
          "quickUnlockEnabled": false,
          "updatedAt": 700000000
        }
        """
        try Data(json.utf8).write(to: directory.appendingPathComponent("security.json"))
    }

    func testVersionOneLocalProfileFromThePreviousBuildOpensAndRotates() throws {
        let id = "v1-local"
        let masterKey = try ProfileVaultCrypto.randomBytes(count: 32)
        try writeVersionOneManifest(id: id, mode: "local", extra: "", key: masterKey)
        // A database and asset written with the previous layout (same key
        // derivation: nothing about the on-disk data format changed).
        let directory = root.appendingPathComponent(id, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory.appendingPathComponent("assets"), withIntermediateDirectories: true)
        let database = try ProfileDatabase(
            url: directory.appendingPathComponent("profile.sqlite3"),
            key: ProfileVaultCrypto.deriveKey(masterKey: masterKey, purpose: "database"))
        try database.saveRecord(key: "value", payload: Data("\"old data\"".utf8))
        database.close()

        let store = makeStore(id)
        XCTAssertNotNil(try store.prepareForLaunch())
        XCTAssertEqual(try ProfileRepository(store: store).rawRecord(key: "value"),
                       Data("\"old data\"".utf8))
        try store.protect(passphrase: passphrase, enableQuickUnlock: false)
        store.lockVault()
        let reopened = makeStore(id)
        _ = try reopened.prepareForLaunch()
        _ = try reopened.unlock(passphrase: passphrase)
        XCTAssertEqual(try ProfileRepository(store: reopened).rawRecord(key: "value"),
                       Data("\"old data\"".utf8))
        XCTAssertThrowsError(try ProfileDatabase(
            url: directory.appendingPathComponent("profile.sqlite3"),
            key: ProfileVaultCrypto.deriveKey(masterKey: masterKey, purpose: "database")))
    }

    func testVersionOnePassphraseProfileFromThePreviousBuildStillUnlocks() throws {
        let id = "v1-pass"
        let masterKey = try ProfileVaultCrypto.randomBytes(count: 32)
        let salt = try ProfileVaultCrypto.randomBytes(count: 16)
        let wrapping = try ProfileVaultCrypto.derivePassphraseKey(
            passphrase: passphrase, salt: salt, rounds: 1)
        let wrapped = try ProfileVaultCrypto.wrapMasterKey(masterKey, with: wrapping)
        try writeVersionOneManifest(
            id: id, mode: "passphrase",
            extra: """
            "kdf": { "algorithm": "PBKDF2-HMAC-SHA256", "rounds": 1, "salt": "\(salt.base64EncodedString())" },
            "wrappedMasterKey": "\(wrapped.base64EncodedString())",
            """)
        let directory = root.appendingPathComponent(id, isDirectory: true)
        let database = try ProfileDatabase(
            url: directory.appendingPathComponent("profile.sqlite3"),
            key: ProfileVaultCrypto.deriveKey(masterKey: masterKey, purpose: "database"))
        try database.saveRecord(key: "value", payload: Data("\"kept\"".utf8))
        database.close()

        let store = makeStore(id)
        XCTAssertNil(try store.prepareForLaunch())
        _ = try store.unlock(passphrase: passphrase)
        XCTAssertEqual(try ProfileRepository(store: store).rawRecord(key: "value"),
                       Data("\"kept\"".utf8))
        // No rotation for an already-wrapped key, and the file stays version 1.
        XCTAssertEqual(try diskManifest(id).formatVersion, 1)
    }

    func testAnExactBackupMadeBeforeRotationStillRestores() async throws {
        let store = makeStore("pre-backup")
        let seeded = try seedLocalProfile("pre-backup", store: store)
        let archive = root.appendingPathComponent("before.turbospark-profile")
        _ = try EncryptedProfileBackup.export(
            profile: UserProfile(id: "pre-backup", name: "Pre"),
            destination: archive, passphrase: "portable backup password",
            appVersion: "tests", store: store)
        // Rotate after the backup was taken: the archive carries its own key.
        try store.protect(passphrase: passphrase, enableQuickUnlock: false)

        let destination = root.appendingPathComponent("restored-pre", isDirectory: true)
        _ = try await EncryptedProfileBackup.restore(
            archive: archive, destination: destination, newProfileID: "restored-pre",
            displayName: "Restored", passphrase: "portable backup password")
        let restored = ProfileVaultStore(
            rootProvider: { destination.appendingPathComponent("private-vault") },
            profileIDProvider: { "restored-pre" },
            migrateLegacyData: false)
        _ = try restored.prepareForLaunch()
        _ = try restored.unlock(passphrase: "portable backup password")
        try assertFullyReadable(restored, seeded: seeded)
        // No plaintext key was left in the restored manifest.
        let manifestText = try String(
            contentsOf: destination.appendingPathComponent("private-vault/security.json"),
            encoding: .utf8)
        XCTAssertFalse(manifestText.contains("localMasterKey"))
    }

    // MARK: Item 2: key derivation and prompts stay outside the lock

    /// Reads `manifest` from another thread while `work` runs. If the work held
    /// the store lock the read would block; the semaphore would time out.
    private func assertManifestReadableDuring(
        _ store: ProfileVaultStore, _ message: String, file: StaticString = #filePath, line: UInt = #line
    ) {
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            _ = store.manifest
            done.signal()
        }
        XCTAssertEqual(done.wait(timeout: .now() + 10), .success, message, file: file, line: line)
    }

    func testPassphraseDerivationNeverRunsUnderTheStoreLock() throws {
        let store = makeStore("lockfree", fast: false)
        _ = try store.prepareForLaunch()
        try ProfileRepository(store: store).save("v", key: "value")
        nonisolated(unsafe) var calls = 0
        store.passphraseKeyDeriver = { [unowned self] pass, salt, _ in
            calls += 1
            self.assertManifestReadableDuring(store, "derivation held the store lock")
            return try ProfileVaultCrypto.derivePassphraseKey(passphrase: pass, salt: salt, rounds: 1)
        }
        try store.protect(passphrase: passphrase, enableQuickUnlock: false)
        store.lockVault()
        _ = try store.unlock(passphrase: passphrase)
        try store.changePassphrase(current: passphrase, replacement: "a different passphrase")
        try store.disableProtection(passphrase: "a different passphrase")
        // protect + unlock + (unlock, replace) + disable
        XCTAssertEqual(calls, 5)
    }

    func testKeychainPromptNeverRunsUnderTheStoreLock() throws {
        let keychain = RecordingKeychain()
        let store = makeStore("prompt", keychain: keychain)
        _ = try store.prepareForLaunch()
        try store.protect(passphrase: passphrase, enableQuickUnlock: true)
        store.lockVault()
        nonisolated(unsafe) var prompted = false
        keychain.onLoad = { [unowned self] in
            prompted = true
            self.assertManifestReadableDuring(store, "keychain prompt held the store lock")
        }
        _ = try store.unlockWithSystemAuthentication(context: LAContext())
        XCTAssertTrue(prompted)
    }

    func testChangePassphraseRefusesAConcurrentManifestChange() throws {
        let store = makeStore("racing")
        _ = try store.prepareForLaunch()
        try store.protect(passphrase: passphrase, enableQuickUnlock: false)
        // Another change lands while the replacement key is being derived.
        nonisolated(unsafe) var calls = 0
        store.passphraseKeyDeriver = { pass, salt, rounds in
            calls += 1
            if calls == 2 {
                try? store.setQuickUnlock(enabled: false)
            }
            return try ProfileVaultCrypto.derivePassphraseKey(
                passphrase: pass, salt: salt, rounds: 1)
        }
        XCTAssertThrowsError(try store.changePassphrase(
            current: passphrase, replacement: "a different passphrase")) { error in
            XCTAssertEqual(error as? ProfileVaultStore.VaultError, .concurrentChange)
        }
        // The original passphrase still works.
        store.lockVault()
        _ = try store.unlock(passphrase: passphrase)
    }

    // MARK: Item 2: lock versus in-flight asset work

    func testLockWaitsForAnInFlightOperationInsteadOfWipingItsKey() throws {
        let store = makeStore("gate")
        _ = try store.prepareForLaunch()
        let started = DispatchSemaphore(value: 0)
        let operationFinished = DispatchSemaphore(value: 0)
        let lockFinished = DispatchSemaphore(value: 0)
        let lockReturned = LockedFlag()
        nonisolated(unsafe) var keyAtCancel: Data?
        nonisolated(unsafe) var lockDoneWhileOperationRan = true
        let original = try XCTUnwrap(store.session).masterKey

        DispatchQueue.global().async {
            try? store.withOperation { operation in
                started.signal()
                // Spin (no sleep) until the lock request is visible.
                spin(until: { operation.isCancelled })
                keyAtCancel = operation.masterKey
                lockDoneWhileOperationRan = lockReturned.value
                XCTAssertThrowsError(try operation.checkNotCancelled())
            }
            operationFinished.signal()
        }
        XCTAssertEqual(started.wait(timeout: .now() + 10), .success)
        DispatchQueue.global().async {
            store.lockVault()
            lockReturned.set()
            lockFinished.signal()
        }
        XCTAssertEqual(operationFinished.wait(timeout: .now() + 10), .success)
        XCTAssertEqual(lockFinished.wait(timeout: .now() + 10), .success)
        XCTAssertEqual(keyAtCancel, original)
        XCTAssertFalse(lockDoneWhileOperationRan, "lock returned before the operation ended")
        XCTAssertFalse(store.isUnlocked)
        XCTAssertThrowsError(try store.withOperation { _ in }) { error in
            XCTAssertEqual(error as? ProfileVaultStore.VaultError, .locked)
        }
    }

    func testLockCancelsALargeImportBetweenChunksAndLeavesNoOrphan() throws {
        let store = makeStore("import-lock")
        _ = try store.prepareForLaunch()
        let assets = ManagedAssetStore(vault: store)
        let source = root.appendingPathComponent("big.bin")
        try Data(repeating: 0x42, count: 4 * 1_048_576 + 3).write(to: source)

        let lockFinished = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var chunks = 0
        assets.chunkObserver = {
            chunks += 1
            // Hash pass is chunks 1-5; request the lock on the first sealing
            // chunk, then wait until the request is visible to the import.
            guard chunks == 6 else { return }
            DispatchQueue.global().async {
                store.lockVault()
                lockFinished.signal()
            }
            spin(until: { store.gate.isCancelRequested })
        }
        XCTAssertThrowsError(try assets.store(fileURL: source)) { error in
            XCTAssertEqual(error as? ManagedAssetStore.AssetError, .locked)
        }
        XCTAssertEqual(lockFinished.wait(timeout: .now() + 10), .success)
        XCTAssertFalse(store.isUnlocked)
        let leftovers = try FileManager.default.subpathsOfDirectory(
            atPath: root.appendingPathComponent("import-lock/assets").path)
            .filter { $0.hasSuffix(".tsasset") || $0.hasSuffix(".tmp") }
        XCTAssertEqual(leftovers, [], "a cancelled import must leave no ciphertext behind")
    }

    func testAssetOperationRacingAFinishedLockFailsClosedWithoutAnEmptyKey() throws {
        let store = makeStore("late")
        _ = try store.prepareForLaunch()
        store.lockVault()
        XCTAssertThrowsError(try ManagedAssetStore(vault: store).store(
            data: Data("x".utf8), fileName: "x")) { error in
            XCTAssertEqual(error as? ManagedAssetStore.AssetError, .locked)
        }
    }

    func testGateLetsANestedOperationThroughAPendingLock() throws {
        let store = makeStore("nested")
        _ = try store.prepareForLaunch()
        let started = DispatchSemaphore(value: 0)
        let finished = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var nestedWorked = false
        DispatchQueue.global().async {
            try? store.withOperation { outer in
                started.signal()
                spin(until: { outer.isCancelled })
                // A consumer closure touching a gated helper must not deadlock
                // against the lock that is waiting for this very operation.
                nestedWorked = (try? store.withOperation { _ in true }) == true
            }
            finished.signal()
        }
        XCTAssertEqual(started.wait(timeout: .now() + 10), .success)
        let lockDone = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            store.lockVault()
            lockDone.signal()
        }
        XCTAssertEqual(finished.wait(timeout: .now() + 10), .success)
        XCTAssertEqual(lockDone.wait(timeout: .now() + 10), .success)
        XCTAssertTrue(nestedWorked)
    }

    // MARK: Item 4: consistent export snapshot

    func testExportSnapshotIgnoresAssetChurnAfterTheSnapshotAndAuthenticatesEachAsset() async throws {
        let store = makeStore("export")
        let seeded = try seedLocalProfile("export", store: store)
        let assets = ManagedAssetStore(vault: store)
        let archive = root.appendingPathComponent("churn.turbospark-profile")
        var lateDescriptor: ManagedAssetDescriptor?
        EncryptedProfileBackup.afterSnapshotHook = {
            // After the snapshot is frozen: add one asset and delete another.
            lateDescriptor = try? assets.store(
                data: Data("added during export".utf8), fileName: "late.bin")
            try? assets.release(reference: seeded.assets[0].descriptor.storedReference)
        }
        defer { EncryptedProfileBackup.afterSnapshotHook = nil }
        _ = try EncryptedProfileBackup.export(
            profile: UserProfile(id: "export", name: "Exp"),
            destination: archive, passphrase: "portable backup password",
            appVersion: "tests", store: store)
        XCTAssertNotNil(lateDescriptor)

        let destination = root.appendingPathComponent("restored-export", isDirectory: true)
        _ = try await EncryptedProfileBackup.restore(
            archive: archive, destination: destination, newProfileID: "restored-export",
            displayName: "R", passphrase: "portable backup password")
        let restored = ProfileVaultStore(
            rootProvider: { destination.appendingPathComponent("private-vault") },
            profileIDProvider: { "restored-export" },
            migrateLegacyData: false)
        _ = try restored.prepareForLaunch()
        _ = try restored.unlock(passphrase: "portable backup password")
        // The snapshot has the asset that was deleted afterwards (its bytes
        // were frozen) and not the one added afterwards.
        try assertFullyReadable(restored, seeded: seeded)
        let lateID = try XCTUnwrap(lateDescriptor).id
        XCTAssertNil(try ManagedAssetStore(vault: restored).descriptor(
            for: ManagedAssetDescriptor(id: lateID, fileName: "", mimeType: nil, byteCount: 0)
                .storedReference))
    }

    func testExportRefusesAnAssetThatFailsAuthentication() throws {
        let store = makeStore("export-bad")
        let seeded = try seedLocalProfile("export-bad", store: store)
        let victim = seeded.assets[1].descriptor
        let url = root.appendingPathComponent(
            "export-bad/assets/\(victim.id.prefix(2))/\(victim.id).tsasset")
        var bytes = try Data(contentsOf: url)
        bytes[bytes.index(before: bytes.endIndex)] ^= 0x01
        try bytes.write(to: url)
        let archive = root.appendingPathComponent("bad.turbospark-profile")
        XCTAssertThrowsError(try EncryptedProfileBackup.export(
            profile: UserProfile(id: "export-bad", name: "Bad"),
            destination: archive, passphrase: "portable backup password",
            appVersion: "tests", store: store)) { error in
            XCTAssertEqual(
                error as? EncryptedProfileBackup.BackupError,
                .assetAuthenticationFailed(victim.id))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: archive.path))
    }

    // MARK: Low items: payload determinism and legacy asset retention

    func testChatPayloadsAreEncodedWithSortedKeys() throws {
        let store = makeStore("sorted")
        _ = try store.prepareForLaunch()
        let chat = AppChat(
            title: "Sorted", messages: [AppChatMessage(role: .user, content: "hi")])
        try ProfileRepository(store: store).saveChatArchive(
            AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.sortedKeys]
        let stored = try XCTUnwrap(
            try XCTUnwrap(store.session).database.chatPayload(id: chat.id.uuidString))
        XCTAssertEqual(stored, try encoder.encode(chat))
    }

    func testMigratedButUnreferencedAssetsSurviveGarbageCollectionWhileRecorded() throws {
        let store = makeStore("legacy-gc")
        let session = try XCTUnwrap(try store.prepareForLaunch())
        let assets = ManagedAssetStore(vault: store)
        let orphan = try assets.store(data: Data("old generated image".utf8), fileName: "old.png")
        // Age the row past the 60 s pending-write grace window without sleeping.
        try session.database.execute("UPDATE assets SET created_at = 0")
        let repository = ProfileRepository(store: store)
        try repository.save([orphan], key: ProfileRepository.legacyAssetsRecordKey)
        let chat = AppChat(title: "Empty", messages: [])
        let archive = AppChatArchive(selectedChatID: chat.id, chats: [chat])
        try repository.saveChatArchive(archive)
        XCTAssertNotNil(try assets.descriptor(for: orphan.storedReference),
                        "a migrated file is the only copy and must not be collected")

        try repository.deleteRecord(key: ProfileRepository.legacyAssetsRecordKey)
        try repository.saveChatArchive(archive)
        XCTAssertNil(try assets.descriptor(for: orphan.storedReference))
    }
}

/// Busy-waits (no sleep) for a condition another thread will make true, with
/// a deadline so a regression fails the test instead of hanging it.
private func spin(until condition: () -> Bool, timeout: TimeInterval = 10) {
    let deadline = Date().addingTimeInterval(timeout)
    while !condition(), Date() < deadline { sched_yield() }
}

private final class LockedFlag: @unchecked Sendable {
    private let lock = NSLock()
    private var flag = false
    var value: Bool { lock.lock(); defer { lock.unlock() }; return flag }
    func set() { lock.lock(); flag = true; lock.unlock() }
}
