import Foundation
import LocalAuthentication
import XCTest
@testable import TurboSparkApp

/// A lock that completes while an unlock is still authenticating must win
/// over that unlock: PBKDF2 and the Keychain prompt deliberately run outside
/// the store lock, so the install has to be rejected as stale instead of
/// reopening the vault behind a finished lock. Every test pauses
/// authentication deterministically (semaphores, no sleeps), locks, and then
/// lets the authentication finish.
private final class LockRaceKeychain: ProfileVaultKeychainProtocol, @unchecked Sendable {
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

final class ProfileVaultLockRaceTests: XCTestCase {
    private var root: URL!
    private let passphrase = "correct horse battery"

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProfileVaultLockRace-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    /// One PBKDF2 round keeps the suite fast; the race does not depend on
    /// the round count.
    private static let fastDeriver: @Sendable (String, Data, UInt32) throws -> Data = {
        passphrase, salt, _ in
        try ProfileVaultCrypto.derivePassphraseKey(passphrase: passphrase, salt: salt, rounds: 1)
    }

    private func makeStore(
        _ id: String, keychain: (any ProfileVaultKeychainProtocol)? = nil
    ) -> ProfileVaultStore {
        let vault = root.appendingPathComponent(id, isDirectory: true)
        let store = ProfileVaultStore(
            rootProvider: { vault },
            profileIDProvider: { id },
            keychain: keychain ?? LockRaceKeychain(),
            migrateLegacyData: false)
        store.passphraseKeyDeriver = Self.fastDeriver
        return store
    }

    /// A deriver that signals `started` and blocks on `release`, standing in
    /// for the real 600k-round derivation window.
    private func blockingDeriver(
        started: DispatchSemaphore, release: DispatchSemaphore
    ) -> @Sendable (String, Data, UInt32) throws -> Data {
        { pass, salt, _ in
            started.signal()
            release.wait()
            return try ProfileVaultCrypto.derivePassphraseKey(
                passphrase: pass, salt: salt, rounds: 1)
        }
    }

    private func unlockOnWorker(
        _ work: @escaping @Sendable () throws -> Void
    ) -> (@Sendable () -> Error?) {
        let returned = DispatchSemaphore(value: 0)
        nonisolated(unsafe) var failure: Error?
        DispatchQueue.global().async {
            do {
                try work()
            } catch {
                failure = error
            }
            returned.signal()
        }
        return {
            XCTAssertEqual(returned.wait(timeout: .now() + 10), .success)
            return failure
        }
    }

    private func assertStaleUnlockFailed(
        _ failure: Error?, _ store: ProfileVaultStore,
        file: StaticString = #filePath, line: UInt = #line
    ) {
        XCTAssertFalse(store.isUnlocked, "a stale unlock reopened the vault", file: file, line: line)
        XCTAssertEqual(
            failure as? ProfileVaultStore.VaultError, .locked,
            "the stale unlock must be refused as locked, not fail some other way",
            file: file, line: line)
    }

    private func assertRelockable(
        _ store: ProfileVaultStore, file: StaticString = #filePath, line: UInt = #line
    ) throws {
        store.passphraseKeyDeriver = Self.fastDeriver
        XCTAssertNotNil(try store.unlock(passphrase: passphrase), file: file, line: line)
        XCTAssertTrue(store.isUnlocked, "a fresh unlock after the lock must still open the vault",
                      file: file, line: line)
    }

    func testLockDuringPassphraseDerivationBeatsTheInFlightUnlock() throws {
        let store = makeStore("lock-wins")
        _ = try store.prepareForLaunch()
        try store.protect(passphrase: passphrase, enableQuickUnlock: false)
        store.lockVault()

        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        store.passphraseKeyDeriver = blockingDeriver(started: started, release: release)
        let finish = unlockOnWorker { try store.unlock(passphrase: self.passphrase) }
        XCTAssertEqual(started.wait(timeout: .now() + 10), .success)

        // The manifest is snapshotted and the store lock released; the lock
        // completes while authentication is still suspended in derivation.
        store.lockVault()
        release.signal()
        assertStaleUnlockFailed(finish(), store)
        try assertRelockable(store)
    }

    func testLockDuringTheKeychainPromptBeatsTheInFlightQuickUnlock() throws {
        let keychain = LockRaceKeychain()
        let store = makeStore("lock-wins-quick", keychain: keychain)
        _ = try store.prepareForLaunch()
        try store.protect(passphrase: passphrase, enableQuickUnlock: true)
        store.lockVault()

        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        keychain.onLoad = {
            started.signal()
            release.wait()
        }
        let finish = unlockOnWorker {
            try store.unlockWithSystemAuthentication(context: LAContext())
        }
        XCTAssertEqual(started.wait(timeout: .now() + 10), .success)

        store.lockVault()
        release.signal()
        assertStaleUnlockFailed(finish(), store)
        // Quick unlock after the lock still works (the prompt no longer
        // blocks).
        keychain.onLoad = nil
        _ = try store.unlockWithSystemAuthentication(context: LAContext())
        XCTAssertTrue(store.isUnlocked)
    }

    func testLockBeatsAnInFlightRotationResume() throws {
        struct Crash: Error {}
        let id = "lock-wins-rotation"
        let store = makeStore(id)
        _ = try store.prepareForLaunch()
        store.rotationHook = { if $0 == .databaseRekeyed { throw Crash() } }
        XCTAssertThrowsError(try store.protect(passphrase: passphrase, enableQuickUnlock: false))
        XCTAssertNotNil(try JSONDecoder().decode(
            ProfileSecurityManifest.self,
            from: Data(contentsOf: root.appendingPathComponent("\(id)/security.json"))
        ).wrappedPreviousKey)
        store.lockVault()

        // A relaunch resumes the interrupted rotation on unlock; a lock that
        // lands during the resumed derivation must beat it too.
        let relaunched = makeStore(id)
        XCTAssertNil(try relaunched.prepareForLaunch())
        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        relaunched.passphraseKeyDeriver = blockingDeriver(started: started, release: release)
        let finish = unlockOnWorker { try relaunched.unlock(passphrase: self.passphrase) }
        XCTAssertEqual(started.wait(timeout: .now() + 10), .success)

        relaunched.lockVault()
        release.signal()
        assertStaleUnlockFailed(finish(), relaunched)
        XCTAssertFalse(relaunched.isUnlocked)

        // The rotation is still resumable afterwards and runs to completion.
        try assertRelockable(relaunched)
        let finished = try JSONDecoder().decode(
            ProfileSecurityManifest.self,
            from: Data(contentsOf: root.appendingPathComponent("\(id)/security.json")))
        XCTAssertNil(finished.wrappedPreviousKey)
    }

    func testLockNowLeavesTheCoordinatorLockedOverAStaleUnlockCompletion() async throws {
        let store = makeStore("lock-wins-coordinator")
        _ = try store.prepareForLaunch()
        try store.protect(passphrase: passphrase, enableQuickUnlock: false)
        store.lockVault()

        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        store.passphraseKeyDeriver = blockingDeriver(started: started, release: release)
        let coordinator = await ProfileVaultCoordinator(store: store)
        let state = await MainActor.run { coordinator.state }
        XCTAssertEqual(state, .locked)

        let unlockTask = Task { await coordinator.unlock(passphrase: passphrase) }
        XCTAssertEqual(started.wait(timeout: .now() + 10), .success)
        // The person leaves the Mac while authentication is still running.
        await coordinator.lockNow()
        release.signal()
        await unlockTask.value

        // The store must not reopen and the coordinator must stay locked:
        // not .unlocked (stale completion completed) and not .error (the
        // stale failure surfaced over the lock either).
        let (finalState, unlocked) = await MainActor.run {
            (coordinator.state, store.isUnlocked)
        }
        XCTAssertEqual(finalState, .locked)
        XCTAssertFalse(unlocked)
    }
}
