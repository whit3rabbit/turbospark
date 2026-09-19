import Foundation
@testable import TurboSparkApp
import XCTest

final class ProfileVaultTests: XCTestCase {
    private var root: URL!

    override func setUpWithError() throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("ProfileVaultTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    func testPBKDF2KnownVectorAndCanonicalUnicode() throws {
        let derived = try ProfileVaultCrypto.derivePassphraseKey(
            passphrase: "passwordpassword", salt: Data("salt".utf8), rounds: 1)
        XCTAssertEqual(
            hex(derived),
            "67e5786e265622603e1815e56744e237d7dd6a44d14a9a9c4d0f23b3f7174411")

        let composed = try ProfileVaultCrypto.derivePassphraseKey(
            passphrase: "ééééééééééééééé", salt: Data("salt".utf8), rounds: 2)
        let decomposed = try ProfileVaultCrypto.derivePassphraseKey(
            passphrase: String(repeating: "e\u{301}", count: 15),
            salt: Data("salt".utf8), rounds: 2)
        XCTAssertEqual(composed, decomposed)
    }

    func testProtectionRejectsWrongPasswordAndWrapperTampering() throws {
        let store = makeStore(id: "protected")
        _ = try store.prepareForLaunch()
        try ProfileRepository(store: store).save("private", key: "value")
        try store.protect(passphrase: "correct horse battery", enableQuickUnlock: false)
        store.lockVault()
        XCTAssertThrowsError(try store.unlock(passphrase: "incorrect passphrase"))
        XCTAssertEqual(
            try ProfileRepository(store: storeAfterUnlock(store, "correct horse battery"))
                .load(String.self, key: "value"),
            "private")

        store.lockVault()
        let manifestURL = root.appendingPathComponent("protected/security.json")
        var manifest = try JSONDecoder().decode(
            ProfileSecurityManifest.self, from: Data(contentsOf: manifestURL))
        manifest.wrappedMasterKey![manifest.wrappedMasterKey!.startIndex] ^= 0x01
        try JSONEncoder().encode(manifest).write(to: manifestURL, options: .atomic)
        store.resetForTests()
        XCTAssertThrowsError(try store.unlock(passphrase: "correct horse battery"))
    }

    func testSQLCipherAndManagedAssetsHidePlaintextAndDetectTampering() throws {
        let store = makeStore(id: "asset")
        let session = try XCTUnwrap(try store.prepareForLaunch())
        let secret = "recognizable-chat-secret-41d7"
        let chat = AppChat(
            title: "Private",
            messages: [AppChatMessage(role: .user, content: secret)])
        try ProfileRepository(store: store).saveChatArchive(
            AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        let bytes = Data(repeating: 0x5a, count: 1_048_576 + 73)
        let assets = ManagedAssetStore(vault: store)
        let descriptor = try assets.store(
            data: bytes, fileName: "picture.png", mimeType: "image/png")
        var opened = Data()
        try assets.streamDecrypted(reference: descriptor.storedReference) { opened.append($0) }
        XCTAssertEqual(opened, bytes)
        try session.database.checkpoint()
        store.lockVault()

        for suffix in ["", "-wal", "-shm"] {
            let url = root.appendingPathComponent("asset/profile.sqlite3\(suffix)")
            if let data = try? Data(contentsOf: url) {
                XCTAssertNil(data.range(of: Data(secret.utf8)))
            }
        }
        let wrongKey = Data(repeating: 0x22, count: 32)
        XCTAssertThrowsError(try ProfileDatabase(
            url: root.appendingPathComponent("asset/profile.sqlite3"), key: wrongKey))

        _ = try store.prepareForLaunch()
        let assetURL = root.appendingPathComponent(
            "asset/assets/\(descriptor.id.prefix(2))/\(descriptor.id).tsasset")
        var encrypted = try Data(contentsOf: assetURL)
        encrypted[encrypted.index(before: encrypted.endIndex)] ^= 0x01
        try encrypted.write(to: assetURL, options: .atomic)
        XCTAssertThrowsError(try assets.streamDecrypted(
            reference: descriptor.storedReference) { _ in })
    }

    func testEncryptedBackupRoundTripWrongPasswordAndTampering() async throws {
        let source = makeStore(id: "source")
        _ = try source.prepareForLaunch()
        let assetStore = ManagedAssetStore(vault: source)
        let asset = try assetStore.store(
            data: Data("image bytes".utf8), fileName: "generated.png", mimeType: "image/png")
        let chat = AppChat(
            title: "Backup secret",
            messages: [AppChatMessage(
                role: .assistant, content: "restored text", imagePaths: [asset.storedReference])])
        try ProfileRepository(store: source).saveChatArchive(
            AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        let archive = root.appendingPathComponent("profile.turbospark-profile")
        let password = "portable backup password"
        _ = try EncryptedProfileBackup.export(
            profile: UserProfile(id: "source", name: "Secret Name"),
            destination: archive,
            passphrase: password,
            appVersion: "tests",
            store: source)
        XCTAssertNil(try Data(contentsOf: archive).range(of: Data("restored text".utf8)))

        let wrongOutput = root.appendingPathComponent("wrong.zip")
        XCTAssertThrowsError(try ProfileEncryptedChunkWriter.decrypt(
            archive: archive,
            passphrase: "wrong backup password",
            destination: wrongOutput))

        let destination = root.appendingPathComponent("restored", isDirectory: true)
        _ = try await EncryptedProfileBackup.restore(
            archive: archive,
            destination: destination,
            newProfileID: "restored",
            displayName: "Restored Name",
            passphrase: password)
        let restored = ProfileVaultStore(
            rootProvider: { destination.appendingPathComponent("private-vault") },
            profileIDProvider: { "restored" },
            migrateLegacyData: false)
        XCTAssertNil(try restored.prepareForLaunch())
        _ = try restored.unlock(passphrase: password)
        let restoredArchive = try XCTUnwrap(
            try ProfileRepository(store: restored).loadChatArchive())
        XCTAssertEqual(restoredArchive.chats.first?.messages.first?.content, "restored text")
        var restoredAsset = Data()
        try ManagedAssetStore(vault: restored).streamDecrypted(
            reference: asset.storedReference) { restoredAsset.append($0) }
        XCTAssertEqual(restoredAsset, Data("image bytes".utf8))

        var tampered = try Data(contentsOf: archive)
        tampered[tampered.index(before: tampered.endIndex)] ^= 0x01
        let tamperedURL = root.appendingPathComponent("tampered.turbospark-profile")
        try tampered.write(to: tamperedURL)
        XCTAssertThrowsError(try ProfileEncryptedChunkWriter.decrypt(
            archive: tamperedURL,
            passphrase: password,
            destination: root.appendingPathComponent("tampered.zip")))
    }

    func testOpenExportUsesStrictCategoriesAndStreamsOriginalAssets() async throws {
        let store = makeStore(id: "open")
        _ = try store.prepareForLaunch()
        let assetStore = ManagedAssetStore(vault: store)
        let asset = try assetStore.store(
            data: Data("portable-image".utf8), fileName: "image.png", mimeType: "image/png")
        let chat = AppChat(
            title: "Portable",
            messages: [AppChatMessage(
                role: .assistant, content: "plain transcript", imagePaths: [asset.storedReference])],
            artifacts: [AppArtifact(
                chatID: UUID(), path: asset.storedReference, title: "Image",
                origin: .imageGeneration)])
        let snapshot = ProfileExportSnapshot(
            profile: UserProfile(id: "open", name: "Open"),
            exportedAt: Date(timeIntervalSince1970: 1_700_000_000),
            appVersion: "tests",
            modelAlias: "model",
            chats: [chat],
            projects: .empty(),
            settingsJSON: Data("{\"private\":true}".utf8),
            memoryFiles: [])
        let zipURL = root.appendingPathComponent("open.zip")
        try OpenProfileExport.export(
            snapshot: snapshot,
            included: ["chats", "generated-images"],
            destination: zipURL,
            assets: assetStore)
        let listing = try await ProcessExecutor.run(
            executableURL: ProfileBackupImport.zipinfoURL,
            arguments: ["-1", zipURL.path], timeoutSeconds: 30)
        let entries = try ProfileBackupImport.entries(from: listing)
        XCTAssertTrue(entries.contains(where: { $0.hasSuffix("/messages.jsonl") }))
        XCTAssertTrue(entries.contains(where: { $0.hasPrefix("assets/generated-images/") }))
        XCTAssertFalse(entries.contains("settings/settings.json"))
        XCTAssertFalse(entries.contains("projects/projects.json"))

        let emptyURL = root.appendingPathComponent("empty.zip")
        try OpenProfileExport.export(
            snapshot: snapshot, included: [], destination: emptyURL, assets: assetStore)
        let emptyListing = try await ProcessExecutor.run(
            executableURL: ProfileBackupImport.zipinfoURL,
            arguments: ["-1", emptyURL.path], timeoutSeconds: 30)
        XCTAssertEqual(
            Set(try ProfileBackupImport.entries(from: emptyListing)),
            Set(["manifest.json", "checksums.sha256"]))
    }

    private func makeStore(id: String) -> ProfileVaultStore {
        let vault = root.appendingPathComponent(id, isDirectory: true)
        return ProfileVaultStore(
            rootProvider: { vault },
            profileIDProvider: { id },
            migrateLegacyData: false)
    }

    private func storeAfterUnlock(
        _ store: ProfileVaultStore,
        _ passphrase: String
    ) throws -> ProfileVaultStore {
        _ = try store.unlock(passphrase: passphrase)
        return store
    }

    private func hex(_ data: Data) -> String {
        data.map { String(format: "%02x", $0) }.joined()
    }
}
