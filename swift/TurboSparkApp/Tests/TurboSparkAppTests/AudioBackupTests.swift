import Foundation
import XCTest
@testable import TurboSparkApp

final class AudioBackupTests: XCTestCase {
    func testAudioRecordsAndReferencesSurviveEncryptedBackupRestore() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = ProfileVaultStore(rootProvider: { root.appendingPathComponent("source") }, profileIDProvider: { "source" }, migrateLegacyData: false)
        _ = try source.prepareForLaunch()
        let assets = ManagedAssetStore(vault: source)
        let bytes = try AudioMediaIO.wav(samples: [0.1, 0.2, 0.3], sampleRate: 16_000, channels: 1)
        let asset = try assets.store(data: bytes, fileName: "meeting.wav", mimeType: "audio/wav")
        var item = AudioLibraryItem(title: "Private meeting", kind: .recording)
        item.status = .recording
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 16_000, channels: 1, frameCount: 3)]
        item.addTranscript([AudioLibrarySegment(start: 0, end: 1, text: "Private transcript")], source: .user)
        try AudioLibraryStore(repository: ProfileRepository(store: source)).save(item)
        let archive = root.appendingPathComponent("audio.turbospark-profile")
        let passphrase = "audio backup test passphrase"
        _ = try EncryptedProfileBackup.export(profile: UserProfile(id: "source", name: "Audio"), destination: archive, passphrase: passphrase, appVersion: "test", store: source)
        XCTAssertNil(try Data(contentsOf: archive).range(of: Data(item.title.utf8)))
        let destination = root.appendingPathComponent("restore")
        _ = try await EncryptedProfileBackup.restore(archive: archive, destination: destination, newProfileID: "restored", displayName: "Restored", passphrase: passphrase)
        let restored = ProfileVaultStore(rootProvider: { destination.appendingPathComponent("private-vault") }, profileIDProvider: { "restored" }, migrateLegacyData: false)
        XCTAssertNil(try restored.prepareForLaunch())
        _ = try restored.unlock(passphrase: passphrase)
        let library = AudioLibraryStore(repository: ProfileRepository(store: restored))
        var reopened = try XCTUnwrap(library.load().first)
        XCTAssertEqual(reopened.id, item.id)
        XCTAssertEqual(reopened.preferredTranscript?.source, .user)
        XCTAssertTrue(reopened.recoverInterrupted())
        try library.save(reopened)
        try restored.session?.database.execute("UPDATE assets SET created_at = 0")
        try library.collectUnusedAssets()
        var output = Data()
        try ManagedAssetStore(vault: restored).streamDecrypted(reference: asset.storedReference) { output.append($0) }
        XCTAssertEqual(output, bytes)
    }

    func testOpenAudioExportContainsPortableRecordsAndAssets() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root.appendingPathComponent("vault") }, profileIDProvider: { "audio" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let assets = ManagedAssetStore(vault: vault)
        let descriptor = try assets.store(data: Data([1, 2, 3]), fileName: "section.wav", mimeType: "audio/wav")
        var item = AudioLibraryItem(title: "Section", kind: .voiceover)
        item.localModelBookmark = Data("private checkpoint bookmark".utf8)
        item.clips = [AudioLibraryClip(assetReference: descriptor.storedReference, sampleRate: 24_000, channels: 1, frameCount: 3)]
        let snapshot = ProfileExportSnapshot(profile: UserProfile(id: "audio", name: "Audio"), exportedAt: Date(), appVersion: "test", modelAlias: nil, chats: [], projects: .empty(), settingsFiles: [:], chatFiles: [], memoryFiles: [], audioItems: [item])
        let archive = root.appendingPathComponent("open.zip")
        try OpenProfileExport.export(snapshot: snapshot, included: ["audio"], destination: archive, assets: assets)
        let recordPath = "audio/projects/\(item.id.uuidString.lowercased()).json"
        let result = try await ProcessExecutor.run(executableURL: URL(fileURLWithPath: "/usr/bin/unzip"), arguments: ["-p", archive.path, recordPath], timeoutSeconds: 30)
        let encoder = JSONDecoder(); encoder.dateDecodingStrategy = .iso8601
        let portable = try encoder.decode(AudioLibraryItem.self, from: Data(result.stdout.utf8))
        XCTAssertNil(portable.localModelBookmark)
        XCTAssertTrue(portable.clips[0].assetReference.hasPrefix("../../assets/audio/"))
        let assetPath = String(portable.clips[0].assetReference.dropFirst(6))
        let listing = try await ProcessExecutor.run(executableURL: ProfileBackupImport.zipinfoURL, arguments: ["-1", archive.path], timeoutSeconds: 30)
        XCTAssertTrue(try ProfileBackupImport.entries(from: listing).contains(assetPath))
    }
}
