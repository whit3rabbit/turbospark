import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

final class AudioLibraryTests: XCTestCase {
    func testAudioReferencesSurviveChatReconciliationAndReopen() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "audio-test" }, migrateLegacyData: false)
        let session = try XCTUnwrap(try vault.prepareForLaunch())
        let assets = ManagedAssetStore(vault: vault)
        let repository = ProfileRepository(store: vault)
        let library = AudioLibraryStore(repository: repository)
        let bytes = Data("recording must survive a chat save".utf8)
        let asset = try assets.store(data: bytes, fileName: "meeting.wav", mimeType: "audio/wav")
        var item = AudioLibraryItem(title: "Meeting", kind: .recording)
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 48_000, channels: 1, frameCount: 100)]
        try library.save(item)
        try session.database.execute("UPDATE assets SET created_at = 0")
        let chat = AppChat(title: "Other work", messages: [])
        try repository.saveChatArchive(AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        XCTAssertEqual(try library.load().first?.clips.first?.assetReference, asset.storedReference)
        var reopened = Data()
        try assets.streamDecrypted(reference: asset.storedReference) { reopened.append($0) }
        XCTAssertEqual(reopened, bytes)
        try library.delete(item.id)
        try repository.saveChatArchive(AppChatArchive(selectedChatID: chat.id, chats: [chat]))
        XCTAssertNil(try assets.descriptor(for: asset.storedReference))
    }

    func testAudioOnlyProfileDeletionCollectsUnreferencedAsset() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "audio-only" }, migrateLegacyData: false)
        let session = try XCTUnwrap(try vault.prepareForLaunch())
        let assets = ManagedAssetStore(vault: vault)
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let asset = try assets.store(data: Data([1, 2, 3]), fileName: "meeting.wav")
        var item = AudioLibraryItem(title: "Meeting", kind: .recording)
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 16_000, channels: 1, frameCount: 1)]
        try library.save(item)
        try session.database.execute("UPDATE assets SET created_at = 0")
        XCTAssertNil(try session.database.loadChatArchive())
        try library.delete(item.id)
        XCTAssertNil(try assets.descriptor(for: asset.storedReference))
        XCTAssertThrowsError(try assets.materializedURL(for: asset.storedReference))
        XCTAssertTrue(try library.load().isEmpty)
    }

    func testFailedReferenceUpdatePreservesPreviousRecord() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "rollback" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        var item = AudioLibraryItem(title: "Original", kind: .voiceover)
        try library.save(item)
        item.title = "Broken update"
        item.clips = [AudioLibraryClip(assetReference: "turbospark-asset:" + String(repeating: "a", count: 64), sampleRate: 24_000, channels: 1, frameCount: 1)]
        XCTAssertThrowsError(try library.save(item))
        XCTAssertEqual(try library.load().first?.title, "Original")
    }

    func testDirectEncryptedWriteAcceptsDataSliceAndDeduplicatesFileImport() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "slice" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let assets = ManagedAssetStore(vault: vault)
        let slice = Data([0, 12, 34, 56]).dropFirst()
        XCTAssertEqual(slice.startIndex, 1)
        let direct = try assets.store(data: slice, fileName: "slice.wav")
        let source = root.appendingPathComponent("input.wav")
        try slice.write(to: source)
        let imported = try assets.store(fileURL: source)
        XCTAssertEqual(imported.id, direct.id)
        var decoded = Data()
        try assets.streamDecrypted(reference: direct.storedReference) { decoded.append($0) }
        XCTAssertEqual(decoded, Data([12, 34, 56]))
    }

    func testDraftRefinementPreservesUserRevision() {
        var item = AudioLibraryItem(title: "Meeting", kind: .recording)
        item.addTranscript([AudioLibrarySegment(start: 0, end: 2, text: "draft")], source: .draft)
        item.addTranscript([AudioLibrarySegment(start: 0, end: 2, text: "my correction")], source: .user)
        item.addTranscript([AudioLibrarySegment(start: 0, end: 2, text: "final model text")], source: .final)
        XCTAssertEqual(item.preferredTranscript?.segments.first?.text, "my correction")
        XCTAssertEqual(item.transcripts.count, 3)
    }

    func testRecoveryAndSubtitleTiming() throws {
        var item = AudioLibraryItem(title: "Interrupted", kind: .recording)
        item.status = .recording
        XCTAssertTrue(item.recoverInterrupted())
        XCTAssertEqual(item.status, .interrupted)
        XCTAssertFalse(item.recoverInterrupted())
        let segments = [AudioLibrarySegment(start: 61.125, end: 63.5, text: "Hello", speaker: "Speaker 1")]
        XCTAssertTrue(AudioTranscriptExport.srt(segments).contains("00:01:01,125 --> 00:01:03,500"))
        XCTAssertTrue(AudioTranscriptExport.vtt(segments).hasPrefix("WEBVTT\n"))
        XCTAssertTrue(AudioTranscriptExport.srt(segments).contains("Speaker 1: Hello"))
    }

    func testSummaryRetainsSpeakerNamesAndBoundsLongTranscriptWindows() {
        let segments = [
            AudioLibrarySegment(start: 12, end: 15, text: "I will send the draft.", speaker: "Alex"),
            AudioLibrarySegment(start: 30, end: 60, text: String(repeating: "word ", count: 4_000)),
        ]
        let chunks = AudioMeetingSummary.chunks(segments)
        XCTAssertTrue(chunks[0].text.hasPrefix("Alex: I will send the draft."))
        XCTAssertEqual(chunks[0].start, 12)
        XCTAssertTrue(chunks.dropFirst().allSatisfy { $0.start == 30 })
        XCTAssertTrue(chunks.allSatisfy { $0.text.count <= 8_000 })
        XCTAssertEqual(chunks.map(\.text).joined().components(separatedBy: "word").count - 1, 4_000)
    }
}

extension AudioLibraryTests {
    @MainActor func testShutdownFlushesStagedTranscriptAndLateCaptureCannotRollBackStatus() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "editor" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault))
        var item = AudioLibraryItem(title: "Recording", kind: .recording)
        item.status = .recording
        try controller.save(item)
        controller.recordingID = item.id
        try library.update(item.id) { $0.title = "Renamed while capturing" }
        XCTAssertTrue(controller.publishCaptureUpdate(item.id, token: controller.epoch))
        XCTAssertEqual(controller.items.first?.title, "Renamed while capturing")
        controller.recordingID = nil
        controller.update(item.id) { $0.status = .completed }
        XCTAssertFalse(controller.publishCaptureUpdate(item.id, token: controller.epoch))
        XCTAssertEqual(controller.items.first?.status, .completed)
        let correction = [AudioLibrarySegment(start: 0, end: 1, text: "Last edit before lock", speaker: "Alex")]
        controller.stageTranscriptEdits(correction, for: item.id)
        controller.shutdown()
        XCTAssertEqual(try library.item(item.id)?.preferredTranscript?.segments, correction)
        XCTAssertTrue(controller.pendingTranscriptEdits.isEmpty)
    }

    @MainActor func testSavedModelSourceDoesNotFollowChangedPreferences() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "provenance" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        let profile = AudioWorkspaceController.localQwenProfile
        var item = AudioLibraryItem(title: "Old take", kind: .transcription)
        item.modelFamily = profile.family; item.localModelBookmark = Data([1])
        controller.preferences.localModels[profile.identity.alias] = Data([2])
        XCTAssertEqual(try controller.modelBookmark(for: profile, source: item), Data([1]))
        XCTAssertEqual(try controller.modelBookmark(for: profile), Data([2]))
        item.modelFamily = "whisper"
        XCTAssertThrowsError(try controller.modelBookmark(for: profile, source: item))
    }

    @MainActor func testPresetsAndRepeatsRestoreExperimentalBackendAndSavedInputs() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "recipe" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        var recipe = AudioRecipe()
        recipe.task = AudioTask.speechToText.rawValue
        recipe.modelID = "local-qwen3-asr"; recipe.experimentalBackend = "experimental_metal"
        controller.applyPreset(AudioPreset(name: "Local ASR", recipe: recipe))
        XCTAssertEqual(controller.page, .advanced)
        XCTAssertEqual(controller.selectedProfile?.family, "qwen3_asr")
        XCTAssertTrue(controller.allowExperimentalMetal)
        var take = AudioLibraryItem(title: "Saved script", kind: .voiceover)
        take.recipe.task = AudioTask.textToSpeech.rawValue
        take.recipe.text = "Keep the original script"
        take.recipe.experimentalBackend = "portable"
        try controller.save(take)
        controller.recipe.text = "Unsaved current form"
        controller.runAgain(take)
        XCTAssertEqual(controller.recipe.text, "Keep the original script")
        XCTAssertEqual(controller.task, .textToSpeech)
        XCTAssertEqual(controller.page, .advanced)
        XCTAssertTrue(controller.allowPortableExperiment)
    }

    @MainActor func testTaskSelectionUsesCompatibleLanguageAndVoice() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "defaults" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        controller.profiles = try AudioCatalog.profiles()
        controller.page = .advanced
        controller.recipe.language = "unsupported-language"
        controller.recipe.voice = "unsupported-voice"
        controller.selectTask(AudioTask.textToSpeech.rawValue)
        let capabilities = try XCTUnwrap(controller.selectedProfile?.capabilities)
        XCTAssertTrue(capabilities.languages.contains(controller.recipe.language))
        XCTAssertTrue(capabilities.voices.contains(controller.recipe.voice))
        controller.selectTask(AudioTask.speechToText.rawValue)
        XCTAssertEqual(controller.selectedProfile?.identity.task, .speechToText)
    }

    @MainActor func testFailedTakeSaveSurvivesRenameAndRetry() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "pending" }, migrateLegacyData: false)
        let session = try XCTUnwrap(try vault.prepareForLaunch())
        let assets = ManagedAssetStore(vault: vault)
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: assets)
        defer { controller.shutdown() }
        var item = AudioLibraryItem(title: "Take", kind: .music)
        item.status = .processing
        try controller.save(item)
        let asset = try assets.store(data: Data([1, 2, 3]), fileName: "take.wav")
        try session.database.execute("CREATE TRIGGER audio_fail BEFORE UPDATE ON private_records BEGIN SELECT RAISE(ABORT, 'simulated full disk'); END")
        XCTAssertThrowsError(try controller.persistMutation(item.id) {
            $0.status = .completed
            $0.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 44_100, channels: 1, frameCount: 1)]
        })
        XCTAssertEqual(controller.pendingSaves[item.id]?.clips.first?.assetReference, asset.storedReference)
        try session.database.execute("DROP TRIGGER audio_fail")
        controller.rename(item, title: "Keep this take")
        controller.retryAutosave()
        let reopened = try XCTUnwrap(library.load().first)
        XCTAssertEqual(reopened.title, "Keep this take")
        XCTAssertEqual(reopened.status, .completed)
        XCTAssertEqual(reopened.clips.first?.assetReference, asset.storedReference)
        XCTAssertTrue(controller.pendingSaves.isEmpty)
    }

    func testRecentDeletedAudioIsCollectedAfterPendingWriteGrace() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "grace" }, migrateLegacyData: false)
        let session = try XCTUnwrap(try vault.prepareForLaunch())
        let assets = ManagedAssetStore(vault: vault)
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let asset = try assets.store(data: Data([9]), fileName: "clip.wav")
        var item = AudioLibraryItem(title: "Delete", kind: .imported)
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 16_000, channels: 1, frameCount: 1)]
        try library.save(item)
        try library.delete(item.id)
        XCTAssertNotNil(try assets.descriptor(for: asset.storedReference))
        try session.database.execute("UPDATE assets SET created_at = 0")
        try library.collectUnusedAssets()
        XCTAssertNil(try assets.descriptor(for: asset.storedReference))
    }
}
