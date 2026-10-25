import AVFoundation
import Foundation
import MLX
import Qwen3TTS
import TurboSpark
import XCTest
@testable import TurboSparkApp

final class AudioVoiceTests: XCTestCase {
    @MainActor func testModesSelectOnlyCompatibleProfilesAndRememberModels() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "voices" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        controller.profiles = try AudioCatalog.profiles()
        controller.selectPage(.voiceClone)
        // Two Qwen3-TTS Base profiles clone in Swift MLX; MegaTTS3, both OmniVoice
        // rows and Drama 3 clone natively on Metal, each with its own request shape.
        XCTAssertEqual(Set(controller.availableProfiles.map(\.identity.alias)), [
            "qwen3-tts-12hz-06b-base-8bit", "qwen3-tts-12hz-17b-base-8bit",
            "megatts3-f32", "omnivoice-8bit", "omnivoice-bf16", "fish-drama3-4bit",
        ])
        XCTAssertTrue(controller.availableProfiles.allSatisfy { AudioWorkspaceController.speechMode(for: $0) == .clone })
        let base = try XCTUnwrap(controller.availableProfiles.last { $0.family == "qwen3_tts" })
        controller.selectModel(base.identity.alias)
        controller.recipe.text = "A new take."
        XCTAssertNotNil(controller.speechValidationMessage)
        controller.selectPage(.voiceDesign)
        XCTAssertEqual(controller.availableProfiles.count, 1)
        XCTAssertEqual(controller.speechSettings.mode, .design)
        controller.speechSettings.instruction = "Warm and calm"
        XCTAssertNil(controller.speechValidationMessage)
        controller.selectPage(.voiceover)
        // Named-voice models, plus both OmniVoice rows and Drama 3, which
        // speak without a reference here. MegaTTS3 cannot, and the Qwen3
        // Base and VoiceDesign rows belong to the other pages.
        XCTAssertEqual(Set(controller.availableProfiles.map(\.identity.alias)), [
            "kokoro-82m-bf16", "qwen3-tts-12hz-06b-customvoice-8bit", "qwen3-tts-12hz-17b-customvoice-8bit",
            "omnivoice-8bit", "omnivoice-bf16", "fish-drama3-4bit",
        ])
        XCTAssertTrue(controller.availableProfiles.allSatisfy {
            AudioWorkspaceController.speechMode(for: $0) == .customVoice
                || AudioWorkspaceController.referenceFreeSpeechFamilies.contains($0.family)
        })
        // Catalog order keeps Qwen3-TTS the default ahead of the native cloners.
        XCTAssertEqual(controller.selectedProfile?.family, "qwen3_tts")
        XCTAssertTrue(controller.selectedProfileIsRunnable)
        controller.selectPage(.voiceClone)
        XCTAssertEqual(controller.recipe.modelID, base.identity.alias)
        XCTAssertEqual(controller.recipe.text, "A new take.")
        controller.selectPage(.voiceDesign)
        XCTAssertEqual(controller.speechSettings.instruction, "Warm and calm")
    }

    func testReferenceIsPinnedByPreferencesAndSavedTakeAfterOriginalDeletion() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "voice-pins" }, migrateLegacyData: false)
        let session = try XCTUnwrap(try vault.prepareForLaunch())
        let assets = ManagedAssetStore(vault: vault)
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let bytes = try AudioMediaIO.wav(samples: Array(repeating: Float(0.1), count: 24_000), sampleRate: 24_000, channels: 1)
        let asset = try assets.store(data: bytes, fileName: "reference.wav")
        var source = AudioLibraryItem(title: "Reference", kind: .imported)
        source.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 24_000, channels: 1, frameCount: 24_000)]
        try library.save(source)
        let reference = AudioVoiceReference(sourceID: source.id, title: source.title, clips: source.clips, duration: 1)
        var settings = AudioSpeechSettings(); settings.mode = .clone; settings.reference = reference
        var preferences = AudioWorkspacePreferences(); preferences.recipe.speech = settings
        try library.savePreferences(preferences)
        try session.database.execute("UPDATE assets SET created_at = 0")
        try library.delete(source.id)
        XCTAssertNotNil(try assets.descriptor(for: asset.storedReference))
        var take = AudioLibraryItem(title: "Cloned take", kind: .voiceover); take.status = .draft; take.recipe = preferences.recipe
        try library.save(take)
        try library.savePreferences(AudioWorkspacePreferences())
        try library.collectUnusedAssets()
        XCTAssertEqual(try library.load().first?.recipe.speech?.reference, reference)
        XCTAssertNotNil(try assets.descriptor(for: asset.storedReference))
        try library.delete(take.id)
        XCTAssertNil(try assets.descriptor(for: asset.storedReference))
    }

    func testStereoReferenceTrimResamplesTo24kMono() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let frames = 48_000 * 3
        var samples = [Float](); samples.reserveCapacity(frames * 2)
        for i in 0..<frames { samples.append(i < 48_000 ? 0 : 0.2); samples.append(i < 48_000 ? 0 : 0.4) }
        try AudioMediaIO.wav(samples: samples, sampleRate: 48_000, channels: 2).write(to: url)
        let clip = AudioLibraryClip(assetReference: "reference", sampleRate: 48_000, channels: 2, frameCount: Int64(frames))
        let reference = try AudioMediaIO.readMono24k([clip], start: 1, duration: 1, resolve: { _ in url })
        XCTAssertEqual(reference.count, 24_000)
        XCTAssertEqual(reference[12_000], 0.3, accuracy: 0.001)
        XCTAssertThrowsError(try AudioMediaIO.readMono24k([clip], start: 0, duration: 31, resolve: { _ in url }))
    }

    func testCancelledSpeechWorkerDrainsBeforeReturning() async throws {
        let job = try Qwen3TTSSpeechJob()
        let worker = try job.start {
            while true { try Task.checkCancellation(); try await Task.sleep(for: .milliseconds(10)) }
        }
        job.stopAndDrain()
        do { _ = try await worker.value; XCTFail("Cancelled worker returned audio") }
        catch { XCTAssertTrue(error is CancellationError) }
        XCTAssertThrowsError(try job.start { AudioResult() })
    }

    func testRealCloningThroughAppWorkerAndEncryptedLibrary() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["TURBOSPARK_TEST_QWEN3_TTS"],
              let referencePath = environment["TURBOSPARK_TEST_VOICE_REFERENCE"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_QWEN3_TTS and TURBOSPARK_TEST_VOICE_REFERENCE for the real Base checkpoint gate")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "real-voice" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let assets = ManagedAssetStore(vault: vault)
        var clips: [AudioLibraryClip] = []
        try AudioMediaIO.importFile(URL(fileURLWithPath: referencePath)) { chunk in
            let asset = try assets.store(data: chunk.wav, fileName: "reference.wav")
            clips.append(AudioLibraryClip(assetReference: asset.storedReference, sampleRate: chunk.sampleRate,
                channels: chunk.channels, frameCount: chunk.frameCount, offset: chunk.offset, source: chunk.source))
        }
        let duration = clips.map { $0.offset + $0.duration }.max() ?? 0
        var settings = AudioSpeechSettings(); settings.mode = .clone
        settings.reference = AudioVoiceReference(sourceID: UUID(), title: "Local reference", clips: clips, duration: min(30, duration))
        settings.referenceText = "Flying is only possible with wings this big. Clear skies ahead!"
        settings.maxTokens = 512
        var recipe = AudioRecipe(); recipe.speech = settings; recipe.seed = 42; recipe.language = "en"
        recipe.text = "The sky is clear. We are ready for the next flight."; recipe.task = "text_to_speech"
        let profile = try XCTUnwrap(try AudioCatalog.profiles().first { $0.identity.alias == "qwen3-tts-12hz-06b-base-8bit" })
        let bookmark = try URL(fileURLWithPath: modelPath).bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        let job = try Qwen3TTSSpeechJob()
        let savedRecipe = recipe
        let worker = try job.start {
            try await Qwen3TTSSpeechJob.generate(recipe: savedRecipe, profile: profile, bookmark: bookmark, installedPath: nil,
                assets: assets, lifetime: AudioOperationLifetime(), openToken: job.openToken, report: { _ in })
        }
        let result = try await worker.value
        XCTAssertEqual(MLX.Memory.cacheMemory, 0, "A completed speech job must release cached GPU buffers")
        let wrongSize = try XCTUnwrap(try AudioCatalog.profiles().first { $0.identity.alias == "qwen3-tts-12hz-17b-base-8bit" })
        do {
            _ = try await Qwen3TTSSpeechJob.generate(recipe: savedRecipe, profile: wrongSize, bookmark: bookmark,
                installedPath: nil, assets: assets, lifetime: AudioOperationLifetime(), openToken: job.openToken, report: { _ in })
            XCTFail("A 0.6B local checkpoint must not be presented as a 1.7B model")
        } catch {
            XCTAssertEqual(error as? Qwen3TTSError, .incompatibleProfile)
        }
        let pcm = try XCTUnwrap(result.pcm)
        XCTAssertEqual(pcm.format.sampleRate, 24_000); XCTAssertEqual(pcm.format.channels, 1)
        XCTAssertTrue(pcm.samples.allSatisfy(\.isFinite)); XCTAssertGreaterThan(pcm.durationSeconds, 1)
        let data = try AudioMediaIO.wav(samples: pcm.samples, sampleRate: 24_000, channels: 1)
        let asset = try assets.store(data: data, fileName: "cloned.wav")
        var item = AudioLibraryItem(title: "Real clone", kind: .voiceover); item.recipe = recipe
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 24_000, channels: 1, frameCount: Int64(pcm.samples.count))]
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        try library.save(item)
        XCTAssertEqual(try library.load().first?.recipe.speech?.reference, settings.reference)
        let export = root.appendingPathComponent("cloned.wav")
        try AudioMediaIO.render(item.clips, store: assets, to: export)
        XCTAssertEqual(try AVAudioFile(forReading: export).length, Int64(pcm.samples.count))
        job.stopAndDrain()
    }
}
