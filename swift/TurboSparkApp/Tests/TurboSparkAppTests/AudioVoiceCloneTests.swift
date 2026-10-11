import AVFoundation
import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Native voice cloning (MegaTTS3): the Voice Clone page follows the
/// catalog's `voice_clone` capability, the request carries the clone options,
/// and consent holds only for the exact reference region the user confirmed.
final class AudioVoiceCloneTests: XCTestCase {
    private func profile(family: String, operations: [String], canRun: Bool = true,
                         acknowledgments: [AudioAcknowledgment] = []) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .textToSpeech, alias: family + "-test", repository: "someone/" + family,
                                           revision: String(repeating: "0", count: 40), assetFingerprint: family),
            family: family, displayName: family,
            capabilities: AudioCapabilities(languages: ["en"], operations: operations, backend: "portable_cpu",
                                            cancellation: "diffusion_step", canRun: canRun),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "implemented_unqualified",
            acknowledgments: acknowledgments)
    }

    private func reference(duration: Double = 5) -> AudioVoiceReference {
        AudioVoiceReference(sourceID: UUID(), title: "me",
                            clips: [AudioLibraryClip(assetReference: "a", sampleRate: 24_000, channels: 1, frameCount: 240_000)],
                            duration: duration)
    }

    func testNativeCloningIsACapabilityNotAFamilyName() {
        let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"])
        let kokoro = profile(family: "kokoro", operations: ["synthesize"], canRun: false)
        XCTAssertTrue(AudioWorkspaceController.clonesNatively(mega))
        XCTAssertEqual(AudioWorkspaceController.speechMode(for: mega), .clone)
        XCTAssertFalse(AudioWorkspaceController.clonesNatively(kokoro))
        XCTAssertEqual(AudioWorkspaceController.speechMode(for: kokoro), .customVoice)
        // Qwen3-TTS clones in Swift MLX, never through a native session.
        XCTAssertFalse(AudioWorkspaceController.clonesNatively(profile(family: "qwen3_tts", operations: ["synthesize", "voice_clone"])))
        XCTAssertTrue(AudioWorkspaceController.runsOnDeclaredCPU(mega))
        XCTAssertFalse(AudioWorkspaceController.runsOnDeclaredCPU(kokoro), "a closed CPU profile still needs the explicit opt-in")
    }

    func testCloneRequestCarriesOptionsAndConsentOnlyForTheConsentedRegion() {
        let voice = reference()
        var speech = AudioSpeechSettings()
        speech.mode = .clone
        speech.reference = voice
        speech.instruction = "whisper"
        speech.referenceText = "hello"
        var options = AudioCloneOptions()
        options.steps = 12
        options.intelligibility = 2
        options.similarity = 4
        speech.cloneOptions = options
        var base = AudioRequest(task: .textToSpeech)
        base.steps = 30
        let unconsented = AudioWorkspaceController.nativeCloneRequest(base, speech: speech)
        XCTAssertFalse(unconsented.consent, "no consent was given")
        XCTAssertEqual(unconsented.steps, 12)
        XCTAssertEqual(unconsented.pW, 2)
        XCTAssertEqual(unconsented.tW, 4)
        XCTAssertEqual(unconsented.referenceSampleRate, 24_000)
        XCTAssertNil(unconsented.instruction)
        XCTAssertNil(unconsented.referenceText)
        speech.cloneOptions?.consentedReference = voice
        XCTAssertTrue(AudioWorkspaceController.nativeCloneRequest(base, speech: speech).consent)
        // A different trim is a different region: consent lapses.
        speech.reference?.start = 1
        XCTAssertFalse(AudioWorkspaceController.nativeCloneRequest(base, speech: speech).consent)
        // Without saved options the model defaults apply.
        speech.cloneOptions = nil
        XCTAssertEqual(AudioWorkspaceController.nativeCloneRequest(base, speech: speech).steps, CloneQuality.balanced.steps)
    }

    /// The pinned profile declares Metal, so the app opens it without the
    /// portable opt-in and the runtime never falls back to the CPU on its own.
    func testCatalogMegaTTS3RunsOnMetalWithoutThePortableOptIn() throws {
        let mega = try XCTUnwrap(try AudioCatalog.profiles().first { $0.family == "megatts3" })
        XCTAssertEqual(mega.capabilities.backend, "metal")
        XCTAssertTrue(mega.capabilities.canRun)
        XCTAssertFalse(AudioWorkspaceController.runsOnDeclaredCPU(mega))
    }

    /// The presets mirror the audio crate's `megatts3::Quality`, and a count
    /// between them is custom, which the picker shows as its own segment.
    func testQualityPresetsMapToStepCountsAndCustomCountsToNone() {
        XCTAssertEqual(CloneQuality.allCases.map(\.steps), [8, 16, 32])
        for preset in CloneQuality.allCases { XCTAssertEqual(CloneQuality(steps: preset.steps), preset) }
        XCTAssertNil(CloneQuality(steps: 20))
        XCTAssertEqual(CloneQuality(steps: AudioCloneOptions().steps), .balanced)
    }

    func testSpeechSettingsSavedBeforeCloneOptionsStillDecode() throws {
        let saved = #"{"mode":"clone","instruction":"","referenceText":"","embeddingOnly":false,"temperature":0.9,"maxTokens":2048}"#
        let settings = try JSONDecoder().decode(AudioSpeechSettings.self, from: Data(saved.utf8))
        XCTAssertNil(settings.cloneOptions)
        XCTAssertEqual(settings.mode, .clone)
    }

    @MainActor func testInstallTermsLocalizeKnownIdsAndFallBackToTheSummary() {
        let terms = profile(family: "megatts3", operations: ["voice_clone"], acknowledgments: [
            AudioAcknowledgment(id: "megatts3_community_encoder", summary: "catalog wording"),
            AudioAcknowledgment(id: "other_terms", summary: "Other terms apply."),
        ])
        let text = AudioModelPickerView.acknowledgmentText(terms)
        XCTAssertFalse(text.contains("catalog wording"))
        XCTAssertTrue(text.contains("community re-upload"))
        XCTAssertTrue(text.contains("Other terms apply."))
    }

    @MainActor func testVoiceClonePageListsNativeClonersAndGatesOnConsent() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "clone-consent" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"])
        controller.profiles = [mega, profile(family: "kokoro", operations: ["synthesize"], canRun: false)]
        controller.selectPage(.voiceClone)
        XCTAssertEqual(controller.availableProfiles.map(\.family), ["megatts3"])
        controller.selectModel(mega.identity.alias)
        XCTAssertTrue(controller.selectedProfileClonesNatively)
        XCTAssertNotNil(controller.speechSettings.cloneOptions, "a cloning profile gets its defaults")
        controller.recipe.text = "Hello."
        XCTAssertNotNil(controller.speechValidationMessage, "no reference yet")
        controller.speechSettings.reference = reference()
        XCTAssertEqual(controller.speechValidationMessage,
                       String(localized: "Confirm you have the right to use this voice.", bundle: .module))
        controller.setCloneConsent(true)
        XCTAssertTrue(controller.cloneConsentGiven)
        XCTAssertNil(controller.speechValidationMessage)
        // Retrimming the reference withdraws the confirmation.
        controller.speechSettings.reference?.duration = 4
        XCTAssertFalse(controller.cloneConsentGiven)
        XCTAssertNotNil(controller.speechValidationMessage)
        // The voiceover page does not offer a model that needs a reference.
        controller.selectPage(.voiceover)
        XCTAssertFalse(controller.availableProfiles.contains { $0.family == "megatts3" })
    }

    /// The Voice Clone page clones a real voice end to end: the catalog
    /// profile, a local model folder, an imported reference, consent, the
    /// native session on Metal (the backend the catalog declares), and the
    /// encrypted take.
    /// Set TURBOSPARK_TEST_MEGATTS3 (a converted model folder) and
    /// TURBOSPARK_TEST_VOICE_REFERENCE (a WAV of 1 to 28 seconds).
    @MainActor func testRealVoiceClonePageClonesWithMegaTTS3() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["TURBOSPARK_TEST_MEGATTS3"],
              let referencePath = environment["TURBOSPARK_TEST_VOICE_REFERENCE"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_MEGATTS3 and TURBOSPARK_TEST_VOICE_REFERENCE for the real MegaTTS3 app gate")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "real-megatts3" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let assets = ManagedAssetStore(vault: vault)
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: assets)
        defer { controller.shutdown() }
        var clips: [AudioLibraryClip] = []
        try AudioMediaIO.importFile(URL(fileURLWithPath: referencePath)) { chunk in
            let asset = try assets.store(data: chunk.wav, fileName: "reference.wav")
            clips.append(AudioLibraryClip(assetReference: asset.storedReference, sampleRate: chunk.sampleRate,
                channels: chunk.channels, frameCount: chunk.frameCount, offset: chunk.offset, source: chunk.source))
        }
        let duration = clips.map { $0.offset + $0.duration }.max() ?? 0

        controller.profiles = try AudioCatalog.profiles()
        controller.selectPage(.voiceClone)
        let mega = try XCTUnwrap(controller.availableProfiles.first { $0.family == "megatts3" })
        controller.selectModel(mega.identity.alias)
        controller.preferences.localModels[mega.identity.alias] = try URL(fileURLWithPath: modelPath)
            .bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        controller.recipe.text = "Hello world, this is a short test."
        controller.recipe.seed = 7
        controller.speechSettings.reference = AudioVoiceReference(sourceID: UUID(), title: "Local reference",
                                                                  clips: clips, duration: min(28, duration))
        controller.speechSettings.cloneOptions?.steps = 4
        XCTAssertFalse(controller.canRun, "consent is required before a clone runs")
        controller.setCloneConsent(true)
        XCTAssertNil(controller.speechValidationMessage)
        XCTAssertTrue(controller.canRun)

        controller.run()
        await controller.jobTask?.value
        XCTAssertNil(controller.error)
        let take = try XCTUnwrap(controller.items.first { $0.modelFamily == "megatts3" })
        XCTAssertEqual(take.status, .completed)
        XCTAssertEqual(take.resolvedSeed, 7)
        let clip = try XCTUnwrap(take.clips.first)
        XCTAssertEqual(clip.sampleRate, 24_000)
        XCTAssertEqual(clip.channels, 1)
        XCTAssertGreaterThan(clip.duration, 1)
        XCTAssertEqual(try library.load().first { $0.id == take.id }?.recipe.speech?.reference, controller.speechSettings.reference)
        let export = root.appendingPathComponent("clone.wav")
        try AudioMediaIO.render(take.clips, store: assets, to: export)
        XCTAssertEqual(try AVAudioFile(forReading: export).length, clip.frameCount)
    }
}
