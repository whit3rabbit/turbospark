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
        let unconsented = AudioWorkspaceController.nativeCloneRequest(base, family: "megatts3", speech: speech)
        XCTAssertFalse(unconsented.consent, "no consent was given")
        XCTAssertEqual(unconsented.steps, 12)
        XCTAssertEqual(unconsented.pW, 2)
        XCTAssertEqual(unconsented.tW, 4)
        XCTAssertEqual(unconsented.referenceSampleRate, 24_000)
        XCTAssertNil(unconsented.instruction)
        XCTAssertNil(unconsented.referenceText)
        speech.cloneOptions?.consentedReference = voice
        XCTAssertTrue(AudioWorkspaceController.nativeCloneRequest(base, family: "megatts3", speech: speech).consent)
        // A different trim is a different region: consent lapses.
        speech.reference?.start = 1
        XCTAssertFalse(AudioWorkspaceController.nativeCloneRequest(base, family: "megatts3", speech: speech).consent)
        // Without saved options the model defaults apply.
        speech.cloneOptions = nil
        XCTAssertEqual(AudioWorkspaceController.nativeCloneRequest(base, family: "megatts3", speech: speech).steps, CloneQuality.balanced.steps)
        // OmniVoice refuses MegaTTS3's guidance weights, so they never
        // cross the wire; the transcript and instruction pass through.
        speech.cloneOptions = options
        speech.reference?.start = 0
        speech.cloneOptions?.consentedReference = voice
        let omni = AudioWorkspaceController.nativeCloneRequest(base, family: "omnivoice", speech: speech)
        XCTAssertNil(omni.pW)
        XCTAssertNil(omni.tW)
        XCTAssertEqual(omni.steps, 12)
        XCTAssertEqual(omni.referenceSampleRate, 24_000)
        XCTAssertTrue(omni.consent)
    }

    /// Drama 3 refuses speed other than 1.0, an instruction and the guidance
    /// weights, and a clone needs the reference's transcript
    /// (`fish_drama3_request` in the runtime). Checked on the wire too, since
    /// a nil field must be absent rather than null.
    func testDrama3RequestCarriesTheTranscriptAndNothingTheRuntimeRefuses() throws {
        let voice = reference()
        var speech = AudioSpeechSettings()
        speech.mode = .clone
        speech.reference = voice
        speech.instruction = "whisper"
        speech.referenceText = "  What I said in the recording.\n"
        var options = AudioCloneOptions()
        options.steps = 12
        options.intelligibility = 2
        options.similarity = 4
        options.consentedReference = voice
        speech.cloneOptions = options
        var base = AudioRequest(task: .textToSpeech)
        base.steps = 30
        base.speed = 1.3
        base.instruction = "calm"
        // MegaTTS3's weights, as a request that took its path would carry.
        base.pW = 1.4
        base.tW = 3
        let drama = AudioWorkspaceController.nativeCloneRequest(base, family: "fish_drama3", speech: speech)
        XCTAssertEqual(drama.referenceText, "What I said in the recording.")
        XCTAssertTrue(drama.consent)
        XCTAssertEqual(drama.referenceSampleRate, 24_000)
        XCTAssertEqual(drama.speed, 1)
        XCTAssertNil(drama.pW)
        XCTAssertNil(drama.tW)
        XCTAssertNil(drama.instruction)
        XCTAssertNil(drama.steps)

        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        let wire = try XCTUnwrap(try JSONSerialization.jsonObject(with: encoder.encode(drama)) as? [String: Any])
        for refused in ["p_w", "t_w", "instruction", "steps"] { XCTAssertNil(wire[refused], refused) }
        XCTAssertEqual(wire["reference_text"] as? String, "What I said in the recording.")
        XCTAssertEqual(wire["speed"] as? Double, 1)
        XCTAssertEqual(wire["consent"] as? Bool, true)

        // A blank transcript is no transcript, and consent still follows the region.
        speech.referenceText = " \n"
        speech.cloneOptions?.consentedReference = nil
        let blank = AudioWorkspaceController.nativeCloneRequest(base, family: "fish_drama3", speech: speech)
        XCTAssertNil(blank.referenceText)
        XCTAssertFalse(blank.consent)
    }

    /// The pinned profile declares Metal, so the app opens it without the
    /// portable opt-in and the runtime never falls back to the CPU on its own.
    func testCatalogMegaTTS3RunsOnMetalWithoutThePortableOptIn() throws {
        let mega = try XCTUnwrap(try AudioCatalog.profiles().first { $0.family == "megatts3" })
        XCTAssertEqual(mega.capabilities.backend, "metal")
        XCTAssertTrue(mega.capabilities.canRun)
        XCTAssertFalse(AudioWorkspaceController.runsOnDeclaredCPU(mega))
    }

    /// The pinned OmniVoice profile declares Metal, so the app opens it without the
    /// portable opt-in and offers it on the Voice Cloning page.
    func testCatalogOmniVoiceRunsOnMetalWithoutThePortableOptIn() throws {
        let omni = try XCTUnwrap(try AudioCatalog.profiles().first { $0.family == "omnivoice" })
        XCTAssertEqual(omni.capabilities.backend, "metal")
        XCTAssertTrue(omni.capabilities.canRun)
        XCTAssertFalse(AudioWorkspaceController.runsOnDeclaredCPU(omni))
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
            AudioAcknowledgment(id: "omnivoice_higgs_tokenizer", summary: "catalog wording"),
            AudioAcknowledgment(id: "other_terms", summary: "Other terms apply."),
        ])
        let text = AudioModelPickerView.acknowledgmentText(terms)
        XCTAssertFalse(text.contains("catalog wording"))
        XCTAssertTrue(text.contains("community re-upload"))
        XCTAssertTrue(text.contains("community-licensed"))
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

    /// Drama 3 asks for consent and for the reference's transcript, which
    /// MegaTTS3 never does, and Qwen3's embedding-only choice does not
    /// excuse it.
    @MainActor func testVoiceClonePageAsksDrama3ForTheReferenceTranscript() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "clone-drama3" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        let drama = profile(family: "fish_drama3", operations: ["synthesize", "voice_clone"])
        let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"])
        controller.profiles = [drama, mega]
        controller.selectPage(.voiceClone)
        XCTAssertEqual(controller.availableProfiles.map(\.family), ["fish_drama3", "megatts3"])
        controller.selectModel(drama.identity.alias)
        XCTAssertTrue(controller.selectedProfileClonesNatively)
        XCTAssertTrue(controller.selectedProfileNeedsReferenceTranscript)
        controller.recipe.text = "Hello."
        controller.speechSettings.reference = reference()
        controller.setCloneConsent(true)
        controller.speechSettings.embeddingOnly = true
        XCTAssertEqual(controller.speechValidationMessage,
                       String(localized: "Enter the exact transcript of the selected reference region.", bundle: .module))
        XCTAssertEqual(controller.voiceCloneSteps.map(\.step), [.voice, .consent, .transcript, .script, .generate])
        XCTAssertEqual(controller.voiceCloneMissingSteps, [.transcript])
        controller.speechSettings.referenceText = "Hi there."
        XCTAssertNil(controller.speechValidationMessage)
        XCTAssertEqual(controller.voiceCloneMissingSteps, [])

        // MegaTTS3 aligns the reference itself: no transcript step.
        controller.speechSettings.referenceText = ""
        controller.selectModel(mega.identity.alias)
        XCTAssertFalse(controller.selectedProfileNeedsReferenceTranscript)
        controller.setCloneConsent(true)
        XCTAssertNil(controller.speechValidationMessage)
        XCTAssertFalse(controller.voiceCloneSteps.contains { $0.step == .transcript })

        // Drama 3 can speak without a reference, but Voiceover's run path
        // still treats every native cloner as a clone, so it stays off.
        controller.selectPage(.voiceover)
        XCTAssertFalse(controller.availableProfiles.contains { $0.family == "fish_drama3" })
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

    /// The Voice Clone page clones with Drama 3 end to end: the catalog
    /// profile, a local install folder, an imported reference with its
    /// transcript, consent, the Metal session, and the saved 44.1 kHz take.
    /// Set TURBOSPARK_TEST_FISH_DRAMA3 (an install folder from
    /// `turbospark-model pull-drama3`), TURBOSPARK_TEST_VOICE_REFERENCE (a
    /// WAV of 1 to 30 seconds) and TURBOSPARK_TEST_VOICE_REFERENCE_TEXT (its
    /// exact words).
    @MainActor func testRealVoiceClonePageClonesWithDrama3() async throws {
        let environment = ProcessInfo.processInfo.environment
        guard let modelPath = environment["TURBOSPARK_TEST_FISH_DRAMA3"],
              let referencePath = environment["TURBOSPARK_TEST_VOICE_REFERENCE"],
              let referenceText = environment["TURBOSPARK_TEST_VOICE_REFERENCE_TEXT"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_FISH_DRAMA3, TURBOSPARK_TEST_VOICE_REFERENCE and TURBOSPARK_TEST_VOICE_REFERENCE_TEXT for the real Drama 3 app gate")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "real-drama3" }, migrateLegacyData: false)
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
        let drama = try XCTUnwrap(controller.availableProfiles.first { $0.family == "fish_drama3" })
        controller.selectModel(drama.identity.alias)
        controller.preferences.localModels[drama.identity.alias] = try URL(fileURLWithPath: modelPath)
            .bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        controller.recipe.text = "Hello world, this is a short test."
        controller.recipe.seed = 7
        controller.speechSettings.reference = AudioVoiceReference(sourceID: UUID(), title: "Local reference",
                                                                  clips: clips, duration: min(30, duration))
        controller.setCloneConsent(true)
        XCTAssertFalse(controller.canRun, "Drama 3 needs the reference transcript before a clone runs")
        controller.speechSettings.referenceText = referenceText
        XCTAssertNil(controller.speechValidationMessage)
        XCTAssertTrue(controller.canRun)

        controller.run()
        await controller.jobTask?.value
        XCTAssertNil(controller.error)
        let take = try XCTUnwrap(controller.items.first { $0.modelFamily == "fish_drama3" })
        XCTAssertEqual(take.status, .completed)
        XCTAssertEqual(take.resolvedSeed, 7)
        let clip = try XCTUnwrap(take.clips.first)
        XCTAssertEqual(clip.sampleRate, 44_100)
        XCTAssertEqual(clip.channels, 1)
        XCTAssertGreaterThan(clip.duration, 1)
    }
}
