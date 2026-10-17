import AVFoundation
import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Native voice cloning (MegaTTS3): the Voice Clone page follows the
/// catalog's `voice_clone` capability, the request carries the clone options,
/// and consent holds only for the exact reference region the user confirmed.
final class AudioVoiceCloneTests: XCTestCase {
    private func profile(family: String, operations: [String], canRun: Bool = true, backend: String = "portable_cpu",
                         acknowledgments: [AudioAcknowledgment] = []) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .textToSpeech, alias: family + "-test", repository: "someone/" + family,
                                           revision: String(repeating: "0", count: 40), assetFingerprint: family),
            family: family, displayName: family,
            capabilities: AudioCapabilities(languages: ["en"], operations: operations, backend: backend,
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

    /// On Voiceover, Drama 3 and OmniVoice speak with no reference: no
    /// consent, reference rate or transcript, and only the fields each
    /// runtime accepts with no PCM (`fish_drama3_request`,
    /// `omnivoice_request`). Checked on the wire, where a nil field must be
    /// absent. MegaTTS3 cannot speak without a reference, so it always clones.
    @MainActor func testVoiceoverRequestSendsNoReferenceInEachFamilysShape() throws {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        func wire(_ request: AudioRequest) throws -> [String: Any] {
            try XCTUnwrap(try JSONSerialization.jsonObject(with: encoder.encode(request)) as? [String: Any])
        }
        let refused = ["p_w", "t_w", "instruction", "steps", "reference_text", "reference_sample_rate"]
        // Every clone field set, as a request that took a clone path would carry.
        var dirty = AudioRequest(task: .textToSpeech)
        dirty.speed = 1.3
        dirty.steps = 12
        dirty.instruction = "calm"
        dirty.referenceText = "What I said."
        dirty.pW = 1.4
        dirty.tW = 3
        dirty.referenceSampleRate = 24_000
        dirty.consent = true
        for family in ["fish_drama3", "omnivoice"] {
            let request = AudioWorkspaceController.referenceFreeSpeechRequest(dirty, family: family)
            XCTAssertFalse(request.consent, family)
            let fields = try wire(request)
            for name in refused { XCTAssertNil(fields[name], family + " " + name) }
            XCTAssertEqual(fields["consent"] as? Bool, false, family)
        }
        XCTAssertEqual(AudioWorkspaceController.referenceFreeSpeechRequest(dirty, family: "fish_drama3").speed, 1,
                       "Drama 3 refuses any other speed")
        XCTAssertEqual(AudioWorkspaceController.referenceFreeSpeechRequest(dirty, family: "omnivoice").speed, 1.3,
                       "OmniVoice keeps the slider's speed")

        // The run routes by the recipe's speech mode: none on Voiceover.
        var recipe = AudioRecipe()
        recipe.text = "Hello there."
        recipe.speed = 1.3
        recipe.steps = 12 // the music page's count, which the engine must not take
        recipe.seed = 7
        let drama = profile(family: "fish_drama3", operations: ["synthesize", "voice_clone"], backend: "metal")
        let omni = profile(family: "omnivoice", operations: ["synthesize", "voice_clone"], backend: "metal")
        let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"], backend: "metal")
        for model in [drama, omni] {
            XCTAssertFalse(AudioWorkspaceController.runClonesNatively(model, speech: recipe.speech), model.family)
            let request = AudioWorkspaceController.runRequest(recipe, task: .textToSpeech, profile: model)
            XCTAssertFalse(request.consent, model.family)
            XCTAssertNil(request.steps, model.family)
            XCTAssertNil(request.referenceSampleRate, model.family)
            XCTAssertEqual(request.text, "Hello there.", model.family)
            XCTAssertEqual(request.seed, 7, model.family)
        }
        XCTAssertEqual(AudioWorkspaceController.runRequest(recipe, task: .textToSpeech, profile: drama).speed, 1)
        XCTAssertEqual(AudioWorkspaceController.runRequest(recipe, task: .textToSpeech, profile: omni).speed, 1.3)
        XCTAssertTrue(AudioWorkspaceController.runClonesNatively(mega, speech: recipe.speech))
        XCTAssertEqual(AudioWorkspaceController.runRequest(recipe, task: .textToSpeech, profile: mega).referenceSampleRate, 24_000)

        // In clone mode (Voice Cloning, Advanced) both still clone.
        var speech = AudioSpeechSettings()
        speech.mode = .clone
        speech.reference = reference()
        speech.referenceText = "Hi."
        var options = AudioCloneOptions()
        options.consentedReference = speech.reference
        speech.cloneOptions = options
        recipe.speech = speech
        for model in [drama, omni] {
            XCTAssertTrue(AudioWorkspaceController.runClonesNatively(model, speech: recipe.speech), model.family)
            let clone = AudioWorkspaceController.runRequest(recipe, task: .textToSpeech, profile: model)
            XCTAssertTrue(clone.consent, model.family)
            XCTAssertEqual(clone.referenceSampleRate, 24_000, model.family)
        }
    }

    /// Voiceover lists Drama 3 and OmniVoice and runs them without a
    /// reference or consent, in a voice the model picks. MegaTTS3 needs a
    /// reference and stays off. Voice Cloning still asks both for one.
    @MainActor func testVoiceoverSpeaksWithDrama3AndOmniVoiceWithoutAReference() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "voiceover-native" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        let drama = profile(family: "fish_drama3", operations: ["synthesize", "voice_clone"], backend: "metal")
        let omni = profile(family: "omnivoice", operations: ["synthesize", "voice_clone"], backend: "metal")
        let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"], backend: "metal")
        controller.profiles = [drama, omni, mega]
        controller.selectPage(.voiceover)
        XCTAssertEqual(controller.availableProfiles.map(\.family), ["fish_drama3", "omnivoice"])
        for model in [drama, omni] {
            controller.selectModel(model.identity.alias)
            controller.preferences.localModels[model.identity.alias] = Data([1])
            XCTAssertFalse(controller.selectedProfileClonesNatively, model.family)
            XCTAssertTrue(controller.selectedProfileSpeaksWithoutReference, model.family)
            XCTAssertNil(controller.recipe.speech, "no clone mode, so the run reads no reference")
            controller.recipe.text = "Hello."
            XCTAssertNil(controller.speechValidationMessage, model.family)
            XCTAssertTrue(controller.canRun, model.family + " needs no reference or consent")
            controller.recipe.text = String(repeating: "a", count: AudioWorkspaceController.speechTextCharacterLimit + 1)
            XCTAssertEqual(controller.speechValidationMessage,
                           String(localized: "Split speech text longer than 6,000 characters into sections.", bundle: .module))
        }

        controller.recipe.text = "Hello."
        controller.selectPage(.voiceClone)
        XCTAssertEqual(Set(controller.availableProfiles.map(\.family)), ["fish_drama3", "omnivoice", "megatts3"])
        for model in [drama, omni] {
            controller.selectModel(model.identity.alias)
            XCTAssertTrue(controller.selectedProfileClonesNatively, model.family)
            XCTAssertFalse(controller.selectedProfileSpeaksWithoutReference, model.family)
            XCTAssertEqual(controller.speechValidationMessage,
                           String(localized: "Choose 1 to 30 seconds of reference speech.", bundle: .module))
        }
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

        // Voiceover lists Drama 3 too, speaking without a reference, and
        // asks for no transcript there.
        controller.selectPage(.voiceover)
        XCTAssertEqual(controller.availableProfiles.map(\.family), ["fish_drama3"])
        XCTAssertNil(controller.speechValidationMessage)
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

    /// Voiceover speaks with Drama 3 end to end and no reference: the
    /// catalog profile, a local install folder, the Metal session and the
    /// saved 44.1 kHz take, which stays a Voiceover take.
    /// Set TURBOSPARK_TEST_FISH_DRAMA3 (an install folder from
    /// `turbospark-model pull-drama3`).
    @MainActor func testRealVoiceoverSpeaksWithDrama3WithoutAReference() async throws {
        guard let modelPath = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_FISH_DRAMA3"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_FISH_DRAMA3 for the real Drama 3 Voiceover gate")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "real-drama3-voiceover" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }

        controller.profiles = try AudioCatalog.profiles()
        controller.selectPage(.voiceover)
        let drama = try XCTUnwrap(controller.availableProfiles.first { $0.family == "fish_drama3" })
        controller.selectModel(drama.identity.alias)
        controller.preferences.localModels[drama.identity.alias] = try URL(fileURLWithPath: modelPath)
            .bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        controller.recipe.text = "Hello world, this is a short test."
        controller.recipe.seed = 7
        XCTAssertTrue(controller.selectedProfileSpeaksWithoutReference)
        XCTAssertNil(controller.speechValidationMessage)
        XCTAssertTrue(controller.canRun)

        controller.run()
        await controller.jobTask?.value
        XCTAssertNil(controller.error)
        let take = try XCTUnwrap(controller.items.first { $0.modelFamily == "fish_drama3" })
        XCTAssertEqual(take.status, .completed)
        XCTAssertEqual(take.resolvedSeed, 7)
        XCTAssertNil(take.recipe.speech, "a Voiceover take, not a clone")
        let clip = try XCTUnwrap(take.clips.first)
        XCTAssertEqual(clip.sampleRate, 44_100)
        XCTAssertEqual(clip.channels, 1)
        XCTAssertGreaterThan(clip.duration, 1)
    }
}
