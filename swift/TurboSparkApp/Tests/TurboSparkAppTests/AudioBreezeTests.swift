import Combine
import Foundation
import Metal
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Breeze TTS 2 runs in the vendored Swift MLX package. One profile serves
/// Voice Cloning (clone and direction), Voice Design and Voiceover, and the
/// request comes from the shared native-clone path, so consent and the
/// reference transcript are enforced the same way as for the Rust cloners.
final class AudioBreezeTests: XCTestCase {
    private static let alias = "breeze-tts-2-mxfp4"

    private func profile(backend: String = "swift_mlx") -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .textToSpeech, alias: Self.alias,
                                           repository: "mlx-community/Breeze-TTS-2-mlx-4bit",
                                           revision: "3a06d26b172ea4ae1da2f42d708383e9c79d5526", assetFingerprint: "b"),
            family: "breeze_tts", displayName: "Breeze TTS 2 MXFP4",
            capabilities: AudioCapabilities(languages: ["en", "zh"], voices: (0 ..< 10).map { "S\($0)" },
                                            operations: ["synthesize", "voice_clone"], backend: backend,
                                            cancellation: "frame", canRun: false),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "runtime_pending")
    }

    private func reference(duration: Double = 5) -> AudioVoiceReference {
        AudioVoiceReference(sourceID: UUID(), title: "me",
                            clips: [AudioLibraryClip(assetReference: "a", sampleRate: 24_000, channels: 1, frameCount: 240_000)],
                            duration: duration)
    }

    @MainActor private func controller(_ id: String) throws -> (AudioWorkspaceController, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { id }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        return (controller, root)
    }

    func testRoutesToTheSwiftProviderOnlyFromSwiftMLX() {
        let breeze = profile()
        XCTAssertEqual(AudioRuntimeProvider(breeze), .breezeMLX)
        // A Rust session cannot run it, but the Swift provider can.
        XCTAssertTrue(AudioRuntimeProvider(breeze).canRun(breeze))
        XCTAssertTrue(AudioRuntimeProvider(breeze).acceptsLocalFolder)
        XCTAssertNotEqual(AudioRuntimeProvider(profile(backend: "metal")), .breezeMLX)
        XCTAssertTrue(AudioWorkspaceController.clonesNatively(breeze))
        XCTAssertTrue(AudioWorkspaceController.curatedCloneAliases.contains(Self.alias))
    }

    @MainActor func testVoiceCloningNeedsConsentAndTheExactTranscript() throws {
        let (controller, root) = try controller("breeze-clone")
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        let breeze = profile()
        controller.profiles = [breeze]
        controller.selectPage(.voiceClone)
        XCTAssertEqual(controller.availableProfiles.map(\.family), ["breeze_tts"])
        controller.selectModel(Self.alias)
        XCTAssertEqual(controller.speechSettings.mode, .clone)
        XCTAssertTrue(controller.selectedProfileClonesNatively)
        XCTAssertTrue(controller.selectedProfileNeedsReferenceTranscript)
        XCTAssertTrue(controller.selectedProfileTakesInstruction, "direction is offered while cloning")
        controller.recipe.text = "Hello."
        controller.speechSettings.reference = reference()
        XCTAssertEqual(controller.speechValidationMessage,
                       String(localized: "Confirm you have the right to use this voice.", bundle: .module))
        controller.setCloneConsent(true)
        XCTAssertEqual(controller.speechValidationMessage,
                       String(localized: "Enter the exact transcript of the selected reference region.", bundle: .module))
        XCTAssertEqual(controller.voiceCloneSteps.map(\.step), [.voice, .consent, .transcript, .script, .generate])
        controller.speechSettings.referenceText = "Hi there."
        XCTAssertNil(controller.speechValidationMessage)
    }

    @MainActor func testVoiceDesignNeedsADescriptionAndVoiceoverNeedsNone() throws {
        let (controller, root) = try controller("breeze-design")
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile()]
        controller.selectPage(.voiceDesign)
        XCTAssertEqual(controller.availableProfiles.map(\.family), ["breeze_tts"])
        controller.selectModel(Self.alias)
        XCTAssertEqual(controller.speechSettings.mode, .design)
        XCTAssertFalse(controller.selectedProfileClonesNatively, "design uses no reference")
        XCTAssertTrue(controller.selectedProfileTakesInstruction)
        controller.recipe.text = "Hello."
        XCTAssertEqual(controller.speechValidationMessage,
                       String(localized: "Describe the voice before generating speech.", bundle: .module))
        controller.speechSettings.instruction = "A calm voice."
        XCTAssertNil(controller.speechValidationMessage)

        controller.selectPage(.voiceover)
        XCTAssertEqual(controller.availableProfiles.map(\.family), ["breeze_tts"])
        controller.selectModel(Self.alias)
        XCTAssertTrue(controller.selectedProfileSpeaksWithoutReference)
        XCTAssertFalse(controller.selectedProfileTakesInstruction, "Voiceover speaks with the preset speaker")
        XCTAssertEqual(controller.recipe.voice, "S0", "the speaker comes from the profile's voices")
        XCTAssertNil(controller.speechValidationMessage)
    }

    func testCloneRequestKeepsTranscriptAndDirectionAndConsentOnlyForTheRegion() {
        let voice = reference()
        var speech = AudioSpeechSettings()
        speech.mode = .clone
        speech.reference = voice
        speech.referenceText = "  Exact words.  "
        speech.instruction = " Speak excitedly. "
        var options = AudioCloneOptions()
        options.steps = 12
        speech.cloneOptions = options
        var base = AudioRequest(task: .textToSpeech)
        base.speed = 1.5
        let request = AudioWorkspaceController.nativeCloneRequest(base, family: "breeze_tts", speech: speech, freshSeed: { 7 })
        XCTAssertFalse(request.consent, "no consent was given")
        XCTAssertEqual(request.referenceText, "Exact words.")
        XCTAssertEqual(request.instruction, "Speak excitedly.")
        XCTAssertEqual(request.referenceSampleRate, 24_000)
        XCTAssertNil(request.steps)
        XCTAssertNil(request.pW)
        XCTAssertNil(request.tW)
        XCTAssertEqual(request.speed, 1)
        XCTAssertEqual(request.seed, 7, "seed 0 is a new take each run")
        speech.cloneOptions?.consentedReference = voice
        XCTAssertTrue(AudioWorkspaceController.nativeCloneRequest(base, family: "breeze_tts", speech: speech).consent)
        speech.reference?.start = 1
        XCTAssertFalse(AudioWorkspaceController.nativeCloneRequest(base, family: "breeze_tts", speech: speech).consent)
        speech.instruction = "   "
        XCTAssertNil(AudioWorkspaceController.nativeCloneRequest(base, family: "breeze_tts", speech: speech).instruction)
    }

    @MainActor func testDesignRequestKeepsTheDescriptionAndVoiceoverDropsIt() {
        var recipe = AudioRecipe()
        recipe.text = "Hello."
        recipe.voice = "S3"
        var speech = AudioSpeechSettings()
        speech.mode = .design
        speech.instruction = "A warm voice."
        speech.reference = reference()
        recipe.speech = speech
        let design = AudioWorkspaceController.runRequest(recipe, task: .textToSpeech, profile: profile())
        XCTAssertEqual(design.instruction, "A warm voice.")
        XCTAssertFalse(design.consent)
        XCTAssertNil(design.referenceText)
        XCTAssertEqual(design.voice, "S3")
        recipe.speech = nil
        let voiceover = AudioWorkspaceController.runRequest(recipe, task: .textToSpeech, profile: profile())
        XCTAssertNil(voiceover.instruction)
        XCTAssertEqual(voiceover.speed, 1)
    }

    func testModelRequestRefusesAReferenceWithoutConsentOrTranscript() throws {
        var request = AudioRequest(task: .textToSpeech)
        request.text = "Hello."
        request.voice = "S4"
        request.seed = 9
        let audio = [Float](repeating: 0.1, count: 24_000)
        // Each refusal is checked with the other input present, so one guard
        // cannot stand in for the other.
        request.referenceText = "Exact words."
        XCTAssertThrowsError(try BreezeTTSSpeech.modelRequest(request, reference: audio), "no consent")
        request.consent = true
        request.referenceText = "   "
        XCTAssertThrowsError(try BreezeTTSSpeech.modelRequest(request, reference: audio), "no transcript")
        request.referenceText = "Exact words."
        let clone = try BreezeTTSSpeech.modelRequest(request, reference: audio)
        XCTAssertEqual(clone.referenceAudio?.count, 24_000)
        XCTAssertEqual(clone.referenceText, "Exact words.")
        XCTAssertEqual(clone.speaker, 4)
        XCTAssertEqual(clone.seed, 9)
        XCTAssertNil(clone.cfgScale, "no instruction, no guidance")
        request.instruction = "Whisper."
        XCTAssertEqual(try BreezeTTSSpeech.modelRequest(request, reference: audio).cfgScale, BreezeTTSSpeech.instructionGuidance)
        // Without a reference there is nothing to consent to.
        request.consent = false
        XCTAssertNil(try BreezeTTSSpeech.modelRequest(request, reference: []).referenceAudio)
        XCTAssertEqual(BreezeTTSSpeech.speaker("S9"), 9)
        XCTAssertEqual(BreezeTTSSpeech.speaker("af_heart"), 0)
        XCTAssertEqual(BreezeTTSSpeech.speaker("S10"), 0)
    }

    /// A managed install loads only after its bytes verify, as for Qwen3-TTS
    /// (`AudioQwenLoadShutdownTests`); the refusal must end the barrier.
    @MainActor func testManagedInstallWithoutATokenRefusesToLoad() async throws {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "the load's memory budget needs a Metal device")
        let (controller, root) = try controller("breeze-token")
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        let model = root.appendingPathComponent("model", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        try Data(count: 16).write(to: model.appendingPathComponent("model.safetensors"))
        controller.qwenLoadDependencies.openToken = { nil }
        controller.qwenLoadDependencies.verify = { _, _, _ in
            XCTFail("there is no token to verify with")
            throw CancellationError()
        }
        controller.breezeLoad = { _ in
            XCTFail("a managed install must not load unverified")
            throw CancellationError()
        }
        var openings: [AudioModelOpening] = []
        let watch = controller.$opening.compactMap { $0 }.sink { openings.append($0) }
        defer { watch.cancel() }
        controller.installedPaths[Self.alias] = model.path
        do {
            _ = try await controller.residentBreezeSession(for: profile())
            XCTFail("a managed install without a token must not load")
        } catch {}
        XCTAssertNil(controller.breezeModel)
        XCTAssertNil(controller.opening)
        let opening = try XCTUnwrap(openings.first, "the load must create an opening")
        let drained = DispatchSemaphore(value: 0)
        DispatchQueue.global().async { opening.stopAndDrain(); drained.signal() }
        XCTAssertEqual(drained.wait(timeout: .now() + 5), .success, "the refusal left the opening's barrier entered")
    }

    /// The app's own load and job on a real checkpoint: a chosen folder opens
    /// through `residentBreezeSession`, then a consented clone with direction
    /// and a design run go through `BreezeTTSSpeech.generate`.
    /// Set TURBOSPARK_TEST_BREEZE to a Breeze TTS 2 checkpoint folder.
    @MainActor func testRealCheckpointClonesAndDesignsThroughTheAppJob() async throws {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_BREEZE"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_BREEZE to a Breeze TTS 2 checkpoint")
        }
        let (controller, root) = try controller("breeze-real")
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        let folder = URL(fileURLWithPath: path, isDirectory: true)
        controller.preferences.localModels[Self.alias] = try folder.bookmarkData(
            options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        let model = try await controller.residentBreezeSession(for: profile())
        XCTAssertNotNil(controller.breezeModel)
        XCTAssertEqual(controller.loadedModelAlias, Self.alias)

        var design = AudioRequest(task: .textToSpeech)
        design.text = "Good morning, everyone."
        design.instruction = "A calm adult voice."
        design.seed = 42
        let designed = try await BreezeTTSSpeech.generate(design, reference: [], model: model, lifetime: controller.lifetime) { _ in }
        let voice = try XCTUnwrap(designed.pcm?.samples)
        XCTAssertGreaterThan(voice.count, 24_000 / 2, "half a second at least")
        XCTAssertTrue(voice.allSatisfy(\.isFinite) && voice.contains { abs($0) > 1e-3 })

        var clone = AudioRequest(task: .textToSpeech)
        clone.text = "The weather today is sunny."
        clone.referenceText = "Good morning, everyone."
        clone.instruction = "Speak excitedly."
        clone.consent = true
        clone.seed = 7
        let cloned = try await BreezeTTSSpeech.generate(clone, reference: Array(voice.prefix(24_000 * 30)), model: model,
                                                        lifetime: controller.lifetime) { _ in }
        let samples = try XCTUnwrap(cloned.pcm?.samples)
        XCTAssertEqual(cloned.pcm?.format.sampleRate, 24_000)
        XCTAssertGreaterThan(samples.count, 24_000 / 2)
        XCTAssertTrue(samples.allSatisfy(\.isFinite) && samples.contains { abs($0) > 1e-3 })

        controller.unloadModel()
        XCTAssertNil(controller.breezeModel, "unload releases the Breeze slot")
    }
}
