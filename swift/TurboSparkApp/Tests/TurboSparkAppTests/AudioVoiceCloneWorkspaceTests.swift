import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The controller seams behind the Voice Cloning page: dropping a file as the
/// reference, recording a reference in place, reusing a take's reference and
/// drawing it. None of them may grant consent or touch the user's settings.
final class AudioVoiceCloneWorkspaceTests: XCTestCase {
    private var root: URL!

    override func tearDown() {
        if let root { try? FileManager.default.removeItem(at: root) }
        super.tearDown()
    }

    private func profile(family: String, operations: [String]) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .textToSpeech, alias: family + "-test", repository: "someone/" + family + "-Base-",
                                           revision: String(repeating: "0", count: 40), assetFingerprint: family),
            family: family, displayName: family,
            capabilities: AudioCapabilities(languages: ["en"], operations: operations, backend: "portable_cpu",
                                            cancellation: "diffusion_step", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "implemented_unqualified")
    }

    /// A controller on the Voice Cloning page with MegaTTS3 selected, so the
    /// speech settings are in clone mode.
    @MainActor private func makeController() async throws -> AudioWorkspaceController {
        root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { self.root }, profileIDProvider: { "voice-clone-workspace" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        controller.liveTranscription = false
        while controller.isRefreshingModels { try await Task.sleep(for: .milliseconds(10)) }
        let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"])
        controller.profiles = [mega]
        controller.selectPage(.voiceClone)
        controller.selectModel(mega.identity.alias)
        XCTAssertEqual(controller.speechSettings.mode, .clone)
        return controller
    }

    @MainActor private func storedClip(_ controller: AudioWorkspaceController, samples: [Float]) throws -> AudioLibraryClip {
        let bytes = try AudioMediaIO.wav(samples: samples, sampleRate: 24_000, channels: 1)
        let asset = try controller.assets.store(data: bytes, fileName: "clip.wav")
        return AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 24_000, channels: 1, frameCount: Int64(samples.count))
    }

    @MainActor private func stubCapture(_ controller: AudioWorkspaceController, record: @escaping (AudioCapture.Configuration) -> Void = { _ in }) {
        controller.recordingSetupDependencies.requestMicrophonePermission = { true }
        controller.recordingSetupDependencies.start = { _, configuration in record(configuration) }
    }

    @MainActor private func savedTake(_ controller: AudioWorkspaceController) throws -> AudioLibraryItem {
        var take = AudioLibraryItem(title: "Take", kind: .voiceover)
        take.recipe.speech = controller.speechSettings
        take.clips = [try storedClip(controller, samples: Array(repeating: 0.1, count: 24_000))]
        try controller.library.save(take)
        controller.publish(take)
        controller.select(take)
        return take
    }

    // MARK: Import

    @MainActor func testImportedFileBecomesTheReferenceWithoutChangingTheSelection() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        let take = try savedTake(controller)
        let file = root.appendingPathComponent("my voice.wav")
        try AudioMediaIO.wav(samples: Array(repeating: 0.2, count: 48_000), sampleRate: 24_000, channels: 1).write(to: file)
        XCTAssertTrue(controller.importAudio(from: file, asVoiceReference: true))
        try await XCTUnwrap(controller.importTask).value
        XCTAssertEqual(controller.speechSettings.reference?.title, "my voice")
        XCTAssertEqual(controller.speechSettings.reference?.duration ?? 0, 2, accuracy: 0.01)
        XCTAssertEqual(controller.selectedID, take.id, "the latest take stays on screen")
    }

    @MainActor func testImportFinishingOnAnotherPageDoesNotAttachThere() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        let file = root.appendingPathComponent("late.wav")
        try AudioMediaIO.wav(samples: Array(repeating: 0.2, count: 24_000), sampleRate: 24_000, channels: 1).write(to: file)
        XCTAssertTrue(controller.importAudio(from: file, asVoiceReference: true))
        let task = try XCTUnwrap(controller.importTask)
        controller.selectPage(.voiceover)
        await task.value
        XCTAssertNil(controller.speechSettings.reference, "the voiceover page's recipe must not receive the reference")
        XCTAssertTrue(controller.items.contains { $0.title == "late" }, "the file itself is still imported")
    }

    @MainActor func testImportRefusesWhileRecordingOrForANonFileURL() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        let count = controller.items.count
        XCTAssertFalse(controller.importAudio(from: try XCTUnwrap(URL(string: "https://example.com/a.wav")), asVoiceReference: true))
        controller.recordingSetupStatus = "Preparing"
        XCTAssertFalse(controller.importAudio(from: root.appendingPathComponent("a.wav"), asVoiceReference: true))
        controller.recordingSetupStatus = nil
        XCTAssertEqual(controller.items.count, count)
        XCTAssertNil(controller.importTask)
    }

    // MARK: Recording a reference

    @MainActor func testReferenceRecordingIsMicrophoneOnlyAndLeavesSettingsAndSelection() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        let take = try savedTake(controller)
        controller.captureMicrophone = false
        controller.captureSystemAudio = true
        var configuration: AudioCapture.Configuration?
        stubCapture(controller) { configuration = $0 }
        controller.startVoiceReferenceRecording()
        try await XCTUnwrap(controller.recordingSetupTask).value
        XCTAssertTrue(controller.isRecording)
        XCTAssertTrue(controller.isRecordingVoiceReference)
        XCTAssertEqual(configuration?.capturesMicrophone, true)
        XCTAssertEqual(configuration?.systemSource, AudioCapture.SystemSource.none)
        XCTAssertFalse(controller.captureMicrophone, "the Record page's sources are the user's")
        XCTAssertTrue(controller.captureSystemAudio)
        XCTAssertEqual(controller.selectedID, take.id)
        let recording = try XCTUnwrap(controller.items.first { $0.id == controller.recordingID })
        XCTAssertNil(recording.recipe.speech, "the old reference's clips must not be pinned to the new clip")
        controller.stopRecording(interrupted: true)
    }

    @MainActor func testStoppedReferenceRecordingBecomesTheReferenceWithoutConsent() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        stubCapture(controller)
        controller.startVoiceReferenceRecording()
        try await XCTUnwrap(controller.recordingSetupTask).value
        let id = try XCTUnwrap(controller.recordingID)
        let clip = try storedClip(controller, samples: Array(repeating: 0.2, count: 48_000))
        try controller.library.update(id) { $0.clips.append(clip) }
        controller.stopRecording()
        XCTAssertEqual(controller.speechSettings.reference?.sourceID, id)
        XCTAssertEqual(controller.speechSettings.reference?.duration ?? 0, 2, accuracy: 0.01)
        XCTAssertFalse(controller.cloneConsentGiven, "a new recording still needs the user's confirmation")
        XCTAssertNil(controller.voiceReferenceRecordingID)
    }

    @MainActor func testMeetingRecordingDoesNotBecomeTheReference() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        stubCapture(controller)
        controller.startRecording()
        try await XCTUnwrap(controller.recordingSetupTask).value
        let id = try XCTUnwrap(controller.recordingID)
        try controller.library.update(id) { $0.clips.append(try self.storedClip(controller, samples: Array(repeating: 0.2, count: 48_000))) }
        controller.stopRecording()
        XCTAssertNil(controller.speechSettings.reference)
    }

    @MainActor func testInterruptedReferenceRecordingDoesNotBecomeTheReference() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        stubCapture(controller)
        controller.startVoiceReferenceRecording()
        try await XCTUnwrap(controller.recordingSetupTask).value
        let id = try XCTUnwrap(controller.recordingID)
        try controller.library.update(id) { $0.clips.append(try self.storedClip(controller, samples: Array(repeating: 0.2, count: 48_000))) }
        controller.stopRecording(interrupted: true)
        XCTAssertNil(controller.speechSettings.reference)
        XCTAssertNil(controller.voiceReferenceRecordingID)
    }

    @MainActor func testReferenceRecordingStoppedAfterLeavingThePageDoesNotAttach() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        stubCapture(controller)
        controller.startVoiceReferenceRecording()
        try await XCTUnwrap(controller.recordingSetupTask).value
        let id = try XCTUnwrap(controller.recordingID)
        try controller.library.update(id) { $0.clips.append(try self.storedClip(controller, samples: Array(repeating: 0.2, count: 48_000))) }
        controller.selectPage(.voiceover)
        controller.stopRecording()
        XCTAssertNil(controller.speechSettings.reference)
    }

    @MainActor func testOnlyAReferenceRecordingStopsAtTheLimit() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        stubCapture(controller)
        controller.startVoiceReferenceRecording()
        try await XCTUnwrap(controller.recordingSetupTask).value
        controller.recordingSeconds = 29.9
        controller.enforceVoiceReferenceRecordingLimit()
        XCTAssertTrue(controller.isRecording)
        controller.recordingSeconds = AudioWorkspaceController.voiceReferenceRecordingLimit
        controller.enforceVoiceReferenceRecordingLimit()
        XCTAssertFalse(controller.isRecording)

        controller.startRecording()
        try await XCTUnwrap(controller.recordingSetupTask).value
        controller.recordingSeconds = 45
        controller.enforceVoiceReferenceRecordingLimit()
        XCTAssertTrue(controller.isRecording, "a meeting has no reference limit")
        controller.stopRecording(interrupted: true)
    }

    // MARK: Reuse and drawing

    @MainActor func testReusingATakeRestoresItsRegionAndTranscriptButNotConsent() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        let clip = try storedClip(controller, samples: Array(repeating: 0.1, count: 120_000))
        let reference = AudioVoiceReference(sourceID: UUID(), title: "Old voice", clips: [clip], start: 1, duration: 3)
        var take = AudioLibraryItem(title: "Take", kind: .voiceover)
        var speech = AudioSpeechSettings(); speech.mode = .clone; speech.reference = reference; speech.referenceText = "hi there"
        var options = AudioCloneOptions(); options.steps = CloneQuality.fast.steps; options.consentedReference = reference
        speech.cloneOptions = options
        take.recipe.speech = speech
        controller.isBusy = true
        controller.useVoiceReference(fromTake: take)
        XCTAssertNil(controller.speechSettings.reference, "refused while a job runs")
        controller.isBusy = false
        controller.useVoiceReference(fromTake: take)
        XCTAssertEqual(controller.speechSettings.reference, reference)
        XCTAssertEqual(controller.speechSettings.referenceText, "hi there")
        XCTAssertFalse(controller.cloneConsentGiven, "consent from history is not consent on this page")
        XCTAssertEqual(controller.cloneOptions.steps, CloneQuality.balanced.steps, "quality is the page's, not the take's")
    }

    @MainActor func testReferencePeaksDrawTheTrimmedRegion() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        let clip = try storedClip(controller, samples: Array(repeating: 0.25, count: 24_000) + Array(repeating: 0.5, count: 24_000))
        let reference = AudioVoiceReference(sourceID: UUID(), title: "Steps", clips: [clip], duration: 2)
        let peaks = try await controller.referencePeaks(for: reference, count: 40)
        XCTAssertEqual(peaks.count, 40)
        XCTAssertEqual(peaks.max() ?? 0, 1, accuracy: 0.001)
        XCTAssertEqual(peaks[5], 0.5, accuracy: 0.05, "the quieter first second draws at half height")
        XCTAssertEqual(peaks[35], 1, accuracy: 0.05)
    }
}
