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

    /// A recorded take of `seconds` of audio with `text` as its transcript.
    @MainActor private func transcribedTake(_ controller: AudioWorkspaceController, kind: AudioLibraryItem.Kind = .recording,
                                            text: String?, seconds: Int = 3) throws -> AudioLibraryItem {
        var take = AudioLibraryItem(title: "Take", kind: kind)
        take.clips = [try storedClip(controller, samples: Array(repeating: 0.1, count: 24_000 * seconds))]
        if let text {
            let segment = AudioLibrarySegment(start: 0, end: Double(seconds), text: text)
            take.transcripts = [AudioTranscriptRevision(source: .final, segments: [segment])]
        }
        return take
    }

    // A transcript that does not match the reference audio is spoken in
    // place of the script (the OmniVoice reference transcript A/B), so these
    // two seams must never leave the previous reference's transcript behind.
    @MainActor func testChoosingAnotherTakeReplacesTheTranscriptAndNeverKeepsTheOldOne() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        controller.setVoiceReference(try transcribedTake(controller, text: "the first take"))
        XCTAssertEqual(controller.speechSettings.referenceText, "the first take")
        controller.setVoiceReference(try transcribedTake(controller, text: "the second take"))
        XCTAssertEqual(controller.speechSettings.referenceText, "the second take")
        controller.setVoiceReference(try transcribedTake(controller, text: nil))
        XCTAssertEqual(controller.speechSettings.referenceText, "", "an untranscribed take sends none, not the last one")
        var spoken = try transcribedTake(controller, kind: .voiceover, text: nil)
        spoken.recipe.text = "the script it spoke"
        controller.setVoiceReference(spoken)
        XCTAssertEqual(controller.speechSettings.referenceText, "the script it spoke")
        controller.setVoiceReference(try transcribedTake(controller, text: "more than the region", seconds: 31))
        XCTAssertEqual(controller.speechSettings.reference?.duration, 30)
        XCTAssertEqual(controller.speechSettings.referenceText, "", "a transcript of more than the 30 s region is dropped")
    }

    @MainActor func testMovingEitherEndOfTheRegionClearsTheTranscript() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        let take = try transcribedTake(controller, text: "the whole take", seconds: 10)
        for path in [\AudioVoiceReference.start, \AudioVoiceReference.duration] {
            controller.setVoiceReference(take)
            XCTAssertEqual(controller.speechSettings.referenceText, "the whole take")
            controller.setVoiceReferenceTrim(path, to: 2)
            XCTAssertEqual(controller.speechSettings.reference?[keyPath: path], 2)
            XCTAssertEqual(controller.speechSettings.referenceText, "", "the old transcript covers another region")
        }
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

    // MARK: Needs attention

    @MainActor func testMissingStepsShrinkAsTheUserFillThemIn() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        XCTAssertEqual(controller.voiceCloneMissingSteps, [.voice, .consent, .script])
        controller.recipe.text = "Hello there."
        XCTAssertEqual(controller.voiceCloneMissingSteps, [.voice, .consent])
        let clip = try storedClip(controller, samples: Array(repeating: 0.1, count: 120_000))
        controller.speechSettings.reference = AudioVoiceReference(sourceID: UUID(), title: "Voice", clips: [clip], duration: 3)
        XCTAssertEqual(controller.voiceCloneMissingSteps, [.consent])
        controller.setCloneConsent(true)
        XCTAssertEqual(controller.voiceCloneMissingSteps, [], "generate is the result of the others, never one of them")
    }

    @MainActor func testRetrimmingTheReferenceFlagsConsentAgain() async throws {
        let controller = try await makeController()
        defer { controller.shutdown() }
        controller.recipe.text = "Hello there."
        let clip = try storedClip(controller, samples: Array(repeating: 0.1, count: 240_000))
        controller.speechSettings.reference = AudioVoiceReference(sourceID: UUID(), title: "Voice", clips: [clip], duration: 4)
        controller.setCloneConsent(true)
        XCTAssertEqual(controller.voiceCloneMissingSteps, [])
        controller.speechSettings.reference?.start = 1
        XCTAssertEqual(controller.voiceCloneMissingSteps, [.consent], "consent lapses when the region changes")
    }

    func testGenerateStaysPressableOnlyForInputTheUserCanGive() {
        let missing: [VoiceCloneStep] = [.consent]
        XCTAssertTrue(AudioWorkspaceController.runNeedsInput(blocker: .invalid("Confirm"), missing: missing))
        XCTAssertTrue(AudioWorkspaceController.runNeedsInput(blocker: .emptyScript, missing: [.script]))
        for blocker in [VoiceCloneRunBlocker.recording, .importing, .installing, .noModel, .notInstalled, .runtimeUnavailable] {
            XCTAssertFalse(AudioWorkspaceController.runNeedsInput(blocker: blocker, missing: missing),
                           "\(blocker) is the app's own state, a press could not help")
        }
        XCTAssertFalse(AudioWorkspaceController.runNeedsInput(blocker: .invalid("Other"), missing: []),
                       "an invalid setting with no missing step keeps its explanation and a disabled button")
        XCTAssertFalse(AudioWorkspaceController.runNeedsInput(blocker: nil, missing: missing))
    }

    @MainActor func testAttentionShowsOnlyAfterAnAttemptAndOnlyForMissingSteps() {
        let attention = VoiceCloneAttention()
        let missing: [VoiceCloneStep] = [.voice, .consent]
        XCTAssertFalse(attention.isOn(.consent, missing: missing), "an untouched page is calm")
        attention.show([.voice, .consent])
        XCTAssertTrue(attention.isOn(.consent, missing: missing))
        XCTAssertFalse(attention.isOn(.script, missing: missing), "a finished or unflagged step stays calm")
        XCTAssertFalse(attention.isOn(.consent, missing: [.voice]), "the edge follows the controller and clears when fixed")
        XCTAssertEqual(attention.ping(.voice), 1)
        attention.show([.voice])
        XCTAssertEqual(attention.ping(.voice), 2, "every attempt pings again, even for a step already flagged")
        XCTAssertEqual(attention.ping(.consent), 1)
        attention.reset()
        XCTAssertFalse(attention.isOn(.voice, missing: missing))
    }

    @MainActor func testStepsStripShowsFlaggedStepsAsNeedingAttention() {
        let attention = VoiceCloneAttention()
        let statuses = [VoiceCloneStepStatus(step: .voice, state: .done), VoiceCloneStepStatus(step: .consent, state: .todo),
                        VoiceCloneStepStatus(step: .generate, state: .todo)]
        XCTAssertEqual(attention.flagged(statuses, missing: [.consent]), statuses, "calm before an attempt")
        attention.show([.consent])
        let flagged = attention.flagged(statuses, missing: [.consent])
        XCTAssertEqual(flagged.map(\.state), [.done, .attention, .todo])
    }

    func testLevelHistoryKeepsTheNewestLevelsAndClampsBadOnes() {
        var history = [Float](repeating: 0, count: 4)
        for level: Float in [0.2, 0.4, 3, -1, .nan, 0.9] { history = AppLevelWaveform.pushed(level, onto: history, count: 4) }
        XCTAssertEqual(history, [1, 0, 0, 0.9])
    }
}
