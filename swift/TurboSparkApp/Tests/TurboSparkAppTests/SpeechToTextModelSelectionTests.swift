import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Each Speech-to-Text mode keeps one model per job (live text, the final
/// transcript, a meeting's speakers). A mode with no choice of its own keeps
/// the app's earlier behavior, and deleting a model moves the modes that
/// used it to another installed model.
@MainActor
final class SpeechToTextModelSelectionTests: XCTestCase {
    private var vault: ProfileVaultStore?
    private var root: URL?

    override func tearDown() {
        if let root { try? FileManager.default.removeItem(at: root) }
        super.tearDown()
    }

    private func controller() throws -> AudioWorkspaceController {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        self.root = root
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "stt-models" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        self.vault = vault
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        addTeardownBlock { controller.shutdown() }
        return controller
    }

    private func profile(_ alias: String, task: AudioTask = .speechToText, operations: [String] = ["transcribe"],
                         family: String? = nil, canRun: Bool = true) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: task, alias: alias, repository: "o/\(alias)", revision: "abcdef0123", assetFingerprint: "f"),
            family: family ?? (alias.hasPrefix("whisper") ? "whisper" : "parakeet_tdt"), displayName: alias,
            capabilities: AudioCapabilities(operations: operations, backend: "coreml", cancellation: "window", canRun: canRun),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "implemented_unqualified")
    }

    private var catalog: [AudioProfile] {
        [
            profile("whisper-base"), profile("parakeet-redux-coreml"), profile("parakeet-ultra-coreml"),
            profile(AudioWorkspaceController.liveTextAlias, operations: ["stream_transcribe"], family: "parakeet_unified"),
            profile("other-live", operations: ["stream_transcribe"], family: "parakeet_unified"),
            profile("diarization-a", task: .diarization, operations: ["diarize"], family: "pyannote_community1"),
            profile("diarization-b", task: .diarization, operations: ["diarize"], family: "pyannote_community1"),
        ]
    }

    private func installed(_ aliases: Set<String>) -> (AudioProfile) -> Bool { { aliases.contains($0.identity.alias) } }

    private func resolve(_ mode: SpeechToTextMode, _ job: SpeechToTextJob, _ selected: [String: String],
                         installed aliases: Set<String>) -> String? {
        SpeechToTextModels.resolve(mode, job, profiles: catalog, selected: selected, isInstalled: installed(aliases))?.identity.alias
    }

    func testEachModeHasItsJobsAndKeys() {
        XCTAssertEqual(SpeechToTextModels.key(.meeting, .final), "speech_to_text:meeting:final")
        for mode in SpeechToTextMode.allCases {
            XCTAssertEqual(mode.jobs, [.live, .final, .speakers], "every mode can label speakers")
        }
        XCTAssertEqual(SpeechToTextMode.anyAudio.titleKey(for: .live), "Live Captions")
        XCTAssertEqual(SpeechToTextMode.notes.titleKey(for: .live), "Live Text")
        XCTAssertEqual(SpeechToTextMode(preset: .meeting), .meeting)
        XCTAssertEqual(SpeechToTextMode(preset: .appAudio), .anyAudio)
    }

    func testJobsOfferOnlyModelsThatCanDoThem() {
        let candidates = { (job: SpeechToTextJob) in SpeechToTextModels.candidates(for: job, in: self.catalog).map(\.identity.alias) }
        XCTAssertEqual(candidates(.live), [AudioWorkspaceController.liveTextAlias, "other-live"], "streaming models only")
        XCTAssertEqual(Set(candidates(.final)), ["whisper-base", "parakeet-redux-coreml", "parakeet-ultra-coreml"],
                       "a streaming-only model never runs a final pass")
        XCTAssertEqual(candidates(.speakers), ["diarization-a", "diarization-b"])
        let unsupported = AudioProfile(
            identity: AudioProfileIdentity(task: .speechToText, alias: "future", repository: "o/f", revision: "r", assetFingerprint: "f"),
            family: "fluidaudio_future", displayName: "Future",
            capabilities: AudioCapabilities(operations: ["transcribe"], backend: "swift_coreml", cancellation: "window", canRun: false),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "implemented_unqualified")
        XCTAssertFalse(SpeechToTextJob.final.accepts(unsupported), "a family this build cannot run is never offered")
    }

    func testEachModeKeepsItsOwnChoice() {
        let all: Set<String> = ["whisper-base", "parakeet-ultra-coreml", "parakeet-redux-coreml"]
        let selected = [SpeechToTextModels.key(.meeting, .final): "whisper-base"]
        XCTAssertEqual(resolve(.meeting, .final, selected, installed: all), "whisper-base")
        XCTAssertEqual(resolve(.notes, .final, selected, installed: all), "parakeet-ultra-coreml", "another mode's choice is not shared")
        XCTAssertEqual(resolve(.anyAudio, .final, selected, installed: all), "parakeet-ultra-coreml")
    }

    func testModesWithoutAChoiceKeepTheEarlierBehavior() {
        // The final pass starts from the one shared choice saved before modes had their own.
        XCTAssertEqual(resolve(.notes, .final, [SpeechToTextModels.sharedKey: "whisper-base"], installed: []), "whisper-base")
        XCTAssertEqual(resolve(.notes, .final, [:], installed: ["parakeet-redux-coreml"]), "parakeet-redux-coreml",
                       "an installed runnable model before a better one that needs a download")
        XCTAssertEqual(resolve(.notes, .final, [:], installed: []), "parakeet-ultra-coreml", "nothing installed: the preferred model")
        // Live text was always Parakeet Unified.
        XCTAssertEqual(resolve(.notes, .live, [:], installed: ["other-live"]), AudioWorkspaceController.liveTextAlias)
        // Speakers were the first installed diarization model.
        XCTAssertEqual(resolve(.meeting, .speakers, [:], installed: ["diarization-b"]), "diarization-b")
        XCTAssertEqual(resolve(.meeting, .speakers, [:], installed: []), "diarization-a", "offered for download")
    }

    func testAChoiceTheCatalogDroppedFallsBack() {
        let selected = [SpeechToTextModels.key(.notes, .final): "removed-model", SpeechToTextModels.key(.notes, .live): "removed-live"]
        XCTAssertEqual(resolve(.notes, .final, selected, installed: ["whisper-base"]), "whisper-base")
        XCTAssertEqual(resolve(.notes, .live, selected, installed: []), AudioWorkspaceController.liveTextAlias)
    }

    func testDeletingAModelMovesEveryModeThatUsedIt() {
        let before: [String: String] = [
            SpeechToTextModels.key(.meeting, .final): "parakeet-ultra-coreml",
            SpeechToTextModels.key(.anyAudio, .final): "whisper-base",
            SpeechToTextModels.key(.meeting, .speakers): "diarization-a",
        ]
        // Voice Notes has no choice of its own and resolved to Ultra too.
        XCTAssertEqual(resolve(.notes, .final, before, installed: ["parakeet-ultra-coreml", "parakeet-redux-coreml", "whisper-base"]),
                       "parakeet-ultra-coreml")
        let after = SpeechToTextModels.selections(afterDeleting: "parakeet-ultra-coreml", from: before, profiles: catalog,
                                                  isInstalled: installed(["parakeet-redux-coreml", "whisper-base"]))
        XCTAssertEqual(after[SpeechToTextModels.key(.meeting, .final)], "parakeet-redux-coreml")
        XCTAssertEqual(after[SpeechToTextModels.key(.notes, .final)], "parakeet-redux-coreml", "an implicit use moves too")
        XCTAssertEqual(after[SpeechToTextModels.key(.anyAudio, .final)], "whisper-base", "a mode that used another model is untouched")
        XCTAssertEqual(after[SpeechToTextModels.key(.meeting, .speakers)], "diarization-a")
    }

    func testDeletingTheLastModelForAJobDropsTheChoice() {
        let before = [SpeechToTextModels.key(.meeting, .speakers): "diarization-a"]
        let after = SpeechToTextModels.selections(afterDeleting: "diarization-a", from: before, profiles: catalog, isInstalled: installed([]))
        XCTAssertNil(after[SpeechToTextModels.key(.meeting, .speakers)], "no installed replacement: the default decides")
    }

    func testAModelStillServedByAFolderIsNotMoved() {
        let before = [SpeechToTextModels.key(.notes, .final): "whisper-base"]
        let after = SpeechToTextModels.selections(afterDeleting: "whisper-base", from: before, profiles: catalog,
                                                  isInstalled: installed(["whisper-base", "parakeet-redux-coreml"]))
        XCTAssertEqual(after, before)
    }

    func testSwitchingModesSavesAndRestoresEachModesFinalModel() throws {
        let controller = try controller()
        controller.profiles = catalog
        for alias in ["whisper-base", "parakeet-ultra-coreml", "parakeet-redux-coreml"] { controller.installedPaths[alias] = "/models/\(alias)" }
        controller.selectPage(.transcribe)
        controller.selectSpeechToTextMode(.notes)
        controller.setSpeechToTextModel("whisper-base", mode: .notes, job: .final)
        XCTAssertEqual(controller.recipe.modelID, "whisper-base", "the current mode's choice is the run's model")

        controller.selectSpeechToTextMode(.meeting)
        XCTAssertEqual(controller.recipe.modelID, "parakeet-ultra-coreml", "Meeting has no choice yet")
        controller.setSpeechToTextModel("parakeet-redux-coreml", mode: .meeting, job: .final)
        controller.selectSpeechToTextMode(.notes)
        XCTAssertEqual(controller.recipe.modelID, "whisper-base")
        controller.selectSpeechToTextMode(.meeting)
        XCTAssertEqual(controller.recipe.modelID, "parakeet-redux-coreml")
        XCTAssertEqual(controller.preferences.selectedModels["speech_to_text:notes:final"], "whisper-base")
        XCTAssertEqual(controller.preferences.selectedModels["speech_to_text:meeting:final"], "parakeet-redux-coreml")
        // A popover change to a mode not on screen does not touch the run's model.
        controller.setSpeechToTextModel("whisper-base", mode: .anyAudio, job: .final)
        XCTAssertEqual(controller.recipe.modelID, "parakeet-redux-coreml")
        XCTAssertEqual(controller.speechToTextProfile(.anyAudio, .final)?.identity.alias, "whisper-base")
    }

    func testDeletingTheCurrentModesModelMovesTheRecipe() throws {
        let controller = try controller()
        controller.profiles = catalog
        for alias in ["whisper-base", "parakeet-redux-coreml"] { controller.installedPaths[alias] = "/models/\(alias)" }
        controller.selectPage(.transcribe)
        controller.setSpeechToTextModel("whisper-base", mode: controller.speechToTextMode, job: .final)
        // What the delete's success path does once the files are gone.
        controller.installedPaths["whisper-base"] = nil
        controller.reassignSpeechToTextModels(afterDeleting: "whisper-base")
        XCTAssertEqual(controller.recipe.modelID, "parakeet-redux-coreml")
        XCTAssertEqual(controller.preferences.selectedModels[controller.modelSelectionKey], "parakeet-redux-coreml")
    }

    func testLiveTextFollowsTheModesChoice() throws {
        let controller = try controller()
        controller.profiles = catalog
        controller.recipe.language = "en"
        controller.installedPaths["other-live"] = "/models/other-live"
        controller.selectPage(.transcribe)
        controller.selectSpeechToTextMode(.notes)
        XCTAssertEqual(controller.liveTextProfile?.identity.alias, AudioWorkspaceController.liveTextAlias)
        XCTAssertNotNil(controller.liveStreamingUnavailableReason, "the default live model is not installed")
        controller.setSpeechToTextModel("other-live", mode: .notes, job: .live)
        XCTAssertEqual(controller.liveTextProfile?.identity.alias, "other-live")
        XCTAssertNil(controller.liveStreamingUnavailableReason)
        XCTAssertEqual(controller.liveTextModelPath, "/models/other-live")
        controller.selectSpeechToTextMode(.meeting)
        XCTAssertEqual(controller.liveTextProfile?.identity.alias, AudioWorkspaceController.liveTextAlias, "Meeting keeps its own")
        controller.selectPage(.record)
        XCTAssertEqual(controller.liveTextProfile?.identity.alias, AudioWorkspaceController.liveTextAlias, "the Record page has no mode")
    }

    func testSpeakersFollowTheMeetingChoice() throws {
        let controller = try controller()
        controller.profiles = catalog
        controller.installedPaths["diarization-a"] = "/models/a"
        controller.installedPaths["diarization-b"] = "/models/b"
        var meeting = AudioLibraryItem(title: "Meeting", kind: .recording)
        meeting.recordingPreset = AudioRecordingPreset.meeting.rawValue
        var note = AudioLibraryItem(title: "Note", kind: .recording)
        note.recordingPreset = AudioRecordingPreset.voiceNote.rawValue
        XCTAssertEqual(controller.diarizationProfile(for: meeting)?.identity.alias, "diarization-a", "no choice: the first installed")
        controller.setSpeechToTextModel("diarization-b", mode: .meeting, job: .speakers)
        XCTAssertEqual(controller.diarizationProfile(for: meeting)?.identity.alias, "diarization-b")
        XCTAssertNil(controller.diarizationProfile(for: note), "Voice Notes labels nothing until its switch is on")
        controller.setLabelsSpeakers(true, in: .notes)
        XCTAssertEqual(controller.diarizationProfile(for: note)?.identity.alias, "diarization-a", "Voice Notes keeps its own choice")
        var sound = AudioLibraryItem(title: "Sound", kind: .recording)
        sound.recordingPreset = AudioRecordingPreset.sound.rawValue
        XCTAssertEqual(controller.diarizationProfile(for: sound)?.identity.alias, "diarization-a", "a take no mode lists keeps the earlier behavior")
        controller.installedPaths["diarization-b"] = nil
        XCTAssertNil(controller.diarizationProfile(for: meeting), "a chosen model that is not installed labels nothing")
    }

    func testOnlyMeetingLabelsSpeakersByDefault() throws {
        let controller = try controller()
        XCTAssertTrue(controller.labelsSpeakers(in: .meeting))
        XCTAssertFalse(controller.labelsSpeakers(in: .notes))
        XCTAssertFalse(controller.labelsSpeakers(in: .anyAudio))
        controller.setLabelsSpeakers(false, in: .meeting)
        XCTAssertFalse(controller.labelsSpeakers(in: .meeting), "a meeting can turn labels off")
        XCTAssertFalse(controller.labelsSpeakers(in: .notes), "one mode's switch is not another's")
    }

    func testTheSpeakerSwitchSurvivesAReloadAndOlderPreferencesDecode() throws {
        let first = try controller()
        first.setLabelsSpeakers(true, in: .notes)
        first.setLabelsSpeakers(false, in: .meeting)
        first.shutdown()
        let vault = try XCTUnwrap(self.vault)
        let second = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                              assets: ManagedAssetStore(vault: vault))
        defer { second.shutdown() }
        XCTAssertTrue(second.labelsSpeakers(in: .notes))
        XCTAssertFalse(second.labelsSpeakers(in: .meeting))
        XCTAssertFalse(second.labelsSpeakers(in: .anyAudio), "a mode never switched keeps its default")

        var json = try XCTUnwrap(JSONSerialization.jsonObject(with: JSONEncoder().encode(AudioWorkspacePreferences())) as? [String: Any])
        json.removeValue(forKey: "speakerLabels")
        let old = try JSONDecoder().decode(AudioWorkspacePreferences.self, from: JSONSerialization.data(withJSONObject: json))
        XCTAssertNil(old.speakerLabels, "preferences saved before the switch still load")
    }

    /// The final pass skips labeling for a mode whose switch is off, before
    /// it opens any model. The diarization install here is a fake path, so a
    /// label attempt would come back as an open failure instead of nil.
    func testANotesTakeIsNotLabeledWhileItsSwitchIsOff() async throws {
        let controller = try controller()
        controller.profiles = catalog
        controller.installedPaths["diarization-a"] = "/nonexistent/diarization-a"
        var note = AudioLibraryItem(title: "Note", kind: .recording)
        note.recordingPreset = AudioRecordingPreset.voiceNote.rawValue
        let failure = try await controller.labelSpeakersBestEffort(item: note, into: note.id, revision: UUID(), words: [], token: controller.epoch)
        XCTAssertNil(failure, "nothing was attempted")
        XCTAssertNil(controller.loadedModelAlias)
        controller.setLabelsSpeakers(true, in: .notes)
        XCTAssertEqual(controller.diarizationProfile(for: note)?.identity.alias, "diarization-a")
    }

    func testImportsFollowAnyAudiosSwitch() throws {
        let controller = try controller()
        controller.profiles = catalog
        controller.installedPaths["diarization-a"] = "/models/a"
        let file = AudioLibraryItem(title: "Podcast", kind: .imported)
        XCTAssertEqual(controller.speechToTextMode(for: file), .anyAudio)
        XCTAssertNil(controller.diarizationProfile(for: file))
        controller.setLabelsSpeakers(true, in: .anyAudio)
        XCTAssertEqual(controller.diarizationProfile(for: file)?.identity.alias, "diarization-a")
    }

    func testModelsCannotChangeDuringARun() throws {
        let controller = try controller()
        controller.profiles = catalog
        controller.isBusy = true
        controller.setSpeechToTextModel("whisper-base", mode: .anyAudio, job: .final)
        XCTAssertNil(controller.preferences.selectedModels[SpeechToTextModels.key(.anyAudio, .final)])
        controller.setLabelsSpeakers(true, in: .notes)
        XCTAssertFalse(controller.labelsSpeakers(in: .notes), "the switch waits for the run too")
        controller.isBusy = false
        controller.setSpeechToTextModel("not-a-candidate", mode: .anyAudio, job: .final)
        XCTAssertNil(controller.preferences.selectedModels[SpeechToTextModels.key(.anyAudio, .final)], "only a catalog candidate is accepted")
    }
}
