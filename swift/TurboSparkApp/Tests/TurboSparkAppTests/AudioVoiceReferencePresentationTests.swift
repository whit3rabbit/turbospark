import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// What the Voice Cloning page shows: recent references, library candidates,
/// takes, the steps strip, why Generate is disabled, and the run's stages.
final class AudioVoiceReferencePresentationTests: XCTestCase {
    private func clip(seconds: Double = 5) -> AudioLibraryClip {
        AudioLibraryClip(assetReference: "a", sampleRate: 24_000, channels: 1, frameCount: Int64(seconds * 24_000))
    }

    private func reference(_ source: UUID, start: Double = 0, duration: Double = 3) -> AudioVoiceReference {
        AudioVoiceReference(sourceID: source, title: "Voice", clips: [clip()], start: start, duration: duration)
    }

    private func take(_ reference: AudioVoiceReference?, mode: AudioSpeechSettings.Mode = .clone, age: TimeInterval,
                      status: AudioLibraryItem.Status = .completed) -> AudioLibraryItem {
        var item = AudioLibraryItem(title: "Take", kind: .voiceover, status: status)
        item.createdAt = Date(timeIntervalSince1970: 1_000_000 - age)
        var speech = AudioSpeechSettings(); speech.mode = mode; speech.reference = reference; speech.referenceText = "t\(Int(age))"
        item.recipe.speech = speech
        item.clips = [clip()]
        return item
    }

    // MARK: Peaks

    func testPeaksNormalizeAndIgnoreNonFiniteSamples() {
        XCTAssertEqual(AudioWorkspaceController.peaks(Array(repeating: 0, count: 100), count: 4), [0, 0, 0, 0])
        XCTAssertEqual(AudioWorkspaceController.peaks([.nan, 0.5, -1, .infinity], count: 2), [0.5, 1])
        XCTAssertEqual(AudioWorkspaceController.peaks([0.2, 0.4], count: 10).count, 2, "never more buckets than samples")
        XCTAssertEqual(AudioWorkspaceController.peaks([], count: 3), [0, 0, 0])
    }

    // MARK: Recent references and candidates

    func testRecentReferencesKeepTheNewestUseOfEachRegion() {
        let source = UUID()
        let newest = take(reference(source), age: 1)
        let older = take(reference(source), age: 5)
        let otherRegion = take(reference(source, start: 1), age: 3)
        let invalid = take(reference(UUID(), duration: 40), age: 0)
        let voiceover = take(reference(UUID()), mode: .customVoice, age: 0)
        let failed = take(reference(UUID()), age: 0, status: .failed)
        let recents = AudioWorkspaceController.recentVoiceReferences(in: [older, otherRegion, invalid, voiceover, failed, newest])
        XCTAssertEqual(recents.map(\.take.id), [newest.id, otherRegion.id])
        XCTAssertEqual(recents.first?.referenceText, newest.recipe.speech?.referenceText)
        XCTAssertEqual(AudioWorkspaceController.recentVoiceReferences(in: [older, otherRegion, newest], limit: 1).count, 1)
    }

    func testCandidatesFilterSearchAndPutFavoritesFirst() {
        var recording = AudioLibraryItem(title: "Morning take", kind: .recording); recording.clips = [clip()]
        recording.createdAt = Date(timeIntervalSince1970: 10)
        var favorite = AudioLibraryItem(title: "Studio", kind: .imported); favorite.clips = [clip()]; favorite.favorite = true
        favorite.createdAt = Date(timeIntervalSince1970: 1)
        var collected = AudioLibraryItem(title: "Other", kind: .imported); collected.clips = [clip()]; collected.collection = "Morning"
        collected.createdAt = Date(timeIntervalSince1970: 5)
        let silent = AudioLibraryItem(title: "Morning empty", kind: .recording)
        var processing = AudioLibraryItem(title: "Morning busy", kind: .imported, status: .processing); processing.clips = [clip()]
        let items = [recording, favorite, collected, silent, processing]
        typealias C = AudioWorkspaceController
        XCTAssertEqual(C.voiceReferenceCandidates(in: items, query: "", favoritesOnly: false).map(\.id), [favorite.id, recording.id, collected.id])
        XCTAssertEqual(C.voiceReferenceCandidates(in: items, query: " morning ", favoritesOnly: false).map(\.id), [recording.id, collected.id])
        XCTAssertEqual(C.voiceReferenceCandidates(in: items, query: "", favoritesOnly: true).map(\.id), [favorite.id])
    }

    // MARK: Run stages

    func testRunStagesTrackVisitedStagesAndEndAtSaving() {
        typealias P = AudioSpeechRunProgress
        XCTAssertEqual(P.stage(forProgress: "reference"), .preparingReference)
        XCTAssertEqual(P.stage(forProgress: "Preparing reference audio"), .preparingReference)
        XCTAssertEqual(P.stage(forProgress: "verifying"), .loadingModel)
        XCTAssertEqual(P.stage(forProgress: "Loading speech model"), .loadingModel)
        XCTAssertEqual(P.stage(forProgress: "diffusion"), .generating)
        XCTAssertEqual(P.stage(forProgress: "Generating speech"), .generating)
        XCTAssertNil(P.stage(forProgress: "ar"))
        // Qwen3 loads its speech model after preparing the reference.
        var run = P()
        run.enter(.loadingModel); run.enter(.preparingReference); run.enter(.loadingModel)
        XCTAssertEqual(run.state(of: .loadingModel), .active)
        XCTAssertEqual(run.state(of: .preparingReference), .done)
        XCTAssertEqual(run.state(of: .generating), .pending)
        run.enter(.generating); run.enter(.saving)
        run.enter(.generating)
        XCTAssertEqual(run.stage, .saving, "a late engine event cannot move a run back from saving")
        XCTAssertEqual(run.state(of: .generating), .done)
    }

    // MARK: Controller-backed presentation

    @MainActor private func withController(_ body: (AudioWorkspaceController) async throws -> Void) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "voice-presentation" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)),
                                                  assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        while controller.isRefreshingModels { try await Task.sleep(for: .milliseconds(10)) }
        try await body(controller)
    }

    private func profile(family: String, operations: [String], repository: String) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .textToSpeech, alias: family + "-test", repository: repository,
                                           revision: String(repeating: "0", count: 40), assetFingerprint: family),
            family: family, displayName: family,
            capabilities: AudioCapabilities(languages: ["en"], operations: operations, backend: "portable_cpu",
                                            cancellation: "diffusion_step", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "implemented_unqualified")
    }

    @MainActor func testRunProgressFromAnOldRunIsIgnored() async throws {
        try await withController { controller in
            controller.isBusy = true
            let runID = UUID(); controller.activeRunID = runID
            controller.speechRun = AudioSpeechRunProgress()
            controller.receiveProgress(AudioProgress(stage: "diffusion", completed: 1, total: 10), runID: UUID(), token: controller.epoch)
            XCTAssertEqual(controller.speechRun?.stage, .waiting)
            controller.receiveProgress(AudioProgress(stage: "diffusion", completed: 1, total: 10), runID: runID, token: controller.epoch)
            XCTAssertEqual(controller.speechRun?.stage, .generating)
            controller.isBusy = false
        }
    }

    @MainActor func testStepsFollowTheSelectedEngine() async throws {
        try await withController { controller in
            let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"], repository: "someone/megatts3")
            let qwen = profile(family: "qwen3_tts", operations: ["synthesize"], repository: "Qwen/Qwen3-TTS-12Hz-0.6B-Base-x")
            controller.profiles = [mega, qwen]
            controller.selectPage(.voiceClone)
            controller.selectModel(nil)
            XCTAssertEqual(controller.voiceCloneSteps.map(\.step), [.voice, .script, .generate])
            controller.selectModel(mega.identity.alias)
            XCTAssertEqual(controller.voiceCloneSteps.map(\.step), [.voice, .consent, .script, .generate])
            XCTAssertEqual(controller.voiceCloneSteps.map(\.state), [.todo, .todo, .todo, .todo])
            controller.speechSettings.reference = reference(UUID(), duration: 0.5)
            XCTAssertEqual(controller.voiceCloneSteps.first?.state, .attention, "a reference outside 1 to 30 seconds")
            controller.speechSettings.reference = reference(UUID())
            controller.setCloneConsent(true)
            controller.recipe.text = String(repeating: "a", count: AudioWorkspaceController.speechTextCharacterLimit + 1)
            XCTAssertEqual(controller.voiceCloneSteps.map(\.state), [.done, .done, .attention, .todo])
            controller.selectModel(qwen.identity.alias)
            XCTAssertEqual(controller.voiceCloneSteps.map(\.step), [.voice, .transcript, .script, .generate])
            controller.speechSettings.referenceText = ""
            controller.speechSettings.embeddingOnly = false
            XCTAssertEqual(controller.voiceCloneSteps[1].state, .todo)
            controller.speechSettings.embeddingOnly = true
            XCTAssertEqual(controller.voiceCloneSteps[1].state, .done, "embedding-only needs no transcript")
        }
    }

    @MainActor func testRunBlockerIsNilExactlyWhenTheIdlePageCanRun() async throws {
        try await withController { controller in
            let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"], repository: "someone/megatts3")
            controller.profiles = [mega]
            controller.selectPage(.voiceClone)
            for installed in [false, true] {
                for text in ["", "Hello."] {
                    for hasReference in [false, true] {
                        for consent in [false, true] {
                            for recording in [false, true] {
                                controller.selectModel(mega.identity.alias)
                                controller.installedPaths = installed ? [mega.identity.alias: "/models/megatts3"] : [:]
                                controller.recipe.text = text
                                controller.speechSettings.reference = hasReference ? reference(UUID()) : nil
                                controller.setCloneConsent(consent)
                                controller.recordingSetupStatus = recording ? "Preparing" : nil
                                XCTAssertEqual(controller.voiceCloneRunBlocker == nil, controller.canRun,
                                               "installed \(installed) text \(text) reference \(hasReference) consent \(consent) recording \(recording)")
                            }
                        }
                    }
                }
            }
            controller.recordingSetupStatus = nil
            XCTAssertNil(controller.voiceCloneRunBlocker)
            XCTAssertTrue(controller.canRun)
            controller.selectModel(nil)
            XCTAssertEqual(controller.voiceCloneRunBlocker, .noModel)
        }
    }

    @MainActor func testTakesKeepTheSelectedTakeAndOnlyThisModesPlayableTakes() async throws {
        try await withController { controller in
            let mega = profile(family: "megatts3", operations: ["synthesize", "voice_clone"], repository: "someone/megatts3")
            controller.profiles = [mega]
            controller.selectPage(.voiceClone)
            controller.selectModel(mega.identity.alias)
            let selected = take(reference(UUID()), age: 1)
            let older = take(reference(UUID()), age: 9)
            let voiceover = take(nil, mode: .customVoice, age: 0)
            let failed = take(reference(UUID()), age: 0, status: .failed)
            var silent = take(reference(UUID()), age: 0); silent.clips = []
            controller.items = [older, voiceover, failed, silent, selected]
            controller.selectedID = selected.id
            XCTAssertEqual(controller.speechTakes.map(\.id), [selected.id, older.id])
            XCTAssertFalse(controller.workspaceHistory.contains { $0.id == selected.id }, "history still excludes it")
        }
    }
}
