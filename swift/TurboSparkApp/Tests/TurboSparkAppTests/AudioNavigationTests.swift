import TurboSpark
import XCTest
@testable import TurboSparkApp

@MainActor
final class AudioNavigationTests: XCTestCase {
    private func withController(_ body: (AudioWorkspaceController) throws -> Void) throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "audio-ui" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        controller.profiles = try AudioCatalog.profiles()
        try body(controller)
    }

    func testTaskNavigationRestoresModelsAndClearsExperimentalBackend() throws {
        try withController { controller in
            controller.selectPage(.transcribe)
            let speech = try XCTUnwrap(controller.availableProfiles.last)
            controller.selectModel(speech.identity.alias)
            controller.selectPage(.music)
            XCTAssertEqual(controller.selectedProfile?.identity.task, .music)
            controller.recipe.caption = "Quiet piano"
            controller.selectPage(.voiceover)
            XCTAssertEqual(controller.selectedProfile?.identity.task, .textToSpeech)
            controller.recipe.text = "Keep this script."
            controller.selectPage(.transcribe)
            XCTAssertEqual(controller.selectedProfile?.id, speech.id)
            controller.selectPage(.advanced)
            controller.allowPortableExperiment = true
            controller.selectPage(.music)
            XCTAssertNil(controller.recipe.experimentalBackend)
            XCTAssertEqual(controller.recipe.caption, "Quiet piano")
            controller.selectPage(.voiceover)
            XCTAssertEqual(controller.recipe.text, "Keep this script.")
        }
    }

    func testMissingSavedChoiceFallsBackToInstalledModelForThisTask() throws {
        try withController { controller in
            let speech = try XCTUnwrap(controller.profiles.last { $0.identity.task == .speechToText })
            controller.installedPaths[speech.identity.alias] = "/test/installed"
            controller.preferences.selectedModels[AudioTask.speechToText.rawValue] = "removed-pin"
            controller.selectPage(.transcribe)
            XCTAssertEqual(controller.selectedProfile?.identity, speech.identity)
            XCTAssertTrue(controller.modelInstalled)
        }
    }

    func testDownloadedModelCanReplaceLocalOverrideOnlyWhenIdle() throws {
        try withController { controller in
            controller.selectPage(.music)
            let id = try XCTUnwrap(controller.recipe.modelID)
            controller.preferences.localModels[id] = Data([1])
            controller.useDownloadedModel()
            XCTAssertNotNil(controller.preferences.localModels[id])
            controller.installedPaths[id] = "/test/installed"
            controller.isBusy = true
            controller.useDownloadedModel()
            XCTAssertNotNil(controller.preferences.localModels[id])
            controller.isBusy = false
            controller.useDownloadedModel()
            XCTAssertNil(controller.preferences.localModels[id])
            XCTAssertTrue(controller.modelInstalled)
        }
    }

    func testMusicAndSpeechOutputsStayWithTheirTask() throws {
        try withController { controller in
            let recording = AudioLibraryItem(title: "Meeting", kind: .recording)
            let music = AudioLibraryItem(title: "Soundtrack", kind: .music)
            let speech = AudioLibraryItem(title: "Narration", kind: .voiceover)
            controller.items = [recording, music, speech]
            controller.selectedID = recording.id
            controller.selectPage(.music)
            XCTAssertNil(controller.workspaceResult)
            XCTAssertEqual(controller.workspaceHistory.map(\.id), [music.id])
            controller.select(music)
            XCTAssertEqual(controller.workspaceResult?.id, music.id)
            controller.selectPage(.voiceover)
            XCTAssertNil(controller.workspaceResult)
            XCTAssertEqual(controller.workspaceHistory.map(\.id), [speech.id])
        }
    }

    func testDownloadStatusMatchesTheWholePinAndNavigationOpensAudio() throws {
        try withController { controller in
            let model = AppModel()
            defer { model.stopCronScheduler() }
            model.downloadHistoryWritable = false
            model.audioWorkspace = controller
            model.openAudio(.music)
            XCTAssertEqual(model.activeSection, .audio)
            XCTAssertEqual(controller.selectedProfile?.identity.task, .music)
            let profile = try XCTUnwrap(controller.selectedProfile)
            let pin = profile.identity
            let oldPin = AudioProfileIdentity(task: pin.task, alias: pin.alias, repository: pin.repository,
                revision: "old-revision", assetFingerprint: pin.assetFingerprint)
            let unrelated = ModelDownload(request: .audio(identity: oldPin), status: .running)
            model.modelDownloads = [unrelated]
            XCTAssertNil(model.audioDownload(for: profile), "An old pin must not borrow this model's progress or controls")
            let current = ModelDownload(request: .audio(identity: pin), status: .paused)
            model.modelDownloads.append(current)
            XCTAssertEqual(model.audioDownload(for: profile)?.id, current.id)
            let queued = ModelDownload(request: .audio(identity: pin), status: .queued)
            model.modelDownloads.append(queued)
            model.activeModelDownloadID = unrelated.id
            model.cancelQueuedAudioDownload(queued)
            XCTAssertEqual(model.modelDownloads.last?.status, .cancelled)
            XCTAssertEqual(model.modelDownloads.first?.status, .running)
            XCTAssertEqual(model.activeModelDownloadID, unrelated.id)
            model.cancelQueuedAudioDownload(current)
            XCTAssertEqual(model.audioDownload(for: profile)?.status, .paused)
        }
    }
}
