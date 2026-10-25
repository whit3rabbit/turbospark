import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

final class AudioMusicExperienceTests: XCTestCase {
    private func event(_ stage: String, _ completed: Int, _ total: Int? = nil) throws -> AudioProgress {
        let json: [String: Any] = ["stage": stage, "completed": completed, "total": total as Any? ?? NSNull()]
        return try JSONDecoder().decode(AudioProgress.self, from: JSONSerialization.data(withJSONObject: json))
    }

    func testDepthDoesNotResetCompletedFramesAndLateEventsCannotRegressPhase() throws {
        var progress = AudioGenerationProgress()
        progress.receive(try event("tokenized", 80))
        progress.receive(try event("depth", 0, 7))
        XCTAssertNil(progress.fraction, "No target is known before the first completed frame")
        progress.receive(try event("ar", 5, 250))
        for count in 0..<7 {
            progress.receive(try event("depth", count, 7))
            XCTAssertEqual(progress.phase, .composing)
            XCTAssertEqual(progress.fraction, 0.02)
            XCTAssertEqual(progress.completed, 5)
        }
        progress.receive(try event("ar", 4, 250))
        XCTAssertEqual(progress.completed, 5, "A queued older event must not move backwards")
        progress.receive(try event("ar", 6, 250))
        XCTAssertEqual(progress.fraction, 0.024)
        progress.receive(try event("flow_chunk", 0, 2))
        XCTAssertEqual(progress.phase, .rendering, "Early EOS may render before the requested frame target")
        progress.receive(try event("ar", 250, 250))
        XCTAssertEqual(progress.phase, .rendering)
        XCTAssertEqual(progress.completed, 0)
        progress.saving()
        progress.receive(try event("depth", 0, 7))
        XCTAssertEqual(progress.phase, .saving)
        XCTAssertNil(progress.fraction)
    }

    func testRenderingCountersDoNotResetWhenDenoisingAndVocoderRepeat() throws {
        var progress = AudioGenerationProgress()
        progress.receive(try event("flow_chunk", 1, 3))
        progress.receive(try event("euler", 29, 30))
        progress.receive(try event("vocoder", 0, 8))
        XCTAssertEqual(progress.fraction, 1.0 / 3.0)
        XCTAssertTrue(try XCTUnwrap(progress.detail).hasSuffix("0 / 8"))
        progress.receive(try event("flow_chunk", 2, 3))
        progress.receive(try event("euler", 0, 30))
        XCTAssertEqual(progress.fraction, 2.0 / 3.0)
        XCTAssertTrue(try XCTUnwrap(progress.detail).hasSuffix("0 / 30"))
    }

    @MainActor func testEmptyMalformedAndNonfiniteGeneratedAudioIsRejected() throws {
        let format = AudioPCMFormat(sampleRate: 24_000, channels: 2)
        for pcm in [nil, AudioPCM(format: format, samples: []),
                    AudioPCM(format: format, samples: [0]),
                    AudioPCM(format: format, samples: [.nan, 0]),
                    AudioPCM(format: AudioPCMFormat(sampleRate: 24_000, channels: 2, interleaved: false), samples: [0, 0]),
                    AudioPCM(format: AudioPCMFormat(sampleRate: 0, channels: 0), samples: [0])] {
            XCTAssertThrowsError(try AudioWorkspaceController.validateGeneratedAudio(pcm))
        }
        XCTAssertEqual(try AudioWorkspaceController.validateGeneratedAudio(AudioPCM(format: format, samples: [0, 0])).samples.count, 2)
    }

    @MainActor private func withController(_ body: (AudioWorkspaceController, AudioLibraryStore) async throws -> Void) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "music-ui" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        while controller.isRefreshingModels { try await Task.sleep(for: .milliseconds(10)) }
        controller.profiles = try AudioCatalog.profiles()
        // The model menu lists installed models, and a page selects only one of those.
        controller.installedPaths = Dictionary(uniqueKeysWithValues: controller.profiles.map { ($0.identity.alias, "/test/\($0.identity.alias)") })
        controller.selectPage(.music)
        try await body(controller, library)
    }

    @MainActor func testFailedAndCancelledGenerationNeverCreateSavedTakes() async throws {
        try await withController { controller, library in
            let profile = try XCTUnwrap(controller.selectedProfile)
            controller.installedPaths[profile.identity.alias] = "/nonexistent-music-test-checkpoint"
            controller.recipe.caption = "Quiet piano"
            XCTAssertTrue(controller.canRun)
            controller.run()
            XCTAssertTrue(try library.load().isEmpty, "A processing placeholder must not be persisted")
            XCTAssertNotNil(controller.activeGeneration)
            await controller.jobTask?.value
            XCTAssertNotNil(controller.error)
            XCTAssertTrue(try library.load().isEmpty)
            XCTAssertTrue(controller.items.isEmpty)
            XCTAssertNil(controller.activeGeneration)
            XCTAssertNil(controller.progress)
            XCTAssertEqual(controller.recipe.caption, "Quiet piano")
            controller.run()
            controller.cancel()
            await controller.jobTask?.value
            XCTAssertTrue(try library.load().isEmpty)
            XCTAssertNil(controller.error)
            XCTAssertNil(controller.activeRunID)
        }
    }

    @MainActor func testCompletedAudioOnlyHistoryAndLegacyFailureFiltering() async throws {
        try await withController { controller, _ in
            var failure = AudioLibraryItem(title: "Failed piano", kind: .music); failure.status = .failed
            var interrupted = failure; interrupted.id = UUID(); interrupted.status = .interrupted
            var draft = failure; draft.id = UUID(); draft.status = .draft
            var success = failure; success.id = UUID(); success.status = .completed
            success.clips = [AudioLibraryClip(assetReference: "fixture", sampleRate: 24_000, channels: 2, frameCount: 24_000)]
            var invalidSuccess = failure; invalidSuccess.id = UUID(); invalidSuccess.status = .completed
            controller.items = [failure, interrupted, draft, success, invalidSuccess]
            XCTAssertEqual(controller.workspaceHistory.map(\.id), [success.id])
            XCTAssertEqual(Set(controller.filteredItems.map(\.id)), Set([draft.id, success.id]))
        }
    }

    @MainActor func testOldRunProgressAndAutosaveCannotOverwriteCurrentStatus() async throws {
        try await withController { controller, _ in
            controller.isBusy = true
            let runID = UUID(); controller.activeRunID = runID
            controller.generationProgress = AudioGenerationProgress()
            controller.receiveProgress(try self.event("ar", 10, 100), runID: runID, token: controller.epoch)
            let status = controller.status
            controller.receiveProgress(try self.event("ar", 99, 100), runID: UUID(), token: controller.epoch)
            controller.flushPreferences()
            XCTAssertEqual(controller.progress, 0.1)
            XCTAssertEqual(controller.status, status)
            controller.isBusy = false
            controller.receiveProgress(try self.event("ar", 99, 100), runID: runID, token: controller.epoch)
            XCTAssertEqual(controller.progress, 0.1)
        }
    }

    @MainActor func testEngineStageIdsNeverReachTheMusicStatus() async throws {
        try await withController { controller, _ in
            controller.isBusy = true
            let runID = UUID(); controller.activeRunID = runID
            controller.generationProgress = AudioGenerationProgress()
            // Every native job starts with "running"; the second id is unknown.
            for stage in ["running", "vocoder_pass"] {
                controller.receiveProgress(try self.event(stage, 0), runID: runID, token: controller.epoch)
                XCTAssertEqual(controller.status, AppStrings.localized("Encoding prompt"), stage)
            }
            controller.isBusy = false
        }
    }

    @MainActor func testPersonalPromptsPersistSearchRenameDeleteAndKeepRunSettings() async throws {
        try await withController { controller, library in
            XCTAssertFalse(controller.savePrompt(name: "Empty"))
            controller.recipe.caption = "Quiet piano"
            controller.recipe.lyrics = "[Verse]\nRain on the window"
            XCTAssertFalse(controller.savePrompt(name: "  "))
            XCTAssertTrue(controller.savePrompt(name: "  Rain  "))
            let prompt = try XCTUnwrap(library.preferences().presets.first)
            XCTAssertEqual(prompt.name, "Rain")
            XCTAssertEqual(controller.prompts(matching: "WINDOW").map(\.id), [prompt.id])
            controller.recipe.caption = "Different"
            controller.recipe.lyrics = ""
            controller.recipe.durationSeconds = 23; controller.recipe.seed = 84; controller.recipe.modelID = "another-model"
            controller.loadPrompt(prompt)
            XCTAssertEqual(controller.recipe.caption, "Quiet piano")
            XCTAssertEqual(controller.recipe.lyrics, prompt.recipe.lyrics)
            XCTAssertEqual(controller.recipe.modelID, "another-model")
            XCTAssertEqual(controller.recipe.durationSeconds, 23)
            XCTAssertEqual(controller.recipe.seed, 84)
            XCTAssertTrue(controller.renamePrompt(prompt, name: "Storm"))
            XCTAssertEqual(try library.preferences().presets.first?.name, "Storm")
            controller.selectPage(.voiceover)
            XCTAssertTrue(controller.prompts(matching: "").isEmpty)
            controller.loadPrompt(prompt)
            XCTAssertEqual(controller.recipe.text, "")
            XCTAssertTrue(controller.deletePrompt(prompt))
            XCTAssertTrue(try library.preferences().presets.isEmpty)
            controller.preferencesTask?.cancel()
        }
    }
}
