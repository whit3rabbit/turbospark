import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A Transcribe or Record page preselects the most accurate runnable
/// transcriber that is installed, not the catalog's first row (Whisper Base).
/// With nothing installed it preselects nothing, and a saved choice wins.
@MainActor
final class AudioSpeechModelPreferenceTests: XCTestCase {
    private func controller() throws -> (AudioWorkspaceController, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "speech-preference" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault))
        controller.page = .transcribe
        controller.recipe.task = AudioTask.speechToText.rawValue
        return (controller, root)
    }

    private func profile(_ alias: String, canRun: Bool = true, operations: [String] = ["transcribe"]) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(
                task: .speechToText, alias: alias, repository: "o/\(alias)", revision: "abc", assetFingerprint: "f"),
            family: alias.hasPrefix("whisper") ? "whisper" : "parakeet_tdt", displayName: alias,
            capabilities: AudioCapabilities(
                operations: operations, backend: "coreml", cancellation: "window", canRun: canRun),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "implemented_unqualified")
    }

    private func select(_ controller: AudioWorkspaceController) -> String? {
        controller.recipe.modelID = nil
        controller.restoreModelSelection()
        return controller.recipe.modelID
    }

    func testFreshSpeechSelectionPrefersUltraThenReduxThenWhisper() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        // Catalog order: Whisper first, as audio_assets.json lists it.
        controller.profiles = [profile("whisper-base"), profile("parakeet-redux-coreml"), profile("parakeet-ultra-coreml")]
        XCTAssertNil(select(controller), "nothing installed: nothing to preselect, Discover offers the downloads")

        controller.installedPaths["whisper-base"] = "/models/whisper-base"
        XCTAssertEqual(select(controller), "whisper-base", "only an installed model is a choice")

        controller.installedPaths["parakeet-redux-coreml"] = "/models/redux"
        XCTAssertEqual(select(controller), "parakeet-redux-coreml", "among installed models the preference order holds")
    }

    func testUnrunnableParakeetFallsBackToWhisper() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        // macOS 14: the Core ML profiles report canRun false.
        controller.profiles = [
            profile("whisper-base"),
            profile("parakeet-redux-coreml", canRun: false),
            profile("parakeet-ultra-coreml", canRun: false),
        ]
        controller.installedPaths["parakeet-ultra-coreml"] = "/models/ultra"
        XCTAssertNil(select(controller), "an installed model that cannot run is not preselected, and a download is not a choice")
    }

    func testStreamingOnlyModelIsNeverABatchChoice() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        let live = profile("parakeet-unified-en-coreml", operations: ["stream_transcribe"])
        controller.profiles = [live, profile("whisper-base")]
        controller.installedPaths["parakeet-unified-en-coreml"] = "/models/unified"
        controller.installedPaths["whisper-base"] = "/models/whisper-base"
        for page in [AudioWorkspacePage.transcribe, .record] {
            controller.page = page
            XCTAssertFalse(controller.availableProfiles.contains(live), "\(page) must not offer a streaming-only model")
        }
        XCTAssertEqual(select(controller), "whisper-base")
    }

    func testSavedSpeechChoiceIsKept() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile("whisper-base"), profile("parakeet-ultra-coreml")]
        controller.installedPaths = ["whisper-base": "/models/whisper-base", "parakeet-ultra-coreml": "/models/ultra"]
        controller.preferences.selectedModels[controller.modelSelectionKey] = "whisper-base"
        XCTAssertEqual(select(controller), "whisper-base", "an existing user's choice is never switched silently")
    }
}
