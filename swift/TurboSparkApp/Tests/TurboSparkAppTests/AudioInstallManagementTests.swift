import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The adopt and delete controls decide what to offer from the controller's
/// published state. Driving the real `AudioCatalog.adopt`/`delete` needs a
/// model install, so these pin the decisions around them.
@MainActor
final class AudioInstallManagementTests: XCTestCase {
    private func controller() throws -> (AudioWorkspaceController, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "installs" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        return (AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault)), root)
    }

    private func legacy(alias: String) throws -> AudioLegacyInstall {
        let json = """
        {"record":{"alias":"\(alias)","path":"/models/\(alias)","family":"minimax_music3",
                   "status":"runs","install_bytes":10},
         "identity":{"task":"music","alias":"\(alias)","repository":"o/r",
                     "revision":"abc","asset_fingerprint":"f"}}
        """
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(AudioLegacyInstall.self, from: Data(json.utf8))
    }

    private func profile(alias: String) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(
                task: .music, alias: alias, repository: "o/r", revision: "abc", assetFingerprint: "f"),
            family: "minimax_music3", displayName: alias,
            capabilities: AudioCapabilities(
                operations: ["generate"], backend: "metal", cancellation: "step", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 44_100, channels: 2), readiness: "qualified")
    }

    func testAdoptIsOfferedOnlyForTheSelectedModelsUnadoptedInstall() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.needsAdoption = [try legacy(alias: "music-a")]
        controller.recipe.modelID = "music-b"
        XCTAssertNil(controller.selectedAdoptable, "another model's legacy install must not show")
        controller.recipe.modelID = "music-a"
        XCTAssertEqual(controller.selectedAdoptable?.identity.alias, "music-a")
    }

    func testDeleteIsOfferedOnlyForAnInstalledIdleSelection() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile(alias: "music-a")]
        controller.recipe.task = AudioTask.music.rawValue
        controller.recipe.modelID = "music-a"
        XCTAssertFalse(controller.canDeleteSelectedModel, "nothing installed yet")
        controller.installedPaths["music-a"] = "/models/music-a"
        XCTAssertTrue(controller.canDeleteSelectedModel)
        controller.isBusy = true
        XCTAssertFalse(controller.canDeleteSelectedModel, "never delete under a running job")
        controller.isBusy = false
        controller.isManagingInstall = true
        XCTAssertFalse(controller.canDeleteSelectedModel, "one install operation at a time")
    }

    func testLoadAndUnloadModelGating() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile(alias: "music-a")]
        controller.recipe.task = AudioTask.music.rawValue
        controller.recipe.modelID = "music-a"
        XCTAssertFalse(controller.canLoadModel, "model is not installed yet")
        XCTAssertFalse(controller.canUnloadModel, "session is not loaded")
        XCTAssertFalse(controller.isModelLoaded)

        controller.installedPaths["music-a"] = "/models/music-a"
        XCTAssertTrue(controller.canLoadModel, "installed model can be loaded")
        XCTAssertFalse(controller.canUnloadModel)

        controller.isBusy = true
        XCTAssertFalse(controller.canLoadModel, "cannot load while busy")
        controller.isBusy = false

        // Unloading becomes available when a session exists
        // (controller.session is non-nil).
        XCTAssertFalse(controller.canUnloadModel)
    }

    func testQwenOrResidentModelLoadAndUnloadGating() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        let qwenProfile = AudioProfile(
            identity: AudioProfileIdentity(
                task: .textToSpeech, alias: "qwen3-base", repository: "Qwen/Qwen3-TTS-12Hz-1.7B-Base", revision: "main", assetFingerprint: "f"),
            family: "qwen3_tts", displayName: "Qwen3-TTS Base",
            capabilities: AudioCapabilities(
                operations: ["synthesize"], backend: "metal", cancellation: "token", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 24_000, channels: 1), readiness: "qualified")
        controller.profiles = [qwenProfile]
        controller.recipe.task = AudioTask.textToSpeech.rawValue
        controller.recipe.modelID = "qwen3-base"

        XCTAssertFalse(controller.canLoadModel)
        XCTAssertFalse(controller.canUnloadModel)
        XCTAssertFalse(controller.isModelLoaded)
        XCTAssertFalse(controller.isProfileLoaded(qwenProfile))

        controller.installedPaths["qwen3-base"] = "/models/qwen3-base"
        XCTAssertTrue(controller.canLoadModel)
        XCTAssertFalse(controller.canUnloadModel)

        // Simulate model loaded
        controller.loadedModelAlias = "qwen3-base"
        XCTAssertTrue(controller.isModelLoaded)
        XCTAssertTrue(controller.isProfileLoaded(qwenProfile))
        XCTAssertFalse(controller.canLoadModel)
        XCTAssertTrue(controller.canUnloadModel)

        // Switching selection to a different model means isModelLoaded is false for the new selection
        controller.recipe.modelID = "another-model"
        XCTAssertFalse(controller.isModelLoaded)

        // Switching back allows unloading the selected model
        controller.recipe.modelID = "qwen3-base"
        XCTAssertTrue(controller.isModelLoaded)
        XCTAssertTrue(controller.canUnloadModel)

        // Unloading clears the state
        controller.unloadModel()
        XCTAssertFalse(controller.isModelLoaded)
        XCTAssertFalse(controller.isProfileLoaded(qwenProfile))
        XCTAssertTrue(controller.canLoadModel)
        XCTAssertFalse(controller.canUnloadModel)
    }
}
