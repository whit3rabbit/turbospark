import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A deleted audio model leaves the model menu and stops being the saved choice.
/// The engine delete needs a real install, so these drive the decisions around it:
/// what the menu lists, what `forgetChoices` does after a delete, and the refusal
/// that comes before the engine is asked.
@MainActor
final class AudioModelDeleteTests: XCTestCase {
    private func controller() throws -> (AudioWorkspaceController, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "deletes" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        return (AudioWorkspaceController(library: library, assets: ManagedAssetStore(vault: vault)), root)
    }

    private func profile(alias: String, repository: String = "o/r") -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(
                task: .music, alias: alias, repository: repository, revision: "abc", assetFingerprint: "f"),
            family: "minimax_music3", displayName: alias,
            capabilities: AudioCapabilities(
                operations: ["generate"], backend: "metal", cancellation: "step", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 44_100, channels: 2), readiness: "qualified")
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

    private func aliases(_ profiles: [AudioProfile]) -> [String] { profiles.map(\.identity.alias) }

    func testMenuListsOnlyWhatIsOnThisMac() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = ["music-a", "music-b", "music-c", "music-d", "music-e"].map { profile(alias: $0) }
        controller.selectPage(.music)
        XCTAssertEqual(aliases(controller.pickerProfiles), [], "nothing installed lists nothing")
        XCTAssertEqual(aliases(controller.availableProfiles).count, 5, "the catalog list itself is unchanged")

        controller.installedPaths["music-a"] = "/models/music-a"
        controller.preferences.localModels["music-b"] = Data([1])
        controller.needsAdoption = [try legacy(alias: "music-c")]
        controller.installingAlias = { "music-d" }
        XCTAssertEqual(aliases(controller.pickerProfiles), ["music-a", "music-b", "music-c", "music-d"])
    }

    func testDriveThatIsAwayStaysListed() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile(alias: "music-a"), profile(alias: "music-b")]
        controller.selectPage(.music)
        let json = """
        {"record":{"alias":"music-a","path":"/Volumes/Away/music-a","family":"minimax_music3",
                   "status":"runs","install_bytes":10},
         "library_name":"Away"}
        """
        let decoder = JSONDecoder(); decoder.keyDecodingStrategy = .convertFromSnakeCase
        controller.notConnectedInstalls = [try decoder.decode(AudioNotConnectedInstall.self, from: Data(json.utf8))]
        XCTAssertEqual(aliases(controller.pickerProfiles), ["music-a"])
    }

    func testNothingInstalledLeavesNoSelection() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile(alias: "music-a")]
        controller.selectPage(.music)
        XCTAssertNil(controller.recipe.modelID)
    }

    func testDeletedSelectionMovesToAnotherInstalledModel() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile(alias: "music-a"), profile(alias: "music-b")]
        controller.installedPaths = ["music-a": "/models/music-a", "music-b": "/models/music-b"]
        controller.selectPage(.music)
        controller.selectModel("music-a")
        controller.flushPreferences()
        XCTAssertEqual(controller.preferences.selectedModels[controller.modelSelectionKey], "music-a")

        // What deleteModel does once the engine says yes.
        controller.installedPaths["music-a"] = nil
        controller.forgetChoices(of: "music-a")
        controller.flushPreferences()

        XCTAssertEqual(aliases(controller.pickerProfiles), ["music-b"])
        XCTAssertEqual(controller.recipe.modelID, "music-b")
        XCTAssertFalse(controller.preferences.selectedModels.values.contains("music-a"), "a flush must not save the deleted id again")
        controller.restoreModelSelection()
        XCTAssertEqual(controller.recipe.modelID, "music-b", "a refresh does not bring it back")
    }

    func testDeletingTheLastModelEmptiesTheMenuAndTheChoice() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile(alias: "music-a")]
        controller.installedPaths = ["music-a": "/models/music-a"]
        controller.selectPage(.music)
        XCTAssertEqual(controller.recipe.modelID, "music-a")
        controller.flushPreferences()

        controller.installedPaths["music-a"] = nil
        controller.forgetChoices(of: "music-a")
        controller.flushPreferences()

        XCTAssertEqual(aliases(controller.pickerProfiles), [])
        XCTAssertNil(controller.recipe.modelID)
        XCTAssertNil(controller.selectedProfile)
        XCTAssertFalse(controller.preferences.selectedModels.values.contains("music-a"))
        controller.restoreModelSelection()
        XCTAssertNil(controller.recipe.modelID, "no default pulls the deleted model back as a download")
    }

    func testCheckpointFolderKeepsTheChoiceOfADeletedManagedInstall() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile(alias: "music-a")]
        controller.installedPaths = ["music-a": "/models/music-a"]
        controller.preferences.localModels["music-a"] = Data([1])
        controller.selectPage(.music)
        controller.installedPaths["music-a"] = nil
        controller.forgetChoices(of: "music-a")
        XCTAssertEqual(controller.recipe.modelID, "music-a", "the folder the user chose still serves it")
        XCTAssertEqual(aliases(controller.pickerProfiles), ["music-a"])
    }

    func testSpeechChoiceMovesOffADeletedModel() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = try AudioCatalog.profiles()
        controller.selectPage(.voiceover)
        let runnable = controller.availableProfiles.filter { AudioRuntimeProvider($0).canRun($0) }
        let first = try XCTUnwrap(runnable.first).identity.alias
        let second = try XCTUnwrap(runnable.dropFirst().first).identity.alias
        controller.installedPaths = [first: "/models/\(first)", second: "/models/\(second)"]
        controller.selectModel(first)
        controller.flushPreferences()

        controller.installedPaths[first] = nil
        controller.forgetChoices(of: first)
        controller.flushPreferences()

        XCTAssertEqual(controller.recipe.modelID, second)
        XCTAssertFalse(controller.pickerProfiles.contains { $0.identity.alias == first })
        XCTAssertFalse(controller.preferences.selectedModels.values.contains(first))
    }

    func testServedModelIsNotDeleted() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        let music = profile(alias: "music-a")
        controller.profiles = [music]
        controller.installedPaths = ["music-a": "/models/music-a"]
        controller.selectPage(.music)
        XCTAssertTrue(controller.canDelete(music))
        controller.deleteBlockedReason = { $0 == "music-a" ? "attached to the server" : nil }
        controller.deleteModel(music)
        XCTAssertEqual(controller.error, "attached to the server")
        XCTAssertFalse(controller.isManagingInstall, "refused before the engine is asked")
        XCTAssertEqual(controller.installedPaths["music-a"], "/models/music-a")
        XCTAssertEqual(controller.recipe.modelID, "music-a")
    }

    func testRemovingACheckpointFolderMovesTheChoice() throws {
        let (controller, root) = try controller()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.profiles = [profile(alias: "music-a"), profile(alias: "music-b")]
        controller.installedPaths = ["music-b": "/models/music-b"]
        controller.preferences.localModels["music-a"] = Data([1])
        controller.selectPage(.music)
        controller.selectModel("music-a")
        var told: [String] = []
        controller.onInstalledModelsChanged = { told.append($0) }
        controller.removeLocalModel("music-a")
        XCTAssertEqual(controller.recipe.modelID, "music-b")
        XCTAssertEqual(told, ["music-a"])
    }
}
