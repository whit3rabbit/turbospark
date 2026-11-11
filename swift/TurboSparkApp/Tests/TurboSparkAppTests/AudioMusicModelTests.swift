import TurboSpark
import XCTest

@testable import TurboSparkApp

@MainActor
final class AudioMusicModelTests: XCTestCase {
    private let turbo = AudioWorkspaceController.defaultMusicProfileAlias
    private let standard = "minimax-music3-4bit"

    private func withController(
        installed: String = "[]", notConnected: String = "[]",
        _ body: (AudioWorkspaceController) async throws -> Void
    ) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "music-model" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let report = try decoder.decode(AudioInstalledReport.self, from: Data("""
            {"installed":\(installed),"not_connected":\(notConnected),"needs_adoption":[],"incompatible":[],"other_audio":[]}
            """.utf8))
        let snapshot = AudioCatalogSnapshot(profiles: try AudioCatalog.profiles(), installed: report, capabilities: [])
        let controller = AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault),
            catalogReader: { snapshot })
        defer { controller.shutdown() }
        let refreshed = await eventually { !controller.isRefreshingModels }
        XCTAssertTrue(refreshed)
        try await body(controller)
    }

    private func record(_ alias: String) -> String {
        """
        {"alias":"\(alias)","path":"/models/\(alias)","family":"minimax_music3","status":"runs","install_bytes":100}
        """
    }

    func testFreshMusicShowsTurboBeforeItIsInstalledAndDownloadsThatPin() async throws {
        try await withController { controller in
            controller.selectPage(.music)
            controller.recipe.caption = "Quiet piano"
            let profile = try XCTUnwrap(controller.selectedProfile)
            XCTAssertEqual(profile.identity.alias, turbo)
            XCTAssertEqual(controller.availableProfiles.map(\.identity.alias), [turbo])
            XCTAssertGreaterThan(profile.downloadBytes, 0)
            XCTAssertEqual(controller.modelState(profile, download: nil), .notInstalled)
            XCTAssertFalse(controller.canRun)

            var requested: AudioProfileIdentity?
            var library: String?
            controller.installHandler = { profile, _, destination in
                requested = profile.identity
                library = destination
                return true
            }
            controller.installSelectedModel(library: "external-library")
            XCTAssertEqual(requested, profile.identity)
            XCTAssertEqual(library, "external-library")
        }
    }

    func testMusicIgnoresInstalledStandardAndPreservesAdvancedChoice() async throws {
        try await withController(installed: "[\(record(standard))]") { controller in
            controller.preferences.selectedModels[AudioTask.music.rawValue] = standard
            controller.selectPage(.music)
            XCTAssertEqual(controller.recipe.modelID, turbo)
            XCTAssertFalse(controller.modelInstalled)
            controller.selectModel(standard)
            XCTAssertEqual(controller.recipe.modelID, turbo)
            controller.flushPreferences()
            XCTAssertEqual(controller.preferences.selectedModels[AudioTask.music.rawValue], standard)

            controller.selectPage(.advanced)
            XCTAssertEqual(controller.selectedProfile?.identity.alias, standard)
            XCTAssertTrue(controller.modelInstalled)
        }
    }

    func testInstallRefreshAndDeletionKeepTheTurboCardSelected() async throws {
        try await withController(installed: "[\(record(turbo))]") { controller in
            controller.selectPage(.music)
            controller.recipe.caption = "Quiet piano"
            let pin = try XCTUnwrap(controller.selectedProfile).identity
            controller.installedPaths.removeAll()
            XCTAssertFalse(controller.canRun)

            controller.refreshModels()
            let refreshed = await eventually { !controller.isRefreshingModels }
            XCTAssertTrue(refreshed)
            XCTAssertEqual(controller.selectedProfile?.identity, pin)
            XCTAssertTrue(controller.canRun)

            // The successful delete callback removes the install before forgetting its choice.
            controller.installedPaths[turbo] = nil
            controller.installedRecords.removeAll()
            controller.forgetChoices(of: turbo)
            controller.flushPreferences()
            XCTAssertEqual(controller.selectedProfile?.identity, pin)
            XCTAssertEqual(controller.modelState(try XCTUnwrap(controller.selectedProfile), download: nil), .notInstalled)
            XCTAssertFalse(controller.canRun)
        }
    }

    func testDisconnectedTurboNamesItsLibraryAndRefusesAnotherDownload() async throws {
        let offline = """
        [{"record":\(record(turbo)),"library_name":"Music SSD"}]
        """
        try await withController(notConnected: offline) { controller in
            controller.selectPage(.music)
            controller.recipe.caption = "Quiet piano"
            let profile = try XCTUnwrap(controller.selectedProfile)
            XCTAssertEqual(profile.identity.alias, turbo)
            XCTAssertEqual(controller.installStatus(for: profile), .notConnected(libraryName: "Music SSD"))
            XCTAssertEqual(AudioOfflineInstalls.statement(for: controller.installStatus(for: profile), language: .english),
                           "Installed on Music SSD, which is not connected. Connect the drive to use it.")
            XCTAssertFalse(controller.canRun)
            var queued = false
            controller.installHandler = { _, _, _ in queued = true; return true }
            controller.installSelectedModel()
            XCTAssertFalse(queued)
        }
    }

    func testRestoringAMusicPresetUsesTurboAndRetainsItsInputs() async throws {
        try await withController { controller in
            var recipe = AudioRecipe()
            recipe.task = AudioTask.music.rawValue
            recipe.modelID = standard
            recipe.caption = "Quiet piano"
            recipe.lyrics = "[instrumental]"
            recipe.durationSeconds = 12
            recipe.seed = 7
            controller.applyPreset(AudioPreset(name: "Older preset", recipe: recipe))
            XCTAssertEqual(controller.page, .music)
            XCTAssertEqual(controller.selectedProfile?.identity.alias, turbo)
            recipe.modelID = turbo
            XCTAssertEqual(controller.recipe, recipe)
            XCTAssertFalse(controller.canRun)
        }
    }

    func testRunAgainRoutesStandardToAdvancedAndRetainsCheckpointAndInputs() async throws {
        try await withController(installed: "[\(record(standard))]") { controller in
            let profile = try XCTUnwrap(controller.profiles.first { $0.identity.alias == standard })
            var take = AudioLibraryItem(title: "Older take", kind: .music)
            take.recipe.task = AudioTask.music.rawValue
            take.recipe.modelID = standard
            take.recipe.seed = 7
            take.recipe.durationSeconds = 12
            take.modelFamily = profile.family
            take.modelIdentity = try JSONEncoder().encode(profile.identity)
            // No caption, so this routing test does not start model inference.
            controller.selectPage(.music)
            controller.runAgain(take)
            XCTAssertEqual(controller.page, .advanced)
            XCTAssertEqual(controller.selectedProfile?.identity, profile.identity)
            XCTAssertEqual(controller.recipe, take.recipe)
            XCTAssertNil(controller.jobTask)
            XCTAssertNil(controller.error)
        }
    }

    func testRunAgainRefusesAnOlderPinInsteadOfSubstitutingTurbo() async throws {
        try await withController(installed: "[\(record(standard))]") { controller in
            let profile = try XCTUnwrap(controller.profiles.first { $0.identity.alias == standard })
            var take = AudioLibraryItem(title: "Older pin", kind: .music)
            take.recipe.task = AudioTask.music.rawValue
            take.recipe.modelID = standard
            take.recipe.caption = "Quiet piano"
            take.modelFamily = profile.family
            let pin = profile.identity
            take.modelIdentity = try JSONEncoder().encode(AudioProfileIdentity(
                task: pin.task, alias: pin.alias, repository: pin.repository,
                revision: "earlier-revision", assetFingerprint: pin.assetFingerprint))
            controller.selectPage(.music)
            controller.runAgain(take)
            XCTAssertEqual(controller.page, .advanced)
            XCTAssertEqual(controller.recipe.modelID, standard)
            XCTAssertNotNil(controller.error)
            XCTAssertNil(controller.jobTask)
        }
    }

    func testRunAgainGeneratesANewTakeWithTheStandardLocalCheckpoint() async throws {
        try await withController { controller in
            let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            let model = root.appendingPathComponent("crates/audio/testdata/minimax_music3/converted_plain")
            var take = AudioLibraryItem(title: "Earlier local take", kind: .music)
            take.recipe.task = AudioTask.music.rawValue
            take.recipe.modelID = standard
            take.recipe.caption = "piano"
            take.recipe.lyrics = "[instrumental]"
            take.recipe.durationSeconds = 0.12
            take.recipe.steps = 1
            take.recipe.seed = 7
            take.modelFamily = "minimax_music3"
            take.localModelBookmark = try model.bookmarkData(
                options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
            try controller.save(take)
            controller.selectPage(.music)
            controller.runAgain(take)
            await controller.jobTask?.value
            XCTAssertNil(controller.error)
            XCTAssertEqual(controller.page, .advanced)
            let repeated = try XCTUnwrap(controller.library.load().first { $0.id != take.id && $0.kind == .music })
            XCTAssertEqual(repeated.status, .completed)
            XCTAssertEqual(repeated.recipe.modelID, standard)
            XCTAssertEqual(repeated.localModelBookmark, take.localModelBookmark)
            XCTAssertEqual(repeated.sourceID, take.id)
            XCTAssertEqual(repeated.resolvedSeed, 7)
        }
    }
}
