import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The model gate over Music and Voice Cloning, and the curated clone list it shows.
@MainActor
final class AudioModelGateTests: XCTestCase {
    private let turbo = AudioWorkspaceController.defaultMusicProfileAlias
    private let curated = AudioWorkspaceController.curatedCloneAliases

    private func withController(
        installed: String = "[]", notConnected: String = "[]",
        _ body: (AudioWorkspaceController) async throws -> Void
    ) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "model-gate" }, migrateLegacyData: false)
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

    private func record(_ alias: String, family: String) -> String {
        """
        {"alias":"\(alias)","path":"/models/\(alias)","family":"\(family)","status":"runs","install_bytes":100}
        """
    }

    func testFreshVoiceCloningGatesExactlyTheCuratedModelsInOrder() async throws {
        try await withController { controller in
            controller.selectPage(.voiceClone)
            XCTAssertEqual(controller.availableProfiles.map(\.identity.alias), curated)
            let gate = try XCTUnwrap(controller.modelGate)
            XCTAssertEqual(gate.page, .voiceClone)
            XCTAssertEqual(gate.entries.map(\.identity.alias), curated)
            // The same engines at another size are not offered.
            XCTAssertFalse(gate.entries.contains { $0.identity.alias == "omnivoice-bf16" })
            XCTAssertFalse(gate.entries.contains { $0.identity.alias == "qwen3-tts-12hz-06b-base-8bit" })
            XCTAssertFalse(controller.canRun)
        }
    }

    func testAnyInstalledCuratedModelLiftsTheGate() async throws {
        for (alias, family) in [("omnivoice-8bit", "omnivoice"), ("megatts3-f32", "megatts3")] {
            try await withController(installed: "[\(record(alias, family: family))]") { controller in
                controller.selectPage(.voiceClone)
                XCTAssertNil(controller.modelGate, alias)
                XCTAssertEqual(controller.selectedProfile?.identity.alias, alias)
            }
        }
    }

    func testAnInstallOutsideTheCuratedListDoesNotLiftTheGate() async throws {
        try await withController(installed: "[\(record("omnivoice-bf16", family: "omnivoice"))]") { controller in
            controller.selectPage(.voiceClone)
            XCTAssertNotNil(controller.modelGate)
            XCTAssertNil(controller.selectedProfile)
        }
    }

    func testAnInstallOnAnAwayDriveKeepsThePageOpenForItsStatement() async throws {
        let offline = """
        [{"record":\(record("omnivoice-8bit", family: "omnivoice")),"library_name":"Voice SSD"}]
        """
        try await withController(notConnected: offline) { controller in
            controller.selectPage(.voiceClone)
            XCTAssertNil(controller.modelGate, "the card says which drive to connect; a gate would hide that")
        }
    }

    func testOnlyMusicAndVoiceCloningAreGated() async throws {
        try await withController { controller in
            for page in AudioWorkspacePage.allCases where page != .music && page != .voiceClone {
                controller.selectPage(page)
                XCTAssertNil(controller.modelGate, page.rawValue)
            }
        }
    }

    func testMusicGateHasTheTurboModelAndLiftsWhenItIsInstalled() async throws {
        try await withController { controller in
            controller.selectPage(.music)
            XCTAssertEqual(controller.modelGate?.entries.map(\.identity.alias), [turbo])
        }
        try await withController(installed: "[\(record(turbo, family: "minimax_music3"))]") { controller in
            controller.selectPage(.music)
            XCTAssertNil(controller.modelGate)
        }
    }

    func testGateWaitsForTheCatalogInsteadOfFlashingOverAnInstall() async throws {
        try await withController { controller in
            controller.selectPage(.voiceClone)
            XCTAssertNotNil(controller.modelGate)
            controller.isRefreshingModels = true
            XCTAssertNil(controller.modelGate)
            controller.isRefreshingModels = false
            controller.modelCatalogError = "offline"
            XCTAssertNil(controller.modelGate)
        }
    }

    func testChoosingARowSelectsItAndTheDownloadTakesThatPin() async throws {
        try await withController { controller in
            controller.selectPage(.voiceClone)
            var requested: [String] = []
            controller.installHandler = { profile, acknowledged, _ in
                requested.append(profile.identity.alias)
                XCTAssertTrue(acknowledged.isEmpty)
                return true
            }
            let fish = "fish-drama3-4bit"
            controller.selectModel(fish)
            XCTAssertEqual(controller.recipe.modelID, fish)
            controller.installSelectedModel()
            XCTAssertEqual(requested, [fish])
        }
    }

    func testEveryCuratedModelAsksForItsTermsBeforeInstalling() async throws {
        try await withController { controller in
            controller.selectPage(.voiceClone)
            let terms = Dictionary(uniqueKeysWithValues: controller.availableProfiles.map {
                ($0.identity.alias, $0.requiredAcknowledgments.map(\.id))
            })
            XCTAssertEqual(terms["megatts3-f32"], ["megatts3_community_encoder"])
            XCTAssertEqual(terms["omnivoice-8bit"], ["omnivoice_higgs_tokenizer"])
            XCTAssertEqual(terms["fish-drama3-4bit"], ["fish_audio_community_license"])
        }
    }

    func testDiscoverOffersTheSameCuratedCloneList() throws {
        let entries = AudioDiscoverCatalog.entries(profiles: try AudioCatalog.profiles(), canRun: { _ in true })
        let clones = entries.filter { $0.group == .voiceCloning }
        XCTAssertEqual(clones.map(\.id), curated)
    }
}
