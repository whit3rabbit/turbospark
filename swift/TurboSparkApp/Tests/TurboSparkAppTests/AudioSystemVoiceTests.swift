import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Text-to-Speech speaks with the macOS voices until a model is installed;
/// Voice Cloning has no such fallback and keeps its gate.
@MainActor
final class AudioSystemVoiceTests: XCTestCase {
    private let speechModel = "qwen3-tts-12hz-06b-customvoice-8bit"

    private func withController(
        installed: String = "[]", _ body: (AudioWorkspaceController) async throws -> Void
    ) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "system-voice" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        let report = try decoder.decode(AudioInstalledReport.self, from: Data("""
            {"installed":\(installed),"not_connected":[],"needs_adoption":[],"incompatible":[],"other_audio":[]}
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

    func testNoInstalledModelSpeaksWithTheSystemVoice() async throws {
        try await withController { controller in
            controller.selectPage(.voiceover)
            XCTAssertEqual(controller.speechEngine, .systemVoice)
            XCTAssertNil(controller.selectedProfile)
            XCTAssertEqual(controller.runTitle, "Read aloud")
            // Runnable with a script and nothing else: no model, no download.
            XCTAssertFalse(controller.canRun)
            controller.recipe.text = "Hello from the system voice."
            XCTAssertTrue(controller.canRun)
        }
    }

    func testAnInstalledModelTakesOverFromTheSystemVoice() async throws {
        let record = """
        [{"alias":"\(speechModel)","path":"/models/\(speechModel)","family":"qwen3_tts","status":"runs","install_bytes":100}]
        """
        try await withController(installed: record) { controller in
            controller.selectPage(.voiceover)
            XCTAssertEqual(controller.speechEngine, .model(speechModel))
            XCTAssertFalse(controller.usesSystemVoice)
            XCTAssertEqual(controller.runTitle, "Generate speech")
        }
    }

    func testVoiceCloningHasNoSystemVoiceFallback() async throws {
        try await withController { controller in
            controller.selectPage(.voiceClone)
            XCTAssertNil(controller.speechEngine)
            XCTAssertNotNil(controller.modelGate)
            controller.recipe.text = "A script."
            XCTAssertFalse(controller.canRun)
        }
    }
}
