import AVFoundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

final class AudioWorkflowTests: XCTestCase {
    @MainActor func testNativeFixtureTakeSavesReopensAndExportsWithoutHTTP() async throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let vault = ProfileVaultStore(rootProvider: { temporary.appendingPathComponent("vault") }, profileIDProvider: { "native-take" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let assets = ManagedAssetStore(vault: vault)
        let library = AudioLibraryStore(repository: ProfileRepository(store: vault))
        let controller = AudioWorkspaceController(library: library, assets: assets)
        defer { controller.shutdown() }
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model = root.appendingPathComponent("crates/audio/testdata/minimax_music3/converted_plain")
        controller.profiles = try AudioCatalog.profiles()
        controller.selectPage(.music)
        let profile = try XCTUnwrap(controller.selectedProfile)
        controller.preferences.localModels[profile.identity.alias] = try model.bookmarkData(options: .withSecurityScope, includingResourceValuesForKeys: nil, relativeTo: nil)
        controller.recipe.caption = "piano"; controller.recipe.lyrics = "[instrumental]"
        controller.recipe.text = "A retained voiceover draft"
        controller.recipe.durationSeconds = 0.12; controller.recipe.steps = 1; controller.recipe.seed = 7
        let unrelated = AudioLibraryItem(title: "Meeting", kind: .recording)
        try controller.save(unrelated)
        controller.selectedID = unrelated.id
        XCTAssertTrue(controller.canRun)
        controller.run()
        await controller.jobTask?.value
        XCTAssertNil(controller.error)
        let reopened = try XCTUnwrap(library.load().first(where: { $0.kind == .music }))
        XCTAssertEqual(reopened.status, .completed)
        XCTAssertEqual(reopened.title, "piano")
        XCTAssertNil(reopened.sourceID, "A new music prompt must not inherit an unrelated selected recording")
        XCTAssertEqual(reopened.resolvedSeed, 7)
        XCTAssertEqual(reopened.modelFamily, "minimax_music3")
        XCTAssertNil(reopened.modelIdentity, "A local folder must not claim a managed checkpoint pin")
        XCTAssertNotNil(reopened.localModelBookmark)
        XCTAssertEqual(reopened.clips.count, 1)
        let output = temporary.appendingPathComponent("video.wav")
        try AudioMediaIO.render(reopened.clips, store: assets, to: output, sampleRate: 48_000)
        let decoded = try AVAudioFile(forReading: output)
        XCTAssertEqual(decoded.processingFormat.sampleRate, 48_000)
        XCTAssertEqual(decoded.processingFormat.channelCount, 2)
        XCTAssertGreaterThan(decoded.length, 0)
        let m4a = temporary.appendingPathComponent("take.m4a")
        try await AudioMediaIO.encodeM4A(from: output, to: m4a)
        XCTAssertGreaterThan(try AVAudioFile(forReading: m4a).length, 0)
    }
}
