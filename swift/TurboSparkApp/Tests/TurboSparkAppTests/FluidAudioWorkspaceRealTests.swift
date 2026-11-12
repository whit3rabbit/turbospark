import AVFoundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// End to end through the workspace with real catalog installs: the
/// Transcribe job, residency (verify, load, unload), the window loop, and the
/// saved transcript. Point TURBOSPARK_HOME at a scratch store.
///
/// - TURBOSPARK_FLUIDAUDIO_INSTALL=1 downloads both speech rows into it
///   (network, about 840 MB).
/// - TURBOSPARK_FLUIDAUDIO_E2E=1 transcribes through the controller. Run it
///   under a network-denied sandbox: nothing here may fetch.
@MainActor
final class FluidAudioWorkspaceRealTests: XCTestCase {
    private static let rows = ["parakeet-ja-coreml", "paraformer-zh-coreml"]

    private func requireStore(_ flag: String) throws {
        let env = ProcessInfo.processInfo.environment
        guard env[flag] == "1" else { throw XCTSkip("set \(flag)=1 to run") }
        guard let home = env["TURBOSPARK_HOME"], home.contains("scratch") || home.contains("tmp") else {
            throw XCTSkip("set TURBOSPARK_HOME to a scratch store, never the real one")
        }
    }

    func testInstallBothSpeechRows() throws {
        try requireStore("TURBOSPARK_FLUIDAUDIO_INSTALL")
        for alias in Self.rows {
            let profile = try XCTUnwrap(AudioCatalog.profiles().first { $0.identity.alias == alias })
            _ = try AudioCatalog.install(profile.identity, acknowledged: profile.requiredAcknowledgments.map(\.id))
        }
        let installed = Set(try AudioCatalog.installed().installed.map(\.alias))
        XCTAssertTrue(installed.isSuperset(of: Self.rows), "\(installed)")
    }

    /// Mono 16 kHz WAV bytes from macOS's own voices, and the frame count.
    private func speech(voice: String, _ text: String) throws -> (Data, Int64) {
        let wav = FileManager.default.temporaryDirectory.appendingPathComponent("say-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: wav) }
        let say = Process()
        say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        say.arguments = ["-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000", "-o", wav.path, text]
        try say.run()
        say.waitUntilExit()
        XCTAssertEqual(say.terminationStatus, 0)
        return (try Data(contentsOf: wav), try AVAudioFile(forReading: wav).length)
    }

    func testTranscribePageRunsBothRowsAndReleasesThem() async throws {
        try requireStore("TURBOSPARK_FLUIDAUDIO_E2E")
        let installed = try AudioCatalog.installed().installed
        guard Set(installed.map(\.alias)).isSuperset(of: Self.rows) else { throw XCTSkip("install both rows first") }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "audio-fluid-e2e" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(
            library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        controller.profiles = try AudioCatalog.profiles()
        controller.installedPaths = Dictionary(installed.map { ($0.alias, $0.path) }, uniquingKeysWith: { first, _ in first })
        controller.selectPage(.transcribe)

        // Long enough for two windows of each plan: Parakeet's 30 s windows
        // overlap, Paraformer's 15 s windows do not.
        let japanese = ["公園", "図書館", "喫茶店", "電車", "博物館", "病院", "映画館", "美術館", "動物園", "郵便局"]
        let chinese = ["公园", "图书馆", "博物馆", "医院", "电影院", "动物园"]
        let cases: [(String, String, String, [String])] = [
            ("parakeet-ja-coreml", "Kyoko", japanese.map { "午前中は友達と一緒に\($0)へ行って、ゆっくり過ごしました。" }.joined(), japanese),
            ("paraformer-zh-coreml", "Tingting", chinese.map { "今天上午我们和朋友一起去\($0)，在那里待了很久。" }.joined(), chinese),
        ]
        for (alias, voice, text, words) in cases {
            let (bytes, frames) = try speech(voice: voice, text)
            let asset = try controller.assets.store(data: bytes, fileName: "take.wav", mimeType: "audio/wav")
            var take = AudioLibraryItem(title: alias, kind: .recording)
            take.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 16_000, channels: 1, frameCount: frames)]
            try controller.save(take)
            controller.selectedID = take.id
            controller.selectModel(alias)
            XCTAssertEqual(controller.selectedProfile?.identity.alias, alias)
            XCTAssertTrue(controller.canRun, "\(alias) is not runnable")
            controller.run()
            let deadline = Date().addingTimeInterval(900)
            while controller.isBusy, Date() < deadline { try await Task.sleep(for: .milliseconds(250)) }
            XCTAssertFalse(controller.isBusy, "\(alias) did not finish")
            XCTAssertNil(controller.error, "\(alias): \(controller.error ?? "")")
            let result = try XCTUnwrap(controller.selectedItem)
            XCTAssertEqual(result.sourceID, take.id)
            XCTAssertEqual(result.status, .completed)
            let saved = try XCTUnwrap(result.preferredTranscript).segments.map(\.text).joined()
            let counts = words.map { saved.components(separatedBy: $0).count - 1 }
            XCTAssertTrue(counts.allSatisfy { $0 <= 1 }, "\(alias) repeated a word: \(saved)")
            XCTAssertGreaterThanOrEqual(counts.filter { $0 == 1 }.count, words.count - 2, "\(alias): \(saved)")
            XCTAssertEqual(controller.loadedModelAlias, alias, "the model stays resident after the job")
            XCTAssertTrue(controller.canUnloadModel)
        }
        controller.unloadModel()
        XCTAssertNil(controller.loadedModelAlias)
        XCTAssertNil(controller.fluidTranscriber)
    }
}
