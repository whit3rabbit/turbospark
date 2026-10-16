import FluidAudio
import XCTest
@testable import TurboSparkApp

/// The offline contract FluidAudio 0.15.5 gives the app once the engine has
/// set its switches (docs/FLUIDAUDIO.md). None of these reaches the network:
/// with the switches set, a fetch fails before any request leaves. A version
/// bump that changes this behavior fails here.
final class FluidAudioEngineTests: XCTestCase {
    private func scratch() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("fluidaudio-offline-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    func testTouchingTheEngineSetsEveryOfflineSwitch() {
        _ = FluidAudioEngine.shared
        XCTAssertTrue(FluidAudioEngine.isOfflineEnforced)
        XCTAssertTrue(ModelHub.offlineMode)
        XCTAssertEqual(ModelRegistry.baseURL, FluidAudioEngine.blockedRegistryURL)
    }

    /// A model that is not on disk is refused, not fetched, and nothing
    /// beside it is deleted.
    func testAMissingModelIsRefusedWithoutAFetch() async throws {
        FluidAudioEngine.enforceOffline()
        let dir = try scratch()
        let sentinel = dir.appendingPathComponent("keep.txt")
        try Data("keep".utf8).write(to: sentinel)
        do {
            _ = try await ModelHub.loadModels(
                .vad, modelNames: ["silero-vad-unified-256ms-v6.2.1.mlmodelc"], directory: dir)
            XCTFail("an absent model loaded")
        } catch let error as DownloadError {
            guard case .modelMissing = error else { return XCTFail("expected modelMissing, got \(error)") }
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: sentinel.path))
    }

    /// `AssetDownloader` ignores offline mode. The blocked registry makes an
    /// unpinned voice fail at once and leaves no file behind.
    func testAnUnpinnedVoiceFailsAtOnceAgainstTheBlockedRegistry() async throws {
        FluidAudioEngine.enforceOffline()
        let dir = try scratch()
        let started = Date()
        do {
            _ = try await KokoroAneResourceDownloader.ensureVoicePack("bf_emma", repoDirectory: dir)
            XCTFail("an unpinned voice was fetched")
        } catch {}
        XCTAssertLessThan(Date().timeIntervalSince(started), 5)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent("bf_emma.bin").path))
    }
}
