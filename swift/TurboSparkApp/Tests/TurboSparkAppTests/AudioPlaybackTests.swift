import AVFoundation
import Foundation
import XCTest
@testable import TurboSparkApp

final class AudioPlaybackTests: XCTestCase {
    @MainActor private func fixture() throws -> (AudioWorkspaceController, URL, URL) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "playback" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        let wav = root.appendingPathComponent("rendered.wav")
        try AudioMediaIO.wav(samples: [Float](repeating: 0, count: 16_000), sampleRate: 16_000, channels: 1).write(to: wav)
        return (controller, root, wav)
    }

    @MainActor func testPendingRecordingPreventsResumeSeekAndNewPlayback() throws {
        let (controller, root, wav) = try fixture()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        let player = try AVAudioPlayer(contentsOf: wav)
        controller.player = player
        controller.recordingSetupStatus = "Awaiting permission"
        controller.playPause()
        controller.seek(to: 0.5)
        XCTAssertFalse(player.isPlaying)
        XCTAssertEqual(player.currentTime, 0)
        var item = AudioLibraryItem(title: "Existing take", kind: .imported)
        item.clips = [AudioLibraryClip(assetReference: "unused", sampleRate: 16_000, channels: 1, frameCount: 16_000)]
        controller.play(item, at: 0)
        controller.compare(with: item)
        XCTAssertTrue(controller.player === player)
        XCTAssertNil(controller.playbackURL)
        XCTAssertNil(controller.error)
    }

    @MainActor func testRenderCompletionAfterRecordingSetupCannotActivatePlayer() throws {
        let (controller, root, wav) = try fixture()
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        controller.playbackURL = wav; controller.scratchURLs.insert(wav)
        let rendering = AudioOperationLifetime(); controller.playbackLifetime = rendering
        controller.recordingSetupStatus = "Awaiting permission"
        controller.finishPreparingPlayback(.success(()), destination: wav, at: 0, autoplay: false, token: controller.epoch)
        XCTAssertNil(controller.player)
        XCTAssertNil(controller.playbackURL)
        XCTAssertFalse(FileManager.default.fileExists(atPath: wav.path))
        XCTAssertFalse(controller.scratchURLs.contains(wav))
        XCTAssertThrowsError(try rendering.check())
        XCTAssertNil(controller.error)
    }
}
