import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Vocal separation on the Clean Up page: the engine refuses a job whose
/// request names no `input_sample_rate` or `input_channels`, and the page
/// reads takes as 44.1 kHz interleaved stereo (`AudioMediaIO.readStereo44kTake`),
/// so the run request must say 44100 and 2 on the wire. Enhancement keeps its
/// mono 48 kHz request with no channel count.
final class AudioSeparationRequestTests: XCTestCase {
    private func profile(task: AudioTask, family: String) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: task, alias: family + "-test", repository: "someone/" + family,
                                           revision: String(repeating: "0", count: 40), assetFingerprint: family),
            family: family, displayName: family,
            capabilities: AudioCapabilities(languages: [], operations: ["separate"], backend: "portable_cpu",
                                            cancellation: "chunk", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 44_100, channels: 2), readiness: "implemented_unqualified")
    }

    private func wire(_ request: AudioRequest) throws -> [String: Any] {
        let encoder = JSONEncoder()
        encoder.keyEncodingStrategy = .convertToSnakeCase
        return try XCTUnwrap(try JSONSerialization.jsonObject(with: encoder.encode(request)) as? [String: Any])
    }

    @MainActor func testSeparationRunSendsTheLayoutItReadsIn() throws {
        XCTAssertTrue(AudioTask.separation.isRunnable)
        XCTAssertTrue(AudioTask.runnable.contains(.separation))
        let recipe = AudioRecipe()
        let separator = profile(task: .separation, family: "mel_roformer")
        let fields = try wire(AudioWorkspaceController.runRequest(recipe, task: .separation, profile: separator))
        XCTAssertEqual(fields["task"] as? String, "separation")
        XCTAssertEqual(fields["input_sample_rate"] as? Int, 44_100)
        XCTAssertEqual(fields["input_channels"] as? Int, 2)

        let enhancer = profile(task: .enhancement, family: "mossformer2_se")
        let enhancement = try wire(AudioWorkspaceController.runRequest(recipe, task: .enhancement, profile: enhancer))
        XCTAssertEqual(enhancement["input_sample_rate"] as? Int, 48_000)
        XCTAssertNil(enhancement["input_channels"], "only separation sends a channel count")
    }

    /// A stereo take comes back interleaved with its channels kept apart
    /// across the 30 s window boundary, a mono take is upmixed to both
    /// channels, and a take over the limit is refused before any read.
    func testStereoTakeReadKeepsChannelsAndUpmixesMono() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root.appendingPathComponent("vault") }, profileIDProvider: { "separation-read" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let assets = ManagedAssetStore(vault: vault)

        let frames = 31 * 44_100
        var stereo = [Float](repeating: 0, count: frames * 2)
        for frame in 0..<frames { stereo[2 * frame] = 0.25; stereo[2 * frame + 1] = -0.125 }
        let stereoAsset = try assets.store(data: AudioMediaIO.wav(samples: stereo, sampleRate: 44_100, channels: 2), fileName: "stereo.wav", mimeType: "audio/wav")
        let stereoClip = AudioLibraryClip(assetReference: stereoAsset.storedReference, sampleRate: 44_100, channels: 2, frameCount: Int64(frames))
        let read = try AudioMediaIO.readStereo44kTake([stereoClip], store: assets, duration: stereoClip.duration, limit: 326)
        XCTAssertEqual(read.count, frames * 2)
        for frame in [0, 30 * 44_100 - 1, 30 * 44_100, frames - 1] {
            XCTAssertEqual(read[2 * frame], 0.25, accuracy: 1e-6, "left at frame \(frame)")
            XCTAssertEqual(read[2 * frame + 1], -0.125, accuracy: 1e-6, "right at frame \(frame)")
        }

        let mono = [Float](repeating: 0.5, count: 44_100)
        let monoAsset = try assets.store(data: AudioMediaIO.wav(samples: mono, sampleRate: 44_100, channels: 1), fileName: "mono.wav", mimeType: "audio/wav")
        let monoClip = AudioLibraryClip(assetReference: monoAsset.storedReference, sampleRate: 44_100, channels: 1, frameCount: 44_100)
        let upmixed = try AudioMediaIO.readStereo44kTake([monoClip], store: assets, duration: monoClip.duration, limit: 326)
        XCTAssertEqual(upmixed.count, 2 * 44_100)
        XCTAssertEqual(upmixed[1000], upmixed[1001], "both channels carry the mono take")
        XCTAssertGreaterThan(abs(upmixed[1000]), 0.1)

        XCTAssertThrowsError(try AudioMediaIO.readStereo44kTake([stereoClip], store: assets, duration: stereoClip.duration, limit: 30))
    }
}
