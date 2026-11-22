import Foundation
import MLX
import os
import TurboSpark
import VoiceIsolatorMLX

/// The realtime Voice Isolator on the vendored MLX port, for live noise
/// reduction (`AudioLiveVoiceCleaner`).
///
/// The Rust stream's fallback network is per-call Metal, about RTF 3 on an
/// M4 Max, so live text gave up on it within a second. This port pushes
/// 20 ms in a median 5.8 ms (crates/audio/src/sts/voice_isolator/README.md).
/// It runs first even beside the Rust Core ML step: at capture packet sizes
/// that step, fixed at two frames a call, measured about 2.6 times slower.
final class VoiceIsolatorMLXLiveModel: @unchecked Sendable {
    nonisolated static let weightFile = "denoiser_realtime.safetensors"
    /// Most frames computed in one network call. A block never waits for
    /// audio (a push computes what it completed, up to this many frames at a
    /// time), and with the cache at the trained 1200-frame window the output
    /// does not depend on it, so this only sets the call count: a capture
    /// tap's 4096 samples complete about 8.5 frames, one call at 10. In the
    /// app's pipeline, 10 halved the median push against 2 (README.md).
    nonisolated static let blockFrames = 10

    private let isolator: VoiceIsolator
    /// One network serves every stream opened from it, and the library is
    /// not reentrant: a new take's stream can start while the last one
    /// flushes, so pushes and flushes of all streams take this lock.
    let lock = NSLock()

    /// Loads the released realtime checkpoint at its default look-ahead.
    /// `windowFrames: 0` streams without the library's crossfaded lanes,
    /// like the Rust port: the checkpoint was trained with a 1200-frame
    /// attention window, which the cache keeps exactly, so lanes would only
    /// change the output relative to the Rust stream.
    init(directory: URL, dtype: DType = .float16, blockFrames: Int = VoiceIsolatorMLXLiveModel.blockFrames) throws {
        let network = try Network(contentsOf: directory.appendingPathComponent(Self.weightFile), dtype: dtype)
        isolator = try VoiceIsolator(network: network, mode: .streaming, blockFrames: blockFrames, windowFrames: 0)
    }

    var latencySamples: Int { isolator.latencySamples }

    func openStream() throws -> VoiceIsolatorMLXLiveStream {
        lock.lock(); defer { lock.unlock() }
        return VoiceIsolatorMLXLiveStream(model: self, stream: try isolator.stream(windowFrames: 0))
    }
}

/// One take's MLX isolation stream, with the Rust stream's push contract:
/// 48 kHz mono in, released voice time-aligned to the input, and `late`
/// when a push took longer than the audio it carried.
final class VoiceIsolatorMLXLiveStream: AudioLiveEnhancing, @unchecked Sendable {
    let backend = "mlx"
    private let model: VoiceIsolatorMLXLiveModel
    private var stream: IsolationStream?

    fileprivate init(model: VoiceIsolatorMLXLiveModel, stream: IsolationStream) {
        self.model = model; self.stream = stream
    }

    func push(_ pcm: [Float]) throws -> AudioEnhancePush {
        // Same as the Rust wrapper: an empty push never reaches the library.
        guard !pcm.isEmpty else { return AudioEnhancePush(voice: [], late: false, processingMs: 0) }
        model.lock.lock(); defer { model.lock.unlock() }
        guard let stream else { throw Self.closedError }
        let start = DispatchTime.now().uptimeNanoseconds
        let voice = try stream.push(pcm)
        let ms = Double(DispatchTime.now().uptimeNanoseconds - start) / 1e6
        let audioMs = Double(pcm.count) * 1000 / Double(AudioEnhanceStream.sampleRate)
        return AudioEnhancePush(voice: voice, late: ms > audioMs, processingMs: ms)
    }

    func flush() throws -> [Float] {
        model.lock.lock(); defer { model.lock.unlock() }
        guard let stream else { throw Self.closedError }
        return try stream.flush()
    }

    func close() {
        model.lock.lock(); defer { model.lock.unlock() }
        stream = nil
    }

    private static let closedError = TurboSparkError(code: .invalidArgument, message: "Voice isolation stream is closed")
}

private let liveIsolatorLog = Logger(subsystem: "com.turbospark.app", category: "VoiceIsolatorMLX")

extension AudioWorkspaceController {
    /// Opens one take's live isolation stream on the install at `path`:
    /// MLX first, whether or not the Core ML step is installed. Only an MLX
    /// step failure opens the Rust stream instead, on `auto` (its Core ML
    /// step when installed on macOS 15, else Metal). Finding, moving and
    /// verification errors propagate, and the transcriber turns them into
    /// "Background noise reduction is off".
    func openLiveCleaner(path: String) async throws -> AudioLiveEnhancing {
        let openRust = liveCleanerDependencies.openRust
        do {
            let model = try await residentLiveVoiceIsolator(path: path)
            do { return try model.openStream() } catch { throw VoiceIsolatorMLXLoadFailure(underlying: error) }
        } catch let failure as VoiceIsolatorMLXLoadFailure {
            liveIsolatorLog.error("MLX live load failed, using the Rust stream: \(String(describing: failure.underlying), privacy: .public)")
            return try await Task.detached { try openRust(path) }.value
        }
    }

    /// The resident MLX realtime isolator, loaded once and kept for later
    /// takes. A managed install loads only after its bytes verify, as
    /// `residentVoiceIsolator` does. It has its own slot rather than the one
    /// resident audio model, so starting a take neither evicts the model the
    /// user loaded nor shows load status over the recording; it is released
    /// wherever `qwenModel` is.
    func residentLiveVoiceIsolator(path: String) async throws -> VoiceIsolatorMLXLiveModel {
        if let live = liveVoiceIsolator, live.path == path { return live.model }
        guard let profile = liveNoiseReductionProfile else {
            throw AudioWorkspaceError.message(AppStrings.localized("Download this model or choose its checkpoint folder."))
        }
        if modelMoving?(path) == true { throw AudioWorkspaceError.message(ModelMoveText.modelBeingMoved()) }
        let token = epoch
        let deps = voiceIsolatorLoadDependencies
        let openToken = deps.openToken()
        let verified = try await Task.detached { () throws -> String in
            guard let openToken else { throw CancellationError() }
            return try deps.verify(profile.identity, openToken) { _ in }
        }.value
        guard verified == path, valid(token) else { throw CancellationError() }
        let directory = URL(fileURLWithPath: path, isDirectory: true)
        let loaded = try await Task.detached { () throws -> VoiceIsolatorMLXLiveModel in
            do { return try deps.loadLive(directory) } catch { throw VoiceIsolatorMLXLoadFailure(underlying: error) }
        }.value
        guard valid(token) else { throw CancellationError() }
        liveVoiceIsolator = (path, loaded)
        return loaded
    }
}

/// The Rust stream open, replaceable so tests can see which engine opened.
struct AudioLiveCleanerDependencies {
    var openRust: @Sendable (String) throws -> AudioLiveEnhancing = { try AudioEnhanceStream(path: $0) }
}
