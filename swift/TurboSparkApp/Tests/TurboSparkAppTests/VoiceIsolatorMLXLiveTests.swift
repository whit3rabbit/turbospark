import Foundation
import MLX
import Metal
import TurboSpark
import VoiceIsolatorMLX
import XCTest

@testable import TurboSparkApp

/// Live noise reduction streams on the vendored MLX port, and opens the
/// Rust stream (its Core ML step when installed, else Metal) only when the
/// MLX step fails (VoiceIsolatorMLXLive.swift).
@MainActor
final class VoiceIsolatorMLXLiveTests: XCTestCase {
    private final class StubStream: AudioLiveEnhancing, @unchecked Sendable {
        let backend: String
        init(_ backend: String) { self.backend = backend }
        func push(_ pcm: [Float]) throws -> AudioEnhancePush { AudioEnhancePush(voice: pcm, late: false, processingMs: 0) }
        func flush() throws -> [Float] { [] }
        func close() {}
    }

    private final class Calls: @unchecked Sendable {
        private let lock = NSLock()
        private var names: [String] = []
        func add(_ n: String) { lock.lock(); names.append(n); lock.unlock() }
        var all: [String] { lock.lock(); defer { lock.unlock() }; return names }
    }

    private func realtimeProfile() -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .enhancement, alias: AudioWorkspaceController.liveNoiseReductionAlias,
                                           repository: "inworld/voice-isolator", revision: String(repeating: "0", count: 40),
                                           assetFingerprint: "f"),
            family: "voice_isolator", displayName: "Realtime",
            capabilities: AudioCapabilities(operations: ["enhance"], backend: "metal", cancellation: "chunk", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 48_000, channels: 1), readiness: "implemented_unqualified")
    }

    private func fixture() throws -> (AudioWorkspaceController, model: URL, root: URL, calls: Calls) {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let vault = ProfileVaultStore(rootProvider: { root.appendingPathComponent("vault") }, profileIDProvider: { "voice-isolator-live" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let model = root.appendingPathComponent("model", isDirectory: true)
        try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        controller.profiles = [realtimeProfile()]
        controller.installedPaths[AudioWorkspaceController.liveNoiseReductionAlias] = model.path
        let calls = Calls()
        controller.liveCleanerDependencies.openRust = { _ in calls.add("rust"); return StubStream("rust") }
        return (controller, model, root, calls)
    }

    // MARK: Selection

    /// MLX runs first even with the Core ML step installed: the step only
    /// matters to the Rust fallback, which opens after the MLX load fails.
    func testACoreMLInstallStillLoadsMLXFirst() async throws {
        let (controller, model, root, calls) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        for program in ["VoiceIsolatorStep.mlmodelc", "VoiceIsolatorStep.mlpackage"] {
            try FileManager.default.createDirectory(at: model.appendingPathComponent(program), withIntermediateDirectories: true)
        }
        struct Broken: Error {}
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in model.path }
        controller.voiceIsolatorLoadDependencies.loadLive = { _ in calls.add("mlx"); throw Broken() }
        let stream = try await controller.openLiveCleaner(path: model.path)
        XCTAssertEqual(stream.backend, "rust")
        XCTAssertEqual(calls.all, ["mlx", "rust"], "MLX was tried before the Rust stream")
    }

    /// An MLX step failure falls back to the Rust stream; nothing resident.
    func testAFailedMLXLoadFallsBackToTheRustStream() async throws {
        let (controller, model, root, calls) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        struct Broken: Error {}
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in model.path }
        controller.voiceIsolatorLoadDependencies.loadLive = { _ in throw Broken() }
        let stream = try await controller.openLiveCleaner(path: model.path)
        XCTAssertEqual(stream.backend, "rust")
        XCTAssertEqual(calls.all, ["rust"])
        XCTAssertNil(controller.liveVoiceIsolator)
    }

    /// A failed verification is not an MLX failure: it propagates rather
    /// than opening the Rust stream on the same unverified bytes.
    func testAVerificationFailureIsNotMaskedByTheFallback() async throws {
        let (controller, model, root, calls) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        struct Corrupt: Error {}
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in throw Corrupt() }
        controller.voiceIsolatorLoadDependencies.loadLive = { _ in XCTFail("an unverified install must not load"); throw CancellationError() }
        do {
            _ = try await controller.openLiveCleaner(path: model.path)
            XCTFail("a verification failure must propagate")
        } catch {
            XCTAssertTrue(error is Corrupt, "\(error)")
        }
        XCTAssertEqual(calls.all, [])
        XCTAssertNil(controller.liveVoiceIsolator)
    }

    /// A model being moved is refused before any verify or load.
    func testAMovingInstallIsRefused() async throws {
        let (controller, model, root, calls) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.modelMoving = { $0 == model.path }
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in XCTFail("must not verify"); return model.path }
        do { _ = try await controller.openLiveCleaner(path: model.path); XCTFail("a moving install must be refused") } catch {}
        XCTAssertEqual(calls.all, [])
    }

    // MARK: Real checkpoint

    /// `TURBOSPARK_VOICE_ISOLATOR_MODEL`: a folder with
    /// `denoiser_realtime.safetensors` and `real_clip.npy` (10 s of speech
    /// in white noise, 48 kHz), the one the Rust parity tests use.
    private func realFolder() throws -> URL {
        try XCTSkipIf(MTLCreateSystemDefaultDevice() == nil, "MLX needs a Metal device")
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_VOICE_ISOLATOR_MODEL"], !path.isEmpty else {
            throw XCTSkip("TURBOSPARK_VOICE_ISOLATOR_MODEL is not set")
        }
        return URL(fileURLWithPath: (path as NSString).expandingTildeInPath, isDirectory: true)
    }

    private func readNpy(_ url: URL) throws -> [Float] {
        let data = try Data(contentsOf: url)
        XCTAssertEqual(data.prefix(6), Data([0x93]) + Data("NUMPY".utf8))
        let headerLength = Int(data[8]) | Int(data[9]) << 8
        let header = String(decoding: data[10 ..< 10 + headerLength], as: UTF8.self)
        XCTAssertTrue(header.contains("'<f4'") && header.contains("False"), header)
        return data.dropFirst(10 + headerLength).withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }
    }

    /// Pushes `clip` in `piece`-sample pushes, then flushes.
    private func run(_ stream: AudioLiveEnhancing, _ clip: [Float], piece: Int) throws -> [Float] {
        var out: [Float] = []
        for start in stride(from: 0, to: clip.count, by: piece) {
            out += try stream.push(Array(clip[start ..< min(clip.count, start + piece)])).voice
        }
        out += try stream.flush()
        stream.close()
        return out
    }

    private func snr(_ reference: [Float], _ test: [Float]) -> Double {
        var signal = 0.0, error = 0.0
        for (a, b) in zip(reference, test) {
            signal += Double(a) * Double(a); error += Double(a - b) * Double(a - b)
        }
        return 10 * log10(signal / error)
    }

    /// The MLX stream against the Rust stream on the same realtime file:
    /// float32 to float rounding, the app's float16 within upstream's 40 dB
    /// fp16 gate. Push sizes differ on purpose (capture's 4096-frame taps on
    /// MLX, 20 ms on Rust): with the cache at the trained window, neither
    /// output depends on them. Rust runs its CPU port here, the reference
    /// path its own parity tests pin.
    func testRealStreamMatchesTheRustStream() throws {
        let folder = try realFolder()
        let clip = try readNpy(folder.appendingPathComponent("real_clip.npy"))
        let rust = try run(AudioEnhanceStream(path: folder.path, backend: .cpu), clip, piece: 960)
        XCTAssertEqual(rust.count, clip.count)

        let fp32 = try VoiceIsolatorMLXLiveModel(directory: folder, dtype: .float32)
        let y32 = try run(fp32.openStream(), clip, piece: 4096)
        XCTAssertEqual(y32.count, clip.count)
        let maxAbs = zip(y32, rust).map { abs($0 - $1) }.max() ?? .infinity
        print("voice isolator live MLX fp32 vs Rust CPU stream: max abs \(maxAbs), SNR \(snr(rust, y32)) dB")
        XCTAssertLessThan(maxAbs, 1e-4)

        let fp16 = try VoiceIsolatorMLXLiveModel(directory: folder)
        let y16 = try run(fp16.openStream(), clip, piece: 4096)
        XCTAssertEqual(y16.count, clip.count)
        let snr16 = snr(rust, y16)
        print("voice isolator live MLX fp16 vs Rust CPU stream: SNR \(snr16) dB")
        XCTAssertGreaterThan(snr16, 40)
    }

    /// The real checkpoint with a Core ML step beside it streams on MLX and
    /// never opens the Rust stream.
    func testRealCoreMLInstallStreamsOnMLX() async throws {
        let folder = try realFolder()
        let (controller, model, root, calls) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createSymbolicLink(at: model.appendingPathComponent(VoiceIsolatorMLXLiveModel.weightFile),
                                                   withDestinationURL: folder.appendingPathComponent(VoiceIsolatorMLXLiveModel.weightFile))
        try FileManager.default.createDirectory(at: model.appendingPathComponent("VoiceIsolatorStep.mlmodelc"), withIntermediateDirectories: true)
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in model.path }
        let stream = try await controller.openLiveCleaner(path: model.path)
        XCTAssertEqual(stream.backend, "mlx")
        XCTAssertEqual(calls.all, [])
        stream.close()
    }

    /// Through the controller's real load: one load for two takes, a stream
    /// per take, and released with the other resident models.
    func testRealLoadIsResidentAcrossTakesAndReleased() async throws {
        let folder = try realFolder()
        let (controller, _, root, calls) = try fixture()
        defer { try? FileManager.default.removeItem(at: root) }
        controller.installedPaths[AudioWorkspaceController.liveNoiseReductionAlias] = folder.path
        // Not a managed install: stand in for the catalog's byte check.
        controller.voiceIsolatorLoadDependencies.verify = { _, _, _ in folder.path }
        let loads = Calls()
        controller.voiceIsolatorLoadDependencies.loadLive = { loads.add("load"); return try VoiceIsolatorMLXLiveModel(directory: $0) }
        let first = try await controller.openLiveCleaner(path: folder.path)
        let second = try await controller.openLiveCleaner(path: folder.path)
        XCTAssertEqual(first.backend, "mlx")
        XCTAssertFalse(first === second, "each take gets its own stream")
        XCTAssertEqual(loads.all, ["load"])
        XCTAssertEqual(calls.all, [])
        let push = try first.push([Float](repeating: 0, count: 4096))
        XCTAssertGreaterThan(push.processingMs, 0)
        first.close(); second.close()
        XCTAssertNotNil(controller.liveVoiceIsolator)
        controller.unloadModel()
        XCTAssertNil(controller.liveVoiceIsolator)
    }

    /// In-app latency: the take's real pipeline (capture feed, transcriber
    /// queue, 100 ms pump, cleaner, mixer) fed in real time with
    /// capture-sized packets (4096 frames at 48 kHz, `AudioCapture`'s tap),
    /// with a stand-in live text stream. Prints the cleaner's own timings.
    /// `TURBOSPARK_LIVE_NOISE_COREML` names a folder with the Core ML step
    /// to time the Rust Core ML stream the same way.
    func testRealLiveLatencyUnderTheCapturePipeline() async throws {
        let folder = try realFolder()
        var clip = try readNpy(folder.appendingPathComponent("real_clip.npy"))
        // 30 s: three passes of the clip.
        clip = clip + clip + clip
        let mlx: (String, @Sendable () throws -> AudioLiveEnhancing) = ("mlx fp16", { try VoiceIsolatorMLXLiveModel(directory: folder).openStream() })
        var engines = [mlx, mlx]
        if let coreml = ProcessInfo.processInfo.environment["TURBOSPARK_LIVE_NOISE_COREML"], !coreml.isEmpty {
            // `.auto`, as the app's fallback opens it: Core ML, the step is there.
            // Interleaved, so the second Core ML take runs on a warm cache.
            let step: (String, @Sendable () throws -> AudioLiveEnhancing) = ("rust coreml", { try AudioEnhanceStream(path: (coreml as NSString).expandingTildeInPath) })
            engines = [mlx, step, mlx, step]
        }
        for (name, open) in engines {
            // Only the MLX arm is gated: the Core ML arm is a measurement.
            let timings = try await timedTake(clip, open: open, gate: name.hasPrefix("mlx"))
            print("live noise reduction in-app, \(name): \(timings.summary)")
            guard name.hasPrefix("mlx") else { continue }
            XCTAssertGreaterThan(timings.processingMs.count, 300)
            // The first push compiles the graph and can run late once; the
            // fallback needs a second of late pushes in a row.
            XCTAssertLessThanOrEqual(timings.lateCount, 2, name)
        }
    }

    private final class SilentLiveText: AudioLiveStreaming, @unchecked Sendable {
        func append(_ pcm: [Float]) throws -> AudioStreamUpdate { AudioStreamUpdate() }
        func finish() throws -> AudioStreamUpdate { AudioStreamUpdate(isFinal: true) }
        func cancel() {}
        func close() {}
    }

    private nonisolated func timedTake(_ clip: [Float], open: @escaping @Sendable () throws -> AudioLiveEnhancing,
                                       gate: Bool) async throws -> AudioLivePushTimings {
        let feed = AudioLiveFeed()
        let stopped = Calls()
        let live = AudioLiveTranscriber(feed: feed, openCleaner: { try open() }, open: { SilentLiveText() }) { event in
            if case .cleaningStopped(let reason) = event { stopped.add(reason) }
        }
        live.start()
        let deadline = Date().addingTimeInterval(60)
        while !live.isCleaningForTesting && Date() < deadline { try await Task.sleep(for: .milliseconds(10)) }
        XCTAssertTrue(live.isCleaningForTesting, "the cleaner opened")
        // Real-time pacing from another thread, as the capture tap delivers.
        let piece = 4096
        let started = DispatchTime.now().uptimeNanoseconds
        for (index, start) in stride(from: 0, to: clip.count, by: piece).enumerated() {
            let due = started + UInt64(Double(index * piece) / 48_000 * 1e9)
            let now = DispatchTime.now().uptimeNanoseconds
            if due > now { try await Task.sleep(nanoseconds: due - now) }
            _ = feed.offer(Array(clip[start ..< min(clip.count, start + piece)]), sampleRate: 48_000, channels: 1,
                           offset: Double(start) / 48_000, source: "microphone")
        }
        // Two pumps, so the last packets reach the cleaner.
        try await Task.sleep(for: .milliseconds(250))
        let timings = live.cleaningTimingsForTesting
        _ = await live.finishAndClose()
        if gate { XCTAssertEqual(stopped.all, [], "cleaning kept up") } else if !stopped.all.isEmpty { print("cleaning stopped: \(stopped.all)") }
        return timings
    }
}
