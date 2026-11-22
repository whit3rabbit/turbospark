import Foundation
import MLX
import XCTest

@testable import VoiceIsolatorMLX

/// Tests that need weights. Set VOICE_ISOLATOR_MODEL to the offline checkpoint
/// (model_fp32.safetensors of VoiceIsolator_v1_48K_MLX, or the released
/// denoiser.safetensors) and VOICE_ISOLATOR_REALTIME_MODEL to the realtime one;
/// under xcodebuild pass them as TEST_RUNNER_VOICE_ISOLATOR_MODEL and
/// TEST_RUNNER_VOICE_ISOLATOR_REALTIME_MODEL.
final class ModelTests: XCTestCase {
    private func path(_ name: String) throws -> URL {
        guard let p = ProcessInfo.processInfo.environment[name], !p.isEmpty else {
            throw XCTSkip("\(name) is not set")
        }
        return URL(fileURLWithPath: p)
    }

    /// A chirp in uniform noise, 48 kHz, the same on every run.
    private func signal(seconds: Double = 3) -> [Float] {
        let n = Int(seconds * 48000)
        var state: UInt64 = 0x9E37_79B9_7F4A_7C15
        return (0 ..< n).map { i in
            state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
            let noise = Float(Double(state >> 11) / Double(1 << 53)) * 0.1 - 0.05
            let t = Double(i) / 48000
            return Float(0.2 * sin(2 * .pi * (200 + 300 * t) * t)) + noise
        }
    }

    private func snr(_ ref: [Float], _ test: [Float]) -> Double {
        var s = 0.0, e = 0.0
        for (a, b) in zip(ref, test) {
            s += Double(a) * Double(a)
            e += Double(a - b) * Double(a - b)
        }
        return 10 * log10(s / e)
    }

    func testOfflineIsolationKeepsLengthAndRate() throws {
        let isolator = try VoiceIsolator(contentsOf: path("VOICE_ISOLATOR_MODEL"))
        let x = signal()

        let at48 = try isolator.isolate(x, sampleRate: 48000)
        let at16 = try isolator.isolate(Array(x.prefix(16000)), sampleRate: 16000)

        XCTAssertEqual(isolator.network.dtype, .float16)
        XCTAssertEqual(at48.count, x.count)
        XCTAssertEqual(at16.count, 16000)
        XCTAssertTrue(at48.allSatisfy { $0.isFinite })
    }

    func testHalfPrecisionStaysCloseToFloat32() throws {
        let url = try path("VOICE_ISOLATOR_MODEL")
        let x = signal()

        let y32 = try VoiceIsolator(contentsOf: url, dtype: .float32).isolate(x, sampleRate: 48000)
        let y16 = try VoiceIsolator(contentsOf: url, dtype: .float16).isolate(x, sampleRate: 48000)

        // Measured against the input's energy. On speech float16 stays 66-75 dB
        // from float32; on this synthetic chirp it measures about 50 dB, so the
        // floor guards against overflow and kernel errors, not rounding.
        let diff = zip(y32, y16).map { $0 - $1 }
        let relative = 10 * log10(x.reduce(0.0) { $0 + Double($1 * $1) }
            / diff.reduce(0.0) { $0 + Double($1 * $1) })
        XCTAssertGreaterThan(relative, 40, "voice SNR \(snr(y32, y16)) dB")
    }

    func testLongInputRendersInWindowsWithProgress() throws {
        let isolator = try VoiceIsolator(contentsOf: path("VOICE_ISOLATOR_MODEL"))
        var reported: [Double] = []

        let y = try isolator.isolate(signal(seconds: 15), sampleRate: 48000) { reported.append($0) }

        XCTAssertEqual(y.count, 15 * 48000)
        XCTAssertEqual(reported.count, 3)
        XCTAssertEqual(reported.last, 1)
    }

    func testCancellationStopsTheRender() async throws {
        let url = try path("VOICE_ISOLATOR_MODEL")
        let x = signal(seconds: 20)
        let task = Task.detached { () throws -> [Float] in
            let isolator = try VoiceIsolator(contentsOf: url)
            withUnsafeCurrentTask { $0?.cancel() }
            return try isolator.isolate(x, sampleRate: 48000)
        }

        do {
            _ = try await task.value
            XCTFail("expected CancellationError")
        } catch is CancellationError {}
    }

    func testStreamOutputDoesNotDependOnPushSizes() throws {
        let network = try Network(contentsOf: path("VOICE_ISOLATOR_REALTIME_MODEL"), dtype: .float32)
        let isolator = try VoiceIsolator(network: network, mode: .streaming)
        let x = signal(seconds: 2)
        var outputs: [[Float]] = []

        for sizes in [[480], [1000, 37, 4800]] {
            let session = try isolator.stream(windowFrames: 0)
            var out: [Float] = [], pos = 0, k = 0
            while pos < x.count {
                let step = sizes[k % sizes.count]
                out += try session.push(Array(x[pos ..< min(x.count, pos + step)]))
                pos += step
                k += 1
            }
            out += try session.flush()
            outputs.append(out)
        }

        XCTAssertEqual(outputs[0].count, x.count)
        XCTAssertEqual(outputs[1].count, x.count)
        var err: Float = 0
        for (a, b) in zip(outputs[0], outputs[1]) { err = max(err, abs(a - b)) }
        XCTAssertLessThan(err, 1e-5)
    }

    func testHalfPrecisionCheckpointKeepsFloat32Parts() throws {
        let src = try path("VOICE_ISOLATOR_MODEL")
        let dst = FileManager.default.temporaryDirectory.appendingPathComponent("vi-test-fp16.safetensors")
        defer { try? FileManager.default.removeItem(at: dst) }
        let loaded = try Weights.load(src)
        guard loaded.quantization == nil, loaded.arrays["blocks.0.time.qkv.weight"]?.dtype == .float32 else {
            throw XCTSkip("needs a float32 checkpoint")
        }

        let cast = Weights.cast(loaded.arrays, to: .float16)
        try save(arrays: cast, metadata: ["format": Weights.format,
                                          "format_version": "\(Weights.formatVersion)",
                                          "config": loaded.config.json(), "quantization": "null",
                                          "in_pad_multiple": "\(loaded.inMultiple)"], url: dst)
        let back = try Weights.load(dst)

        XCTAssertEqual(back.arrays["blocks.0.time.qkv.weight"]?.dtype, .float16)
        XCTAssertEqual(back.arrays["blocks.1.time.out.weight"]?.dtype, .float32)
        XCTAssertEqual(back.arrays["blocks.1.time.A"]?.dtype, .float32)
        XCTAssertEqual(back.arrays["stft.window"]?.dtype, .float32)
    }
}
