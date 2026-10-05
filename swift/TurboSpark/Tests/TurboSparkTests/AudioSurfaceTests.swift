import Foundation
import XCTest

@testable import TurboSpark

/// The Swift half of the `ts_audio_*` contract, linked against the real
/// staticlib. Model-free: fixtures are WAVs the engine itself writes.
final class AudioSurfaceTests: XCTestCase {
    private var scratch: URL!

    override func setUpWithError() throws {
        scratch = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-audio-swift-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: scratch)
    }

    /// One second of 440 Hz at 48 kHz, written as a float WAV, then
    /// re-encoded by the engine so the fixture never comes from Swift code.
    private func tone() throws -> URL {
        let raw = scratch.appendingPathComponent("raw.wav")
        var samples = [Float](repeating: 0, count: 48_000)
        for i in samples.indices {
            samples[i] = 0.5 * sin(2 * Float.pi * 440 * Float(i) / 48_000)
        }
        // Minimal float WAV header, written here only to seed the engine.
        var data = Data()
        func append<T: FixedWidthInteger>(_ value: T) {
            withUnsafeBytes(of: value.littleEndian) { data.append(contentsOf: $0) }
        }
        data.append(contentsOf: Array("RIFF".utf8)); append(UInt32(36 + samples.count * 4))
        data.append(contentsOf: Array("WAVEfmt ".utf8)); append(UInt32(16)); append(UInt16(3))
        append(UInt16(1)); append(UInt32(48_000)); append(UInt32(48_000 * 4)); append(UInt16(4))
        append(UInt16(32)); data.append(contentsOf: Array("data".utf8)); append(UInt32(samples.count * 4))
        for sample in samples { append(sample.bitPattern) }
        try data.write(to: raw)
        return raw
    }

    func testCapabilitiesDecodeAndRefuseEveryModelTask() throws {
        let caps = try TurboSparkAudio.capabilities()
        XCTAssertTrue(caps.decodeExtensions.contains("m4a"))
        XCTAssertTrue(caps.refusedExtensions.contains("opus"))
        XCTAssertFalse(caps.speechToText.active)
        XCTAssertFalse(caps.textToSpeech.active)
        XCTAssertNotNil(caps.speechToText.reason)
    }

    func testProbePeaksAndSpeechConversion() throws {
        let source = try tone()
        let probe = try TurboSparkAudio.probe(source)
        XCTAssertEqual(probe.sampleRate, 48_000)
        XCTAssertEqual(probe.durationSeconds, 1.0, accuracy: 1e-9)

        let peaks = try TurboSparkAudio.peaks(source, buckets: 64)
        XCTAssertEqual(peaks.peaks.count, 64)

        let target = scratch.appendingPathComponent("speech.wav")
        let report = try TurboSparkAudio.convert(source, to: target, options: .speech)
        XCTAssertEqual(report.sampleRate, 16_000)
        XCTAssertEqual(report.channels, 1)
        XCTAssertEqual(report.frames, 16_000)
    }

    func testARefusedFormatThrowsUnsupportedWithASentence() throws {
        let opus = scratch.appendingPathComponent("voice.opus")
        try Data("OggS".utf8).write(to: opus)
        XCTAssertThrowsError(try TurboSparkAudio.probe(opus)) { error in
            let error = error as? TurboSparkError
            XCTAssertEqual(error?.code, .unsupportedPlatform)
            XCTAssertTrue(error?.message.contains("convert") ?? false)
        }
    }

    func testDisplayLevelMapsFullScaleToOne() {
        let square: [Float] = [1, -1, 1, -1]
        let level = square.withUnsafeBufferPointer {
            TurboSparkAudio.displayLevel($0.baseAddress!, count: $0.count)
        }
        XCTAssertEqual(level, 1, accuracy: 1e-6)
    }

    func testSessionOpenIsRefusedWithTheEngineReason() async throws {
        do {
            _ = try await TurboSparkAudioSession(modelPath: "/nonexistent/audio-model")
            XCTFail("an audio session must not open while no family exists")
        } catch let error as TurboSparkError {
            XCTAssertEqual(error.code, .unsupportedPlatform)
            XCTAssertEqual(error.message, try TurboSparkAudio.capabilities().speechToText.reason)
        }
    }
}
