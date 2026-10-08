import AVFoundation
import XCTest
@testable import TurboSparkApp

final class AudioMediaIOTests: XCTestCase {
    func testFloatWAVPreservesStereoAndSampleRate() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        let samples: [Float] = [0.1, -0.2, 0.3, -0.4]
        try AudioMediaIO.wav(samples: samples, sampleRate: 48_000, channels: 2).write(to: url)
        let file = try AVAudioFile(forReading: url)
        XCTAssertEqual(file.length, 2)
        XCTAssertEqual(file.processingFormat.sampleRate, 48_000)
        let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 2)!
        try file.read(into: buffer)
        XCTAssertEqual(buffer.floatChannelData![0][1], 0.3, accuracy: 1e-7)
        XCTAssertEqual(buffer.floatChannelData![1][0], -0.2, accuracy: 1e-7)
    }

    func testNativeAndVideoExportPreserveExactFrameCount() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root.appendingPathComponent("vault") }, profileIDProvider: { "exact-frames" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let assets = ManagedAssetStore(vault: vault)
        let frameCount = 32_400
        let sampleRate = 24_000.0
        let bytes = try AudioMediaIO.wav(samples: [Float](repeating: 0.125, count: frameCount), sampleRate: sampleRate, channels: 1)
        let asset = try assets.store(data: bytes, fileName: "take.wav", mimeType: "audio/wav")
        let clip = AudioLibraryClip(assetReference: asset.storedReference, sampleRate: sampleRate, channels: 1, frameCount: Int64(frameCount))
        // 1.35 * 24000 rounds slightly above 32400 in binary floating point.
        // Export must preserve frames, including when converting to the video rate.
        for (name, rate, expectedFrames) in [("native", sampleRate, Int64(32_400)), ("video", 48_000.0, Int64(64_800))] {
            let url = root.appendingPathComponent(name + ".wav")
            try AudioMediaIO.render([clip], store: assets, to: url, sampleRate: name == "native" ? nil : rate)
            let file = try AVAudioFile(forReading: url)
            XCTAssertEqual(file.processingFormat.sampleRate, rate)
            XCTAssertEqual(file.length, expectedFrames)
        }
        let window = try AudioMediaIO.readMono16k([clip], store: assets, start: 0, duration: clip.duration)
        XCTAssertEqual(window.count, 21_600)
    }

    func testTimelineMixKeepsSourcesAndSequentialChunksAligned() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        var clips: [AudioLibraryClip] = []
        for (name, value, offset, source) in [("a.wav", Float(0.25), 0.0, "microphone"), ("b.wav", Float(0.5), 1.0, "microphone"), ("c.wav", Float(0.1), 0.5, "system")] {
            try AudioMediaIO.wav(samples: Array(repeating: value, count: 16_000), sampleRate: 16_000, channels: 1).write(to: root.appendingPathComponent(name))
            clips.append(AudioLibraryClip(assetReference: name, sampleRate: 16_000, channels: 1, frameCount: 16_000, offset: offset, source: source))
        }
        let samples = try AudioMediaIO.readMono16k(clips, start: 0, duration: 2, resolve: { root.appendingPathComponent($0) })
        XCTAssertEqual(samples.count, 32_000)
        XCTAssertEqual(samples[4_000], 0.25, accuracy: 1e-5)
        XCTAssertEqual(samples[12_000], 0.35, accuracy: 1e-5)
        XCTAssertEqual(samples[20_000], 0.6, accuracy: 1e-5)
        XCTAssertEqual(samples[28_000], 0.5, accuracy: 1e-5)
        XCTAssertTrue(samples.allSatisfy { $0 > 0.24 })
    }

    func testResamplingDoesNotInsertGapsBetweenSavedChunks() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let first = root.appendingPathComponent("first.wav")
        let second = root.appendingPathComponent("second.wav")
        let data = try AudioMediaIO.wav(samples: Array(repeating: 0.25, count: 48_000), sampleRate: 48_000, channels: 1)
        try data.write(to: first); try data.write(to: second)
        let clips = [
            AudioLibraryClip(assetReference: "first.wav", sampleRate: 48_000, channels: 1, frameCount: 48_000, offset: 0),
            AudioLibraryClip(assetReference: "second.wav", sampleRate: 48_000, channels: 1, frameCount: 48_000, offset: 1)
        ]
        let samples = try AudioMediaIO.readMono16k(clips, start: 0, duration: 2, resolve: { root.appendingPathComponent($0) })
        XCTAssertEqual(samples.count, 32_000)
        XCTAssertTrue(samples[15_990..<16_010].allSatisfy { $0 > 0.1 }, "Boundary samples: \(Array(samples[15_990..<16_010]))")
        XCTAssertEqual(samples[20_000], 0.25, accuracy: 1e-3)
    }

    func testCancellationStopsBoundedReadsBeforeFurtherOutput() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioMediaIO.wav(samples: Array(repeating: 0.25, count: 16_000 * 2), sampleRate: 16_000, channels: 1).write(to: url)
        let clip = AudioLibraryClip(assetReference: "source", sampleRate: 16_000, channels: 1, frameCount: 32_000)
        var checks = 0
        XCTAssertThrowsError(try AudioMediaIO.readMono16k([clip], start: 0, duration: 2, cancel: {
            checks += 1
            if checks >= 4 { throw CancellationError() }
        }, resolve: { _ in url })) { XCTAssertTrue($0 is CancellationError) }
        XCTAssertEqual(checks, 4)
        try AudioMediaIO.wav(samples: Array(repeating: 0.25, count: 16_000 * 6), sampleRate: 16_000, channels: 1).write(to: url)
        var chunks = 0
        XCTAssertThrowsError(try AudioMediaIO.importFile(url, cancel: {
            if chunks > 0 { throw CancellationError() }
        }, consume: { _ in chunks += 1 }))
        XCTAssertEqual(chunks, 1)
    }

    func testM4AExportProducesDecodableAudio() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source.wav")
        let output = root.appendingPathComponent("output.m4a")
        let samples = (0..<48_000).map { Float(sin(Double($0) * 2 * .pi * 440 / 24_000)) * 0.25 }
        try AudioMediaIO.wav(samples: samples, sampleRate: 24_000, channels: 1).write(to: source)
        try await AudioMediaIO.encodeM4A(from: source, to: output)
        let file = try AVAudioFile(forReading: output, commonFormat: .pcmFormatFloat32, interleaved: false)
        XCTAssertEqual(file.processingFormat.channelCount, 1)
        XCTAssertEqual(Double(file.length) / file.processingFormat.sampleRate, 2, accuracy: 0.05)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 2048))
        try file.read(into: buffer)
        let decoded = try AudioMediaIO.interleavedSamples(buffer)
        XCTAssertTrue(decoded.allSatisfy(\.isFinite))
        XCTAssertTrue(decoded.contains { abs($0) > 0.1 })
    }

    func testRejectsMalformedPCM() {
        XCTAssertThrowsError(try AudioMediaIO.wav(samples: [0], sampleRate: 48_000, channels: 2))
        XCTAssertThrowsError(try AudioMediaIO.wav(samples: [.nan], sampleRate: 48_000, channels: 1))
        XCTAssertThrowsError(try AudioMediaIO.wav(samples: [0], sampleRate: 0, channels: 1))
        XCTAssertThrowsError(try AudioMediaIO.wav(samples: [0], sampleRate: 16_000, channels: 0))
    }

    func testImportStaysInFiveSecondChunks() throws {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".wav")
        defer { try? FileManager.default.removeItem(at: url) }
        try AudioMediaIO.wav(samples: Array(repeating: 0.25, count: 16_000 * 12), sampleRate: 16_000, channels: 1).write(to: url)
        var chunks: [AudioCaptureChunk] = []
        try AudioMediaIO.importFile(url, consume: { chunks.append($0) })
        XCTAssertEqual(chunks.map(\.frameCount).reduce(0, +), 192_000)
        XCTAssertTrue(chunks.allSatisfy { $0.frameCount <= 80_000 })
        var end = 0.0
        for chunk in chunks {
            XCTAssertEqual(chunk.offset, end, accuracy: 1.0 / 16_000)
            end += Double(chunk.frameCount) / chunk.sampleRate
        }
        XCTAssertTrue(chunks.allSatisfy { $0.wav.count <= 80_000 * 4 + 44 })
    }
}
