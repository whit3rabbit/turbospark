import XCTest

@testable import TurboSpark

/// The live voice-isolation binding: open failure, and a CPU round trip on
/// the Rust tiny causal fixture when the checkout has it.
final class AudioEnhanceStreamTests: XCTestCase {
    func testOpeningAMissingInstallFailsCleanly() {
        XCTAssertThrowsError(try AudioEnhanceStream(path: "/missing/voice-isolator-realtime", backend: .cpu)) { error in
            XCTAssertTrue(error is TurboSparkError, "a typed library error, not a crash: \(error)")
        }
    }

    func testAsyncOpenOfAMissingInstallThrows() async {
        do {
            _ = try await AudioEnhanceStream.open(path: "/missing/voice-isolator-realtime", backend: .cpu)
            XCTFail("opened a missing install")
        } catch {}
    }

    /// The fixture lives in the Rust tree beside this package.
    private func fixtureInstall() throws -> URL {
        let fixture = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("crates/audio/testdata/voice_isolator/causal/model.safetensors")
        guard FileManager.default.fileExists(atPath: fixture.path) else {
            throw XCTSkip("tiny voice isolator fixture not reachable at \(fixture.path)")
        }
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("ts-enhance-stream-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        try FileManager.default.copyItem(at: fixture, to: dir.appendingPathComponent("denoiser_realtime.safetensors"))
        return dir
    }

    func testFixtureRoundTripReleasesEverySampleOnce() throws {
        let dir = try fixtureInstall()
        let stream = try AudioEnhanceStream(path: dir.path, backend: .cpu)
        XCTAssertEqual(stream.backend, "cpu")
        let audio: [Float] = (0..<4_800).map { sinf(Float($0) * 0.05) * 0.2 }
        var out: [Float] = []
        for start in stride(from: 0, to: audio.count, by: 320) {
            let push = try stream.push(Array(audio[start..<min(audio.count, start + 320)]))
            XCTAssertGreaterThanOrEqual(push.processingMs, 0)
            out += push.voice
        }
        XCTAssertLessThan(out.count, audio.count, "the look-ahead holds the tail back until flush")
        XCTAssertEqual(try stream.push([]).voice, [], "an empty push releases nothing")
        out += try stream.flush()
        XCTAssertEqual(out.count, audio.count, "output is time-aligned: one sample out per sample in")
        XCTAssertTrue(out.allSatisfy(\.isFinite))
        XCTAssertThrowsError(try stream.push(audio), "a flushed stream takes no more audio")
        stream.close()
        XCTAssertThrowsError(try stream.flush(), "a closed stream fails instead of touching a freed handle")
    }

    func testAPushLongerThanOneNativeChunkIsSplit() throws {
        let dir = try fixtureInstall()
        let stream = try AudioEnhanceStream(path: dir.path, backend: .cpu)
        let audio = [Float](repeating: 0.01, count: AudioEnhanceStream.maxPushSamples + 4_800)
        let push = try stream.push(audio)
        let rest = try stream.flush()
        XCTAssertEqual(push.voice.count + rest.count, audio.count)
    }
}
