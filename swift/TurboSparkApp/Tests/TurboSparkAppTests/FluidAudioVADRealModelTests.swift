import AVFoundation
import XCTest
@testable import TurboSparkApp

/// Real-model check for the Silero VAD adapter. Opt in with
/// TURBOSPARK_FLUIDAUDIO_MODELS set to a folder holding the pinned
/// `silero-vad-coreml` install as `silero-vad/`. Speech comes from `say`.
final class FluidAudioVADRealModelTests: XCTestCase {
    private func installRoot() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_FLUIDAUDIO_MODELS"] else {
            throw XCTSkip("set TURBOSPARK_FLUIDAUDIO_MODELS to run the real FluidAudio models")
        }
        let dir = URL(fileURLWithPath: path).appendingPathComponent("silero-vad")
        guard FileManager.default.fileExists(atPath: dir.path) else { throw XCTSkip("no silero-vad install") }
        return dir
    }

    private func speech(_ text: String) throws -> [Float] {
        let wav = FileManager.default.temporaryDirectory.appendingPathComponent("say-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: wav) }
        let say = Process()
        say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        say.arguments = ["-v", "Samantha", "--file-format=WAVE", "--data-format=LEF32@16000", "-o", wav.path, text]
        try say.run()
        say.waitUntilExit()
        XCTAssertEqual(say.terminationStatus, 0)
        let file = try AVAudioFile(forReading: wav)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                                    frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    }

    private static let fluidAudioCaches = [
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/fluidaudio"),
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/FluidAudio"),
    ]

    /// Speech between two silences is found where it is, the same input
    /// gives the same segments twice, silence gives none, and no FluidAudio
    /// cache is created on the way.
    func testSpeechIsFoundBetweenSilencesAndRepeats() async throws {
        let absent = Self.fluidAudioCaches.filter { !FileManager.default.fileExists(atPath: $0.path) }
        let detector = try await FluidAudioEngine.shared.speechDetector(for: .sileroVAD, installRoot: try installRoot())
        let rate = 16_000
        let silence = [Float](repeating: 0, count: 3 * rate)
        let first = try speech("The library opens at nine and closes at five on weekdays.")
        let second = try speech("Please bring your card and a form of identification.")
        let clip = silence + first + silence + second + silence
        let firstRange = (3.0, 3.0 + Double(first.count) / Double(rate))
        let secondStart = firstRange.1 + 3.0
        let secondRange = (secondStart, secondStart + Double(second.count) / Double(rate))
        let duration = Double(clip.count) / Double(rate)

        let segments = try await detector.speechSegments(in: clip, maxSegmentSeconds: 14)
        let again = try await detector.speechSegments(in: clip, maxSegmentSeconds: 14)
        XCTAssertEqual(segments, again, "the same audio gave different segments")
        XCTAssertFalse(segments.isEmpty)
        var previousEnd = 0.0
        for segment in segments {
            XCTAssertLessThan(segment.start, segment.end)
            XCTAssertGreaterThanOrEqual(segment.start, previousEnd, "segments overlap or are out of order")
            XCTAssertLessThanOrEqual(segment.end, duration)
            XCTAssertLessThanOrEqual(segment.end - segment.start, 14.0 + 0.001)
            previousEnd = segment.end
        }
        // Each spoken phrase overlaps a segment, and no segment lies wholly inside a 3 s silence.
        func overlaps(_ range: (Double, Double)) -> Bool { segments.contains { $0.start < range.1 && $0.end > range.0 } }
        XCTAssertTrue(overlaps(firstRange), "\(segments)")
        XCTAssertTrue(overlaps(secondRange), "\(segments)")
        for segment in segments {
            let inSpeech = [firstRange, secondRange].contains { segment.start < $0.1 && segment.end > $0.0 }
            XCTAssertTrue(inSpeech, "a segment sits in silence: \(segment)")
        }

        let quiet = try await detector.speechSegments(in: silence, maxSegmentSeconds: 14)
        let none = try await detector.speechSegments(in: [], maxSegmentSeconds: 14)
        XCTAssertEqual(quiet, [])
        XCTAssertEqual(none, [])
        for cache in absent { XCTAssertFalse(FileManager.default.fileExists(atPath: cache.path), cache.path) }
    }

    /// A segment is capped at the length asked for, so a long monologue
    /// never exceeds a recognizer's window.
    func testLongSpeechIsCutAtTheCap() async throws {
        let detector = try await FluidAudioEngine.shared.speechDetector(for: .sileroVAD, installRoot: try installRoot())
        let sentence = "The committee reviewed the proposal and agreed to meet again next week to finalize the budget. "
        let clip = try speech(String(repeating: sentence, count: 6))
        let duration = Double(clip.count) / 16_000
        XCTAssertGreaterThan(duration, 20, "the clip must be longer than two caps")
        let segments = try await detector.speechSegments(in: clip, maxSegmentSeconds: 8)
        XCTAssertGreaterThan(segments.count, 1)
        XCTAssertTrue(segments.allSatisfy { $0.end - $0.start <= 8.0 + 0.001 }, "\(segments)")
    }
}
