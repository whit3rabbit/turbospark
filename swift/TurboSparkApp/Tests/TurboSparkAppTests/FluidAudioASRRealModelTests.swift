import AVFoundation
import XCTest
@testable import TurboSparkApp

/// Real-model checks for the FluidAudio speech adapters. Opt in with
/// TURBOSPARK_FLUIDAUDIO_MODELS set to a folder holding each pinned install's
/// files as `parakeet-ja/` and `paraformer/` (any subset). Speech comes from
/// `say`, so no audio fixture is committed. First runs compile for the
/// Neural Engine and take minutes.
final class FluidAudioASRRealModelTests: XCTestCase {
    private func installs() throws -> URL {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_FLUIDAUDIO_MODELS"] else {
            throw XCTSkip("set TURBOSPARK_FLUIDAUDIO_MODELS to run the real FluidAudio speech models")
        }
        return URL(fileURLWithPath: path)
    }

    private func install(_ name: String) throws -> URL {
        let dir = try installs().appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: dir.path) else { throw XCTSkip("no \(name) install") }
        return dir
    }

    /// Mono 16 kHz float speech from macOS's own voices.
    private func speech(voice: String, _ text: String) throws -> [Float] {
        let wav = FileManager.default.temporaryDirectory.appendingPathComponent("say-\(UUID().uuidString).wav")
        defer { try? FileManager.default.removeItem(at: wav) }
        let say = Process()
        say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
        say.arguments = ["-v", voice, "--file-format=WAVE", "--data-format=LEF32@16000", "-o", wav.path, text]
        try say.run()
        say.waitUntilExit()
        XCTAssertEqual(say.terminationStatus, 0)
        let file = try AVAudioFile(forReading: wav)
        let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: file.processingFormat,
                                                    frameCapacity: AVAudioFrameCount(file.length)))
        try file.read(into: buffer)
        XCTAssertEqual(file.processingFormat.sampleRate, 16_000)
        return Array(UnsafeBufferPointer(start: buffer.floatChannelData![0], count: Int(buffer.frameLength)))
    }

    /// FluidAudio's own download and G2P caches. Installs come from the
    /// catalog, so a load that creates one went around the offline engine.
    private static let fluidAudioCaches = [
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(".cache/fluidaudio"),
        FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/FluidAudio"),
    ]

    /// The same window twice gives the same text: a Neural Engine run that
    /// varies between calls would make transcripts unrepeatable.
    private func check(_ family: FluidAudioFamily, install name: String, samples: [Float],
                       expect fragments: [String]) async throws {
        let absent = Self.fluidAudioCaches.filter { !FileManager.default.fileExists(atPath: $0.path) }
        let transcriber = try await FluidAudioEngine.shared.transcriber(for: family, installRoot: try install(name))
        let first = try await transcriber.transcribe(samples)
        let second = try await transcriber.transcribe(samples)
        XCTAssertEqual(first, second, "\(family.rawValue) gave different text for the same window")
        for fragment in fragments {
            XCTAssertTrue(first.text.contains(fragment), "\(family.rawValue): \(first.text)")
        }
        for cache in absent {
            XCTAssertFalse(FileManager.default.fileExists(atPath: cache.path), "\(family.rawValue) created \(cache.path)")
        }
    }

    func testParakeetJapaneseTranscribesWithTokenTimings() async throws {
        try await check(.parakeetJa, install: "parakeet-ja",
                        samples: try speech(voice: "Kyoko", "今日は天気がとても良いので、公園を散歩してから図書館に行きます。"),
                        expect: ["天気", "公園", "図書館"])
        let transcriber = try await FluidAudioEngine.shared.transcriber(for: .parakeetJa, installRoot: try install("parakeet-ja"))
        let result = try await transcriber.transcribe(try speech(voice: "Kyoko", "図書館に行きます。"))
        XCTAssertFalse(result.tokens.isEmpty)
        XCTAssertTrue(result.tokens.allSatisfy { $0.start <= $0.end })
    }

    func testParaformerTranscribesMandarin() async throws {
        try await check(.paraformerZh, install: "paraformer",
                        samples: try speech(voice: "Tingting", "今天天气很好，我们下午去公园散步，然后去图书馆看书。"),
                        expect: ["天气", "公园", "图书馆"])
    }

    /// The workspace's windows are longer than Parakeet's 15 s model input.
    /// FluidAudio splits them, and words from both ends survive the merge.
    func testParakeetJapaneseTranscribesAWindowLongerThanItsModelInput() async throws {
        let samples = try speech(voice: "Kyoko", """
            今日は天気がとても良いので、公園を散歩してから図書館に行きます。\
            図書館では歴史の本を二冊借りて、駅の近くの喫茶店で読みました。\
            夕方には友達と会って、新しいレストランで晩ご飯を食べる予定です。\
            明日は朝早く起きて、電車で海の見える町まで出かけるつもりです。
            """)
        XCTAssertGreaterThan(samples.count, 15 * 16_000, "the clip must exceed one model window")
        try await check(.parakeetJa, install: "parakeet-ja", samples: samples, expect: ["公園", "電車"])
    }

    /// A recording longer than one workspace window, planned and joined as
    /// the workspace does: overlapping 30 s windows cut by token start time.
    /// Every word survives once; none repeats across the boundary.
    func testParakeetWindowsJoinWithoutRepeats() async throws {
        let nouns = ["公園", "図書館", "喫茶店", "電車", "博物館", "病院", "映画館", "美術館", "動物園", "郵便局"]
        let samples = try speech(voice: "Kyoko", nouns.map { "午前中は友達と一緒に\($0)へ行って、ゆっくり過ごしました。" }.joined())
        XCTAssertGreaterThan(samples.count, 30 * 16_000, "the clip must span two windows")
        let transcriber = try await FluidAudioEngine.shared.transcriber(for: .parakeetJa, installRoot: try install("parakeet-ja"))
        let windows = AudioTranscriptionPlan.windows(duration: Double(samples.count) / 16_000, length: 30, overlap: 2, skipsCoveredTail: true)
        XCTAssertGreaterThan(windows.count, 1)
        var joined = ""
        for window in windows {
            let from = Int(window.start * 16_000), to = min(samples.count, Int((window.end * 16_000).rounded(.up)))
            joined += try await transcriber.transcribe(Array(samples[from..<to])).transcript(in: window, language: "ja").text
        }
        let counts = nouns.map { joined.components(separatedBy: $0).count - 1 }
        XCTAssertTrue(counts.allSatisfy { $0 <= 1 }, "a word repeated across a window boundary: \(joined)")
        XCTAssertGreaterThanOrEqual(counts.filter { $0 == 1 }.count, 8, joined)
    }

    /// Paraformer would drop the end of a long window without an error, so
    /// the adapter refuses it instead.
    func testParaformerRefusesAWindowItWouldCut() async throws {
        let transcriber = try await FluidAudioEngine.shared.transcriber(for: .paraformerZh,
                                                                        installRoot: try install("paraformer"))
        let limit = ParaformerTranscriber.windowSamples
        XCTAssertEqual(transcriber.maxWindowSamples, limit)
        do {
            _ = try await transcriber.transcribe([Float](repeating: 0, count: limit + 1))
            XCTFail("a window past the limit was accepted")
        } catch {
            XCTAssertEqual(error as? FluidAudioModelError, .windowTooLong(samples: limit + 1, limit: limit))
        }
    }

    /// A model that is not the one a family expects throws before FluidAudio
    /// sees it. Handed to FluidAudio, it would end the process.
    func testAWrongModelIsRefusedNotTrapped() throws {
        let source = try install("paraformer")
        let fake = FileManager.default.temporaryDirectory.appendingPathComponent("fake-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: fake, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: fake) }
        try FileManager.default.copyItem(at: source.appendingPathComponent("ParaformerPreprocessor.mlmodelc"),
                                         to: fake.appendingPathComponent("Preprocessor.mlmodelc"))
        XCTAssertThrowsError(try FluidAudioModelLoader.load(FluidAudioSignatures.ParakeetJa.preprocessor,
                                                            in: fake, units: .cpuOnly)) { error in
            guard case FluidAudioModelError.mismatch(let problems) = error else { return XCTFail("\(error)") }
            XCTAssertTrue(problems.contains("Preprocessor.mlmodelc: no input audio_signal"), "\(problems)")
        }
    }
}
