import AVFoundation
import Foundation
import XCTest

@testable import TurboSparkApp

/// Speaker labels on real models for a take longer than the diarization
/// model accepts at once: three `say` voices over 32 minutes, labeled in two
/// parts that share five minutes. Samantha and Daniel speak throughout.
/// Karen is silent around the shared span, so nothing can match her across
/// it: she shows up as two speakers, never as someone else. Skips unless the
/// model path is set:
///
///     TURBOSPARK_DIARIZATION_COREML_MODEL=~/models/coreml/speaker-diarization-coreml
final class AudioSpeakerPartsRealModelTests: XCTestCase {
    private typealias Line = (voice: Int, start: Double, end: Double)

    /// Each voice's lines, synthesized once at 16 kHz with say's own
    /// silence trimmed.
    private func synthesizeLines(in directory: URL, voices: [String]) throws -> [[[Float]]] {
        let script = [
            "Thanks for joining. I want to settle the release plan before the end of the week.",
            "The database migration is still the main risk, and staging took about three hours.",
            "Can we split the work into smaller batches and run them over the weekend?",
            "I will send the results to the whole group as soon as the dry run finishes.",
        ]
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return try voices.enumerated().map { index, voice in
            try script.enumerated().map { line, text in
                let file = directory.appendingPathComponent("v\(index)-l\(line).wav")
                let say = Process()
                say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
                say.arguments = ["-v", voice, "-o", file.path, "--data-format=LEI16@16000", text]
                try say.run()
                say.waitUntilExit()
                guard say.terminationStatus == 0 else { throw XCTSkip("say -v \(voice) failed") }
                let audio = try AVAudioFile(forReading: file)
                let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: audio.processingFormat, frameCapacity: AVAudioFrameCount(audio.length)))
                try audio.read(into: buffer)
                let samples = Array(UnsafeBufferPointer(start: try XCTUnwrap(buffer.floatChannelData)[0], count: Int(buffer.frameLength)))
                let first = try XCTUnwrap(samples.firstIndex { abs($0) > 1e-3 })
                let last = try XCTUnwrap(samples.lastIndex { abs($0) > 1e-3 })
                return Array(samples[first...last])
            }
        }
    }

    @MainActor func testALongMeetingIsLabeledInPartsAndJoined() async throws {
        guard let diarization = ProcessInfo.processInfo.environment["TURBOSPARK_DIARIZATION_COREML_MODEL"] else {
            throw XCTSkip("set TURBOSPARK_DIARIZATION_COREML_MODEL")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let voices = ["Samantha", "Daniel", "Karen"]
        let clips = try synthesizeLines(in: root.appendingPathComponent("lines"), voices: voices)

        // Turns rotate A, B, A, C, B, C. Karen sits out 780 s to 1140 s,
        // around the shared span of the two parts (810 s to 1110 s).
        let rate = 16_000.0, duration = 1920.0
        let gap = [Float](repeating: 0, count: Int(0.7 * rate))
        let rotation = [0, 1, 0, 2, 1, 2]
        var pcm = gap
        var lines: [Line] = []
        var turn = 0
        while Double(pcm.count) / rate < duration - 12 {
            var voice = rotation[turn % rotation.count]
            let now = Double(pcm.count) / rate
            if voice == 2, now > 770, now < 1140 { voice = turn % 2 }
            let samples = clips[voice][turn % clips[voice].count]
            lines.append((voice, now, now + Double(samples.count) / rate))
            pcm += samples
            pcm += gap
            turn += 1
        }
        pcm += [Float](repeating: 0, count: Int(duration * rate) - pcm.count)
        XCTAssertEqual(Double(pcm.count) / rate, duration, accuracy: 0.01)

        let vault = ProfileVaultStore(rootProvider: { root.appendingPathComponent("profile") }, profileIDProvider: { "speaker-parts" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        defer { controller.shutdown() }
        controller.liveTranscription = false
        while controller.isRefreshingModels || controller.profiles.isEmpty { try await Task.sleep(nanoseconds: 50_000_000) }
        controller.installedPaths["speaker-diarization-coreml"] = diarization
        XCTAssertNotNil(controller.installedDiarizationProfile)

        let asset = try controller.assets.store(data: try AudioMediaIO.wav(samples: pcm, sampleRate: rate, channels: 1), fileName: "meeting.wav", mimeType: "audio/wav")
        var item = AudioLibraryItem(title: "Long meeting", kind: .recording)
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: rate, channels: 1, frameCount: Int64(pcm.count), source: "microphone")]
        let revision = item.addTranscript(lines.enumerated().map { index, line in
            AudioLibrarySegment(start: line.start, end: line.end, text: "\(voices[line.voice]) \(index)", timing: "segment")
        }, source: .final)
        try controller.save(item)
        XCTAssertEqual(AudioSpeakerParts.plan(duration: item.duration, limit: 1800).count, 2)

        let started = Date()
        try await controller.labelSpeakers(item: item, into: item.id, revision: revision, words: [], token: controller.epoch)
        let saved = try XCTUnwrap(controller.library.item(item.id))
        print(String(format: "labeled %.0f s in %.1f s", duration, Date().timeIntervalSince(started)))
        XCTAssertEqual(saved.speakerLabelParts, 2, controller.status)
        let segments = try XCTUnwrap(saved.transcripts.first { $0.id == revision }).segments

        // Labels each voice got, before and after the cut (960 s), where
        // part 1's labels end and part 2's begin.
        func counts(_ voice: Int, _ side: (Double) -> Bool) -> [String: Int] {
            var result: [String: Int] = [:]
            for (line, segment) in zip(lines, segments) where line.voice == voice && side(line.start) {
                result[segment.speakerID ?? "-", default: 0] += 1
            }
            return result
        }
        func majority(_ counts: [String: Int]) -> String? { counts.max { $0.value < $1.value }?.key }
        let before = voices.indices.map { counts($0) { $0 < 960 } }
        let after = voices.indices.map { counts($0) { $0 >= 960 } }
        for voice in voices.indices {
            // Each voice repeats four clips, so a label that follows the clip
            // in both parts is the diarizer's call, not the join's.
            var byClip: [Int: [String: Int]] = [:]
            for (index, pair) in zip(lines, segments).enumerated() where pair.0.voice == voice {
                byClip[index % 4, default: [:]][pair.1.speakerID ?? "-", default: 0] += 1
            }
            print(voices[voice], "before the cut:", before[voice], "after:", after[voice], "by clip:", byClip.sorted { $0.key < $1.key })
        }
        // Samantha and Daniel speak where the parts overlap: one label each
        // on both sides of the cut, and not the same one.
        for voice in [0, 1] {
            let all = before[voice].merging(after[voice], uniquingKeysWith: +)
            let label = try XCTUnwrap(majority(all))
            XCTAssertEqual(majority(before[voice]), label, voices[voice])
            XCTAssertEqual(majority(after[voice]), label, voices[voice])
            XCTAssertGreaterThanOrEqual(Double(all[label] ?? 0) / Double(all.values.reduce(0, +)), 0.98, "\(voices[voice]): \(all)")
        }
        let joined = [majority(before[0]), majority(before[1])].compactMap { $0 }
        XCTAssertEqual(Set(joined).count, 2)
        // Karen is silent where the parts overlap, so nothing can join her
        // across it: after the cut she is a new speaker, never Samantha or
        // Daniel. The diarizer puts one of her clips with Samantha inside
        // each part on its own, so only her majority labels are checked.
        let karenBefore = try XCTUnwrap(majority(before[2])), karenAfter = try XCTUnwrap(majority(after[2]))
        XCTAssertFalse(joined.contains(karenBefore), "Karen before the cut: \(before[2])")
        XCTAssertFalse(joined.contains(karenAfter), "Karen after the cut: \(after[2])")
        XCTAssertNotEqual(karenBefore, karenAfter, "the silent overlap cannot match Karen")
    }
}
