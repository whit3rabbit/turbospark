import AVFoundation
import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

final class AudioSpeakerLabelsTests: XCTestCase {
    private func withLanguage(_ language: AppLanguage, _ body: () -> Void) {
        let defaults = UserDefaults.standard
        let previous = defaults.object(forKey: AppLanguage.storageKey)
        defer {
            if let previous { defaults.set(previous, forKey: AppLanguage.storageKey) }
            else { defaults.removeObject(forKey: AppLanguage.storageKey) }
        }
        defaults.set(language.rawValue, forKey: AppLanguage.storageKey)
        body()
    }

    private func segment(_ start: Double, _ end: Double, speaker: String? = nil) -> AudioLibrarySegment {
        AudioLibrarySegment(start: start, end: end, text: "words", speaker: speaker, timing: "token")
    }

    func testDisplayNameLocalizesRuntimeIdsAndKeepsOtherLabels() {
        withLanguage(.english) {
            XCTAssertEqual(AudioSpeakerLabels.displayName("S2"), "Speaker 2")
            XCTAssertEqual(AudioSpeakerLabels.displayName("Alice"), "Alice")
            XCTAssertEqual(AudioSpeakerLabels.displayName("S0"), "S0")
            XCTAssertEqual(AudioSpeakerLabels.displayName("S"), "S")
            XCTAssertEqual(AudioSpeakerLabels.displayName("Sx"), "Sx")
        }
        withLanguage(.german) {
            XCTAssertEqual(AudioSpeakerLabels.displayName("S1"), "Sprecher 1")
        }
    }

    /// Turns keep the runtime's ids, so a rename on the take can be undone,
    /// and move onto the take's timeline by a part's offset.
    func testTurnsKeepTimedLabeledSegmentsOnly() {
        let transcript = AudioTranscript(text: "", segments: [
            AudioSegment(startSeconds: 0, endSeconds: 2, text: "", timing: "segment", speaker: "S1"),
            AudioSegment(startSeconds: 2, endSeconds: 2, text: "", timing: "segment", speaker: "S2"),
            AudioSegment(startSeconds: 3, endSeconds: 4, text: "", timing: "segment", speaker: nil),
            AudioSegment(startSeconds: 4, endSeconds: 6, text: "", timing: "segment", speaker: "S2"),
        ])
        XCTAssertEqual(AudioSpeakerLabels.turns(transcript), [
            .init(speaker: "S1", start: 0, end: 2),
            .init(speaker: "S2", start: 4, end: 6),
        ])
        XCTAssertEqual(AudioSpeakerLabels.turns(transcript, offset: 1100).first, .init(speaker: "S1", start: 1100, end: 1102))
    }

    func testWordsDecideTheSpeakerAndManualLabelsStay() {
        let turns: [AudioSpeakerLabels.Turn] = [
            .init(speaker: "A", start: 0, end: 4.1),
            .init(speaker: "B", start: 4.1, end: 8.5),
            .init(speaker: "A", start: 8.5, end: 12),
        ]
        let segments = [
            segment(0, 4),
            // Starts inside A's turn, but its words are B's.
            segment(3.9, 8),
            segment(8, 9, speaker: "Alice"),
            // No word timings: the extent votes.
            segment(9, 10),
            // Nothing overlaps: stays unlabeled.
            segment(20, 21),
        ]
        let words: [ClosedRange<Double>] = [0...1, 1...2, 2...3.5, 4.2...5, 5...7.8, 8.1...8.9]
        let labeled = AudioSpeakerLabels.assign(segments, words: words, turns: turns)
        XCTAssertEqual(labeled.map(\.speaker), ["A", "B", "Alice", "A", nil])
        XCTAssertEqual(labeled.map(\.speakerID), ["A", "B", nil, "A", nil], "the manual label gets no id")
        XCTAssertEqual(labeled.map(\.id), segments.map(\.id), "segments keep their identity")
        XCTAssertEqual(labeled.map(\.text), segments.map(\.text))
    }

    func testTiesGoToTheSpeakerHeardFirst() {
        let turns: [AudioSpeakerLabels.Turn] = [
            .init(speaker: "B", start: 0, end: 1),
            .init(speaker: "A", start: 1, end: 2),
        ]
        let labeled = AudioSpeakerLabels.assign([segment(0, 2)], words: [], turns: turns)
        XCTAssertEqual(labeled.first?.speaker, "B")
        XCTAssertEqual(AudioSpeakerLabels.assign([segment(0, 2)], words: [], turns: []).first?.speaker, nil)
    }
}

/// Speaker labels on real models: Parakeet Ultra transcribes a two-voice
/// take, then the diarization model labels its segments. The Record page's
/// final pass reads a saved take from `TURBOSPARK_DIARIZATION_TAKE_WAV`; the
/// Speech-to-Text tests synthesize their call with `say`. Each test skips
/// unless the paths it uses are set:
///
///     TURBOSPARK_DIARIZATION_TAKE_WAV=<16 kHz mono WAV, Samantha first>
///     TURBOSPARK_PARAKEET_COREML_MODEL=~/models/coreml/parakeet-ultra-coreml
///     TURBOSPARK_DIARIZATION_COREML_MODEL=~/models/coreml/speaker-diarization-coreml
final class AudioSpeakerLabelsRealModelTests: XCTestCase {
    private typealias Turn = (voice: Int, start: Double, end: Double)

    /// A controller on a temporary profile with `installs` in place.
    @MainActor private func makeController(root: URL, installs: [String: String]) async throws -> AudioWorkspaceController {
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "speaker-labels" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        controller.liveTranscription = false
        // Imports follow Any Audio, where speaker labels are opt-in.
        controller.preferences.speakerLabels = [SpeechToTextMode.anyAudio.rawValue: true]
        // The init refresh would overwrite the paths set below.
        while controller.isRefreshingModels || controller.profiles.isEmpty { try await Task.sleep(nanoseconds: 50_000_000) }
        for (alias, path) in installs { controller.installedPaths[alias] = path }
        return controller
    }

    /// A call made with `say` at 22.05 kHz, as an import might be: Samantha
    /// opens, Daniel answers, three turns each with 0.7 s of silence between.
    /// Returns the WAV and each turn's voice (0 is Samantha) and extent.
    private func synthesizeCall(in directory: URL) throws -> (url: URL, turns: [Turn]) {
        let rate = 22_050.0
        let voices = ["Samantha", "Daniel"]
        let script = [
            "Good morning everyone, and thanks for joining the planning call. I want to settle the release date before Thursday.",
            "Morning. The database migration is my main worry. Last week it took about three hours on the staging cluster.",
            "That is longer than I expected. Can the team split the work into smaller batches over the weekend?",
            "We can. The migration scripts already support batches, so I will run a dry run tomorrow and report back.",
            "Perfect. Please send the results to the whole group, and we can confirm the date on Thursday.",
            "Will do. I will also check the rollback plan, in case the batches take longer than three hours.",
        ]
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let gap = [Float](repeating: 0, count: Int(0.7 * rate))
        var pcm = gap
        var turns: [Turn] = []
        for (index, line) in script.enumerated() {
            let voice = index % 2
            let file = directory.appendingPathComponent("turn\(index).wav")
            let say = Process()
            say.executableURL = URL(fileURLWithPath: "/usr/bin/say")
            say.arguments = ["-v", voices[voice], "-o", file.path, "--data-format=LEI16@22050", line]
            try say.run()
            say.waitUntilExit()
            guard say.terminationStatus == 0 else { throw XCTSkip("say -v \(voices[voice]) failed") }
            let audio = try AVAudioFile(forReading: file)
            let buffer = try XCTUnwrap(AVAudioPCMBuffer(pcmFormat: audio.processingFormat, frameCapacity: AVAudioFrameCount(audio.length)))
            try audio.read(into: buffer)
            let channel = try XCTUnwrap(buffer.floatChannelData)[0]
            let samples = Array(UnsafeBufferPointer(start: channel, count: Int(buffer.frameLength)))
            // Without say's own silence, each turn's extent is exact.
            let first = try XCTUnwrap(samples.firstIndex { abs($0) > 1e-3 })
            let last = try XCTUnwrap(samples.lastIndex { abs($0) > 1e-3 })
            turns.append((voice, Double(pcm.count) / rate, Double(pcm.count + last + 1 - first) / rate))
            pcm += samples[first...last]
            pcm += gap
        }
        let url = directory.appendingPathComponent("Two voice call.wav")
        try AudioMediaIO.wav(samples: pcm, sampleRate: rate, channels: 1).write(to: url)
        return (url, turns)
    }

    /// Imports `wav` on the Speech-to-Text page with Parakeet Ultra selected.
    @MainActor private func importOnTranscribePage(_ wav: URL, with controller: AudioWorkspaceController) async throws -> AudioLibraryItem {
        controller.page = .transcribe
        controller.recipe.modelID = "parakeet-ultra-coreml"
        XCTAssertTrue(controller.importAudio(from: wav))
        await controller.importTask?.value
        let imported = try XCTUnwrap(controller.selectedItem)
        XCTAssertEqual(imported.kind, .imported)
        XCTAssertEqual(imported.status, .completed, imported.failure ?? "")
        return imported
    }

    /// Imports `wav` on the Speech-to-Text page and transcribes it with
    /// Parakeet Ultra. Returns the transcription the job saved.
    @MainActor private func transcribeImported(_ wav: URL, with controller: AudioWorkspaceController) async throws -> AudioLibraryItem {
        let imported = try await importOnTranscribePage(wav, with: controller)
        XCTAssertTrue(controller.canRun, controller.status)
        controller.run()
        await controller.jobTask?.value
        let saved = try XCTUnwrap(controller.library.item(try XCTUnwrap(controller.selectedID)))
        XCTAssertEqual(saved.kind, .transcription)
        XCTAssertEqual(saved.sourceID, imported.id)
        XCTAssertEqual(saved.status, .completed, saved.failure ?? "")
        XCTAssertNil(controller.error)
        return saved
    }

    private var labelFailurePrefix: String { AppStrings.localized("Transcript saved. Speakers could not be labeled: \("")") }

    /// Windows overlap by a second, so a sentence cut by one window must not
    /// also appear whole at the start of the next.
    private func assertOneCopyAtEachSeam(_ segments: [AudioLibrarySegment]) {
        for (earlier, later) in zip(segments, segments.dropFirst()) {
            let words = earlier.text.lowercased().trimmingCharacters(in: .punctuationCharacters)
            XCTAssertFalse(later.text.lowercased().hasPrefix(words), "\(earlier.text) | \(later.text)")
        }
    }

    /// Waits until labeling has the diarization model open, then cancels.
    /// A cancel during the open is a CancellationError; this one reaches the
    /// running native job.
    @MainActor private func cancelWhileLabeling(_ controller: AudioWorkspaceController) async throws {
        let labeling = AppStrings.localized("Labeling speakers")
        let deadline = Date().addingTimeInterval(120)
        while controller.status != labeling || controller.loadedModelAlias != "speaker-diarization-coreml" {
            guard Date() < deadline else { return XCTFail("labeling never started: \(controller.status)") }
            try await Task.sleep(nanoseconds: 2_000_000)
        }
        controller.cancel()
        await controller.jobTask?.value
    }

    /// A cancel during labeling interrupts the take and leaves the transcript
    /// saved before labeling, unlabeled. It is not a labeling failure.
    @MainActor private func assertCancelledWhileLabeling(_ item: AudioLibraryItem, _ controller: AudioWorkspaceController) throws {
        XCTAssertEqual(item.status, .interrupted, controller.status)
        let final = try XCTUnwrap(item.transcripts.last(where: { $0.source == .final }))
        XCTAssertFalse(final.segments.isEmpty)
        XCTAssertTrue(final.segments.allSatisfy { $0.speaker == nil })
        XCTAssertNil(controller.error)
        XCTAssertFalse(controller.status.hasPrefix(labelFailurePrefix), controller.status)
    }

    @MainActor func testFinalPassLabelsATwoVoiceTake() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let wav = env["TURBOSPARK_DIARIZATION_TAKE_WAV"], let ultra = env["TURBOSPARK_PARAKEET_COREML_MODEL"],
              let diarization = env["TURBOSPARK_DIARIZATION_COREML_MODEL"] else {
            throw XCTSkip("set TURBOSPARK_DIARIZATION_TAKE_WAV, TURBOSPARK_PARAKEET_COREML_MODEL and TURBOSPARK_DIARIZATION_COREML_MODEL")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let controller = try await makeController(root: root, installs: ["parakeet-ultra-coreml": ultra, "speaker-diarization-coreml": diarization])
        defer { controller.shutdown(); try? FileManager.default.removeItem(at: root) }
        XCTAssertNotNil(controller.installedDiarizationProfile)

        let data = try Data(contentsOf: URL(fileURLWithPath: wav))
        let samples = Int64((data.count - 44) / 2)
        let asset = try controller.assets.store(data: data, fileName: "take.wav", mimeType: "audio/wav")
        var item = AudioLibraryItem(title: "Two voices", kind: .recording)
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 16_000, channels: 1, frameCount: samples, source: "microphone")]
        item.status = .completed
        item.recipe.task = AudioTask.speechToText.rawValue
        item.recipe.modelID = "parakeet-ultra-coreml"
        try controller.save(item)

        controller.refineRecording(item.id)
        await controller.jobTask?.value
        let saved = try XCTUnwrap(controller.library.item(item.id))
        XCTAssertEqual(saved.status, .completed, saved.failure ?? "")
        let final = try XCTUnwrap(saved.transcripts.last(where: { $0.source == .final }))
        for segment in final.segments {
            print(String(format: "%6.2f %6.2f", segment.start, segment.end), segment.speaker ?? "-", segment.text)
        }
        XCTAssertGreaterThan(final.segments.count, 3)
        assertOneCopyAtEachSeam(final.segments)
        // A labeling failure only reaches the status line.
        XCTAssertEqual(Set(final.segments.compactMap(\.speaker)).count, 2, controller.status)
        XCTAssertTrue(final.segments.allSatisfy { $0.speaker != nil })
        let first = try XCTUnwrap(final.segments.first?.speaker)
        let second = try XCTUnwrap(final.segments.first { $0.speaker != first }?.speaker)
        // Samantha opens the call and Daniel answers.
        for segment in final.segments {
            let text = segment.text.lowercased()
            if text.contains("planning call") || text.contains("thursday") { XCTAssertEqual(segment.speaker, first, segment.text) }
            if text.contains("migration") || text.contains("three hours") { XCTAssertEqual(segment.speaker, second, segment.text) }
        }
        // Display names in order of first appearance, not the runtime ids,
        // which stay beside them for the take's names and merges.
        XCTAssertEqual(first, AudioSpeakerLabels.displayName("S1"))
        XCTAssertEqual(second, AudioSpeakerLabels.displayName("S2"))
        XCTAssertNotEqual(first, "S1")
        XCTAssertEqual(Set(final.segments.compactMap(\.speakerID)), ["S1", "S2"])
    }

    /// An imported meeting on the Speech-to-Text page gets the same labels as
    /// the Record page's final pass: each segment names the voice that spoke
    /// most of it, Speaker 1 for Samantha, who opens.
    @MainActor func testTranscribePageLabelsAnImportedTwoVoiceCall() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let ultra = env["TURBOSPARK_PARAKEET_COREML_MODEL"], let diarization = env["TURBOSPARK_DIARIZATION_COREML_MODEL"] else {
            throw XCTSkip("set TURBOSPARK_PARAKEET_COREML_MODEL and TURBOSPARK_DIARIZATION_COREML_MODEL")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let call = try synthesizeCall(in: root.appendingPathComponent("call"))
        let controller = try await makeController(root: root.appendingPathComponent("profile"), installs: ["parakeet-ultra-coreml": ultra, "speaker-diarization-coreml": diarization])
        defer { controller.shutdown() }
        XCTAssertNotNil(controller.installedDiarizationProfile)

        let saved = try await transcribeImported(call.url, with: controller)
        let final = try XCTUnwrap(saved.transcripts.last(where: { $0.source == .final }))
        XCTAssertGreaterThanOrEqual(final.segments.count, call.turns.count)
        for segment in final.segments {
            print(String(format: "%6.2f %6.2f", segment.start, segment.end), segment.speaker ?? "-", segment.text)
            let overlap = { (turn: Turn) in min(segment.end, turn.end) - max(segment.start, turn.start) }
            let turn = try XCTUnwrap(call.turns.max { overlap($0) < overlap($1) })
            XCTAssertGreaterThan(overlap(turn), 0, segment.text)
            // A labeling failure only reaches the status line.
            XCTAssertEqual(segment.speaker, AudioSpeakerLabels.displayName("S\(turn.voice + 1)"), "\(segment.text) [\(controller.status)]")
        }
        assertOneCopyAtEachSeam(final.segments)
        XCTAssertEqual(controller.status, AppStrings.localized("Saved in this profile"))
    }

    /// A Meeting take's final pass on the Meeting pane labels its speakers
    /// with raw ids and asks for their names.
    @MainActor func testMeetingFinalPassAsksForNames() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let ultra = env["TURBOSPARK_PARAKEET_COREML_MODEL"], let diarization = env["TURBOSPARK_DIARIZATION_COREML_MODEL"] else {
            throw XCTSkip("set TURBOSPARK_PARAKEET_COREML_MODEL and TURBOSPARK_DIARIZATION_COREML_MODEL")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let call = try synthesizeCall(in: root.appendingPathComponent("call"))
        let controller = try await makeController(root: root.appendingPathComponent("profile"), installs: ["parakeet-ultra-coreml": ultra, "speaker-diarization-coreml": diarization])
        defer { controller.shutdown() }
        controller.selectPage(.transcribe)
        controller.selectSpeechToTextMode(.meeting)

        let data = try Data(contentsOf: call.url)
        let asset = try controller.assets.store(data: data, fileName: "meeting.wav", mimeType: "audio/wav")
        var item = AudioLibraryItem(title: "Planning call", kind: .recording)
        item.recordingPreset = AudioRecordingPreset.meeting.rawValue
        item.clips = [AudioLibraryClip(assetReference: asset.storedReference, sampleRate: 22_050, channels: 1, frameCount: Int64((data.count - 44) / 4), source: "microphone")]
        item.recipe.task = AudioTask.speechToText.rawValue
        item.recipe.modelID = "parakeet-ultra-coreml"
        try controller.save(item)
        XCTAssertNil(controller.speakerNamingID)

        controller.refineRecording(item.id)
        await controller.jobTask?.value
        let saved = try XCTUnwrap(controller.library.item(item.id))
        XCTAssertEqual(saved.status, .completed, saved.failure ?? controller.status)
        let final = try XCTUnwrap(saved.transcripts.last(where: { $0.source == .final }))
        XCTAssertEqual(Set(final.segments.compactMap(\.speakerID)), ["S1", "S2"], controller.status)
        XCTAssertEqual(controller.speakerNamingID, item.id)
    }

    /// The native job reports a cancel as its own error, not CancellationError,
    /// so both pages must check the task itself while labeling.
    @MainActor func testCancelWhileLabelingInterruptsBothPages() async throws {
        let env = ProcessInfo.processInfo.environment
        guard let ultra = env["TURBOSPARK_PARAKEET_COREML_MODEL"], let diarization = env["TURBOSPARK_DIARIZATION_COREML_MODEL"] else {
            throw XCTSkip("set TURBOSPARK_PARAKEET_COREML_MODEL and TURBOSPARK_DIARIZATION_COREML_MODEL")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let call = try synthesizeCall(in: root.appendingPathComponent("call"))
        let controller = try await makeController(root: root.appendingPathComponent("profile"), installs: ["parakeet-ultra-coreml": ultra, "speaker-diarization-coreml": diarization])
        defer { controller.shutdown() }
        let imported = try await importOnTranscribePage(call.url, with: controller)

        // Speech-to-Text page.
        XCTAssertTrue(controller.canRun, controller.status)
        controller.run()
        try await cancelWhileLabeling(controller)
        try assertCancelledWhileLabeling(try XCTUnwrap(controller.library.item(try XCTUnwrap(controller.selectedID))), controller)
        XCTAssertEqual(controller.status, AppStrings.localized("Interrupted"))

        // The Record page's final pass, on the same audio.
        controller.refineRecording(imported.id)
        try await cancelWhileLabeling(controller)
        try assertCancelledWhileLabeling(try XCTUnwrap(controller.library.item(imported.id)), controller)
    }

    /// Labels are extra: a diarization install that cannot open leaves the
    /// transcript saved and unlabeled, and says so on the status line.
    @MainActor func testTranscribePageKeepsTheTranscriptWhenLabelingFails() async throws {
        guard let ultra = ProcessInfo.processInfo.environment["TURBOSPARK_PARAKEET_COREML_MODEL"] else {
            throw XCTSkip("set TURBOSPARK_PARAKEET_COREML_MODEL")
        }
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let call = try synthesizeCall(in: root.appendingPathComponent("call"))
        let broken = root.appendingPathComponent("empty-diarization", isDirectory: true)
        try FileManager.default.createDirectory(at: broken, withIntermediateDirectories: true)
        let controller = try await makeController(root: root.appendingPathComponent("profile"), installs: ["parakeet-ultra-coreml": ultra, "speaker-diarization-coreml": broken.path])
        defer { controller.shutdown() }
        XCTAssertNotNil(controller.installedDiarizationProfile)

        let saved = try await transcribeImported(call.url, with: controller)
        let final = try XCTUnwrap(saved.transcripts.last(where: { $0.source == .final }))
        XCTAssertFalse(final.segments.isEmpty)
        XCTAssertTrue(final.segments.contains { $0.text.lowercased().contains("migration") })
        XCTAssertTrue(final.segments.allSatisfy { $0.speaker == nil })
        XCTAssertTrue(controller.status.hasPrefix(labelFailurePrefix) && controller.status.count > labelFailurePrefix.count, controller.status)
    }
}
