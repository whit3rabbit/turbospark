import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Input that is already 16 kHz: keeps channel 0, so mixer tests control
/// exact sample values.
private struct Passthrough: AudioLiveResampling {
    mutating func mono16k(_ interleaved: [Float], sampleRate: Double, channels: Int) -> [Float] {
        stride(from: 0, to: interleaved.count, by: channels).map { interleaved[$0] }
    }
}

private func packet(_ value: Float, count: Int, at offset: Double, _ source: String = "microphone") -> AudioLiveFeed.Packet {
    AudioLiveFeed.Packet(samples: [Float](repeating: value, count: count), sampleRate: 16_000, channels: 1, offset: offset, source: source)
}

final class AudioLiveFeedTests: XCTestCase {
    func testOfferDropsPastTheBoundPerSourceAndDrainFreesIt() {
        let feed = AudioLiveFeed(limitSeconds: 1)
        let half = [Float](repeating: 0.1, count: 8_000)
        XCTAssertTrue(feed.offer(half, sampleRate: 16_000, channels: 1, offset: 0, source: "microphone"))
        XCTAssertTrue(feed.offer(half, sampleRate: 16_000, channels: 1, offset: 0.5, source: "microphone"))
        XCTAssertFalse(feed.offer(half, sampleRate: 16_000, channels: 1, offset: 1, source: "microphone"), "past the bound it drops, never waits")
        XCTAssertTrue(feed.offer(half, sampleRate: 16_000, channels: 1, offset: 0, source: "system"), "each source has its own bound")
        let (packets, dropped) = feed.drain()
        XCTAssertEqual(packets.count, 3)
        XCTAssertEqual(dropped, 1)
        XCTAssertTrue(feed.offer(half, sampleRate: 16_000, channels: 1, offset: 1, source: "microphone"), "a drain frees the budget")
        XCTAssertFalse(feed.offer([], sampleRate: 16_000, channels: 1, offset: 0, source: "microphone"))
    }
}

final class AudioLiveMixerTests: XCTestCase {
    private func mixer() -> AudioLiveMixer<Passthrough> { AudioLiveMixer(makeResampler: { Passthrough() }) }

    func testOneSourcePassesThroughContiguously() {
        var m = mixer()
        m.add(packet(0.5, count: 1_600, at: 0))
        m.add(packet(0.25, count: 1_600, at: 0.1))
        let out = m.drain()
        XCTAssertEqual(out.count, 3_200)
        XCTAssertEqual(out.first, 0.5)
        XCTAssertEqual(out.last, 0.25)
    }

    func testSourcesAreSummedOnTheTimeline() {
        var m = mixer()
        m.add(packet(0.5, count: 1_600, at: 0))
        m.add(packet(0.25, count: 1_600, at: 0, "system"))
        XCTAssertEqual(m.drain(), [Float](repeating: 0.75, count: 1_600))
    }

    func testOutputWaitsForTheSlowerSourceThenTreatsAStallAsSilence() {
        var m = mixer()
        m.add(packet(0.5, count: 1_600, at: 0))
        m.add(packet(0.25, count: 800, at: 0, "system"))
        XCTAssertEqual(m.drain().count, 800, "waits while the slower source is within the lag")
        m.add(packet(0.5, count: 16_000, at: 0.1))
        let out = m.drain()
        XCTAssertEqual(m.emitted, 17_600, "a source more than 0.5 s behind no longer holds output back")
        XCTAssertEqual(out.last, 0.5)
    }

    func testAGapIsSilenceAndLateAudioIsDropped() {
        var m = mixer()
        m.add(packet(0.5, count: 1_600, at: 0))
        m.add(packet(0.5, count: 1_600, at: 0.5))
        let out = m.drain()
        XCTAssertEqual(out.count, 9_600)
        XCTAssertEqual(out[4_000], 0, "the 0.4 s gap is silence")
        XCTAssertEqual(out[9_000], 0.5)
        m.add(packet(0.9, count: 1_600, at: 0.2))
        XCTAssertTrue(m.drain().isEmpty, "audio behind the output position is gone")
    }

    func testFlushEmitsWhatTheLagWouldHoldBack() {
        var m = mixer()
        m.add(packet(0.5, count: 4_800, at: 0))
        m.add(packet(0.25, count: 1_600, at: 0, "system"))
        XCTAssertEqual(m.drain().count, 1_600)
        let rest = m.drain(flush: true)
        XCTAssertEqual(rest.count, 3_200)
        XCTAssertEqual(rest.first, 0.5)
    }
}

/// One word per second of audio appended, so tests can count words.
private final class FakeStream: AudioLiveStreaming, @unchecked Sendable {
    private let lock = NSLock()
    private var samples = 0
    private var words = 0
    private(set) var calls: [String] = []
    var failAppend = false

    func append(_ pcm: [Float]) throws -> AudioStreamUpdate {
        lock.lock(); defer { lock.unlock() }
        if failAppend { throw TurboSparkError(code: .invalidArgument, message: "boom") }
        calls.append("append")
        samples += pcm.count
        var text = "", closed: [AudioStreamWord] = []
        while (words + 1) * 16_000 <= samples {
            closed.append(AudioStreamWord(text: "w\(words)", startSeconds: Double(words), endSeconds: Double(words) + 0.5))
            text += (words == 0 ? "" : " ") + "w\(words)"
            words += 1
        }
        return AudioStreamUpdate(appendedText: text, closedWords: closed, receivedSeconds: Double(samples) / 16_000)
    }
    func finish() throws -> AudioStreamUpdate {
        lock.lock(); defer { lock.unlock() }
        calls.append("finish")
        return AudioStreamUpdate(appendedText: " end.", closedWords: [AudioStreamWord(text: "end.", startSeconds: 99, endSeconds: 99.5)], isFinal: true)
    }
    func cancel() { lock.lock(); calls.append("cancel"); lock.unlock() }
    func close() { lock.lock(); calls.append("close"); lock.unlock() }
    var appendedSamples: Int { lock.lock(); defer { lock.unlock() }; return samples }
    var callLog: [String] { lock.lock(); defer { lock.unlock() }; return calls }
}

private final class Events: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [AudioLiveTranscriber.Event] = []
    func add(_ e: AudioLiveTranscriber.Event) { lock.lock(); items.append(e); lock.unlock() }
    var all: [AudioLiveTranscriber.Event] { lock.lock(); defer { lock.unlock() }; return items }
}

final class AudioLiveTranscriberTests: XCTestCase {
    private func wait(_ condition: () -> Bool, file: StaticString = #filePath, line: UInt = #line) {
        let deadline = Date().addingTimeInterval(5)
        while !condition() && Date() < deadline { usleep(10_000) }
        XCTAssertTrue(condition(), "timed out", file: file, line: line)
    }

    private func offerSeconds(_ feed: AudioLiveFeed, _ seconds: Int, from start: Int = 0) {
        for s in start..<(start + seconds) {
            feed.offer([Float](repeating: 0.1, count: 16_000), sampleRate: 16_000, channels: 1, offset: Double(s), source: "microphone")
        }
    }

    func testCommitsLiveThenFinishesBeforeClosing() async {
        let feed = AudioLiveFeed(limitSeconds: 30)
        let stream = FakeStream(), events = Events()
        let live = AudioLiveTranscriber(feed: feed, open: { stream }, onEvent: events.add)
        live.start(pumpInterval: .seconds(3_600))
        wait { live.isOpenForTesting }
        offerSeconds(feed, 3)
        live.pumpForTesting()
        XCTAssertEqual(stream.appendedSamples, 48_000)
        XCTAssertEqual(live.snapshot().words.map(\.text), ["w0", "w1", "w2"])
        let final = await live.finishAndClose()
        XCTAssertEqual(final.text, "w0 w1 w2 end.")
        XCTAssertEqual(final.words.last?.text, "end.", "the finish's tail is in the returned transcript")
        XCTAssertEqual(Array(stream.callLog.suffix(2)), ["finish", "close"], "the model is closed only after the tail")
        XCTAssertTrue(events.all.contains(.preparing))
    }

    func testAudioWaitsDuringASlowOpenAndIsCaughtUp() async {
        let feed = AudioLiveFeed(limitSeconds: 30)
        let stream = FakeStream(), events = Events()
        let gate = DispatchSemaphore(value: 0)
        let live = AudioLiveTranscriber(feed: feed, open: { gate.wait(); return stream }, onEvent: events.add)
        live.start(pumpInterval: .seconds(3_600))
        offerSeconds(feed, 4)
        live.pumpForTesting()
        XCTAssertEqual(stream.appendedSamples, 0, "nothing reaches the model before it opens")
        gate.signal()
        wait { stream.appendedSamples == 64_000 }
        _ = await live.finishAndClose()
    }

    func testABacklogPastItsBoundStopsLiveTextInsteadOfGrowing() {
        let feed = AudioLiveFeed(limitSeconds: 30)
        let events = Events()
        let gate = DispatchSemaphore(value: 0)
        let live = AudioLiveTranscriber(feed: feed, backlogSeconds: 1, open: { gate.wait(); return FakeStream() }, onEvent: events.add)
        live.start(pumpInterval: .seconds(3_600))
        offerSeconds(feed, 2)
        live.pumpForTesting()
        XCTAssertTrue(events.all.contains { if case .stopped = $0 { return true }; return false })
        gate.signal()
        live.stopAndDrain()
    }

    func testAFailedOpenStopsLiveTextWithAReason() {
        let events = Events()
        let live = AudioLiveTranscriber(feed: AudioLiveFeed(), open: { throw TurboSparkError(code: .invalidArgument, message: "no model") }, onEvent: events.add)
        live.start(pumpInterval: .seconds(3_600))
        wait { events.all.contains { if case .stopped(let reason) = $0 { return reason.contains("no model") }; return false } }
        live.stopAndDrain()
    }
}

@MainActor
final class AudioLiveTextModeTests: XCTestCase {
    func testEnglishDetectionNeverGuessesANonEnglishTake() {
        XCTAssertTrue(AudioWorkspaceController.isEnglish("en"))
        XCTAssertTrue(AudioWorkspaceController.isEnglish("en-GB"))
        XCTAssertTrue(AudioWorkspaceController.isEnglish("auto", preferred: ["en-US", "de-DE"]))
        XCTAssertFalse(AudioWorkspaceController.isEnglish("auto", preferred: ["de-DE", "en-US"]))
        XCTAssertFalse(AudioWorkspaceController.isEnglish("auto", preferred: []))
        XCTAssertFalse(AudioWorkspaceController.isEnglish("fr"))
    }

    func testStreamingNeedsTheInstalledRunnableModelAndAnEnglishTake() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "live-mode" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        let controller = AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
        controller.recipe.language = "en"
        XCTAssertNotNil(controller.liveStreamingUnavailableReason, "not in the catalog")
        let profile = AudioProfile(
            identity: AudioProfileIdentity(task: .speechToText, alias: AudioWorkspaceController.liveTextAlias, repository: "o/r", revision: "abc", assetFingerprint: "f"),
            family: "parakeet_unified", displayName: "Live",
            capabilities: AudioCapabilities(operations: ["stream_transcribe"], backend: "coreml", cancellation: "chunk", canRun: true),
            pcmFormat: AudioPCMFormat(sampleRate: 16_000, channels: 1), readiness: "implemented_unqualified")
        controller.profiles = [profile]
        XCTAssertNotNil(controller.liveStreamingUnavailableReason, "not installed")
        controller.installedPaths[AudioWorkspaceController.liveTextAlias] = "/models/live"
        XCTAssertNil(controller.liveStreamingUnavailableReason)
        XCTAssertTrue(controller.streamsLiveText)
        controller.liveTranscription = false
        XCTAssertFalse(controller.streamsLiveText, "the live text toggle still decides")
        controller.liveTranscription = true
        controller.recipe.language = "fr"
        XCTAssertNotNil(controller.liveStreamingUnavailableReason, "English only")
    }

    func testLiveWordsBecomeSentenceSegments() {
        func w(_ t: String, _ s: Double) -> AudioStreamWord { AudioStreamWord(text: t, startSeconds: s, endSeconds: s + 0.4) }
        let segments = AudioWorkspaceController.liveSegments([w("Hello", 0), w("there.", 0.5), w("How", 1), w("are", 1.5), w("you?", 2)])
        XCTAssertEqual(segments.map(\.text), ["Hello there.", "How are you?"])
        XCTAssertEqual(segments[1].start, 1)
        XCTAssertEqual(segments[1].end, 2.4, accuracy: 1e-9)
        let long = (0..<40).map { w("word", Double($0)) }
        XCTAssertTrue(AudioWorkspaceController.liveSegments(long).allSatisfy { $0.end - $0.start <= 15.5 }, "unpunctuated speech still splits")
        XCTAssertTrue(AudioWorkspaceController.liveSegments([]).isEmpty)
    }

    func testLiveDraftUpsertKeepsOneDraftRevision() {
        var item = AudioLibraryItem(title: "Take", kind: .recording)
        let id = UUID()
        item.upsertTranscript(id: id, [AudioLibrarySegment(start: 0, end: 1, text: "one")], source: .draft, language: "en")
        item.upsertTranscript(id: id, [AudioLibrarySegment(start: 0, end: 2, text: "one two")], source: .draft, language: "en")
        XCTAssertEqual(item.transcripts.count, 1, "each save replaces the take's draft, it does not append")
        XCTAssertEqual(item.preferredTranscript?.segments.map(\.text), ["one two"])
    }
}
