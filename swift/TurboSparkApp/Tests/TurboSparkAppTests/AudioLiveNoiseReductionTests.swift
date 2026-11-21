import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A live isolation stream stand-in: holds `delay` samples back like the
/// model's look-ahead, releases `value` for each sample (so cleaned audio is
/// recognizable), and reports `late` as scripted.
private final class FakeEnhancer: AudioLiveEnhancing, @unchecked Sendable {
    private let lock = NSLock()
    let delay: Int
    let value: Float
    var late: (Int) -> Bool
    var failPush = false
    private var held = 0
    private(set) var pushes = 0
    private(set) var flushed = false
    private(set) var closed = false

    init(delay: Int = 3_024, value: Float = 0.5, late: @escaping (Int) -> Bool = { _ in false }) {
        self.delay = delay; self.value = value; self.late = late
    }
    func push(_ pcm: [Float]) throws -> AudioEnhancePush {
        lock.lock(); defer { lock.unlock() }
        if failPush { throw TurboSparkError(code: .invalidArgument, message: "isolator broke") }
        let total = held + pcm.count
        let release = max(0, total - delay)
        held = total - release
        defer { pushes += 1 }
        return AudioEnhancePush(voice: [Float](repeating: value, count: release), late: late(pushes), processingMs: 1)
    }
    func flush() throws -> [Float] {
        lock.lock(); defer { lock.unlock() }
        flushed = true
        let rest = held; held = 0
        return [Float](repeating: value, count: rest)
    }
    func close() { lock.lock(); closed = true; lock.unlock() }
    var isClosed: Bool { lock.lock(); defer { lock.unlock() }; return closed }
}

private func mic(_ seconds: Double, at offset: Double, value: Float = 0.1, rate: Double = 48_000) -> AudioLiveFeed.Packet {
    AudioLiveFeed.Packet(samples: [Float](repeating: value, count: Int(seconds * rate)), sampleRate: rate, channels: 1,
                         offset: offset, source: "microphone")
}

final class AudioLiveVoiceCleanerTests: XCTestCase {
    func testWithoutAStreamEverythingPassesRaw() {
        var cleaner = AudioLiveVoiceCleaner()
        let input = [mic(0.1, at: 0)]
        let out = cleaner.process(input, dropped: 0)
        XCTAssertNil(out.fallback)
        XCTAssertEqual(out.packets.map(\.samples), input.map(\.samples))
    }

    func testCleanedVoiceKeepsTheInputTimelineAndOnlyTheMicrophoneIsCleaned() {
        let fake = FakeEnhancer()
        var cleaner = AudioLiveVoiceCleaner()
        cleaner.attach(fake)
        let system = AudioLiveFeed.Packet(samples: [0.2, 0.2], sampleRate: 16_000, channels: 1, offset: 0, source: "system")
        let first = cleaner.process([mic(0.1, at: 1), system], dropped: 0)
        XCTAssertNil(first.fallback)
        XCTAssertEqual(first.packets.filter { $0.source == "system" }.map(\.samples), [[0.2, 0.2]], "system audio is never cleaned")
        let firstMic = first.packets.filter { $0.source == "microphone" }
        XCTAssertEqual(firstMic.first?.offset, 1, "released voice sits where its input was, not where it was released")
        XCTAssertEqual(firstMic.reduce(0) { $0 + $1.samples.count }, 4_800 - 3_024)
        let second = cleaner.process([mic(0.1, at: 1.1)], dropped: 0).packets
        // The first 3,024 samples of this batch finish the first packet.
        XCTAssertEqual(second.first?.offset ?? 0, 1 + Double(4_800 - 3_024) / 48_000, accuracy: 1e-9)
        XCTAssertEqual(second.last?.offset ?? 0, 1.1, accuracy: 1e-9)
        let tail = cleaner.finish()
        XCTAssertTrue(fake.flushed)
        XCTAssertEqual((firstMic + second + tail).reduce(0) { $0 + $1.samples.count }, 9_600, "every pushed sample comes back once")
        XCTAssertTrue((firstMic + second + tail).allSatisfy { $0.samples.allSatisfy { $0 == 0.5 } && $0.sampleRate == 48_000 })
    }

    func testRunningLateForMoreThanASecondFallsBackToRawForTheRestOfTheTake() {
        let fake = FakeEnhancer(late: { _ in true })
        var cleaner = AudioLiveVoiceCleaner()
        cleaner.attach(fake)
        // Four 0.25 s pushes are exactly one second late: not past the bound.
        for i in 0..<4 {
            XCTAssertNil(cleaner.process([mic(0.25, at: Double(i) * 0.25)], dropped: 0).fallback, "push \(i)")
        }
        let fifth = cleaner.process([mic(0.25, at: 1), mic(0.25, at: 1.25, value: 0.3)], dropped: 0)
        XCTAssertEqual(fifth.fallback, .fellBehind)
        XCTAssertTrue(fake.flushed, "the held-back voice is delivered before switching")
        XCTAssertTrue(fake.isClosed)
        XCTAssertEqual(fake.pushes, 5, "the packet after the switch is not pushed")
        XCTAssertEqual(fifth.packets.last?.samples.first, 0.3, "the rest of the batch passes raw")
        let later = cleaner.process([mic(0.25, at: 1.5, value: 0.3)], dropped: 0)
        XCTAssertNil(later.fallback, "the switch is reported once")
        XCTAssertEqual(later.packets.map(\.samples.first), [0.3])
        XCTAssertTrue(cleaner.gaveUp)
        cleaner.attach(FakeEnhancer())
        XCTAssertNil(cleaner.stream, "cleaning does not come back within the take")
    }

    func testLatePushesThatAreNotInARowKeepCleaning() {
        let fake = FakeEnhancer(late: { $0 % 4 != 3 })
        var cleaner = AudioLiveVoiceCleaner()
        cleaner.attach(fake)
        for i in 0..<20 {
            XCTAssertNil(cleaner.process([mic(0.25, at: Double(i) * 0.25)], dropped: 0).fallback)
        }
        XCTAssertNotNil(cleaner.stream)
    }

    func testABackedUpQueueFallsBackWithoutPushingMore() {
        let fake = FakeEnhancer()
        var cleaner = AudioLiveVoiceCleaner()
        cleaner.attach(fake)
        let out = cleaner.process([mic(0.1, at: 0)], dropped: 1)
        XCTAssertEqual(out.fallback, .fellBehind)
        XCTAssertEqual(fake.pushes, 0)
        XCTAssertEqual(out.packets.last?.samples.first, 0.1)
        var slow = AudioLiveVoiceCleaner(backlogLimitSeconds: 2)
        slow.attach(FakeEnhancer())
        XCTAssertEqual(slow.process([mic(2.5, at: 0)], dropped: 0).fallback, .fellBehind)
    }

    func testAFailedPushFallsBackWithTheReason() {
        let fake = FakeEnhancer()
        fake.failPush = true
        var cleaner = AudioLiveVoiceCleaner()
        cleaner.attach(fake)
        let out = cleaner.process([mic(0.1, at: 0)], dropped: 0)
        guard case .failed(let reason)? = out.fallback else { return XCTFail("no failure reported") }
        XCTAssertTrue(reason.contains("isolator broke"))
        XCTAssertEqual(out.packets.map(\.samples.first), [0.1], "the packet still reaches live text, raw")
        XCTAssertTrue(fake.isClosed)
    }
}

/// Records what live text heard.
private final class RecordingStream: AudioLiveStreaming, @unchecked Sendable {
    private let lock = NSLock()
    private var heard: [Float] = []
    func append(_ pcm: [Float]) throws -> AudioStreamUpdate { lock.lock(); heard += pcm; lock.unlock(); return AudioStreamUpdate() }
    func finish() throws -> AudioStreamUpdate { AudioStreamUpdate(isFinal: true) }
    func cancel() {}
    func close() {}
    var samples: [Float] { lock.lock(); defer { lock.unlock() }; return heard }
}

private final class EventLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [AudioLiveTranscriber.Event] = []
    func add(_ e: AudioLiveTranscriber.Event) { lock.lock(); items.append(e); lock.unlock() }
    var all: [AudioLiveTranscriber.Event] { lock.lock(); defer { lock.unlock() }; return items }
}

final class AudioLiveNoiseReductionTranscriberTests: XCTestCase {
    private func wait(_ condition: () -> Bool, file: StaticString = #filePath, line: UInt = #line) {
        let deadline = Date().addingTimeInterval(5)
        while !condition() && Date() < deadline { usleep(10_000) }
        XCTAssertTrue(condition(), "timed out", file: file, line: line)
    }

    /// Capture hands every packet to its own chunker and to the live feed
    /// (`AudioCapture.enqueue`). Live text must hear the cleaned voice while
    /// the chunk written to disk is the original.
    func testLiveTextHearsCleanedAudioWhileTheSavedChunkIsTheOriginal() async throws {
        let feed = AudioLiveFeed(limitSeconds: 30)
        let stt = RecordingStream(), fake = FakeEnhancer(delay: 0, value: 0)
        let events = EventLog()
        let live = AudioLiveTranscriber(feed: feed, openCleaner: { fake }, open: { stt }, onEvent: { events.add($0) })
        live.start(pumpInterval: .seconds(3_600))
        wait { live.isOpenForTesting && live.isCleaningForTesting }
        let chunker = AudioCapture.Chunker(source: "microphone")
        var written: [AudioCaptureChunk] = []
        let original = [Float](repeating: 0.25, count: 48_000)
        for second in 0..<2 {
            feed.offer(original, sampleRate: 48_000, channels: 1, offset: Double(second), source: "microphone")
            try chunker.append(original, sampleRate: 48_000, channels: 1, offset: Double(second)) { written.append($0) }
        }
        try chunker.flush { written.append($0) }
        live.pumpForTesting()
        _ = await live.finishAndClose()
        XCTAssertGreaterThan(stt.samples.count, 30_000, "live text got the take")
        XCTAssertTrue(stt.samples.allSatisfy { abs($0) < 1e-6 }, "live text heard the cleaned (silent) voice")
        XCTAssertEqual(written.map(\.wav), [try AudioMediaIO.wav(samples: original + original, sampleRate: 48_000, channels: 1)],
                       "the saved chunk is the uncleaned capture")
        XCTAssertTrue(fake.isClosed)
        XCTAssertFalse(events.all.contains { if case .cleaningStopped = $0 { return true }; return false })
    }

    func testFallingBehindReportsOnceAndLiveTextGetsRawAudio() async {
        let feed = AudioLiveFeed(limitSeconds: 30)
        let stt = RecordingStream(), fake = FakeEnhancer(delay: 0, value: 0, late: { _ in true })
        let events = EventLog()
        let live = AudioLiveTranscriber(feed: feed, openCleaner: { fake }, open: { stt }, onEvent: { events.add($0) })
        live.start(pumpInterval: .seconds(3_600))
        wait { live.isOpenForTesting && live.isCleaningForTesting }
        for i in 0..<12 {
            feed.offer([Float](repeating: 0.25, count: 12_000), sampleRate: 48_000, channels: 1, offset: Double(i) * 0.25, source: "microphone")
            live.pumpForTesting()
        }
        _ = await live.finishAndClose()
        let stops = events.all.filter { if case .cleaningStopped = $0 { return true }; return false }
        XCTAssertEqual(stops.count, 1)
        XCTAssertFalse(live.isCleaningForTesting)
        XCTAssertTrue(stt.samples.contains { abs($0 - 0.25) < 0.05 }, "raw microphone reached live text after the switch")
    }

    func testAFailedCleanerOpenKeepsLiveTextRunning() {
        let events = EventLog()
        let stt = RecordingStream()
        let live = AudioLiveTranscriber(feed: AudioLiveFeed(), openCleaner: { throw TurboSparkError(code: .invalidArgument, message: "no isolator") },
                                        open: { stt }, onEvent: { events.add($0) })
        live.start(pumpInterval: .seconds(3_600))
        wait { events.all.contains { if case .cleaningStopped(let r) = $0 { return r.contains("no isolator") }; return false } }
        XCTAssertFalse(events.all.contains { if case .stopped = $0 { return true }; return false })
        live.stopAndDrain()
    }
}

@MainActor
final class AudioLiveNoiseReductionSettingTests: XCTestCase {
    private func makeController(_ root: URL) throws -> AudioWorkspaceController {
        let vault = ProfileVaultStore(rootProvider: { root }, profileIDProvider: { "noise" }, migrateLegacyData: false)
        _ = try vault.prepareForLaunch()
        return AudioWorkspaceController(library: AudioLibraryStore(repository: ProfileRepository(store: vault)), assets: ManagedAssetStore(vault: vault))
    }

    private func profile(canRun: Bool) -> AudioProfile {
        AudioProfile(
            identity: AudioProfileIdentity(task: .enhancement, alias: AudioWorkspaceController.liveNoiseReductionAlias, repository: "inworld/voice-isolator", revision: "abc", assetFingerprint: "f"),
            family: "voice_isolator", displayName: "Realtime",
            capabilities: AudioCapabilities(operations: ["enhance"], backend: "metal", cancellation: "chunk", canRun: canRun),
            pcmFormat: AudioPCMFormat(sampleRate: 48_000, channels: 1), readiness: "implemented_unqualified")
    }

    func testOffByDefaultAndPersisted() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(root)
        XCTAssertFalse(controller.reduceLiveNoise, "off by default")
        controller.reduceLiveNoise = true
        controller.flushPreferences()
        XCTAssertTrue(try makeController(root).reduceLiveNoise, "the choice survives a relaunch")
    }

    func testShownAndUsedOnlyWithTheRealtimeInstallRunnable() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let controller = try makeController(root)
        controller.profiles = []
        controller.reduceLiveNoise = true
        XCTAssertFalse(controller.offersLiveNoiseReduction, "not in the catalog")
        XCTAssertFalse(controller.reducesLiveNoise)
        controller.profiles = [profile(canRun: true)]
        XCTAssertFalse(controller.offersLiveNoiseReduction, "not installed")
        controller.installedPaths[AudioWorkspaceController.liveNoiseReductionAlias] = "/models/isolator"
        XCTAssertTrue(controller.offersLiveNoiseReduction)
        XCTAssertTrue(controller.reducesLiveNoise)
        XCTAssertEqual(controller.liveNoiseReductionPath, "/models/isolator")
        controller.profiles = [profile(canRun: false)]
        XCTAssertFalse(controller.offersLiveNoiseReduction, "cannot run on this Mac")
        XCTAssertFalse(controller.reducesLiveNoise)
        controller.profiles = [profile(canRun: true)]
        controller.reduceLiveNoise = false
        XCTAssertFalse(controller.reducesLiveNoise, "the setting still decides")
    }
}
