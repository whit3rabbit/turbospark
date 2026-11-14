import AVFoundation
import CoreGraphics
import XCTest
@testable import TurboSparkApp

/// A sink that records what the controller did and writes real bytes, because
/// the controller judges "saved" by the file on disk.
private actor StubSink: BrowserRecordingSink {
    let url: URL
    let bytesPerFrame: Int
    private(set) var timestamps: [TimeInterval] = []
    private(set) var finishCount = 0
    private var written = 0

    init(url: URL, bytesPerFrame: Int) {
        self.url = url
        self.bytesPerFrame = bytesPerFrame
        FileManager.default.createFile(atPath: url.path, contents: Data())
    }

    func append(_ image: CGImage, atSeconds seconds: TimeInterval) async throws -> Bool {
        timestamps.append(seconds)
        written += bytesPerFrame
        try Data(count: written).write(to: url)
        return true
    }

    func byteCount() async -> Int64 { Int64(written) }

    func finish(endSeconds: TimeInterval) async throws { finishCount += 1 }
}

/// Holds the sinks a factory made so the test can inspect them afterwards.
private final class SinkLog: @unchecked Sendable {
    private let lock = NSLock()
    private var all: [StubSink] = []
    var sinks: [StubSink] { lock.withLock { all } }
    func add(_ sink: StubSink) { lock.withLock { all.append(sink) } }
}

@MainActor
final class BrowserRecordingTests: XCTestCase {
    private var directory: URL!

    override func setUp() async throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("BrowserRecordingTests-\(UUID().uuidString)", isDirectory: true)
    }

    override func tearDown() async throws {
        try? FileManager.default.removeItem(at: directory)
    }

    // MARK: helpers

    private static func image(width: Int = 64, height: Int = 48) -> CGImage {
        let context = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0,
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        context.setFillColor(CGColor(red: 0.2, green: 0.5, blue: 0.9, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        return context.makeImage()!
    }

    private func makeController(
        limits: BrowserRecordingController.Limits = .fast,
        freeSpace: Int64? = Int64.max,
        bytesPerFrame: Int = 10,
        log: SinkLog = SinkLog()
    ) -> (BrowserRecordingController, SinkLog) {
        let directory = directory!
        let controller = BrowserRecordingController(
            limits: limits,
            directory: { directory },
            freeSpace: { _ in freeSpace },
            makeSink: { url, _ in
                let sink = StubSink(url: url, bytesPerFrame: bytesPerFrame)
                log.add(sink)
                return sink
            })
        return (controller, log)
    }

    private func waitForIdle(_ controller: BrowserRecordingController, timeout: TimeInterval = 5) async {
        let deadline = Date().addingTimeInterval(timeout)
        while controller.phase != .idle, Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    private func waitUntil(timeout: TimeInterval = 5, _ condition: @MainActor () async -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while await !condition(), Date() < deadline {
            try? await Task.sleep(nanoseconds: 5_000_000)
        }
    }

    private let goodFrames: BrowserRecordingController.FrameSource = { _ in
        BrowserRecordingTests.image()
    }

    private func savedResult(_ controller: BrowserRecordingController) -> BrowserRecordingResult? {
        if case .saved(let result)? = controller.notice { return result }
        return nil
    }

    // MARK: state machine

    func testUserStopFinalizesAndReportsSavedFile() async throws {
        let (controller, log) = makeController()
        let tab = BrowserTabID(rawValue: UUID())
        try controller.start(tabID: tab, frameSource: goodFrames)
        XCTAssertEqual(controller.recordingTabID, tab)

        await waitUntil { await log.sinks.first?.timestamps.count ?? 0 >= 3 }
        await controller.stop()

        XCTAssertEqual(controller.phase, .idle)
        let result = try XCTUnwrap(savedResult(controller))
        XCTAssertEqual(result.reason, .user)
        XCTAssertGreaterThanOrEqual(result.frameCount, 3)
        XCTAssertGreaterThan(result.byteCount, 0)
        XCTAssertTrue(FileManager.default.fileExists(atPath: result.url.path))
        XCTAssertEqual(result.url.pathExtension, "mp4")
        XCTAssertEqual(result.url.deletingLastPathComponent().path, directory.path)
        let sink = try XCTUnwrap(log.sinks.first)
        let finishes = await sink.finishCount
        XCTAssertEqual(finishes, 1)
        let stamps = await sink.timestamps
        XCTAssertEqual(stamps, stamps.sorted(), "timestamps must not go backwards")
    }

    func testSecondStartWhileRecordingIsRefusedAndFirstContinues() async throws {
        let (controller, log) = makeController()
        let first = BrowserTabID(rawValue: UUID())
        let second = BrowserTabID(rawValue: UUID())
        try controller.start(tabID: first, frameSource: goodFrames)

        XCTAssertThrowsError(try controller.start(tabID: second, frameSource: goodFrames)) {
            XCTAssertEqual($0 as? BrowserRecordingStartError, .alreadyRecording)
        }
        XCTAssertEqual(controller.recordingTabID, first)
        await waitUntil { await log.sinks.first?.timestamps.count ?? 0 >= 1 }
        XCTAssertEqual(log.sinks.count, 1, "the refused start must not open a second file")
        await controller.stop()
        XCTAssertEqual(savedResult(controller)?.reason, .user)
    }

    func testStartRefusedWithoutDiskHeadroomAndNothingIsCreated() throws {
        let limits = BrowserRecordingController.Limits.fast
        let (controller, log) = makeController(
            limits: limits, freeSpace: limits.maxBytes + limits.diskHeadroomBytes - 1)
        XCTAssertThrowsError(
            try controller.start(tabID: BrowserTabID(rawValue: UUID()), frameSource: goodFrames)
        ) {
            XCTAssertEqual($0 as? BrowserRecordingStartError, .insufficientDiskSpace)
        }
        XCTAssertEqual(controller.phase, .idle)
        XCTAssertEqual(controller.notice, .couldNotStart(.insufficientDiskSpace))
        XCTAssertTrue(log.sinks.isEmpty)
    }

    func testStartRefusedWhenFreeSpaceIsUnknown() {
        let (controller, _) = makeController(freeSpace: nil)
        XCTAssertThrowsError(
            try controller.start(tabID: BrowserTabID(rawValue: UUID()), frameSource: goodFrames))
        XCTAssertEqual(controller.phase, .idle)
    }

    func testSizeCapStopsAndKeepsFinalizedFile() async throws {
        var limits = BrowserRecordingController.Limits.fast
        limits.maxBytes = 50
        let (controller, log) = makeController(limits: limits, bytesPerFrame: 20)
        try controller.start(tabID: BrowserTabID(rawValue: UUID()), frameSource: goodFrames)
        await waitForIdle(controller)

        let result = try XCTUnwrap(savedResult(controller))
        XCTAssertEqual(result.reason, .sizeLimit)
        XCTAssertGreaterThanOrEqual(result.byteCount, 50)
        let finishes = await log.sinks[0].finishCount
        XCTAssertEqual(finishes, 1, "a capped recording is still finalized")
    }

    func testDurationCapStops() async throws {
        var limits = BrowserRecordingController.Limits.fast
        limits.maxDuration = 0.08
        let (controller, _) = makeController(limits: limits)
        try controller.start(tabID: BrowserTabID(rawValue: UUID()), frameSource: goodFrames)
        await waitForIdle(controller)
        XCTAssertEqual(savedResult(controller)?.reason, .timeLimit)
    }

    func testPersistentCaptureFailureWithNoFramesDeletesAndReportsCouldNotSave() async throws {
        let (controller, log) = makeController()
        try controller.start(tabID: BrowserTabID(rawValue: UUID())) { _ in
            throw WebKitBrowserEngineError.unapprovedFrameOrigin(origin: "https://other.test")
        }
        await waitForIdle(controller)

        XCTAssertEqual(controller.notice, .couldNotSave)
        XCTAssertTrue(log.sinks.isEmpty, "an unauthorized frame is never handed to the encoder")
        let leftovers = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        XCTAssertTrue(leftovers.isEmpty)
    }

    func testCaptureFailureAfterFramesSavesWhatWasRecorded() async throws {
        var calls = 0
        let (controller, _) = makeController()
        try controller.start(tabID: BrowserTabID(rawValue: UUID())) { _ in
            calls += 1
            if calls > 3 { throw WebKitBrowserEngineError.screenshotUnavailable }
            return Self.image()
        }
        await waitForIdle(controller)
        let result = try XCTUnwrap(savedResult(controller))
        XCTAssertEqual(result.reason, .captureFailed)
        XCTAssertEqual(result.frameCount, 3)
    }

    func testClosingTheRecordedTabFinalizesAndClosingAnotherDoesNot() async throws {
        let (controller, log) = makeController()
        let recorded = BrowserTabID(rawValue: UUID())
        try controller.start(tabID: recorded, frameSource: goodFrames)
        await waitUntil { await log.sinks.first?.timestamps.count ?? 0 >= 1 }

        await controller.tabClosed(BrowserTabID(rawValue: UUID()))
        XCTAssertEqual(controller.recordingTabID, recorded, "another tab closing must not stop it")

        await controller.tabClosed(recorded)
        XCTAssertEqual(controller.phase, .idle)
        XCTAssertEqual(savedResult(controller)?.reason, .tabClosed)
    }

    func testVanishedTabErrorEndsRecording() async throws {
        let tab = BrowserTabID(rawValue: UUID())
        var calls = 0
        let (controller, _) = makeController()
        try controller.start(tabID: tab) { id in
            calls += 1
            if calls > 2 { throw BrowserTabStoreError.tabNotFound(id) }
            return Self.image()
        }
        await waitForIdle(controller)
        XCTAssertEqual(savedResult(controller)?.reason, .tabClosed)
    }

    func testStopWhenIdleIsANoOp() async {
        let (controller, _) = makeController()
        await controller.stop()
        XCTAssertEqual(controller.phase, .idle)
        XCTAssertNil(controller.notice)
    }

    func testNewStartClearsThePreviousNotice() async throws {
        let (controller, _) = makeController()
        try controller.start(tabID: BrowserTabID(rawValue: UUID()), frameSource: goodFrames)
        await controller.stop()
        XCTAssertNotNil(controller.notice)
        try controller.start(tabID: BrowserTabID(rawValue: UUID()), frameSource: goodFrames)
        XCTAssertNil(controller.notice)
        await controller.stop()
    }

    // MARK: real encoder

    func testAVAssetWriterSinkProducesPlayableEvenSizedMP4() async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("real.mp4")
        // Odd source size: H.264 needs even dimensions, so the sink must round.
        let sink = try AVAssetWriterRecordingSink(url: url, size: CGSize(width: 301, height: 201))
        for index in 0..<6 {
            // A real-time encoder may be briefly not ready and drop a frame
            // (false, not an error); wait for it to take this one.
            var appended = false
            for _ in 0..<200 where !appended {
                appended = try await sink.append(
                    Self.image(width: 301, height: 201), atSeconds: Double(index) * 0.25)
                if !appended { try await Task.sleep(nanoseconds: 20_000_000) }
            }
            XCTAssertTrue(appended)
        }
        try await sink.finish(endSeconds: 1.5)

        let asset = AVURLAsset(url: url)
        let tracks = try await asset.loadTracks(withMediaType: .video)
        let track = try XCTUnwrap(tracks.first)
        let size = try await track.load(.naturalSize)
        XCTAssertEqual(Int(size.width) % 2, 0)
        XCTAssertEqual(Int(size.height) % 2, 0)
        let duration = try await asset.load(.duration).seconds
        XCTAssertGreaterThan(duration, 1.0)
        let bytes = await sink.byteCount()
        XCTAssertGreaterThan(bytes, 0)
    }

    func testAVAssetWriterSinkAcceptsAResizedFrameWithoutEndingTheFile() async throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let url = directory.appendingPathComponent("resize.mp4")
        let sink = try AVAssetWriterRecordingSink(url: url, size: CGSize(width: 200, height: 100))
        _ = try await sink.append(Self.image(width: 200, height: 100), atSeconds: 0)
        // Same timestamp again plus a different size: the clock must be forced forward.
        _ = try await sink.append(Self.image(width: 120, height: 160), atSeconds: 0)
        try await sink.finish(endSeconds: 1)
        let tracks = try await AVURLAsset(url: url).loadTracks(withMediaType: .video)
        XCTAssertEqual(tracks.count, 1)
    }

    // MARK: pane model and privacy

    func testPaneCannotStartWithoutATabOrWhileAnotherRecordingRuns() async throws {
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store, authorizeNavigation: { _ in .allow }, onUserTakeover: { _ in })
        let (controller, _) = makeController()
        let pane = BrowserPaneModel(engine: engine, recording: controller)
        XCTAssertFalse(pane.canStartRecording)
        XCTAssertFalse(pane.startRecording())

        _ = pane.createTab()
        XCTAssertTrue(pane.canStartRecording)

        try controller.start(tabID: BrowserTabID(rawValue: UUID()), frameSource: goodFrames)
        XCTAssertFalse(pane.canStartRecording)
        await controller.stop()
    }

    func testClosingTheRecordedPaneTabStopsTheRecording() async throws {
        let store = BrowserTabStore()
        let engine = WebKitBrowserEngine(
            tabStore: store, authorizeNavigation: { _ in .allow }, onUserTakeover: { _ in })
        let (controller, _) = makeController()
        let pane = BrowserPaneModel(engine: engine, recording: controller)
        let tab = pane.createTab()
        try controller.start(tabID: tab, frameSource: goodFrames)

        pane.closeTab(tab)
        await waitForIdle(controller)
        XCTAssertEqual(controller.phase, .idle)
        XCTAssertEqual(savedResult(controller)?.reason, .tabClosed)
    }

    /// The recorder must stay a snapshot recorder: no screen-capture API means
    /// no Screen Recording prompt and no capture of anything outside the view,
    /// and no path that hands the file to a prompt or a model.
    func testRecorderSourceUsesNoScreenCaptureOrPromptAttachmentAPI() throws {
        let root = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp/Browser")
        let source = try String(
            contentsOf: root.appendingPathComponent("BrowserRecordingController.swift"),
            encoding: .utf8)
        for forbidden in [
            "ScreenCaptureKit", "SCStream", "SCShareableContent", "CGRequestScreenCaptureAccess",
            "CGWindowListCreateImage", "addPromptAttachment", "AppPromptAttachment",
        ] {
            XCTAssertFalse(source.contains(forbidden), "recorder must not use \(forbidden)")
        }
    }

    func testElapsedTextFormatting() {
        XCTAssertEqual(BrowserPaneView.elapsedText(seconds: 0), "0:00")
        XCTAssertEqual(BrowserPaneView.elapsedText(seconds: 75.9), "1:15")
        XCTAssertEqual(BrowserPaneView.elapsedText(seconds: -3), "0:00")
        XCTAssertEqual(BrowserPaneView.elapsedText(seconds: .nan), "0:00")
    }
}

private extension BrowserRecordingController.Limits {
    /// Small caps and a fast tick so the loop runs in milliseconds.
    static var fast: Self {
        var limits = Self()
        limits.framesPerSecond = 200
        limits.maxBytes = 1_000_000
        limits.maxDuration = 60
        limits.diskHeadroomBytes = 1_000
        return limits
    }
}
