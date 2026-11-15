import AVFoundation
import AppKit
import CoreGraphics
import CoreVideo
import Foundation
import os

/// One finished recording. Plain values so a notice and a test can hold it.
struct BrowserRecordingResult: Equatable, Sendable {
    let url: URL
    let duration: TimeInterval
    let byteCount: Int64
    let frameCount: Int
    let reason: BrowserRecordingStopReason
}

enum BrowserRecordingStopReason: Equatable, Sendable {
    /// The user pressed Stop.
    case user
    /// The file reached `Limits.maxBytes`.
    case sizeLimit
    /// The elapsed time reached `Limits.maxDuration`.
    case timeLimit
    /// Frames could not be captured for `maxConsecutiveCaptureFailures` ticks
    /// in a row (page crashed, a frame origin was not authorized, tab gone).
    case captureFailed
    /// The recorded tab was closed.
    case tabClosed
    /// The encoder reported an error (disk full, writer failure).
    case writerFailed
}

enum BrowserRecordingStartError: Error, Equatable {
    case alreadyRecording
    case insufficientDiskSpace
    case cannotCreateOutput
}

enum BrowserRecordingNotice: Equatable {
    case saved(BrowserRecordingResult)
    case couldNotStart(BrowserRecordingStartError)
    /// A stop that produced no playable file (no frame was ever captured, or
    /// the encoder failed before the first frame was written).
    case couldNotSave
}

/// Where encoded frames go. An actor-isolated or otherwise thread-safe
/// object, so the pixel copy and encode stay off the main thread.
protocol BrowserRecordingSink: Sendable {
    /// Appends one frame. Returns false when the encoder was not ready and the
    /// frame was dropped (not an error: the previous frame holds on screen).
    func append(_ image: CGImage, atSeconds seconds: TimeInterval) async throws -> Bool
    /// Bytes on disk so far.
    func byteCount() async -> Int64
    /// Finalizes the file so it plays to the last written frame.
    func finish(endSeconds: TimeInterval) async throws
}

/// Records one browser tab to a local MP4 by stitching WKWebView snapshots.
///
/// Why snapshots and not a screen-capture stream (decision, ZP-15): `takeSnapshot`
/// renders the web view's own surface, so the file can never contain another
/// pane, a chat, a menu or an overlay, and it needs no Screen Recording grant
/// (no `RecordingPermissions` prompt, nothing for a decline to block). The
/// frame source is `WebKitBrowserEngine.captureScreenshot`, which already
/// refuses frames whose iframe origins were not authorized, so recording adds
/// no new way to read a page the screenshot tool could not. The price is a
/// low frame rate (default 4 fps) with wall-clock timestamps, so playback
/// timing is truthful even when a capture is slow.
///
/// Files stay in the profile's `BrowserRecordings` folder, are never attached
/// to a prompt and never sent to a model.
@MainActor
final class BrowserRecordingController: ObservableObject {
    static let shared = BrowserRecordingController()

    struct Limits: Equatable {
        /// Default size cap, 500 MB.
        var maxBytes: Int64 = 500 * 1024 * 1024
        /// Default duration cap, 30 minutes.
        var maxDuration: TimeInterval = 30 * 60
        var framesPerSecond: Double = 4
        /// Free space required beyond the size cap before a start is allowed.
        var diskHeadroomBytes: Int64 = 512 * 1024 * 1024
        var maxConsecutiveCaptureFailures = 8
    }

    enum Phase: Equatable {
        case idle
        case recording(tabID: BrowserTabID, startedAt: Date)
        case finalizing
    }

    typealias FrameSource = @MainActor (BrowserTabID) async throws -> CGImage
    typealias SinkFactory = @Sendable (_ url: URL, _ firstFrameSize: CGSize) throws -> any BrowserRecordingSink

    @Published private(set) var phase: Phase = .idle
    @Published private(set) var notice: BrowserRecordingNotice?

    let limits: Limits

    private let directory: () -> URL
    private let freeSpace: (URL) -> Int64?
    private let makeSink: SinkFactory
    private let now: () -> Date
    private let log = Logger(subsystem: "com.whit3rabbit.turbospark", category: "browser-recording")
    private var runTask: Task<Void, Never>?
    private var requestedStop: BrowserRecordingStopReason?

    init(
        limits: Limits = Limits(),
        directory: @escaping () -> URL = { BrowserRecordingController.defaultDirectory() },
        freeSpace: @escaping (URL) -> Int64? = { BrowserRecordingController.availableBytes(at: $0) },
        makeSink: @escaping SinkFactory = { url, size in try AVAssetWriterRecordingSink(url: url, size: size) },
        now: @escaping () -> Date = Date.init
    ) {
        self.limits = limits
        self.directory = directory
        self.freeSpace = freeSpace
        self.makeSink = makeSink
        self.now = now
    }

    var isRecording: Bool {
        if case .recording = phase { return true }
        return false
    }

    var recordingTabID: BrowserTabID? {
        if case .recording(let tabID, _) = phase { return tabID }
        return nil
    }

    nonisolated static func defaultDirectory() -> URL {
        AppStorageRoot.subdirectory("BrowserRecordings")
    }

    nonisolated static func availableBytes(at url: URL) -> Int64? {
        let values = try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey])
        return values?.volumeAvailableCapacityForImportantUsage
    }

    /// Starts recording `tabID`. One recording at a time across every tab: a
    /// second start throws rather than replacing the first, because the
    /// indicator and the single output file would no longer match.
    func start(tabID: BrowserTabID, frameSource: @escaping FrameSource) throws {
        guard phase == .idle else { throw BrowserRecordingStartError.alreadyRecording }
        notice = nil

        let folder = directory()
        guard let free = freeSpace(folder),
              free >= limits.maxBytes + limits.diskHeadroomBytes
        else {
            notice = .couldNotStart(.insufficientDiskSpace)
            throw BrowserRecordingStartError.insufficientDiskSpace
        }
        do {
            try FileManager.default.createDirectory(
                at: folder, withIntermediateDirectories: true,
                attributes: [.posixPermissions: 0o700])
        } catch {
            notice = .couldNotStart(.cannotCreateOutput)
            throw BrowserRecordingStartError.cannotCreateOutput
        }

        let startedAt = now()
        let url = folder.appendingPathComponent(Self.fileName(for: startedAt))
        phase = .recording(tabID: tabID, startedAt: startedAt)
        requestedStop = nil
        // Origins and page titles are deliberately not logged.
        log.info("recording started")
        runTask = Task { [weak self] in
            await self?.run(tabID: tabID, url: url, frameSource: frameSource)
        }
    }

    /// Stops and finalizes. A no-op unless a recording is active.
    func stop() async {
        await finish(reason: .user)
    }

    /// The pane calls this when a tab closes. A recording of another tab is untouched.
    func tabClosed(_ tabID: BrowserTabID) async {
        guard recordingTabID == tabID else { return }
        await finish(reason: .tabClosed)
    }

    func dismissNotice() {
        notice = nil
    }

    private func finish(reason: BrowserRecordingStopReason) async {
        guard isRecording, let task = runTask else { return }
        if requestedStop == nil { requestedStop = reason }
        task.cancel()
        await task.value
    }

    private static func fileName(for date: Date) -> String {
        let formatter = DateFormatter()
        formatter.locale = Locale(identifier: "en_US_POSIX")
        formatter.dateFormat = "yyyyMMdd-HHmmss"
        return "browser-recording-\(formatter.string(from: date))-\(UUID().uuidString.prefix(8)).mp4"
    }

    private func run(tabID: BrowserTabID, url: URL, frameSource: FrameSource) async {
        let clock = ContinuousClock()
        let origin = clock.now
        let interval = Duration.seconds(1 / max(limits.framesPerSecond, 0.1))
        var sink: (any BrowserRecordingSink)?
        var frameCount = 0
        var lastSeconds: TimeInterval = 0
        var failures = 0
        var stopReason: BrowserRecordingStopReason?

        loop: while stopReason == nil {
            if Task.isCancelled {
                stopReason = requestedStop ?? .user
                break
            }
            let tickStart = clock.now
            let elapsed = Self.seconds(origin.duration(to: tickStart))
            if elapsed >= limits.maxDuration {
                stopReason = .timeLimit
                break
            }

            do {
                let image = try await frameSource(tabID)
                failures = 0
                if Task.isCancelled {
                    // A frame that lands after Stop is still the user's page as
                    // of the click, but it would postdate the stop: drop it.
                    stopReason = requestedStop ?? .user
                    break loop
                }
                if sink == nil {
                    sink = try makeSink(url, CGSize(width: image.width, height: image.height))
                }
                if let sink {
                    let at = Self.seconds(origin.duration(to: clock.now))
                    if try await sink.append(image, atSeconds: at) {
                        frameCount += 1
                        lastSeconds = at
                    }
                    if await sink.byteCount() >= limits.maxBytes {
                        stopReason = .sizeLimit
                        break loop
                    }
                }
            } catch is CancellationError {
                stopReason = requestedStop ?? .user
                break loop
            } catch let error as BrowserTabStoreError {
                _ = error
                stopReason = .tabClosed
                break loop
            } catch let capture as WebKitBrowserEngineError {
                // An unauthorized frame origin or an unavailable view: the
                // frame is skipped, never written. Repeated failures stop.
                _ = capture
                failures += 1
                if failures >= limits.maxConsecutiveCaptureFailures {
                    stopReason = .captureFailed
                    break loop
                }
            } catch {
                if sink != nil || frameCount > 0 {
                    stopReason = .writerFailed
                    break loop
                }
                failures += 1
                if failures >= limits.maxConsecutiveCaptureFailures {
                    stopReason = .captureFailed
                    break loop
                }
            }

            let next = tickStart.advanced(by: interval)
            do {
                try await clock.sleep(until: next)
            } catch {
                stopReason = requestedStop ?? .user
            }
        }

        phase = .finalizing
        let reason = stopReason ?? .user
        var finishFailed = false
        if let sink {
            do {
                try await sink.finish(endSeconds: lastSeconds + 1 / max(limits.framesPerSecond, 0.1))
            } catch {
                finishFailed = true
            }
        }
        let bytes = Self.fileSize(url)
        if frameCount > 0, !finishFailed, bytes > 0 {
            let result = BrowserRecordingResult(
                url: url, duration: lastSeconds, byteCount: bytes,
                frameCount: frameCount, reason: reason)
            notice = .saved(result)
        } else {
            try? FileManager.default.removeItem(at: url)
            notice = .couldNotSave
        }
        log.info("recording stopped frames=\(frameCount, privacy: .public) bytes=\(bytes, privacy: .public)")
        runTask = nil
        requestedStop = nil
        phase = .idle
    }

    private static func seconds(_ duration: Duration) -> TimeInterval {
        let parts = duration.components
        return TimeInterval(parts.seconds) + TimeInterval(parts.attoseconds) / 1e18
    }

    private static func fileSize(_ url: URL) -> Int64 {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber
        return size?.int64Value ?? 0
    }
}

/// H.264 MP4 output through AVAssetWriter, hardware encoded where available.
///
/// Fragmented every 10 seconds, so a crash or force quit leaves a file that
/// plays up to the last fragment instead of an unreadable one (the movie
/// header of a non-fragmented file is written only at finish).
actor AVAssetWriterRecordingSink: BrowserRecordingSink {
    enum SinkError: Error { case writerRejected, pixelBufferUnavailable }

    private let url: URL
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    private let width: Int
    private let height: Int
    private var lastTimestamp = CMTime(value: -1, timescale: 600)
    private var sessionStarted = false

    init(url: URL, size: CGSize) throws {
        // H.264 needs even dimensions. A very large viewport is scaled down so
        // a 4K page does not produce an unplayable or enormous stream.
        let maxEdge: CGFloat = 1920
        let scale = min(1, maxEdge / max(size.width, size.height, 1))
        func even(_ value: CGFloat) -> Int { max(2, (Int((value * scale).rounded()) / 2) * 2) }
        width = even(size.width)
        height = even(size.height)
        self.url = url

        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        writer.movieFragmentInterval = CMTime(seconds: 10, preferredTimescale: 600)
        input = AVAssetWriterInput(
            mediaType: .video,
            outputSettings: [
                AVVideoCodecKey: AVVideoCodecType.h264,
                AVVideoWidthKey: width,
                AVVideoHeightKey: height,
                AVVideoCompressionPropertiesKey: [
                    AVVideoAverageBitRateKey: max(500_000, width * height * 2),
                    AVVideoExpectedSourceFrameRateKey: 4,
                    AVVideoMaxKeyFrameIntervalKey: 40,
                ],
            ])
        input.expectsMediaDataInRealTime = true
        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA,
                kCVPixelBufferWidthKey as String: width,
                kCVPixelBufferHeightKey as String: height,
            ])
        guard writer.canAdd(input) else { throw SinkError.writerRejected }
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? SinkError.writerRejected }
    }

    func append(_ image: CGImage, atSeconds seconds: TimeInterval) async throws -> Bool {
        if writer.status == .failed { throw writer.error ?? SinkError.writerRejected }
        guard input.isReadyForMoreMediaData else { return false }
        var time = CMTime(seconds: max(0, seconds), preferredTimescale: 600)
        // Presentation times must strictly increase.
        if time <= lastTimestamp { time = lastTimestamp + CMTime(value: 1, timescale: 600) }
        if !sessionStarted {
            writer.startSession(atSourceTime: time)
            sessionStarted = true
        }
        guard let pool = adaptor.pixelBufferPool else { throw SinkError.pixelBufferUnavailable }
        var buffer: CVPixelBuffer?
        CVPixelBufferPoolCreatePixelBuffer(nil, pool, &buffer)
        guard let buffer else { throw SinkError.pixelBufferUnavailable }

        CVPixelBufferLockBaseAddress(buffer, [])
        defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        guard let context = CGContext(
            data: CVPixelBufferGetBaseAddress(buffer),
            width: width, height: height, bitsPerComponent: 8,
            bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
            space: CGColorSpaceCreateDeviceRGB(),
            bitmapInfo: CGImageAlphaInfo.premultipliedFirst.rawValue
                | CGBitmapInfo.byteOrder32Little.rawValue)
        else { throw SinkError.pixelBufferUnavailable }

        // Aspect-fit into the first frame's canvas: a resized tab keeps a
        // constant video size instead of ending the file.
        context.setFillColor(CGColor(gray: 0, alpha: 1))
        context.fill(CGRect(x: 0, y: 0, width: width, height: height))
        let fit = min(CGFloat(width) / CGFloat(image.width), CGFloat(height) / CGFloat(image.height))
        let drawWidth = CGFloat(image.width) * fit
        let drawHeight = CGFloat(image.height) * fit
        context.interpolationQuality = .medium
        context.draw(image, in: CGRect(
            x: (CGFloat(width) - drawWidth) / 2, y: (CGFloat(height) - drawHeight) / 2,
            width: drawWidth, height: drawHeight))

        guard adaptor.append(buffer, withPresentationTime: time) else {
            throw writer.error ?? SinkError.writerRejected
        }
        lastTimestamp = time
        return true
    }

    func byteCount() async -> Int64 {
        let size = (try? FileManager.default.attributesOfItem(atPath: url.path)[.size]) as? NSNumber
        return size?.int64Value ?? 0
    }

    func finish(endSeconds: TimeInterval) async throws {
        guard sessionStarted else {
            writer.cancelWriting()
            throw SinkError.writerRejected
        }
        input.markAsFinished()
        let end = max(CMTime(seconds: endSeconds, preferredTimescale: 600), lastTimestamp)
        writer.endSession(atSourceTime: end)
        await writer.finishWriting()
        if writer.status != .completed { throw writer.error ?? SinkError.writerRejected }
    }
}
