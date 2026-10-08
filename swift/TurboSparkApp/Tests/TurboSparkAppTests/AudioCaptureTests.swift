import XCTest
@testable import TurboSparkApp

final class AudioCaptureTests: XCTestCase {
    func testCancelledStartCannotReserveAfterStop() async {
        let capture = AudioCapture(onChunk: { _ in XCTFail("Cancelled start must not capture audio") }, onEvent: { _ in })
        capture.stopAcceptingAndDrain()
        let outcome = await Task { () -> String in
            withUnsafeCurrentTask { $0?.cancel() }
            do {
                // No-source configuration makes the RED phase safe: even a broken
                // cancellation guard cannot touch permissions or start a device.
                try await capture.start(configuration: .init(capturesMicrophone: false))
                return "started"
            } catch is CancellationError { return "cancelled" }
            catch { return String(reflecting: error) }
        }.value
        XCTAssertEqual(outcome, "cancelled", "Cancellation must win before source validation or device setup")
        XCTAssertEqual(capture.currentElapsed(), 0)
    }

    func testTimelineRemovesPausesAndUsesOneClockForSources() {
        var timeline = AudioCaptureTimeline(start: 100)
        XCTAssertEqual(timeline.offset(at: 103), 3)
        timeline.pause(at: 104)
        XCTAssertNil(timeline.offset(at: 108))
        XCTAssertEqual(timeline.elapsed(at: 108), 4)
        timeline.resume(at: 110)
        XCTAssertEqual(timeline.offset(at: 111), 5)
        timeline.pause(at: 112)
        timeline.pause(at: 114)
        timeline.resume(at: 117)
        XCTAssertEqual(timeline.offset(at: 120), 9)
    }

    func testBudgetRejectsOverflowAndReleasesReservations() {
        let budget = AudioCaptureBudget(limit: 1024)
        XCTAssertTrue(budget.reserve(800))
        XCTAssertFalse(budget.reserve(225))
        XCTAssertFalse(budget.reserve(-1))
        budget.release(800)
        XCTAssertTrue(budget.reserve(1024))
        budget.release(1024)
        XCTAssertEqual(budget.pendingBytes, 0)
    }

    func testChunkerBoundsMemoryAndRestoresOffsetsAfterDiscontinuity() throws {
        let chunker = AudioCapture.Chunker(source: "microphone")
        var chunks: [AudioCaptureChunk] = []
        let sink: (AudioCaptureChunk) -> Void = { chunks.append($0) }
        try chunker.append(Array(repeating: 0.1, count: 64_000), sampleRate: 16_000, channels: 1, offset: 0, emit: sink)
        try chunker.append(Array(repeating: 0.2, count: 64_000), sampleRate: 16_000, channels: 1, offset: 4, emit: sink)
        XCTAssertEqual(chunks.map(\.frameCount), [80_000])
        XCTAssertEqual(chunker.samples.count, 48_000)
        try chunker.append(Array(repeating: 0.3, count: 8_000), sampleRate: 8_000, channels: 1, offset: 9, emit: sink)
        try chunker.flush(emit: sink)
        XCTAssertEqual(chunks.map(\.offset), [0, 5, 9])
        XCTAssertEqual(chunks.map(\.sampleRate), [16_000, 16_000, 8_000])
        XCTAssertTrue(chunker.samples.isEmpty)
    }

    func testFailedChunkCommitDoesNotAdvanceDurableTimeline() throws {
        let chunker = AudioCapture.Chunker(source: "microphone")
        try chunker.append([0.1, 0.2], sampleRate: 16_000, channels: 1, offset: 0) { _ in XCTFail("Not a full chunk") }
        XCTAssertThrowsError(try chunker.flush { _ in throw CocoaError(.fileWriteOutOfSpace) })
        XCTAssertEqual(chunker.lastEnd, 0)
        XCTAssertEqual(chunker.samples, [0.1, 0.2])
    }

    /// Accelerated synthetic duration test, not a real microphone or system capture gate.
    func testTwoHourSyntheticCaptureKeepsFiveSecondBufferBound() throws {
        let rate = 8_000.0
        let framesPerPacket = 40_000
        let packet = [Float](repeating: 0.125, count: framesPerPacket)
        let chunker = AudioCapture.Chunker(source: "microphone")
        let timeline = AudioCaptureTimeline(start: 100)
        var chunks = 0
        var totalFrames: Int64 = 0
        let sink: (AudioCaptureChunk) -> Void = { chunk in
            XCTAssertEqual(chunk.offset, Double(chunks * 5), accuracy: 1.0 / rate)
            XCTAssertEqual(chunk.frameCount, Int64(framesPerPacket))
            XCTAssertEqual(chunk.wav.count, framesPerPacket * MemoryLayout<Float>.size + 44)
            totalFrames += chunk.frameCount
            chunks += 1
        }
        for index in 0..<1440 {
            let hostTime = 100 + Double(index * 5)
            let offset = try XCTUnwrap(timeline.offset(at: hostTime))
            try autoreleasepool {
                try chunker.append(packet, sampleRate: rate, channels: 1, offset: offset, emit: sink)
            }
            XCTAssertLessThanOrEqual(chunker.samples.count, framesPerPacket)
        }
        try chunker.flush(emit: sink)
        XCTAssertEqual(chunks, 1440)
        XCTAssertEqual(totalFrames, 57_600_000)
        XCTAssertEqual(chunker.lastEnd, 7200)
        XCTAssertTrue(chunker.samples.isEmpty)
        XCTAssertEqual(timeline.elapsed(at: 7300), 7200)
    }

    func testTimelineRejectsOldAndInvalidPackets() {
        let timeline = AudioCaptureTimeline(start: 100)
        XCTAssertNil(timeline.offset(at: 99))
        XCTAssertNil(timeline.offset(at: .nan))
    }
}
