import XCTest
@testable import TurboSparkApp

final class AudioTranscriptionPlanTests: XCTestCase {
    private let durations: [Double] = [0.2, 1, 14.99, 15, 29, 29.5, 29.8, 30, 30.5, 58, 59, 60, 61.25, 120.7, 300, 1_800]

    /// The Rust session loop was a 30 s window every 29 s. Planning it through
    /// the shared function must not move a single window.
    func testRustWindowsAreTheFormerStride() {
        for duration in durations {
            let planned = AudioTranscriptionPlan.windows(duration: duration, length: 30, overlap: 1)
            let former = Array(stride(from: 0.0, to: duration, by: 29.0))
            XCTAssertEqual(planned.map(\.start), former, "\(duration)")
            XCTAssertEqual(planned.map(\.duration), former.map { min(30, duration - $0) }, "\(duration)")
        }
    }

    /// Every instant of the recording belongs to exactly one window, and each
    /// cut lies inside the overlap it splits, so no word is kept twice or lost.
    /// Back-to-back windows can be an ulp apart in floating point, hence the
    /// tolerance.
    func testOwnershipPartitionsTheRecording() {
        for (length, overlap) in [(30.0, 2.0), (14.99, 0.0), (30.0, 1.0)] {
            for duration in durations {
                let windows = AudioTranscriptionPlan.windows(duration: duration, length: length, overlap: overlap, skipsCoveredTail: true)
                guard let first = windows.first, let last = windows.last else { return XCTFail("\(duration)") }
                XCTAssertEqual(first.ownedFrom, 0)
                XCTAssertEqual(last.ownedUntil, .infinity)
                XCTAssertGreaterThanOrEqual(last.end, duration - 1e-9, "the plan must reach the end")
                for (previous, next) in zip(windows, windows.dropFirst()) {
                    XCTAssertEqual(previous.ownedUntil, next.ownedFrom)
                    XCTAssertGreaterThanOrEqual(next.ownedFrom, next.start)
                    XCTAssertLessThanOrEqual(next.ownedFrom, previous.end + 1e-9)
                }
            }
        }
    }

    /// Paraformer's windows never reach its 240,000-sample limit, even after
    /// the PCM reader rounds a window up, and they do not overlap: without
    /// token times an overlap could not be cut.
    func testParaformerWindowsStayUnderItsLimitWithoutOverlap() {
        let length = Double(ParaformerTranscriber.windowSamples) / 16_000 - 0.01
        for duration in durations {
            let windows = AudioTranscriptionPlan.windows(duration: duration, length: length, overlap: 0, skipsCoveredTail: true)
            for window in windows {
                XCTAssertLessThanOrEqual(Int((window.duration * 16_000).rounded(.up)) + 1, ParaformerTranscriber.windowSamples)
            }
            for (previous, next) in zip(windows, windows.dropFirst()) {
                XCTAssertEqual(next.start, previous.end, accuracy: 1e-9)
            }
        }
    }

    /// A last window whose audio the previous one already read is not run
    /// again when the caller can cut overlaps itself. The Rust loop still
    /// gets it.
    func testACoveredTailIsReadOnlyByTheRustLoop() {
        XCTAssertEqual(AudioTranscriptionPlan.windows(duration: 29.5, length: 30, overlap: 2, skipsCoveredTail: true).count, 1)
        XCTAssertEqual(AudioTranscriptionPlan.windows(duration: 29.5, length: 30, overlap: 2).count, 2)
    }

    func testDegenerateInputsPlanNothing() {
        XCTAssertEqual(AudioTranscriptionPlan.windows(duration: 0, length: 30, overlap: 1), [])
        XCTAssertEqual(AudioTranscriptionPlan.windows(duration: 10, length: 30, overlap: 30), [])
        XCTAssertEqual(AudioTranscriptionPlan.windows(duration: 10, length: 0, overlap: 0), [])
    }
}
