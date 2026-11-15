import XCTest
@testable import TurboSparkApp

/// OC-14. Every test drives an injected clock and calls `tick()` itself, so
/// nothing waits, hangs or polls for real.
final class GenerationStallMonitorTests: XCTestCase {
    func testSilencePastWindowReportsOnceThenStaysQuiet() {
        var m = GenerationStallMonitor(window: 90)
        m.begin(now: 0)
        XCTAssertNil(m.poll(now: 89))
        XCTAssertEqual(m.poll(now: 90)?.silenceSeconds, 90)
        // Once per window: later polls say nothing until Keep Waiting.
        XCTAssertNil(m.poll(now: 91))
        XCTAssertNil(m.poll(now: 500))
    }

    func testAnyActivityResetsTheWindow() {
        var m = GenerationStallMonitor(window: 90)
        m.begin(now: 0)
        m.noteActivity(now: 80)
        XCTAssertNil(m.poll(now: 160))
        XCTAssertNotNil(m.poll(now: 170))
    }

    /// CUJ 3. Prefill progress goes through the same activity path as a
    /// token, so a long prefill that keeps reporting never raises a panel.
    func testPrefillProgressEveryTenSecondsNeverStalls() {
        var m = GenerationStallMonitor(window: 90)
        m.begin(now: 0)
        var t = 0.0
        while t < 900 {
            t += 10
            m.noteActivity(now: t)
            XCTAssertNil(m.poll(now: t))
        }
    }

    func testKeepWaitingRearmsAFullWindowAndReportsTotalSilence() {
        var m = GenerationStallMonitor(window: 90)
        m.begin(now: 0)
        XCTAssertNotNil(m.poll(now: 90))
        m.keepWaiting(now: 95)
        XCTAssertNil(m.poll(now: 184))
        // Silence is measured from the last real event, not the re-arm.
        XCTAssertEqual(m.poll(now: 185)?.silenceSeconds, 185)
    }

    func testDisarmedMonitorNeverReportsAndEndClearsReported() {
        var m = GenerationStallMonitor(window: 90)
        XCTAssertNil(m.poll(now: 1000))
        m.begin(now: 0)
        m.end()
        XCTAssertNil(m.poll(now: 1000))
        m.noteActivity(now: 1001)
        m.keepWaiting(now: 1002)
        XCTAssertNil(m.poll(now: 5000))
    }

    func testBackwardsClockNeverReports() {
        var m = GenerationStallMonitor(window: 90)
        m.begin(now: 1000)
        XCTAssertNil(m.poll(now: 10))
    }

    func testDefaultWindowIsNinetySeconds() {
        XCTAssertEqual(GenerationStallMonitor.defaultWindow, 90)
    }
}

@MainActor
final class GenerationStallWatchdogTests: XCTestCase {
    private var now: TimeInterval = 0
    private var stalls: [TimeInterval] = []
    private var recovered = 0

    private func makeWatchdog() -> GenerationStallWatchdog {
        now = 0
        stalls = []
        recovered = 0
        return GenerationStallWatchdog(
            window: 90,
            clock: { [unowned self] in self.now },
            onStall: { [unowned self] in self.stalls.append($0) },
            onRecovered: { [unowned self] in self.recovered += 1 })
    }

    func testStallRaisesOnceAndActivityRecovers() {
        let w = makeWatchdog()
        w.begin(pollInterval: nil)
        now = 90; w.tick()
        now = 91; w.tick()
        XCTAssertEqual(stalls, [90])
        XCTAssertTrue(w.isShowingStall)
        now = 100; w.noteActivity()
        XCTAssertEqual(recovered, 1)
        XCTAssertFalse(w.isShowingStall)
    }

    func testEndBeforeTickPresentsNothing() {
        // The stale-panel race: the turn ended between two ticks.
        let w = makeWatchdog()
        w.begin(pollInterval: nil)
        now = 200
        w.end()
        w.tick()
        XCTAssertTrue(stalls.isEmpty)
    }

    func testKeepWaitingHidesWithoutRecoveredCallbackAndRearms() {
        let w = makeWatchdog()
        w.begin(pollInterval: nil)
        now = 90; w.tick()
        w.keepWaiting()
        XCTAssertFalse(w.isShowingStall)
        XCTAssertEqual(recovered, 0)
        now = 179; w.tick()
        XCTAssertEqual(stalls.count, 1)
        now = 180; w.tick()
        XCTAssertEqual(stalls, [90, 180])
    }

    func testEndWhileShowingClearsTheNotice() {
        let w = makeWatchdog()
        w.begin(pollInterval: nil)
        now = 90; w.tick()
        w.end()
        XCTAssertEqual(recovered, 1)
    }
}
