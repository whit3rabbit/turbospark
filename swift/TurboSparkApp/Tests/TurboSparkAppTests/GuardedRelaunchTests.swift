import XCTest
@testable import TurboSparkApp

/// OC-14. No test quits, exits or spawns anything: every side effect is a
/// recording seam, and the hard exit is captured instead of scheduled.
final class RelaunchGuardTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("RelaunchGuardTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    func testThirdAttemptInsideTheWindowIsRefused() {
        let g = RelaunchGuard(directory: directory, maxAttempts: 2, window: 600)
        let t = Date(timeIntervalSince1970: 1_000_000)
        XCTAssertTrue(g.recordAttempt(now: t))
        XCTAssertTrue(g.recordAttempt(now: t.addingTimeInterval(30)))
        XCTAssertFalse(g.recordAttempt(now: t.addingTimeInterval(60)))
        // The refusal is not a loophole: it stays refused until the window moves.
        XCTAssertFalse(g.recordAttempt(now: t.addingTimeInterval(599)))
    }

    func testCapSurvivesANewGuardOverTheSameFile() {
        let t = Date(timeIntervalSince1970: 1_000_000)
        let first = RelaunchGuard(directory: directory, maxAttempts: 2, window: 600)
        XCTAssertTrue(first.recordAttempt(now: t))
        XCTAssertTrue(first.recordAttempt(now: t.addingTimeInterval(1)))
        // A relaunched process builds a fresh guard.
        let second = RelaunchGuard(directory: directory, maxAttempts: 2, window: 600)
        XCTAssertFalse(second.recordAttempt(now: t.addingTimeInterval(2)))
    }

    func testAttemptsAgeOutOfTheWindow() {
        let g = RelaunchGuard(directory: directory, maxAttempts: 2, window: 600)
        let t = Date(timeIntervalSince1970: 1_000_000)
        XCTAssertTrue(g.recordAttempt(now: t))
        XCTAssertTrue(g.recordAttempt(now: t.addingTimeInterval(1)))
        XCTAssertTrue(g.recordAttempt(now: t.addingTimeInterval(601)))
    }

    func testCorruptRecordIsReplacedAndStillBoundsTheNextAttempts() throws {
        try Data("not json".utf8).write(to: directory.appendingPathComponent(RelaunchGuard.fileName))
        let g = RelaunchGuard(directory: directory, maxAttempts: 2, window: 600)
        let t = Date(timeIntervalSince1970: 1_000_000)
        XCTAssertTrue(g.recordAttempt(now: t))
        XCTAssertTrue(g.recordAttempt(now: t.addingTimeInterval(1)))
        XCTAssertFalse(g.recordAttempt(now: t.addingTimeInterval(2)))
    }

    func testUnwritableRecordFailsClosed() {
        let missing = directory.appendingPathComponent("no-such-dir", isDirectory: true)
        let g = RelaunchGuard(directory: missing, maxAttempts: 2, window: 600)
        XCTAssertFalse(g.recordAttempt(now: Date()))
    }

    func testFutureDatedAttemptsDoNotCountAsRecent() throws {
        // A clock set back must not lock the user out, nor erase the cap for
        // real recent attempts: only attempts at or before `now` count.
        let g = RelaunchGuard(directory: directory, maxAttempts: 2, window: 600)
        let t = Date(timeIntervalSince1970: 1_000_000)
        XCTAssertTrue(g.recordAttempt(now: t.addingTimeInterval(10_000)))
        XCTAssertTrue(g.recordAttempt(now: t))
        XCTAssertTrue(g.recordAttempt(now: t.addingTimeInterval(1)))
        XCTAssertFalse(g.recordAttempt(now: t.addingTimeInterval(2)))
    }
}

@MainActor
final class GuardedRelaunchTests: XCTestCase {
    private var directory: URL!
    private var events: [String] = []
    private var hardExit: (delay: TimeInterval, work: @Sendable () -> Void)?
    private var canRelaunch = true
    private var spawnThrows = false
    private let t0 = Date(timeIntervalSince1970: 1_000_000)

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("GuardedRelaunchTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        events = []
        hardExit = nil
        canRelaunch = true
        spawnThrows = false
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private struct Boom: Error {}

    private func makeRelaunch(now: @escaping () -> Date = { Date(timeIntervalSince1970: 1_000_000) }) -> GuardedRelaunch {
        GuardedRelaunch(
            attemptGuard: RelaunchGuard(directory: directory, maxAttempts: 2, window: 600),
            seams: .init(
                now: now,
                canRelaunch: { [unowned self] in self.canRelaunch },
                spawnHelper: { [unowned self] in
                    self.events.append("spawn")
                    if self.spawnThrows { throw Boom() }
                },
                flush: { [unowned self] in self.events.append("flush") },
                terminate: { [unowned self] in self.events.append("terminate") },
                endSession: { [unowned self] in self.events.append("endSession") },
                forceExit: { [unowned self] in self.events.append("forceExit") },
                scheduleHardExit: { [unowned self] delay, work in
                    self.events.append("arm")
                    self.hardExit = (delay, work)
                }))
    }

    func testOrderIsSpawnArmFlushTerminate() {
        let r = makeRelaunch()
        XCTAssertEqual(r.perform(), .relaunching)
        // Flush strictly precedes terminate: persist first, then quit. The
        // hard exit is armed before the flush so a wedged flush is bounded.
        XCTAssertEqual(events, ["spawn", "arm", "flush", "terminate"])
        XCTAssertEqual(hardExit?.delay, 10)
    }

    func testHardExitAfterAFinishedFlushClearsTheMarkerThenExits() {
        let r = makeRelaunch()
        _ = r.perform()
        events = []
        hardExit?.work()
        XCTAssertEqual(events, ["endSession", "forceExit"])
    }

    func testHardExitWithAHungFlushKeepsTheMarkerSoNextLaunchSaysSo() {
        // The flush never returns: simulate by capturing the hard exit from
        // inside the flush seam, before the flag is set.
        var captured: (@Sendable () -> Void)?
        let r = GuardedRelaunch(
            attemptGuard: RelaunchGuard(directory: directory, maxAttempts: 2, window: 600),
            seams: .init(
                now: { self.t0 },
                canRelaunch: { true },
                spawnHelper: {},
                flush: { [unowned self] in
                    // The timer fires while the flush is still running.
                    captured?()
                    self.events.append("flush-returned")
                },
                terminate: {},
                endSession: { [unowned self] in self.events.append("endSession") },
                forceExit: { [unowned self] in self.events.append("forceExit") },
                scheduleHardExit: { _, work in captured = work }))
        _ = r.perform()
        XCTAssertEqual(events.first, "forceExit")
        XCTAssertFalse(events.contains("endSession"))
    }

    func testThirdRelaunchInTheWindowIsRefusedAndTouchesNothing() {
        let r1 = makeRelaunch()
        XCTAssertEqual(r1.perform(), .relaunching)
        XCTAssertEqual(makeRelaunch().perform(), .relaunching)
        events = []
        hardExit = nil
        XCTAssertEqual(makeRelaunch().perform(), .refusedTooManyAttempts)
        XCTAssertEqual(events, [], "a refused attempt must not spawn, flush or quit")
        XCTAssertNil(hardExit)
    }

    func testSecondClickOnTheSameRelaunchIsLatched() {
        let r = makeRelaunch()
        XCTAssertEqual(r.perform(), .relaunching)
        events = []
        XCTAssertEqual(r.perform(), .alreadyInProgress)
        XCTAssertEqual(events, [])
    }

    func testNotABundleIsUnavailableAndCostsNoAttempt() {
        canRelaunch = false
        let r = makeRelaunch()
        XCTAssertEqual(r.perform(), .unavailable)
        XCTAssertEqual(events, [])
        canRelaunch = true
        // Not counted: both real attempts are still available.
        XCTAssertEqual(makeRelaunch().perform(), .relaunching)
        XCTAssertEqual(makeRelaunch().perform(), .relaunching)
    }

    func testFailedSpawnDoesNotFlushOrQuitAndCanBeRetried() {
        spawnThrows = true
        let r = makeRelaunch()
        XCTAssertEqual(r.perform(), .unavailable)
        XCTAssertEqual(events, ["spawn"])
        XCTAssertNil(hardExit)
        spawnThrows = false
        events = []
        // The latch was released, so a retry can proceed (and is counted).
        XCTAssertEqual(r.perform(), .relaunching)
    }

    func testFailedSpawnsStillCountTowardTheCap() {
        spawnThrows = true
        XCTAssertEqual(makeRelaunch().perform(), .unavailable)
        XCTAssertEqual(makeRelaunch().perform(), .unavailable)
        XCTAssertEqual(makeRelaunch().perform(), .refusedTooManyAttempts)
    }

    func testRelaunchHelperScriptTakesPidAndPathAsArgumentsOnly() throws {
        // The live seam's script text is a constant; the path never lands in
        // it. Guard the property by reading the source of truth.
        let src = try String(
            contentsOf: URL(fileURLWithPath: #filePath)
                .deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent(
                    "Sources/TurboSparkApp/Diagnostics/GuardedRelaunch.swift"))
        XCTAssertTrue(src.contains("exec /usr/bin/open \\\"$2\\\""))
        XCTAssertFalse(src.contains("open -n"))
    }
}
