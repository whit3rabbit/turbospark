import XCTest
@testable import TurboSparkApp

/// The Scheduled Tasks row's last-run outcome comes from the scheduler's
/// recorded status, and an unknown status shows nothing.
final class CronLastRunResultTests: XCTestCase {
    func testMapsRecordedStatuses() {
        XCTAssertEqual(CronLastRunResult(.init(status: "fired")), .delivered)
        XCTAssertEqual(CronLastRunResult(.init(status: "missed")), .missed)
    }

    func testNilWithoutARunOrForUnknownStatus() {
        XCTAssertNil(CronLastRunResult(nil))
        XCTAssertNil(CronLastRunResult(.init(status: "skipped")))
    }

    /// End to end through the scheduler's own fire path: a run whose
    /// delivery failed is the newest record and reads as missed.
    func testFirePathRecordsTheNewestOutcome() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cron-last-run-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let scheduler = CronScheduler(directory: dir)
        try scheduler.create(cron: "*/5 * * * *", prompt: "p", recurring: true, chatID: UUID())

        scheduler.fireHandler = { _ in true }
        await scheduler.fireDueJobs(now: Date().addingTimeInterval(3_600))
        XCTAssertEqual(CronLastRunResult(scheduler.list().first?.recentRuns.first), .delivered)

        scheduler.fireHandler = { _ in false }
        await scheduler.fireDueJobs(now: Date().addingTimeInterval(7_200))
        XCTAssertEqual(CronLastRunResult(scheduler.list().first?.recentRuns.first), .missed)
    }
}
