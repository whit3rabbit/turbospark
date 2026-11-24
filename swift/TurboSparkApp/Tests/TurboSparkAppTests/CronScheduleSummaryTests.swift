import XCTest

@testable import TurboSparkApp

/// The `CronCreate` card's schedule summary (component audit,
/// schedule-card). Runs come from `CronSchedule.nextFire`, the function the
/// scheduler fires from, so these pin what the card promises against what
/// will run, in a fixed zone so DST boundaries are deterministic.
final class CronScheduleSummaryTests: XCTestCase {

    private var newYork: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "America/New_York")!
        return calendar
    }

    private func date(_ y: Int, _ mo: Int, _ d: Int, _ h: Int, _ mi: Int = 0) -> Date {
        newYork.date(from: DateComponents(year: y, month: mo, day: d, hour: h, minute: mi))!
    }

    private func make(_ arguments: [String: String], hasRun: Bool = false, now: Date)
        -> CronScheduleSummary?
    {
        CronScheduleSummary.make(arguments: arguments, hasRun: hasRun, now: now, calendar: newYork)
    }

    // MARK: - Named shapes and next runs

    func testWeekdayMorningNamesTheShapeAndSkipsTheWeekend() throws {
        // Friday 2026-11-20 10:00, after that day's run.
        let summary = try XCTUnwrap(
            make(["cron": "0 9 * * 1-5", "prompt": "standup"], now: date(2026, 11, 20, 10)))
        XCTAssertEqual(summary.description, CronSchedule("0 9 * * 1-5")?.localizedDescription())
        XCTAssertNotNil(summary.description)
        XCTAssertEqual(
            summary.nextRuns,
            [date(2026, 11, 23, 9), date(2026, 11, 24, 9), date(2026, 11, 25, 9)])
    }

    func testParsedCustomShapeShowsRunsButNoDescription() throws {
        // Two hours a day has no plain-language name, but the parser fully
        // understands it, so the runs are still exact.
        let summary = try XCTUnwrap(make(["cron": "0 9,17 * * *"], now: date(2026, 11, 20, 10)))
        XCTAssertNil(summary.description)
        XCTAssertEqual(
            summary.nextRuns,
            [date(2026, 11, 20, 17), date(2026, 11, 21, 9), date(2026, 11, 21, 17)])
    }

    func testReadsTheSameArgumentAliasAsTheExecutor() throws {
        let summary = try XCTUnwrap(make(["expression": "0 9 * * *"], now: date(2026, 11, 20, 10)))
        XCTAssertEqual(summary.expression, "0 9 * * *")
        XCTAssertEqual(summary.nextRuns.count, 3)
    }

    func testMissingExpressionIsNoSummary() {
        XCTAssertNil(make(["prompt": "hi"], now: date(2026, 11, 20, 10)))
    }

    // MARK: - One-shot

    func testOneShotShowsOneRunUntilItHasRunThenNone() throws {
        let args = ["cron": "0 9 * * *", "recurring": "false"]
        let pending = try XCTUnwrap(make(args, now: date(2026, 11, 20, 10)))
        XCTAssertFalse(pending.recurring)
        XCTAssertEqual(pending.nextRuns, [date(2026, 11, 21, 9)])
        // Once the call ran, the one run is past or pending in the store; a
        // time computed from now would be a run that will never happen.
        let ran = try XCTUnwrap(make(args, hasRun: true, now: date(2026, 11, 20, 10)))
        XCTAssertEqual(ran.nextRuns, [])
        XCTAssertNotNil(ran.description)
    }

    // MARK: - DST

    func testSpringForwardSkipsTheMissingLocalTime() throws {
        // 2026-03-08 02:00 EST jumps to 03:00 EDT in New York: 02:30 does not
        // exist that day, and the scheduler does not fire then.
        let summary = try XCTUnwrap(make(["cron": "30 2 * * *"], now: date(2026, 3, 7, 12)))
        XCTAssertEqual(
            summary.nextRuns,
            [date(2026, 3, 9, 2, 30), date(2026, 3, 10, 2, 30), date(2026, 3, 11, 2, 30)])
    }

    func testDailyRunKeepsItsWallClockTimeAcrossSpringForward() throws {
        let summary = try XCTUnwrap(make(["cron": "0 9 * * *"], now: date(2026, 3, 7, 10)))
        XCTAssertEqual(summary.nextRuns.map { newYork.component(.hour, from: $0) }, [9, 9, 9])
        XCTAssertEqual(summary.nextRuns[0], date(2026, 3, 8, 9))
        XCTAssertEqual(summary.nextRuns[1].timeIntervalSince(summary.nextRuns[0]), 24 * 3600)
        // Saturday 09:00 EST to Sunday 09:00 EDT is 23 hours of absolute
        // time: the run follows the wall clock, not a fixed interval.
        XCTAssertEqual(
            summary.nextRuns[0].timeIntervalSince(date(2026, 3, 7, 9)), 23 * 3600)
    }

    // MARK: - Nil on unknown

    func testUnsupportedSyntaxShowsTheRawExpressionOnly() throws {
        let unsupported = [
            "@daily", "@hourly", "0 9 * * MON", "0 9 * JAN *", "0 9 L * *",
            "0 9 * * 1#2", "0 9 ? * *", "0 9 * *", "0 9 * * * 2026", "60 * * * *",
            "0 24 * * *", "*/0 * * * *", "5-1 * * * *",
        ]
        for expression in unsupported {
            let summary = try XCTUnwrap(
                make(["cron": expression], now: date(2026, 11, 20, 10)), expression)
            XCTAssertEqual(summary.expression, expression)
            XCTAssertNil(summary.description, expression)
            XCTAssertEqual(summary.nextRuns, [], expression)
        }
    }

    // MARK: - Formatting

    func testRunLabelUsesTheGivenZoneAndLocale() {
        let label = CronScheduleSummary.runLabel(
            date(2026, 11, 23, 9), locale: Locale(identifier: "en_US_POSIX"),
            timeZone: newYork.timeZone)
        XCTAssertTrue(label.contains("Mon"), label)
        XCTAssertTrue(label.contains("9:00"), label)
        let tokyo = CronScheduleSummary.runLabel(
            date(2026, 11, 23, 9), locale: Locale(identifier: "en_US_POSIX"),
            timeZone: TimeZone(identifier: "Asia/Tokyo")!)
        XCTAssertTrue(tokyo.contains("23:00") || tokyo.contains("11:00"), tokyo)
    }
}
