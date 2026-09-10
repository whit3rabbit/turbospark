import Foundation
import XCTest
@testable import TurboSparkApp

@MainActor
final class CronTimerLifecycleTests: XCTestCase {
    func testRestartInvalidatesPreviousTimer() throws {
        let model = AppModel()
        defer { model.stopCronScheduler() }
        let first = try XCTUnwrap(model.cronPollTimer)
        XCTAssertTrue(first.isValid)

        model.startCronScheduler()
        let second = try XCTUnwrap(model.cronPollTimer)
        XCTAssertFalse(first.isValid)
        XCTAssertTrue(second.isValid)
        XCTAssertFalse(first === second)
    }

    func testStopInvalidatesAndClearsTimer() throws {
        let model = AppModel()
        let timer = try XCTUnwrap(model.cronPollTimer)
        XCTAssertTrue(timer.isValid)

        model.stopCronScheduler()
        XCTAssertFalse(timer.isValid)
        XCTAssertNil(model.cronPollTimer)
        model.stopCronScheduler()
        XCTAssertNil(model.cronPollTimer)
    }

    func testModelReleaseInvalidatesTimer() async throws {
        var model: AppModel? = AppModel()
        weak var releasedModel = model
        let timer = try XCTUnwrap(model?.cronPollTimer)
        defer { timer.invalidate() }
        XCTAssertTrue(timer.isValid)
        model = nil

        // Startup work can temporarily retain the model across an await.
        for _ in 0..<200 {
            if releasedModel == nil { break }
            try await Task.sleep(nanoseconds: 10_000_000)
        }
        XCTAssertNil(releasedModel, "startup work must release the model")
        XCTAssertFalse(timer.isValid, "the run loop must not keep an orphan poller")
    }
}
