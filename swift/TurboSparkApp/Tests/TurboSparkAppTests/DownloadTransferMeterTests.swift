import XCTest
@testable import TurboSparkApp

final class DownloadTransferMeterTests: XCTestCase {
    func testSpeedTracksTransferThenFallsToZeroWithoutNewCallbacks() throws {
        var meter = DownloadTransferMeter()
        meter.reset(bytes: 0, at: 100)
        XCTAssertNil(meter.bytesPerSecond(at: 100))
        meter.record(bytes: 4_000_000, at: 104)
        XCTAssertEqual(try XCTUnwrap(meter.bytesPerSecond(at: 104)), 1_000_000, accuracy: 1)
        XCTAssertEqual(try XCTUnwrap(meter.bytesPerSecond(at: 108)), 500_000, accuracy: 1)
        XCTAssertEqual(meter.bytesPerSecond(at: 115), 0)
        meter.record(bytes: 6_000_000, at: 116)
        XCTAssertGreaterThan(try XCTUnwrap(meter.bytesPerSecond(at: 116)), 0)
    }

    func testResumeExcludesPausedTimeAndBytesAndIgnoresStaleCallbacks() throws {
        var meter = DownloadTransferMeter()
        meter.reset(bytes: 0, at: 0)
        meter.record(bytes: 4_000_000, at: 4)
        meter.reset(bytes: 5_000_000, at: 100)
        XCTAssertNil(meter.bytesPerSecond(at: 101))
        meter.record(bytes: 7_000_000, at: 102)
        meter.record(bytes: 6_000_000, at: 103)
        meter.record(bytes: 8_000_000, at: 101)
        XCTAssertEqual(try XCTUnwrap(meter.bytesPerSecond(at: 102)), 1_000_000, accuracy: 1)
        XCTAssertNil(meter.bytesPerSecond(at: 99), "a clock regression must not produce an invalid speed")
    }
}
