import XCTest
import TurboSpark

@testable import TurboSparkApp

/// OC-13: which version line the menu bar shows for a daemon status. Stubbed
/// payloads, no daemon spawned.
final class ServerAttachGuardTests: XCTestCase {
    private func status(_ engine: String?) -> TurboSparkDaemonStatus {
        TurboSparkDaemonStatus(running: true, pid: 1, port: 8080, engineVersion: engine)
    }

    func testMatchingVersionShowsTheVersionWithoutAWarning() {
        let line = DaemonVersionLine.make(for: status("0.8.2"), minimum: "0.8")
        XCTAssertEqual(line, .version("0.8.2"))
        XCTAssertFalse(line.isWarning)
    }

    func testStaleDaemonNamesBothVersions() {
        let line = DaemonVersionLine.make(for: status("0.7.4"), minimum: "0.8")
        XCTAssertEqual(line, .mismatch(daemon: "0.7.4", minimum: "0.8"))
        XCTAssertTrue(line.isWarning)
    }

    func testMajorSkewIsAMismatchEvenWithAHigherMinor() {
        XCTAssertEqual(
            DaemonVersionLine.make(for: status("1.9.0"), minimum: "0.8"),
            .mismatch(daemon: "1.9.0", minimum: "0.8"))
    }

    func testPreStampDaemonIsAWarningNotAPass() {
        let line = DaemonVersionLine.make(for: status(nil), minimum: "0.8")
        XCTAssertEqual(line, .unknown)
        XCTAssertTrue(line.isWarning)
    }

    func testGarbageVersionIsAMismatch() {
        XCTAssertEqual(
            DaemonVersionLine.make(for: status("banana"), minimum: "0.8"),
            .mismatch(daemon: "banana", minimum: "0.8"))
    }
}
