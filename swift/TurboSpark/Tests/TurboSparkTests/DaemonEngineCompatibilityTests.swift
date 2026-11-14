import XCTest

@testable import TurboSpark

/// OC-13: the app-versus-daemon version rule and the status decode that feeds
/// it. The matrix is pure, so no daemon is spawned.
final class DaemonEngineCompatibilityTests: XCTestCase {
    private func verdict(_ daemon: String?, min: String = "0.8") -> DaemonEngineCompatibility.Verdict {
        DaemonEngineCompatibility.verdict(daemon: daemon, minimum: min)
    }

    func testEqualAndNewerMinorAreCompatible() {
        XCTAssertEqual(verdict("0.8.0"), .compatible)
        XCTAssertEqual(verdict("0.8.7"), .compatible)
        XCTAssertEqual(verdict("0.9.0"), .compatible)
        XCTAssertEqual(verdict("0.12.0"), .compatible)
        XCTAssertEqual(verdict("0.8.1-rc1"), .compatible)
    }

    func testOlderMinorIsRefused() {
        XCTAssertEqual(verdict("0.7.9"), .tooOld(daemon: "0.7.9", minimum: "0.8"))
        // Numeric, not lexicographic: 0.10 is above 0.9.
        XCTAssertEqual(verdict("0.10.0", min: "0.9"), .compatible)
        XCTAssertEqual(verdict("0.9.0", min: "0.10"), .tooOld(daemon: "0.9.0", minimum: "0.10"))
    }

    func testDifferentMajorIsRefusedBothWays() {
        // A higher major must not pass on the strength of a higher minor.
        XCTAssertEqual(verdict("1.9.0"), .majorMismatch(daemon: "1.9.0", minimum: "0.8"))
        XCTAssertEqual(verdict("0.9.0", min: "1.0"), .majorMismatch(daemon: "0.9.0", minimum: "1.0"))
    }

    func testMissingVersionIsUnknownNotCompatible() {
        XCTAssertEqual(verdict(nil), .unknownVersion)
        XCTAssertEqual(verdict(""), .unknownVersion)
        XCTAssertFalse(verdict(nil).isCompatible)
    }

    func testMalformedVersionIsRefused() {
        XCTAssertEqual(verdict("banana"), .unparseable("banana"))
        XCTAssertEqual(verdict("8"), .unparseable("8"))
        XCTAssertEqual(verdict("0.x.1"), .unparseable("0.x.1"))
        XCTAssertEqual(verdict("0.8.0", min: "nope"), .unparseable("nope"))
    }

    func testStatusDecodesTheStampAndAnOlderDaemonDecodesAsUnknown() throws {
        let stamped = try JSONDecoder().decode(
            TurboSparkDaemonStatus.self,
            from: Data(#"{"running":true,"pid":1,"port":8080,"engineVersion":"0.8.0"}"#.utf8))
        XCTAssertEqual(stamped.engineVersion, "0.8.0")
        XCTAssertEqual(stamped.compatibility(), .compatible)

        let old = try JSONDecoder().decode(
            TurboSparkDaemonStatus.self,
            from: Data(#"{"running":true,"pid":1,"port":8080,"model":null}"#.utf8))
        XCTAssertNil(old.engineVersion)
        XCTAssertEqual(old.compatibility(), .unknownVersion)
    }

    /// The app's own build must satisfy its own minimum, or a freshly started
    /// daemon would be refused the moment it came up.
    func testThisBuildSatisfiesItsOwnMinimum() throws {
        let own = try TurboSparkRuntime.buildInfo().version
        XCTAssertEqual(
            DaemonEngineCompatibility.verdict(
                daemon: own, minimum: DaemonEngineCompatibility.minimumEngineVersion),
            .compatible)
    }
}
