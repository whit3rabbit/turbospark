import XCTest

@testable import TurboSpark

/// **These tests exist to prove the HAND-WRITTEN header matches the Rust
/// side.** Nothing else in the repository can: the Rust tests call the same
/// function bodies through the `rlib`, so they would pass even if
/// `turbospark.h` declared a wrong signature. Only linking the `staticlib`
/// and calling through the header can catch that, and a mismatch shows up
/// here as a link error or a wrong answer.
///
/// They deliberately need no model: opening one costs gigabytes, and the
/// question here is whether the two sides agree on the ABI.
final class SurfaceTests: XCTestCase {

    func testTheCatalogDecodesIntoSwiftTypes() throws {
        let rows = try TurboSparkCatalog.available()
        XCTAssertFalse(rows.isEmpty, "the embedded catalog should not be empty")
        // Decoding at all is the assertion: every field name below crossed
        // the boundary as camelCase and was matched with no CodingKeys, so a
        // spelling drift on either side fails here.
        let first = try XCTUnwrap(rows.first)
        XCTAssertFalse(first.alias.isEmpty)
        XCTAssertFalse(first.family.isEmpty)
        XCTAssertGreaterThan(first.downloadBytes, 0)
    }

    func testInstallCostIsReportedForACatalogRow() throws {
        let alias = try XCTUnwrap(TurboSparkCatalog.available().first).alias
        let cost = try TurboSparkCatalog.cost(of: alias)
        XCTAssertGreaterThan(cost.downloadBytes, 0)
        XCTAssertGreaterThan(cost.installBytes, 0)
    }

    func testAnErrorCrossesTheBoundaryWithItsMessage() throws {
        // A malformed repository is refused by a shape check BEFORE any
        // network call, which is what makes this safe in a standing suite.
        XCTAssertThrowsError(try TurboSparkCatalog.probe(repo: "nameonly")) { error in
            let e = error as? TurboSparkError
            XCTAssertEqual(e?.code, .json)
            // The message survived the round trip rather than arriving as a
            // bare code, which is the half of `ts_last_error` that a status
            // code alone cannot check.
            XCTAssertTrue(
                e?.message.contains("owner/name") == true,
                "expected the message to name the expected form, got \(e?.message ?? "nil")")
        }
    }

    func testOpeningAMissingModelFailsWithAReadableMessage() async throws {
        do {
            _ = try await TurboSparkSession(modelPath: "/nonexistent/model.gturbo")
            XCTFail("opening a path that does not exist should throw")
        } catch let error as TurboSparkError {
            XCTAssertEqual(error.code, .open)
            XCTAssertTrue(
                error.message.contains("/nonexistent/model.gturbo"),
                "the message should name the path, got \(error.message)")
        }
    }

    func testCatalogRowsKnowWhetherTheyAreInstalled() throws {
        // Cross-checks the two calls against each other: every alias the
        // store reports must be flagged installed in the catalog listing, or
        // one of the two is reading a different store.
        let installed = Set(try TurboSparkCatalog.installed().map(\.alias))
        let flagged = Set(try TurboSparkCatalog.available().filter(\.installed).map(\.alias))
        // A `--repo` pull is in the store without a catalog row, so the
        // catalog's flagged set is a SUBSET rather than equal.
        XCTAssertTrue(
            flagged.isSubset(of: installed),
            "catalog flagged \(flagged.subtracting(installed)) as installed, store disagrees")
    }

    func testPeakFootprintIsReadable() throws {
        // Zero means the counter is unavailable, which on macOS it is not.
        let peak = try XCTUnwrap(TurboSparkSession.peakFootprintBytes)
        XCTAssertGreaterThan(peak, 0)
    }
}
