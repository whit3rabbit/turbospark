import XCTest

import CTurboSpark

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

    /// Tests that the embedded catalog decodes cleanly into Swift types.
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

    /// Tests that install and download costs are reported for a catalog entry.
    func testInstallCostIsReportedForACatalogRow() throws {
        let alias = try XCTUnwrap(TurboSparkCatalog.available().first).alias
        let cost = try TurboSparkCatalog.cost(of: alias)
        XCTAssertGreaterThan(cost.downloadBytes, 0)
        XCTAssertGreaterThan(cost.installBytes, 0)
    }

    /// Tests that structured error codes and messages cross the FFI boundary.
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

    /// Tests that attempting to open a nonexistent model path throws a descriptive error.
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

    /// Tests consistency between installed models and catalog availability flags.
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

    /// Tests that a misspelled speculation option is refused by NAME, with
    /// no model on the machine.
    ///
    /// **This is the only check anywhere that the two sides agree on the
    /// speculation spellings.** The Rust unit tests call the mapper
    /// directly, so they pass against a `turbospark.h` that documents keys
    /// nothing reads; only encoding a Swift `OpenOptions` and handing the
    /// JSON to the staticlib can catch that. It works without an install
    /// because `open` maps every option BEFORE it touches the disk, so the
    /// error naming the option outranks the error naming the path.
    func testAMisspelledSpeculationOptionIsRefusedByName() async throws {
        var options = OpenOptions()
        // `.block(99)` rather than a bad string: an enum cannot misspell
        // itself, so the value has to be one the Swift side can express and
        // the ENGINE rejects. 99 is past the 1-15 the three front ends
        // share, and the message must say so.
        options.speculation = .block(99)
        do {
            _ = try await TurboSparkSession(
                modelPath: "/nonexistent/model.gturbo", options: options)
            XCTFail("a block outside the allowed range should throw")
        } catch let error as TurboSparkError {
            XCTAssertTrue(
                error.message.contains("99") && error.message.contains("speculation"),
                "expected the option and the value, got \(error.message)")
            XCTAssertFalse(
                error.message.contains("/nonexistent"),
                "the option is answerable without the install and should be answered first")
        }
    }

    /// Tests that a drafter name the engine does not know is refused.
    func testAnUnknownDrafterIsRefusedRatherThanDefaulted() async throws {
        // The enum cannot produce this, so it is built by hand: the point
        // is that the RUST side refuses rather than falling back to auto,
        // which would leave a caller thinking they had picked a drafter.
        struct BadOptions: Encodable { let speculativeDrafter = "mpt" }
        let json = String(decoding: try JSONEncoder().encode(BadOptions()), as: UTF8.self)
        var out: OpaquePointer?
        let status = "/nonexistent/model.gturbo".withCString { path in
            json.withCString { opts in ts_session_open(path, opts, &out) }
        }
        XCTAssertNotEqual(status, 0)
        XCTAssertTrue(
            TurboSparkError.fromLastError(status).message.contains("mpt"),
            "the message should name the misspelling")
    }

    /// Tests that the session peak memory footprint counter can be read.
    func testPeakFootprintIsReadable() throws {
        // Zero means the counter is unavailable, which on macOS it is not.
        let peak = try XCTUnwrap(TurboSparkSession.peakFootprintBytes)
        XCTAssertGreaterThan(peak, 0)
    }
}
