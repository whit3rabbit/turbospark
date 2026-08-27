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

    /// Tests that an inverted or malformed steering layer range is refused before disk.
    func testAnInvalidSteeringLayerRangeIsRefused() async throws {
        var options = OpenOptions()
        options.steering = "/path/to/vector.gguf"
        options.steeringLayers = "10:5"
        do {
            _ = try await TurboSparkSession(
                modelPath: "/nonexistent/model.gturbo", options: options)
            XCTFail("an inverted layer range should throw")
        } catch let error as TurboSparkError {
            XCTAssertTrue(
                error.message.contains("steeringLayers"),
                "expected the option in the message, got \(error.message)")
        }
    }

    /// Tests that steering modifiers without a steering vector path are refused.
    func testSteeringModifiersWithoutPathAreRefused() async throws {
        var options = OpenOptions()
        options.steeringMode = .add
        do {
            _ = try await TurboSparkSession(
                modelPath: "/nonexistent/model.gturbo", options: options)
            XCTFail("steering modifiers without path should throw")
        } catch let error as TurboSparkError {
            XCTAssertTrue(
                error.message.contains("steering options given without a steering vector path"),
                "expected path requirement message, got \(error.message)")
        }
    }

    /// Tests that an unknown steering mode is refused rather than defaulted.
    func testAnUnknownSteeringModeIsRefusedRatherThanDefaulted() async throws {
        struct BadOptions: Encodable {
            let steering = "/path/to/vector.gguf"
            let steeringMode = "unknown"
        }
        let json = String(decoding: try JSONEncoder().encode(BadOptions()), as: UTF8.self)
        var out: OpaquePointer?
        let status = "/nonexistent/model.gturbo".withCString { path in
            json.withCString { opts in ts_session_open(path, opts, &out) }
        }
        XCTAssertNotEqual(status, 0)
        XCTAssertTrue(
            TurboSparkError.fromLastError(status).message.contains("steeringMode"),
            "the message should name the misspelling")
    }

    /// Tests that model recommendations can be decoded into Swift types.
    func testCatalogRecommendationsAreDecodable() throws {
        let recs = try TurboSparkCatalog.recommend(context: 4096)
        XCTAssertFalse(recs.isEmpty, "recommendations should return catalog rows")
        let first = try XCTUnwrap(recs.first)
        XCTAssertFalse(first.alias.isEmpty)
        XCTAssertFalse(first.name.isEmpty)
        XCTAssertFalse(first.verdictSummary.isEmpty)
    }

    /// Tests that system hardware and power telemetry is readable.
    func testSystemTelemetryIsReadable() throws {
        let telemetry = try XCTUnwrap(TurboSparkSession.systemTelemetry)
        XCTAssertGreaterThan(telemetry.physicalMemoryBytes, 0)
        XCTAssertFalse(telemetry.thermalLevel.isEmpty)
    }

    /// Tests that deleting a nonexistent model throws an expected error.
    func testDeletingNonexistentModelThrows() throws {
        XCTAssertThrowsError(try TurboSparkCatalog.delete("nonexistent_model_test_123")) { error in
            let e = error as? TurboSparkError
            XCTAssertEqual(e?.code, .invalidArgument)
            XCTAssertTrue(e?.message.contains("not installed") == true)
        }
    }

    /// Tests that ChatMessage convenience constructors correctly assign roles and contents.
    func testChatMessageConvenienceFactories() {
        let sys = ChatMessage.system("sys prompt")
        XCTAssertEqual(sys.role, .system)
        XCTAssertEqual(sys.content, "sys prompt")

        let dev = ChatMessage.developer("dev instruction")
        XCTAssertEqual(dev.role, .developer)
        XCTAssertEqual(dev.content, "dev instruction")

        let usr = ChatMessage.user("user message")
        XCTAssertEqual(usr.role, .user)
        XCTAssertEqual(usr.content, "user message")

        let asst = ChatMessage.assistant("assistant answer")
        XCTAssertEqual(asst.role, .assistant)
        XCTAssertEqual(asst.content, "assistant answer")

        let tool = ChatMessage.tool("tool output")
        XCTAssertEqual(tool.role, .tool)
        XCTAssertEqual(tool.content, "tool output")
    }

    /// Tests that WindowFitOutcome can decode from JSON without coding keys drift.
    func testWindowFitOutcomeDecodable() throws {
        let json = """
        {
            "retained": [
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hello"}
            ],
            "measuredTokens": 18,
            "removedTurnCount": 2,
            "hasRoomForGeneration": true
        }
        """
        let outcome = try JSONDecoder().decode(WindowFitOutcome.self, from: Data(json.utf8))
        XCTAssertEqual(outcome.retained.count, 2)
        XCTAssertEqual(outcome.retained[0].role, .system)
        XCTAssertEqual(outcome.measuredTokens, 18)
        XCTAssertEqual(outcome.removedTurnCount, 2)
        XCTAssertTrue(outcome.hasRoomForGeneration)
    }

    /// Tests that SessionInfo and SpecialTokens decode correctly from JSON.
    func testSessionInfoSpecialTokensDecodable() throws {
        let json = """
        {
            "modelPath": "/path/to/model.gturbo",
            "family": "qwen36",
            "maxContext": 4096,
            "trainedContext": 32768,
            "pastTrainedContext": false,
            "expertCacheSlots": 16,
            "vocabSize": 151936,
            "dialect": "ChatMl",
            "reasoningSupport": "level",
            "steering": { "active": false },
            "speculation": { "block": null, "drafter": null, "reason": null },
            "specialTokens": {
                "bosId": 1,
                "eosId": 2,
                "padId": 0,
                "endOfTurnId": 151645,
                "stopTokenIds": [151643, 151645],
                "thinkStartId": 151648,
                "thinkEndId": 151649
            }
        }
        """
        let info = try JSONDecoder().decode(SessionInfo.self, from: Data(json.utf8))
        XCTAssertEqual(info.family, "qwen36")
        XCTAssertEqual(info.specialTokens.bosId, 1)
        XCTAssertEqual(info.specialTokens.eosId, 2)
        XCTAssertEqual(info.specialTokens.endOfTurnId, 151645)
        XCTAssertEqual(info.specialTokens.stopTokenIds, [151643, 151645])
        XCTAssertEqual(info.specialTokens.thinkStartId, 151648)
        XCTAssertEqual(info.specialTokens.thinkEndId, 151649)
    }

    /// Tests that GenerateOptions encodes custom stopTokens without error.
    func testGenerateOptionsStopTokensEncodable() throws {
        var options = GenerateOptions()
        options.stopTokens = [151643, 151645]
        let data = try JSONEncoder().encode(options)
        let json = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(json.contains("stopTokens"))
        XCTAssertTrue(json.contains("151643"))
    }

    /// Tests that new C ABI symbols in turbospark.h link and handle null arguments.
    func testNewCABISymbolsLinkAndValidateNullArgs() {
        var out: UnsafeMutablePointer<CChar>?
        let statusPrompt = ts_session_render_prompt(nil, nil, nil, &out)
        XCTAssertEqual(statusPrompt, TS_ERR_INVALID_ARGUMENT)

        let statusTokenize = ts_session_tokenize_json(nil, nil, false, &out)
        XCTAssertEqual(statusTokenize, TS_ERR_INVALID_ARGUMENT)

        let statusDetokenize = ts_session_detokenize_json(nil, nil, false, &out)
        XCTAssertEqual(statusDetokenize, TS_ERR_INVALID_ARGUMENT)
    }
}


