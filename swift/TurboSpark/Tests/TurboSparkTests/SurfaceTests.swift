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

    /// A recommendation says WHERE its footprint came from, and a row nothing
    /// has read says `unknown` rather than reporting a zero as a figure.
    ///
    /// This is the field that stops a detail pane advertising "Zero KB" and
    /// "16 slots" for a checkpoint whose header nobody opened -- the 16 being
    /// the slot floor resolved by ignorance, which is indistinguishable from
    /// a measured 16 without this.
    func testRecommendationsCarryTheSourceOfTheirFootprint() throws {
        let recs = try TurboSparkCatalog.recommend(context: 4096)
        XCTAssertFalse(recs.isEmpty)
        for row in recs where row.countedSource == .unknown {
            XCTAssertEqual(
                row.verdict, .unknown,
                "\(row.alias): an unsourced footprint cannot produce a fit verdict")
        }
        XCTAssertTrue(
            recs.contains { $0.countedSource == .measured },
            "the curated table carries frozen rows for this chip")
    }

    /// The slot count reaches the ranking, and an illegal one is refused
    /// rather than panicking the host.
    ///
    /// The engine is linked INTO this process, so an out-of-set count aborts
    /// the whole app rather than raising something a UI can show. Both arms
    /// matter: a validator that rejected everything would satisfy the refusal
    /// alone.
    func testRecommendTakesASlotCountAndRefusesAnIllegalOne() throws {
        let auto = try TurboSparkCatalog.recommend(context: 4096, expertCacheSlots: .auto)
        let fixed = try TurboSparkCatalog.recommend(context: 4096, expertCacheSlots: .fixed(32))
        XCTAssertFalse(auto.isEmpty)
        XCTAssertEqual(auto.count, fixed.count)

        XCTAssertThrowsError(
            try TurboSparkCatalog.recommend(context: 4096, expertCacheSlots: .fixed(12))
        ) { error in
            let e = error as? TurboSparkError
            XCTAssertEqual(e?.code, .invalidArgument)
            XCTAssertTrue(
                e?.message.contains("expertCacheSlots") == true,
                "the message should name the knob, got \(e?.message ?? "")")
        }
    }

    /// The probe report decodes from the bytes the engine actually emits.
    ///
    /// Written as a literal rather than driven through a network probe so it
    /// pins the WIRE SPELLING on both sides: this binding sets no
    /// `keyDecodingStrategy`, which is what makes a drift on either side a
    /// test failure instead of a silent nil.
    func testProbeReportDecodesTheEmittedShape() throws {
        let json = """
            {"repo":"owner/name","revision":"main","file":"m-Q4_K_M.gguf",
             "downloadBytes":17179869184,"architecture":"qwen3moe","family":"qwen3moe",
             "runnable":true,"refusedBecause":null,
             "types":[{"name":"Q4_K","tensors":338,"bytes":16000000000,"executable":true},
                      {"name":"IQ2_XS","tensors":2,"bytes":null,"executable":false}],
             "affine":null,"expertStride":2920000,"trainedContext":40960,
             "slotCacheBytes":[{"slots":8,"bytes":1120000000},{"slots":16,"bytes":2240000000}],
             "sidecarsPresent":["tokenizer.json"],"sidecarsMissing":["merges.txt"],
             "chatTemplate":"tokenizer_config.json:chat_template","warnings":[],
             "fit":{"verdict":"streams","verdictSummary":"fits, streams experts from disk",
                    "runs":true,"countedBytes":2880000000,"countedSource":"estimated",
                    "mappedBytes":17179869184,"mappedSource":"download","slotCacheSlots":16,
                    "slotCacheBytes":2240000000,"kvBytes":640000000,
                    "residentBytes":960000000,"largestContext":40960,"context":4096,
                    "trainedContext":40960,
                    "contextLadder":[
                      {"context":4096,"kvBytes":640000000,"counted":2880000000,
                       "verdict":"streams","runs":true,"pastTrained":false,
                       "isTrainedMax":false,"isLargestFitting":false},
                      {"context":40960,"kvBytes":6400000000,"counted":8640000000,
                       "verdict":"tight","runs":true,"pastTrained":false,
                       "isTrainedMax":true,"isLargestFitting":true},
                      {"context":131072,"kvBytes":20480000000,"counted":22720000000,
                       "verdict":"refused","runs":false,"pastTrained":true,
                       "isTrainedMax":false,"isLargestFitting":false}]}}
            """
        let report = try JSONDecoder().decode(ProbeReport.self, from: Data(json.utf8))
        XCTAssertTrue(report.runnable)
        XCTAssertNil(report.refusedBecause)
        XCTAssertEqual(report.expertStride, 2_920_000)
        XCTAssertEqual(report.slotCacheBytes.count, 2)

        // An unsized type is nil, NOT 0. A zero sorts to the bottom of a
        // share column, which is the inverse of its real rank.
        let unsized = try XCTUnwrap(report.types.first { !$0.executable })
        XCTAssertNil(unsized.bytes)

        let fit = try XCTUnwrap(report.fit)
        XCTAssertEqual(fit.verdict, .streams)
        XCTAssertEqual(fit.countedSource, .estimated)
        // The mapped figure is the published checkpoint, and says so.
        XCTAssertEqual(fit.mappedSource, "download")
        XCTAssertEqual(fit.context, 4096)

        // The ladder is what answers "can I use the full window", and its
        // rungs must NOT be reconstructable by multiplication: the whole
        // reason it is computed in the engine is that a sliding-window layer
        // stops growing while a full one does not.
        XCTAssertEqual(fit.contextLadder.count, 3)
        XCTAssertEqual(report.trainedContext, 40960)
        let trained = try XCTUnwrap(fit.contextLadder.first { $0.isTrainedMax })
        XCTAssertEqual(trained.context, 40960)
        XCTAssertFalse(trained.pastTrained)
        // Past the trained window is REPORTED, never dropped: RoPE
        // extrapolates rather than failing.
        let beyond = try XCTUnwrap(fit.contextLadder.first { $0.pastTrained })
        XCTAssertEqual(beyond.verdict, .refused)
        XCTAssertFalse(beyond.runs)
    }

    /// A band measured on other silicon decodes and says so. Showing nothing
    /// leaves a user on an M1 with no throughput signal at all; showing it
    /// unlabelled would present another machine's number as theirs.
    func testThroughputBandCarriesTheChipItWasMeasuredOn() throws {
        let json = """
            {"minTokensPerSecond":33.039,"maxTokensPerSecond":45.648,
             "chip":"Apple M4 Max","measuredOnThisChip":false}
            """
        let band = try JSONDecoder().decode(ThroughputBand.self, from: Data(json.utf8))
        XCTAssertEqual(band.chip, "Apple M4 Max")
        XCTAssertFalse(band.measuredOnThisChip)
    }

    /// An install whose shape could not be read has an EMPTY ladder, which is
    /// a question nothing answered rather than a model with no memory cost.
    func testAnUnreadableInstallHasAnEmptyLadderRatherThanZeroRungs() throws {
        let json = """
            {"path":"/tmp/nope","trainedContext":null,"rungs":[]}
            """
        let ladder = try JSONDecoder().decode(ContextLadder.self, from: Data(json.utf8))
        XCTAssertTrue(ladder.rungs.isEmpty)
        XCTAssertNil(ladder.trainedContext)
    }

    /// A refused probe carries the reason, and `fit` is absent rather than
    /// zeroed when the header yielded no shape.
    func testProbeReportDecodesARefusalWithNoFit() throws {
        let json = """
            {"repo":"owner/name","revision":"main","file":null,"downloadBytes":null,
             "architecture":"phi3","family":null,"runnable":false,
             "refusedBecause":"GGUF architecture \\"phi3\\" is recognized but has no decode flow here",
             "types":[],"affine":null,"expertStride":null,"slotCacheBytes":[],
             "sidecarsPresent":[],"sidecarsMissing":["tokenizer.json"],
             "chatTemplate":null,"warnings":["no base model named"],"trainedContext":null,
             "fit":null}
            """
        let report = try JSONDecoder().decode(ProbeReport.self, from: Data(json.utf8))
        XCTAssertFalse(report.runnable)
        XCTAssertTrue(try XCTUnwrap(report.refusedBecause).contains("no decode flow"))
        XCTAssertNil(report.fit, "an absent shape is nil, never a zeroed fit")
    }

    /// The variant listing decodes, keeps unrunnable rows, and reports the
    /// shard count that explains a short picker.
    func testRepoVariantsDecodeAndKeepUnrunnableRows() throws {
        let json = """
            {"repo":"owner/name","revision":"main","shardedSkipped":5,
             "variants":[{"file":"m-Q8_0.gguf","bytes":32000000000,"quantLabel":"Q8_0",
                          "ladderRank":0,"executable":true},
                         {"file":"m-Q2_K.gguf","bytes":null,"quantLabel":null,
                          "ladderRank":null,"executable":false}]}
            """
        let listed = try JSONDecoder().decode(RepoVariants.self, from: Data(json.utf8))
        XCTAssertEqual(listed.shardedSkipped, 5)
        XCTAssertEqual(listed.variants.count, 2)
        // A type with no kernels is LISTED, not hidden: a picker showing
        // three of eight files reads as the repository having three.
        XCTAssertFalse(listed.variants[1].executable)
        XCTAssertNil(listed.variants[1].bytes, "an unknown length is nil, not 0")
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
            "reasoningLevels": ["off", "low", "medium", "xhigh"],
            "steering": {
                "active": false, "supported": true, "reason": null,
                "mode": null, "scale": null, "summary": null
            },
            "toolCalling": { "native": true, "reason": null },
            "speculation": { "block": null, "drafter": null, "reason": null },
            "vision": { "active": false, "imageTokenId": null, "reason": null },
            "specialTokens": {
                "bosId": 1,
                "eosId": 2,
                "padId": 0,
                "endOfTurnId": 151645,
                "stopTokenIds": [151643, 151645],
                "thinkStartId": 151648,
                "thinkEndId": 151649
            },
            "kvBits": "off"
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
        XCTAssertEqual(info.kvBits, "off")
        XCTAssertEqual(info.reasoningEfforts, [.off, .low, .medium, .xhigh])
        XCTAssertFalse(
            info.reasoningEfforts.contains(.high),
            "this checkpoint raises on high, and a picker is built from this array"
        )
    }

    /// **AN UNKNOWN LEVEL IS DROPPED, NOT THROWN**, and the difference is a
    /// whole session rather than a menu entry.
    ///
    /// `SessionInfo` is decoded inside `TurboSparkSession.init`, so a throw
    /// here fails the OPEN -- and the failure path is the one that has to
    /// close the C handle by hand (Gotcha 31). An engine that one day reports
    /// a sixth spelling should cost a caller that one entry and nothing else.
    /// Field-NAME drift is a different question and still fails the decode,
    /// which is what Gotcha 5 asks for.
    func testAnUnknownReasoningLevelIsDroppedRatherThanFailingTheOpen() throws {
        let json = """
        {
            "modelPath": "/m.gturbo", "family": "qwen38", "maxContext": 4096,
            "trainedContext": null, "pastTrainedContext": false,
            "expertCacheSlots": 16, "vocabSize": 151936, "dialect": "ChatMl",
            "reasoningSupport": "level",
            "reasoningLevels": ["off", "low", "ludicrous", "xhigh"],
            "steering": {
                "active": false, "supported": true, "reason": null,
                "mode": null, "scale": null, "summary": null
            },
            "toolCalling": { "native": true, "reason": null },
            "speculation": { "block": null, "drafter": null, "reason": null },
            "vision": { "active": false, "imageTokenId": null, "reason": null },
            "specialTokens": {
                "bosId": null, "eosId": null, "padId": null, "endOfTurnId": null,
                "stopTokenIds": [], "thinkStartId": null, "thinkEndId": null
            },
            "kvBits": "off"
        }
        """
        let info = try JSONDecoder().decode(SessionInfo.self, from: Data(json.utf8))
        XCTAssertEqual(info.reasoningEfforts, [.off, .low, .xhigh])
        XCTAssertEqual(
            info.reasoningLevels.count, 4,
            "the raw spellings are kept whole; only the typed view drops one"
        )
    }

    /// **THE COMPATIBILITY GUARD FOR ADDING IMAGES.** A message with none
    /// must encode `content` as a bare STRING, which is byte for byte what
    /// this binding sent before `images` existed. An unconditional parts
    /// array would have been an ABI break dressed as a field.
    func testAMessageWithoutImagesEncodesContentAsABareString() throws {
        let message = ChatMessage.user("hello")
        let json = String(decoding: try JSONEncoder().encode(message), as: UTF8.self)
        XCTAssertTrue(json.contains("\"content\":\"hello\""), json)
        XCTAssertFalse(json.contains("\"type\""), json)
    }

    /// Images encode as ordered parts and are PREPENDED to the text, which is
    /// what the reference processor builds (`[image, text]`). Appending would
    /// move every mRoPE position past the image and change the prompt for the
    /// same request.
    func testImagesEncodeAsOrderedPartsBeforeTheText() throws {
        let message = ChatMessage(
            role: .user,
            content: "Transcribe this.",
            images: [.path("/a.png"), .base64("QQ==")])
        let data = try JSONEncoder().encode(message)
        let parts = try XCTUnwrap(
            (try JSONSerialization.jsonObject(with: data) as? [String: Any])?["content"]
                as? [[String: String]])

        XCTAssertEqual(parts.count, 3)
        // The IMAGES come first, in the order given, and the text last.
        XCTAssertEqual(parts[0]["type"], "image")
        XCTAssertEqual(parts[0]["path"], "/a.png")
        XCTAssertEqual(parts[1]["type"], "image")
        XCTAssertEqual(parts[1]["base64"], "QQ==")
        XCTAssertEqual(parts[2]["type"], "text")
        XCTAssertEqual(parts[2]["text"], "Transcribe this.")

        // Exactly one source per image part, which is what the engine
        // requires -- both or neither is refused there rather than resolved.
        XCTAssertNil(parts[0]["base64"])
        XCTAssertNil(parts[1]["path"])
    }

    /// Both shapes decode, so a message that made a round trip through
    /// `fitWindow` comes back equal to what went in.
    func testBothContentShapesRoundTrip() throws {
        for message in [
            ChatMessage.user("plain"),
            ChatMessage(role: .user, content: "with a picture", images: [.path("/p.png")]),
            // No text beside the image: the parts array is images alone.
            ChatMessage(role: .user, content: "", images: [.base64("QQ==")]),
        ] {
            let data = try JSONEncoder().encode(message)
            let back = try JSONDecoder().decode(ChatMessage.self, from: data)
            XCTAssertEqual(back, message)
        }
    }

    /// `vision.active` decodes, and a refusing install carries its reason.
    /// A host gates its attach control on this pair.
    func testVisionInfoDecodesActiveAndItsRefusalReason() throws {
        let refusing = """
        { "active": false, "imageTokenId": null,
          "reason": "this install declares no image preprocessing config" }
        """
        let off = try JSONDecoder().decode(
            SessionInfo.Vision.self, from: Data(refusing.utf8))
        XCTAssertFalse(off.active)
        XCTAssertNil(off.imageTokenId)
        XCTAssertNotNil(off.reason)

        let serving = #"{ "active": true, "imageTokenId": 151655, "reason": null, "source": "sidecar", "sidecarPath": "/path/to/tower", "maxPixels": 1048576 }"#
        let on = try JSONDecoder().decode(
            SessionInfo.Vision.self, from: Data(serving.utf8))
        XCTAssertTrue(on.active)
        XCTAssertEqual(on.imageTokenId, 151655)
        XCTAssertNil(on.reason)
        XCTAssertEqual(on.source, "sidecar")
        XCTAssertEqual(on.sidecarPath, "/path/to/tower")
        XCTAssertEqual(on.maxPixels, 1048576)
    }

    /// Tests that OpenOptions encodes visionSidecar when provided.
    func testOpenOptionsVisionSidecarEncodable() throws {
        var options = OpenOptions()
        options.visionSidecar = "/tmp/tower.gturbo-vision"
        let encoder = JSONEncoder()
        encoder.outputFormatting = .withoutEscapingSlashes
        let data = try encoder.encode(options)
        let json = String(decoding: data, as: UTF8.self)
        XCTAssertTrue(json.contains("visionSidecar"))
        XCTAssertTrue(json.contains("/tmp/tower.gturbo-vision"))
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

    /// Every documented spelling round-trips, and `nil` is omitted rather
    /// than encoded as `null` -- matching every other optional field on
    /// `OpenOptions`, and what lets an absent key mean "off" on the Rust
    /// side without a caller having to spell it.
    func testKvBitsSpellingsEncodeToTheDocumentedStrings() throws {
        var options = OpenOptions()
        XCTAssertFalse(
            String(decoding: try JSONEncoder().encode(options), as: UTF8.self).contains("kvBits"),
            "an absent kvBits must be OMITTED, not encoded as null"
        )

        for (kvBits, expected) in [
            (OpenOptions.KvBits.off, "\"off\""),
            (.two, "\"2\""),
            (.three, "\"3\""),
            (.threePointFive, "\"3.5\""),
            (.four, "\"4\""),
        ] {
            options.kvBits = kvBits
            let json = String(decoding: try JSONEncoder().encode(options), as: UTF8.self)
            XCTAssertTrue(
                json.contains("\"kvBits\":\(expected)"),
                "expected kvBits \(expected) in \(json)"
            )
        }
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

        let statusReleaseVision = ts_session_release_vision(nil)
        XCTAssertEqual(statusReleaseVision, TS_ERR_INVALID_ARGUMENT)
    }

    /// The in-process server's C ABI entry points, called through the
    /// STATICLIB rather than the rlib (this file, not `c_surface.rs`), which
    /// is the only thing that can catch a signature mismatch between
    /// `turbospark.h` and the Rust side (`crates/ffi/CLAUDE.md` Gotcha 2).
    /// Real usage against an open session is `RealModelTests`'s job, gated
    /// on an install being present.
    ///
    /// **`ts_server_attach_session` AND `ts_server_poll_events_json` ARE WHY
    /// THIS TEST MATTERS MORE THAN THE OTHERS HERE.** Both are new ARITIES
    /// rather than new fields on an existing options bag, and an arity is
    /// precisely what `c_surface.rs` structurally cannot check: it reaches
    /// these bodies through the rlib and would pass against a header
    /// declaring any signature at all.
    func testServerCABISymbolsLinkAndValidateNullArgs() {
        var out: UnsafeMutablePointer<CChar>?

        let statusInfo = ts_server_info_json(nil, &out)
        XCTAssertEqual(statusInfo, TS_ERR_INVALID_ARGUMENT)

        let statusAttach = ts_server_attach_session(nil, nil, &out)
        XCTAssertEqual(statusAttach, TS_ERR_INVALID_ARGUMENT)

        let statusDetach = ts_server_detach_model(nil, nil)
        XCTAssertEqual(statusDetach, TS_ERR_INVALID_ARGUMENT)

        let statusPoll = ts_server_poll_events_json(nil, 16, &out)
        XCTAssertEqual(statusPoll, TS_ERR_INVALID_ARGUMENT)

        // NULL is a documented no-op, not an error to check a status code
        // for.
        ts_server_stop(nil)
    }

    /// **A NULL SESSION STARTS AN EMPTY SERVER RATHER THAN FAILING**, and
    /// this test used to assert the opposite -- correctly, when
    /// `ts_server_start` required a session. It is the order a GUI wants:
    /// the address exists and can be shown before its user has decided what
    /// to load.
    ///
    /// Needs no model, because nothing is attached. That is the point.
    func testStartingWithNoSessionBindsAnEmptyServer() throws {
        let server = try TurboSparkServer.start()
        defer { server.stop() }

        let info = try server.info()
        XCTAssertGreaterThan(info.port, 0, "port 0 must resolve to a bound one")
        XCTAssertEqual(info.host, "127.0.0.1")
        XCTAssertTrue(info.models.isEmpty)
        XCTAssertEqual(info.modelId, "", "no first model when there is none")
        XCTAssertNotNil(info.baseURL)
    }

    /// A poll on a live server with no traffic is empty and lossless, which
    /// is what a host's timer sees between requests.
    func testPollingAnIdleServerIsEmptyRatherThanAnError() throws {
        let server = try TurboSparkServer.start()
        defer { server.stop() }

        let batch = server.poll()
        XCTAssertTrue(batch.events.isEmpty)
        XCTAssertEqual(batch.dropped, 0)
    }

    /// **A POLL AFTER `stop()` IS EMPTY, NOT AN ERROR.** A poll timer and a
    /// Stop button race by nature, so a host should not have to catch an
    /// error for the ordinary case of one tick arriving late. `info()` and
    /// `detach` DO throw there, because those are things a caller asked for
    /// deliberately and getting silence back would be worse.
    func testPollingAfterStopIsEmptyWhileInfoThrows() throws {
        let server = try TurboSparkServer.start()
        server.stop()

        XCTAssertTrue(server.poll().events.isEmpty)
        XCTAssertThrowsError(try server.info())
        XCTAssertThrowsError(try server.detach(modelId: "anything"))
    }

    /// The event decoder keeps the events it DOES know when a newer engine
    /// sends one it does not, rather than failing the whole batch and
    /// blanking a console over one unrecognized row.
    func testAnUnknownEventKindDoesNotDiscardTheKnownOnes() throws {
        let json = """
            {"events":[
              {"kind":"requestStarted","id":1,"atMs":5,"method":"GET","path":"/health"},
              {"kind":"somethingNewer","id":2},
              {"kind":"requestFinished","id":1,"status":200,"durationMs":3}
            ],"dropped":7}
            """
        let batch = try JSONDecoder().decode(ServerEventBatch.self, from: Data(json.utf8))

        XCTAssertEqual(batch.events.count, 3)
        XCTAssertEqual(batch.dropped, 7)
        XCTAssertEqual(batch.events[0], .requestStarted(id: 1, atMs: 5, method: "GET", path: "/health"))
        XCTAssertEqual(batch.events[1], .unknown(kind: "somethingNewer"))
        XCTAssertEqual(batch.events[2], .requestFinished(id: 1, status: 200, durationMs: 3))
        XCTAssertEqual(batch.events[1].requestID, nil, "an unknown event ties to no request")
    }

    /// `requested` and `served` are separate fields, and the case that
    /// matters is the one where they differ.
    func testARoutedEventCarriesBothTheAskedForAndTheServingModel() throws {
        let json = """
            {"kind":"requestRouted","id":4,"requested":"claude-sonnet-4-6",
             "served":"gemma4.gturbo","stream":true}
            """
        let event = try JSONDecoder().decode(ServerEvent.self, from: Data(json.utf8))
        XCTAssertEqual(
            event,
            .requestRouted(
                id: 4, requested: "claude-sonnet-4-6", served: "gemma4.gturbo", stream: true))
        XCTAssertEqual(event.requestID, 4)
    }

    /// `ServerInfo` decodes against an engine that predates `models` and
    /// `uptimeSeconds` rather than throwing and losing every field with
    /// them, and it reconstructs `models` from the one id such an engine
    /// does report.
    func testServerInfoDecodesAgainstAnOlderEngine() throws {
        let json = """
            {"port":8080,"host":"127.0.0.1","modelId":"gemma4.gturbo","authEnabled":false}
            """
        let info = try JSONDecoder().decode(ServerInfo.self, from: Data(json.utf8))
        XCTAssertEqual(info.models, ["gemma4.gturbo"])
        XCTAssertEqual(info.uptimeSeconds, 0)
        XCTAssertEqual(info.baseURL?.absoluteString, "http://127.0.0.1:8080")
    }

    /// **The language rule `TurboSparkSession.init`'s manual
    /// `ts_session_close` rests on, stated as a test because it is the thing
    /// a future reader will doubt.**
    ///
    /// A class initializer that throws BEFORE every stored property is
    /// assigned leaves an instance that was never fully initialized, and
    /// Swift does not run `deinit` on one. So a `deinit` holding the only
    /// release of a C resource releases NOTHING on that path, however
    /// plainly it reads as cleanup. `TurboSparkSession.init` acquires its
    /// handle from `ts_session_open` and then reads `ts_session_info_json`,
    /// which can fail; before the fix that failure stranded an entire open
    /// session -- mapped weights, KV cache and compiled Metal pipelines --
    /// for the life of the process.
    ///
    /// This cannot be asserted against the real initializer from here: making
    /// it throw at that exact point needs `ts_session_open` to SUCCEED and the
    /// info read to fail, which is a fault injection the C ABI does not offer
    /// and a bad path cannot produce (a bad path fails at `open`, before any
    /// handle exists). What IS assertable is the rule itself, on a local
    /// stand-in of the same shape.
    func testAThrowingInitDoesNotRunDeinitSoCLeanupMustBeManual() {
        final class ResourceHolder {
            static var deinitRan = false
            static var manualReleaseRan = false
            let resource: Int

            init(failAfterAcquiring: Bool) throws {
                let acquired = 1
                if failAfterAcquiring {
                    // What the fixed initializer does: release by hand,
                    // because `deinit` below will not be reached.
                    Self.manualReleaseRan = true
                    throw TurboSparkError(code: .open, message: "failed after acquiring")
                }
                self.resource = acquired
            }

            deinit { Self.deinitRan = true }
        }

        ResourceHolder.deinitRan = false
        ResourceHolder.manualReleaseRan = false
        XCTAssertThrowsError(try ResourceHolder(failAfterAcquiring: true))
        XCTAssertFalse(
            ResourceHolder.deinitRan,
            "deinit must NOT run for a partially initialized class -- if this ever "
                + "goes green, TurboSparkSession.init's manual ts_session_close is "
                + "redundant and should be revisited")
        XCTAssertTrue(
            ResourceHolder.manualReleaseRan,
            "the throwing path is the only place the resource can be released")

        // The succeeding path is the control: deinit DOES run there, which is
        // what makes the asymmetry above a real hazard rather than a
        // never-deallocating type.
        ResourceHolder.deinitRan = false
        do {
            _ = try ResourceHolder(failAfterAcquiring: false)
        } catch {
            XCTFail("the succeeding path must not throw: \(error)")
        }
        XCTAssertTrue(ResourceHolder.deinitRan, "a fully initialized instance does run deinit")
    }

    /// A failed open must not accumulate process footprint, whatever stage of
    /// `init` it failed at.
    ///
    /// Deliberately a WEAK bound rather than an equality: `phys_footprint` is
    /// a process-wide peak that other work in this suite also moves, so the
    /// assertion is that twenty failed opens do not add a session's worth of
    /// memory (hundreds of MiB at the very least), not that they add zero.
    /// A regression here is the M1 leak reaching a path a test can see.
    func testRepeatedFailedOpensDoNotAccumulateFootprint() async {
        let before = TurboSparkSession.peakFootprintBytes ?? 0

        for _ in 0..<20 {
            do {
                _ = try await TurboSparkSession(
                    modelPath: "/nonexistent/turbospark-test-\(UUID().uuidString).gturbo")
                XCTFail("opening a nonexistent install must throw")
            } catch {
                // Expected. The code varies by platform (TS_ERR_OPEN on
                // macOS, TS_ERR_UNSUPPORTED elsewhere), so the THROW is the
                // assertion and the code is not.
            }
        }

        let after = TurboSparkSession.peakFootprintBytes ?? 0
        guard before > 0, after > 0 else {
            // The counter is unavailable off macOS; there is nothing to
            // assert rather than something that failed.
            return
        }
        XCTAssertLessThan(
            after - before, 128 * 1_024 * 1_024,
            "twenty failed opens grew peak footprint by \((after - before) / 1_024 / 1_024) MiB")
    }

    /// **`supported` AND `active` ARE DIFFERENT QUESTIONS, and the fixture is
    /// the combination a UI gets wrong.**
    ///
    /// An unsteered session on a family that steers perfectly well answers
    /// `active: false, supported: true`. A caller that gated its control on
    /// `active` would disable it for every model that is not already
    /// steering, i.e. all of them at first open -- which is the control being
    /// permanently off rather than being gated.
    func testAnUnsteeredSessionOnASteerableFamilyStillReportsSupported() throws {
        let info = try decodeInfo(
            steering: """
            { "active": false, "supported": true, "reason": null,
              "mode": null, "scale": null, "summary": null }
            """,
            toolCalling: #"{ "native": true, "reason": null }"#
        )
        XCTAssertFalse(info.steering.active)
        XCTAssertTrue(info.steering.supported)
        XCTAssertNil(info.steering.reason)
    }

    /// A family that cannot steer names itself, so a disabled control can say
    /// why rather than merely being grey.
    func testAnUnsupportedFamilyCarriesTheRefusalsOwnWording() throws {
        let info = try decodeInfo(
            steering: """
            { "active": false, "supported": false,
              "reason": "steering is not wired for family Qwen4Exp: its flow does not dispatch the edit",
              "mode": null, "scale": null, "summary": null }
            """,
            toolCalling: #"{ "native": true, "reason": null }"#
        )
        XCTAssertFalse(info.steering.supported)
        XCTAssertEqual(
            info.steering.reason?.contains("Qwen4Exp"), true,
            "a reason that does not name the family cannot be shown to a user"
        )
    }

    /// `toolCalling.native == false` arrives WITH a reason, because the
    /// correct UI response is to explain rather than to hide: a dialect with
    /// no tool markup is the case a guardrail rescue helps most.
    func testANonNativeToolCallingReportCarriesItsReason() throws {
        let info = try decodeInfo(
            steering: """
            { "active": false, "supported": true, "reason": null,
              "mode": null, "scale": null, "summary": null }
            """,
            toolCalling: """
            { "native": false,
              "reason": "the Mistral dialect defines no tool-call markup" }
            """
        )
        XCTAssertFalse(info.toolCalling.native)
        XCTAssertEqual(info.toolCalling.reason?.contains("Mistral"), true)
    }

    /// **A MISSING `supported` FAILS THE DECODE, DELIBERATELY.** It is a
    /// non-optional `Bool`, so engine-side field drift reddens here rather
    /// than silently defaulting a capability to false and greying out a
    /// control on every model (Gotcha 5's rule: this suite is the only thing
    /// that can catch the two sides disagreeing).
    func testAnAbsentSupportedFlagFailsTheDecodeRatherThanDefaulting() {
        XCTAssertThrowsError(
            try decodeInfo(
                steering: #"{ "active": false }"#,
                toolCalling: #"{ "native": true, "reason": null }"#
            )
        )
    }

    /// A server started with guardrails off says so in the options it encodes,
    /// and absent means the engine default rather than off.
    func testServerOptionsEncodeGuardrailsOnlyWhenAsked() throws {
        let encoder = JSONEncoder()
        let bare = String(data: try encoder.encode(ServerOptions()), encoding: .utf8) ?? ""
        XCTAssertFalse(
            bare.contains("guardrails"),
            "an absent value must not encode, or every existing caller silently changes meaning"
        )
        let off = String(
            data: try encoder.encode(ServerOptions(port: 0, apiKey: nil, guardrails: .off)),
            encoding: .utf8
        ) ?? ""
        XCTAssertTrue(off.contains("\"guardrails\":\"off\""), off)
    }

    /// A control vector reports what the FILE carries. `minLayer` of 1 is the
    /// llama.cpp convention working, not a gap, and `spannedLayers` rather
    /// than `coveredLayers` is what a layer-count check compares against.
    func testControlVectorInfoDecodesTheShapeItReports() throws {
        let json = """
        {
            "hidden": 5120, "coveredLayers": 63, "minLayer": 1, "maxLayer": 63,
            "spannedLayers": 64, "declaredMode": "ablate", "declaredArch": "qwen3_5"
        }
        """
        let info = try JSONDecoder().decode(ControlVectorInfo.self, from: Data(json.utf8))
        XCTAssertEqual(info.hidden, 5120)
        XCTAssertEqual(info.minLayer, 1)
        XCTAssertEqual(info.spannedLayers, 64)
        XCTAssertNotEqual(
            info.coveredLayers, info.spannedLayers,
            "the two differ whenever block 0 is absent, which is every well-formed file"
        )
    }

    /// Builds a `SessionInfo` around the two blocks under test, so a case
    /// states only what it is about.
    private func decodeInfo(steering: String, toolCalling: String) throws -> SessionInfo {
        let json = """
        {
            "modelPath": "/m.gturbo", "family": "qwen38", "maxContext": 4096,
            "trainedContext": null, "pastTrainedContext": false,
            "expertCacheSlots": 16, "vocabSize": 151936, "dialect": "ChatMl",
            "reasoningSupport": "none", "reasoningLevels": ["off"],
            "steering": \(steering),
            "toolCalling": \(toolCalling),
            "speculation": { "block": null, "drafter": null, "reason": null },
            "vision": { "active": false, "imageTokenId": null, "reason": null },
            "specialTokens": {
                "bosId": null, "eosId": null, "padId": null, "endOfTurnId": null,
                "stopTokenIds": [], "thinkStartId": null, "thinkEndId": null
            },
            "kvBits": "off"
        }
        """
        return try JSONDecoder().decode(SessionInfo.self, from: Data(json.utf8))
    }

    // MARK: - TS_EVENT_TOOL

    /// **THE OTHER HALF OF A CONTRACT NEITHER SIDE'S SUITE COULD SEE.** No
    /// `GenerateOptions` field offers tools, so no real turn can carry a
    /// `TS_EVENT_TOOL` payload and no end-to-end case can reach this parser
    /// at all. The Rust side pins what `tool_call_json` EMITS
    /// (`a_tool_call_row_carries_id_name_and_an_object_of_arguments`); this
    /// pins what Swift ACCEPTS, against the same three spellings. Written
    /// from the emitted shape by hand rather than shared, because a fixture
    /// generated by the producer would agree with the producer whatever it
    /// spelled.
    func testAToolCallPayloadParsesIntoItsThreeFields() throws {
        let payload = #"{"id":"toolu_0","name":"get_weather","arguments":{"city":"Oslo","days":3}}"#
        let call = try XCTUnwrap(GenerationToolCall(parsingJSON: payload))
        XCTAssertEqual(call.id, "toolu_0")
        XCTAssertEqual(call.name, "get_weather")
        // `argumentsJSON` is re-serialized rather than sliced out of the
        // payload, so assert the VALUE it decodes to and never its bytes:
        // JSONSerialization does not promise key order.
        let decoded = try JSONSerialization.jsonObject(
            with: Data(call.argumentsJSON.utf8)) as? [String: Any]
        XCTAssertEqual(decoded?["city"] as? String, "Oslo")
        XCTAssertEqual(decoded?["days"] as? Int, 3)
    }

    /// A payload this parser cannot make sense of yields nil, which
    /// `streamCallback` drops. **That is the right failure for a STREAM
    /// event**: the alternative is throwing out of a C callback, and the
    /// turn's own `toolCalls` result field carries the same rows anyway, so
    /// a dropped event costs the live update and not the data.
    ///
    /// `name` is the required field because it is the only one a host can
    /// act on: an id it did not generate and arguments it cannot attribute
    /// are not a call.
    func testAToolCallPayloadWithoutANameIsRejectedRatherThanHalfBuilt() {
        XCTAssertNil(GenerationToolCall(parsingJSON: #"{"id":"toolu_0"}"#))
        XCTAssertNil(GenerationToolCall(parsingJSON: "not json at all"))
        // An id is NOT required: it is generated engine-side and a row
        // missing one is still an actionable call.
        let call = GenerationToolCall(parsingJSON: #"{"name":"ping","arguments":{}}"#)
        XCTAssertEqual(call?.name, "ping")
        XCTAssertEqual(call?.id, "")
    }

    // MARK: - Hugging Face Token Auth

    func testHfTokenLifecycleAndValidationDecoding() throws {
        let baselineToken = try TurboSparkCatalog.getHfToken()
        defer {
            if let baselineToken {
                try? TurboSparkCatalog.setHfToken(baselineToken)
            } else {
                try? TurboSparkCatalog.clearHfToken()
            }
        }

        // Test set and get
        try TurboSparkCatalog.setHfToken("hf_test_swift_token_123")
        XCTAssertEqual(try TurboSparkCatalog.getHfToken(), "hf_test_swift_token_123")

        // Test clear: reverts to whatever ambient token (env/cache) or nil existed
        try TurboSparkCatalog.clearHfToken()
        XCTAssertEqual(try TurboSparkCatalog.getHfToken(), baselineToken)

        // Test validate decoding
        let status = try TurboSparkCatalog.validateHfToken("hf_dummy_invalid_token")
        switch status {
        case .invalid, .unavailable:
            break
        default:
            XCTFail("Expected invalid or unavailable for dummy token, got \(status)")
        }
    }

    // MARK: - Hugging Face Mirror Endpoint

    func testHfEndpointLifecycle() throws {
        let baseline = try TurboSparkCatalog.getHfEndpoint()
        defer {
            try? TurboSparkCatalog.setHfEndpoint(baseline == "https://huggingface.co" ? nil : baseline)
        }

        try TurboSparkCatalog.setHfEndpoint("https://hf-mirror.com")
        XCTAssertEqual(try TurboSparkCatalog.getHfEndpoint(), "https://hf-mirror.com")

        try TurboSparkCatalog.setHfEndpoint(nil)
        XCTAssertEqual(try TurboSparkCatalog.getHfEndpoint(), "https://huggingface.co")
    }

    // MARK: - Embedding and Similarity

    func testCosineSimilarity() {
        let v1: [Float] = [1.0, 0.0, 0.0]
        let v2: [Float] = [1.0, 0.0, 0.0]
        let v3: [Float] = [0.0, 1.0, 0.0]

        let same = TurboSparkEmbedding.cosineSimilarity(v1, v2)
        XCTAssertEqual(same, 1.0, accuracy: 1e-5)

        let ortho = TurboSparkEmbedding.cosineSimilarity(v1, v3)
        XCTAssertEqual(ortho, 0.0, accuracy: 1e-5)

        // Empty and mismatched length checks
        XCTAssertEqual(TurboSparkEmbedding.cosineSimilarity([], []), 0.0)
        XCTAssertEqual(TurboSparkEmbedding.cosineSimilarity([1.0], [1.0, 2.0]), 0.0)
    }

    func testServerAttachEmbeddingModelRefusesMissingPath() throws {
        let server = try TurboSparkServer.start(options: ServerOptions(port: 0))
        defer { server.stop() }

        XCTAssertThrowsError(try server.attachEmbeddingModel("/nonexistent/model/path"))
    }

    // MARK: - TurboSparkAgent launch commands

    /// The plain shape is unchanged: quoted base URL, quoted key, the agent
    /// named by the env vars it reads.
    func testLaunchCommandWrapsBothValuesInDoubleQuotes() {
        let cmd = TurboSparkAgent.launchCommand(
            for: "claude", host: "127.0.0.1", port: 8080, apiKey: "sk-abc")
        XCTAssertEqual(
            cmd,
            "export ANTHROPIC_BASE_URL=\"http://127.0.0.1:8080/v1\""
                + " && export ANTHROPIC_API_KEY=\"sk-abc\" && claude")
    }

    /// **A HAND-TYPED KEY CANNOT BREAK OUT OF THE QUOTES IT IS PASTED
    /// INSIDE.** The command goes straight into a terminal; an unescaped
    /// quote terminates the export and a `$` or backtick runs part of the
    /// key as a substitution. Every metacharacter a double-quoted shell
    /// context treats specially survives as its escaped self.
    func testLaunchCommandEscapesShellMetacharactersInTheKey() {
        let raw = "a\\b\"c$d`e"
        let keySegment = "a\\\\b\\\"c\\$d\\`e"
        for agent in ["claude", "codex", "opencode", "hermes"] {
            let cmd = TurboSparkAgent.launchCommand(for: agent, apiKey: raw)
            XCTAssertTrue(
                cmd.contains(keySegment),
                "\(agent) command must escape every metacharacter: \(cmd)")
            // The unescaped dollar must not survive inside the quotes, where
            // the shell would read it as a substitution.
            XCTAssertFalse(cmd.contains("c$d"), "\(agent) command leaked an unescaped $")
        }
    }

}


