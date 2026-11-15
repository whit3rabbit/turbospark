import XCTest

@testable import TurboSpark

/// The quantizer's Swift surface against the real engine, with no model: the
/// type list, option encoding, error mapping, and the refusal paths. The
/// quantization itself is held to llama.cpp by the Rust tests.
final class GGUFQuantizerTests: XCTestCase {

    private func missing(_ name: String) -> String {
        FileManager.default.temporaryDirectory
            .appendingPathComponent("gguf-quantizer-\(UUID().uuidString)-\(name)").path
    }

    func testSupportedTypesDecodeAndAreConsistent() throws {
        let types = try GGUFQuantizer.supportedTypes()
        XCTAssertGreaterThanOrEqual(types.count, 20)
        XCTAssertEqual(Set(types.map(\.name)).count, types.count, "type names are unique")
        let q4k = try XCTUnwrap(types.first { $0.name == "q4_k" })
        XCTAssertEqual(q4k.blockElements, 256)
        XCTAssertFalse(q4k.requiresImatrix)
        XCTAssertGreaterThan(q4k.bitsPerWeight, 4)
        XCTAssertEqual(q4k.id, "q4_k")
        XCTAssertTrue(try XCTUnwrap(types.first { $0.name == "iq1_s" }).requiresImatrix)
    }

    func testCancelWithNothingRunningIsFalse() {
        XCTAssertFalse(GGUFQuantizer.cancel())
    }

    func testRecipeFromMissingFileThrowsInvalidArgument() {
        XCTAssertThrowsError(try GGUFQuantizer.recipe(fromGGUF: missing("no.gguf"))) { error in
            let e = error as? TurboSparkError
            XCTAssertEqual(e?.code, .invalidArgument)
            XCTAssertFalse(e?.message.isEmpty ?? true)
        }
    }

    func testCheckUpdatesOfMissingFileThrowsWithoutTouchingTheNetwork() async {
        do {
            _ = try await GGUFQuantizer.checkUpdates(provenancePath: missing("p.json"))
            XCTFail("expected a throw")
        } catch let e as TurboSparkError {
            XCTAssertEqual(e.code, .invalidArgument)
        } catch {
            XCTFail("wrong error \(error)")
        }
    }

    // MARK: - Options encoding

    private func wire(_ options: GGUFQuantizer.Options) throws -> [String: Any] {
        let data = Data(try options.wireJSON().utf8)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testBareTypeEncodesAsRecipeString() throws {
        let w = try wire(.init(source: "a", destination: "b", recipe: .type("q4_k")))
        XCTAssertEqual(w["recipe"] as? String, "q4_k")
        XCTAssertNil(w["recipeFromGguf"], "exactly one recipe key is sent")
        XCTAssertNil(w["threads"], "unset options are omitted, not null")
    }

    func testFollowGGUFEncodesAsRecipeFromGguf() throws {
        let w = try wire(
            .init(
                source: "a", destination: "b", recipe: .followGGUF(path: "ref.gguf"),
                imatrix: "im.gguf", threads: 4, iq1TieOrder: .apple, provenance: false,
                huggingFace: .init(source: "o/r:f.gguf")))
        XCTAssertEqual(w["recipeFromGguf"] as? String, "ref.gguf")
        XCTAssertNil(w["recipe"])
        XCTAssertEqual(w["imatrix"] as? String, "im.gguf")
        XCTAssertEqual(w["threads"] as? Int, 4)
        XCTAssertEqual(w["iq1TieOrder"] as? String, "apple")
        XCTAssertEqual(w["provenance"] as? Bool, false)
        XCTAssertEqual((w["hf"] as? [String: Any])?["source"] as? String, "o/r:f.gguf")
    }

    func testObjectRecipeEncodesAsObject() throws {
        let recipe: JSONValue = .object(["default": .string("q4_k")])
        let w = try wire(.init(source: "a", destination: "b", recipe: .object(recipe)))
        XCTAssertEqual((w["recipe"] as? [String: Any])?["default"] as? String, "q4_k")
    }

    // MARK: - Wire decoding

    func testReportAndFreshnessDecode() throws {
        let report = try decode(
            GGUFQuantizer.Report.self,
            from: """
                {"tensors":3,"bytesIn":100,"bytesOut":40,"byType":{"q4_k":2,"f32":1},
                 "keptQuantized":[],"imatrixUsed":2,"imatrixMissing":["blk.0.x"],
                 "notInExecutableList":["q4_1"],"provenancePath":null}
                """)
        XCTAssertEqual(report.byType["q4_k"], 2)
        XCTAssertEqual(report.notInExecutableList, ["q4_1"])
        XCTAssertNil(report.provenancePath)

        let rows = try decode(
            [GGUFQuantizer.InputFreshness].self,
            from: """
                [{"role":"source","fileName":"m.gguf","status":"sizeOnly","detail":"same size",
                  "commit":null,"needsAttention":true}]
                """)
        XCTAssertEqual(rows.first?.status, .sizeOnly)
        XCTAssertEqual(rows.first?.needsAttention, true)
    }

    // MARK: - Engine errors

    private func collect(_ stream: AsyncThrowingStream<GGUFQuantizer.Event, Error>) async
        -> (events: [GGUFQuantizer.Event], error: Error?)
    {
        var events: [GGUFQuantizer.Event] = []
        do {
            for try await e in stream { events.append(e) }
            return (events, nil)
        } catch {
            return (events, error)
        }
    }

    func testExistingDestinationIsRefusedAndLeftAlone() async throws {
        let dst = missing("dst.gguf")
        try Data("keep".utf8).write(to: URL(fileURLWithPath: dst))
        defer { try? FileManager.default.removeItem(atPath: dst) }
        let result = await collect(
            GGUFQuantizer.quantize(.init(source: missing("s.gguf"), destination: dst, recipe: .type("q4_k"))))
        XCTAssertTrue(result.events.isEmpty)
        let e = try XCTUnwrap(result.error as? TurboSparkError)
        XCTAssertEqual(e.code, .invalidArgument)
        XCTAssertTrue(e.message.contains("refusing to overwrite"), e.message)
        XCTAssertEqual(try String(contentsOfFile: dst), "keep")
    }

    func testMissingSourceThrowsAndLeavesNoDestination() async {
        let dst = missing("out.gguf")
        let result = await collect(
            GGUFQuantizer.quantize(.init(source: missing("s.gguf"), destination: dst, recipe: .type("q4_k"))))
        XCTAssertNotNil(result.error as? TurboSparkError)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dst))
        XCTAssertFalse(FileManager.default.fileExists(atPath: dst + ".partial"))
        // The run slot was released: a cancel finds nothing, and a second run
        // is not refused as busy.
        XCTAssertFalse(GGUFQuantizer.cancel())
        let again = await collect(
            GGUFQuantizer.quantize(.init(source: missing("s.gguf"), destination: dst, recipe: .type("q4_k"))))
        XCTAssertNotEqual((again.error as? TurboSparkError)?.code, .busy)
    }

    func testUnknownTypeNameIsRefusedAsAnArgumentError() async throws {
        // A bare type name the engine does not know is an argument error from
        // the recipe parser, not a crash or a silent default.
        let result = await collect(
            GGUFQuantizer.quantize(
                .init(source: missing("s.gguf"), destination: missing("d.gguf"), recipe: .type("not_a_type"))))
        let e = try XCTUnwrap(result.error as? TurboSparkError)
        XCTAssertEqual(e.code, .invalidArgument)
    }
}
