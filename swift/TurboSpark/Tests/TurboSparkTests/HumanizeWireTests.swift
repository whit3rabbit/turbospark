import XCTest

@testable import TurboSpark

/// `ts_humanize` refuses unknown option keys by name, so the encoded names
/// here are the contract, and the result's `audit` object is the only place
/// the rewrite checks travel.
final class HumanizeWireTests: XCTestCase {

    private func keys(_ value: some Encodable) throws -> [String: Any] {
        let data = try JSONEncoder().encode(value)
        return try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
    }

    func testDefaultOptionsEncodeNoKeysAtAll() throws {
        // Nothing but the budget and seed exist, and absent means "the
        // library's rule", so the default must not send a key.
        XCTAssertTrue(try keys(HumanizeOptions()).isEmpty)
    }

    func testBudgetAndSeedEncodeUnderTheirWireNames() throws {
        let json = try keys(HumanizeOptions(maxNewTokens: 512, seed: 7))
        XCTAssertEqual(Set(json.keys), ["maxNewTokens", "seed"])
        XCTAssertEqual(json["maxNewTokens"] as? Int, 512)
        XCTAssertEqual(json["seed"] as? Int, 7)
    }

    private let resultJSON = """
        {"promptTokens":120,"newTokens":80,"reusedPrefixTokens":0,
         "prefillSeconds":1.5,"decodeSeconds":9.0,"stopReason":"eos",
         "toolCalls":[],"tokensPerSecond":8.9,"content":"Rewritten.",
         "reasoning":"","peakMemoryPressure":"normal",
         "audit":{"droppedNumbers":["37"],"inventedNumbers":[],
                  "languageDrift":false,"reuseRatio":0.31,"reuseFlagged":true,
                  "warnings":["numbers dropped from the draft: 37"]}}
        """

    func testResultCarriesTheAuditBesideTheTurn() throws {
        let result = try decode(GenerationResult.self, from: resultJSON)
        XCTAssertEqual(result.content, "Rewritten.")
        XCTAssertEqual(result.stopReason, .eos)
        let audit = try XCTUnwrap(result.audit)
        XCTAssertEqual(audit.droppedNumbers, ["37"])
        XCTAssertTrue(audit.reuseFlagged)
        XCTAssertEqual(audit.reuseRatio, 0.31, accuracy: 1e-9)
        XCTAssertFalse(audit.isClean)
    }

    func testAnOrdinaryTurnHasNoAudit() throws {
        let json = """
            {"promptTokens":1,"newTokens":1,"prefillSeconds":0,"decodeSeconds":0,
             "stopReason":"eos","content":"hi"}
            """
        XCTAssertNil(try decode(GenerationResult.self, from: json).audit)
    }

    func testACleanAuditIsClean() throws {
        let json = """
            {"droppedNumbers":[],"inventedNumbers":[],"languageDrift":false,
             "reuseRatio":0.1,"reuseFlagged":false,"warnings":[]}
            """
        XCTAssertTrue(try decode(HumanizeAudit.self, from: json).isClean)
    }
}
