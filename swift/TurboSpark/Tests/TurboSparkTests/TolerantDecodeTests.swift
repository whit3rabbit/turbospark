import XCTest

@testable import TurboSpark

/// A value the engine adds after this binding was built must degrade one
/// field, not discard the whole payload it arrived in. Fixtures are
/// hand-written in the shape crates/ffi/src/wire.rs serializes.
final class TolerantDecodeTests: XCTestCase {

    private func result(stopReason: String) -> String {
        """
        {"promptTokens":3,"newTokens":2,"prefillSeconds":0.1,"decodeSeconds":0.2,
         "stopReason":"\(stopReason)","tokensPerSecond":10.0,"content":"hi"}
        """
    }

    func testKnownStopReasonStillDecodes() throws {
        let r = try JSONDecoder().decode(
            GenerationResult.self, from: Data(result(stopReason: "maxTokens").utf8))
        XCTAssertEqual(r.stopReason, .maxTokens)
    }

    func testUnknownStopReasonKeepsTheTurn() throws {
        let r = try JSONDecoder().decode(
            GenerationResult.self, from: Data(result(stopReason: "somethingNew").utf8))
        XCTAssertEqual(r.stopReason, .unknown)
        // The point of the fallback: the reply and counts survive.
        XCTAssertEqual(r.content, "hi")
        XCTAssertEqual(r.newTokens, 2)
    }

    func testUnknownFitVerdictDecodesAsUnknown() throws {
        let v = try JSONDecoder().decode(
            [ModelRecommendation.FitVerdict].self,
            from: Data(#"["resident","brandNew","refused"]"#.utf8))
        XCTAssertEqual(v, [.resident, .unknown, .refused])
    }

    func testUnknownImageEventIsSkippedWithoutDroppingItsNeighbours() throws {
        let json = #"""
        [{"kind":"cancel","id":1},
         {"kind":"futureKind","id":2},
         {"kind":"cancel","id":3}]
        """#
        let rows = try JSONDecoder().decode(
            [TurboSparkServer.LossyImageEvent].self, from: Data(json.utf8))
        let events = rows.compactMap(\.event)
        XCTAssertEqual(events.count, 2)
        guard case .cancel(let first) = events[0], case .cancel(let second) = events[1] else {
            return XCTFail("expected two cancel events, got \(events)")
        }
        XCTAssertEqual([first, second], [1, 3])
    }
}
