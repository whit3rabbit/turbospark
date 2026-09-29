import Foundation
import XCTest

@testable import TurboSparkApp

final class TypeSafePlaygroundDraftTests: XCTestCase {
    func testFormBuildsTypedRequestAndRawRoundTrips() throws {
        let draft = TypeSafePlaygroundPreset.mixed.draft(model: "mock")
        let request = try draft.request()
        XCTAssertEqual(request.model, "mock")
        XCTAssertEqual(Set(request.questions.keys), ["is_urgent", "department", "frustration"])

        let raw = try TypeSafePlaygroundDraft.pretty(request)
        let decoded = try JSONDecoder().decode(type(of: request), from: Data(raw.utf8))
        let restored = try TypeSafePlaygroundDraft.from(decoded)
        XCTAssertEqual(restored.questions.count, 3)
        XCTAssertEqual(try TypeSafePlaygroundDraft.pretty(restored.request()), raw)
    }

    func testChoiceWithNullOptionAndSemanticNoneIsPreserved() throws {
        var draft = TypeSafePlaygroundPreset.triage.draft(model: "mock")
        draft.questions[0].options[0].detail = ""
        let encoded = try TypeSafePlaygroundDraft.pretty(draft.request())
        XCTAssertTrue(encoded.contains("\"__none__\""))
        XCTAssertTrue(encoded.contains("\"billing\" : null"))
    }

    func testRejectsDuplicateQuestionAndChoiceKeys() throws {
        var draft = TypeSafePlaygroundPreset.triage.draft(model: "mock")
        draft.questions.append(draft.questions[0])
        XCTAssertThrowsError(try draft.request())
        draft.questions.removeLast()
        draft.questions[0].options[1].key = "billing"
        XCTAssertThrowsError(try draft.request())
    }

    func testStateModesAndScoreValidation() throws {
        var draft = TypeSafePlaygroundPreset.structured.draft(model: "mock")
        XCTAssertNoThrow(try draft.request())
        let restored = try TypeSafePlaygroundDraft.from(draft.request())
        XCTAssertEqual(restored.stateMode, .json)
        XCTAssertEqual(try TypeSafePlaygroundDraft.pretty(restored.request()),
                       try TypeSafePlaygroundDraft.pretty(draft.request()))
        draft.state = "true"
        XCTAssertThrowsError(try draft.request())
        draft.state = "{"
        XCTAssertThrowsError(try draft.request())

        draft = TypeSafePlaygroundPreset.frustration.draft(model: "mock")
        draft.questions[0].levels = ["Only one"]
        XCTAssertThrowsError(try draft.request())
    }

    func testRawRequestWithStructuredInstructionsStaysInRawMode() throws {
        let raw = """
            {"state":{"text":"hello"},"model":"mock","questions":{"q":{"type":"noul","instructions":{"rule":"check"}}}}
            """
        let request = try JSONDecoder().decode(type(of: TypeSafePlaygroundDraft().request()),
                                               from: Data(raw.utf8))
        XCTAssertThrowsError(try TypeSafePlaygroundDraft.from(request))
    }

    func testLatencySummaryUsesNearestRankPercentiles() {
        XCTAssertNil(TypeSafeLatencySummary([]))
        let summary = TypeSafeLatencySummary([100, 10, 70, 20, 40])
        XCTAssertEqual(summary?.count, 5)
        XCTAssertEqual(summary?.p50, 40)
        XCTAssertEqual(summary?.p95, 100)
        XCTAssertEqual(summary?.mean, 48)
        XCTAssertEqual(summary?.minimum, 10)
        XCTAssertEqual(summary?.maximum, 100)
    }
}
