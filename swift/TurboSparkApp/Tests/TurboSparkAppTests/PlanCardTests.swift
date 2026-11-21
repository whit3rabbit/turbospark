import XCTest
@testable import TurboSparkApp

/// The ExitPlanMode plan card: the step outline parsed from the plan
/// markdown (`PlanOutlineParser`) and when its Start / Request changes
/// buttons are offered (`PlanCardActions`).
final class PlanCardTests: XCTestCase {
    // MARK: - Outline parser

    func testNumberedPlanParsesIntoStepsWithDetail() throws {
        let plan = """
            # Rate limit the API

            1. Add a token bucket limiter
               60 requests a minute per key.
            2. **Wrap the routes** in middleware
            3) Return 429 with Retry-After
            """
        let outline = try XCTUnwrap(PlanOutlineParser.parse(plan))
        XCTAssertEqual(outline.intro, "# Rate limit the API")
        XCTAssertEqual(outline.steps.map(\.number), [1, 2, 3])
        XCTAssertEqual(outline.steps[0].title, "Add a token bucket limiter")
        XCTAssertEqual(outline.steps[0].detail, "60 requests a minute per key.")
        // Emphasis markers are stripped from the row title only.
        XCTAssertEqual(outline.steps[1].title, "Wrap the routes in middleware")
        XCTAssertEqual(outline.steps[2].title, "Return 429 with Retry-After")
        XCTAssertEqual(outline.steps[2].detail, "")
    }

    func testTrailingNotesStayWithTheLastStepSoNoTextIsLost() throws {
        let plan = "1. First\n2. Second\n\nRisks: none known."
        let outline = try XCTUnwrap(PlanOutlineParser.parse(plan))
        XCTAssertEqual(outline.steps.count, 2)
        XCTAssertEqual(outline.steps[1].detail, "Risks: none known.")
    }

    func testIndentedSublistIsDetailNotASeparateStep() throws {
        let plan = "1. Parent\n   1. child a\n   2. child b\n2. Next"
        let outline = try XCTUnwrap(PlanOutlineParser.parse(plan))
        XCTAssertEqual(outline.steps.map(\.title), ["Parent", "Next"])
        XCTAssertEqual(outline.steps[0].detail, "1. child a\n2. child b")
    }

    func testNumberedLinesInsideACodeFenceAreNotSteps() throws {
        let plan = "1. Run this\n```\n2. not a step\n```\n2. Then this"
        let outline = try XCTUnwrap(PlanOutlineParser.parse(plan))
        XCTAssertEqual(outline.steps.map(\.title), ["Run this", "Then this"])
        XCTAssertTrue(outline.steps[0].detail.contains("2. not a step"))
    }

    func testUnsurePlansReturnNilSoTheMarkdownFallbackIsUsed() {
        // Prose only.
        XCTAssertNil(PlanOutlineParser.parse("Refactor the parser, then add tests."))
        // A single step is not an outline.
        XCTAssertNil(PlanOutlineParser.parse("1. Only step"))
        // Numbering that does not run 1, 2, 3 is not trusted.
        XCTAssertNil(PlanOutlineParser.parse("1. a\n3. b"))
        XCTAssertNil(PlanOutlineParser.parse("2. a\n3. b"))
        // Bullets are not numbered steps.
        XCTAssertNil(PlanOutlineParser.parse("- a\n- b"))
        XCTAssertNil(PlanOutlineParser.parse("   \n"))
    }

    // MARK: - Action availability

    private func planCall(_ id: UUID = UUID()) -> AppToolCall {
        AppToolCall(id: id, name: "ExitPlanMode", arguments: ["plan": "1. a\n2. b"], status: .completed)
    }

    private func result(for call: AppToolCall, isError: Bool = false) -> AppToolResult {
        AppToolResult(callID: call.id, output: "Exited plan mode.", isError: isError)
    }

    func testOfferedWhenThePlanEndsAnIdleTurn() {
        let call = planCall()
        let messages = [
            AppChatMessage(role: .user, content: "plan it"),
            AppChatMessage(role: .assistant, content: "", toolCalls: [call], toolResults: [result(for: call)]),
            // The model's closing prose after the tool result does not hide it.
            AppChatMessage(role: .assistant, content: "Shall I start?"),
        ]
        XCTAssertTrue(PlanCardActions.isOffered(callID: call.id, in: messages, isIdle: true))
    }

    func testNotOfferedWhileTheTurnIsRunning() {
        let call = planCall()
        let messages = [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call], toolResults: [result(for: call)])
        ]
        XCTAssertFalse(PlanCardActions.isOffered(callID: call.id, in: messages, isIdle: false))
    }

    func testNotOfferedOnceTheUserReplied() {
        let call = planCall()
        let messages = [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call], toolResults: [result(for: call)]),
            AppChatMessage(role: .user, content: "Start"),
        ]
        XCTAssertFalse(PlanCardActions.isOffered(callID: call.id, in: messages, isIdle: true))
    }

    func testNotOfferedOnceTheAgentAlreadyActed() {
        let call = planCall()
        let edit = AppToolCall(name: "write_file", status: .completed)
        // A later call in the same batch.
        XCTAssertFalse(PlanCardActions.isOffered(callID: call.id, in: [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call, edit],
                toolResults: [result(for: call), result(for: edit)])
        ], isIdle: true))
        // A later step of the same turn.
        XCTAssertFalse(PlanCardActions.isOffered(callID: call.id, in: [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call], toolResults: [result(for: call)]),
            AppChatMessage(role: .assistant, content: "", toolCalls: [edit], toolResults: [result(for: edit)]),
        ], isIdle: true))
    }

    func testNotOfferedWithoutASuccessfulResultOrForAnUnknownCall() {
        let call = planCall()
        XCTAssertFalse(PlanCardActions.isOffered(callID: call.id, in: [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call])
        ], isIdle: true))
        XCTAssertFalse(PlanCardActions.isOffered(callID: call.id, in: [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call],
                toolResults: [result(for: call, isError: true)])
        ], isIdle: true))
        XCTAssertFalse(PlanCardActions.isOffered(callID: UUID(), in: [
            AppChatMessage(role: .assistant, content: "", toolCalls: [call], toolResults: [result(for: call)])
        ], isIdle: true))
    }
}
