import XCTest
@testable import TurboSparkApp

/// The chat activity line's status decision (`ChatActivityStatus.resolve`).
/// The order matters: a second Esc or a pending approval needs the user and
/// must outrank whatever the model is doing, and prefill with no count yet
/// must not invent one (the old pill printed "Reading 0/1").
final class ChatActivityStatusTests: XCTestCase {
    private typealias Inputs = ChatActivityStatus.Inputs

    func testPlainDecodeIsGenerating() {
        XCTAssertEqual(ChatActivityStatus.resolve(Inputs()), .generating)
    }

    func testPrefillCarriesItsCountsIncludingAnUnknownTotal() {
        var inputs = Inputs()
        inputs.isPrefill = true
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .readingPrompt(done: 0, total: 0))
        inputs.prefillDone = 1_240
        inputs.prefillTotal = 3_100
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .readingPrompt(done: 1_240, total: 3_100))
    }

    func testCancellationOutranksEverything() {
        var inputs = busyInputs()
        inputs.isCancellationPending = true
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .stopping)
    }

    func testEscArmedOutranksApprovalAndWork() {
        var inputs = busyInputs()
        inputs.isEscCancelArmed = true
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .escArmed)
    }

    func testPendingApprovalOutranksModelWork() {
        let inputs = busyInputs()
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .needsConfirmation(tool: "write_file"))
    }

    func testImageJobOutranksPrefill() {
        var inputs = Inputs()
        inputs.isPrefill = true
        inputs.imageJob = (false, "transformer", 3, 20, 0.15)
        XCTAssertEqual(
            ChatActivityStatus.resolve(inputs),
            .imageStage(stage: "transformer", completed: 3, total: 20, fraction: 0.15))
        inputs.imageJob = (true, nil, 0, 0, nil)
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .imageWaiting)
    }

    func testRunningToolOutranksTaskAndThinking() {
        var inputs = Inputs()
        inputs.runningToolName = "web_search"
        inputs.activeTask = "Checking the forecast"
        inputs.reasoning = "Search first."
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .runningTool(name: "web_search"))
    }

    func testActiveTaskOutranksThinkingButNotWhenEmpty() {
        var inputs = Inputs()
        inputs.reasoning = "Search first."
        inputs.activeTask = "Checking the forecast"
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .task("Checking the forecast"))
        inputs.activeTask = ""
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .thinking(latestLine: "Search first."))
    }

    func testThinkingEndsOnceOutputStreams() {
        var inputs = Inputs()
        inputs.reasoning = "Search first."
        inputs.hasOutput = true
        XCTAssertEqual(ChatActivityStatus.resolve(inputs), .generating)
    }

    func testLatestLineSkipsTrailingBlankLines() {
        XCTAssertEqual(
            ChatActivityStatus.latestLine(of: "First idea.\n  Second idea.  \n\n   \n"),
            "Second idea.")
        XCTAssertNil(ChatActivityStatus.latestLine(of: "\n \n"))
    }

    func testElapsedTextSwitchesToMinutes() {
        XCTAssertEqual(ChatActivityStatus.elapsedText(seconds: -3), "0s")
        XCTAssertEqual(ChatActivityStatus.elapsedText(seconds: 7), "7s")
        XCTAssertEqual(ChatActivityStatus.elapsedText(seconds: 59), "59s")
        XCTAssertEqual(ChatActivityStatus.elapsedText(seconds: 65), "1m 5s")
    }

    /// A turn that is prefilling, running a tool, working a task and
    /// thinking, with an approval pending: every lower branch is live.
    private func busyInputs() -> Inputs {
        var inputs = Inputs()
        inputs.pendingToolName = "write_file"
        inputs.isPrefill = true
        inputs.prefillTotal = 10
        inputs.runningToolName = "web_search"
        inputs.activeTask = "Checking the forecast"
        inputs.reasoning = "Search first."
        return inputs
    }
}
