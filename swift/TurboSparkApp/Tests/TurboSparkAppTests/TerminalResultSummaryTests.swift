import XCTest

@testable import TurboSparkApp

/// The finished shell card's one-line summary and three-line tail
/// (component audit, terminal). Parsed from the text `ShellCommandRunner`
/// and the registry's error wrapper produce, so these fixtures are those
/// exact shapes.
final class TerminalResultSummaryTests: XCTestCase {

    private func make(_ output: String, isError: Bool = false, seconds: Double = 1.23)
        -> TerminalResultSummary?
    {
        TerminalResultSummary.make(output: output, isError: isError, durationSeconds: seconds)
    }

    // MARK: - Outcome

    func testSuccessfulOutputIsExitZeroWithLineCountAndTail() throws {
        let summary = try XCTUnwrap(make("a\nb\nc\nd\n"))
        XCTAssertEqual(summary.outcome, .exited(0))
        XCTAssertEqual(summary.lineCount, 4)
        XCTAssertEqual(summary.tail, ["b", "c", "d"])
        XCTAssertEqual(summary.durationText, "1.2 s")
    }

    func testFailedCommandReadsTheExitCodeAndDropsTheWrapperLines() throws {
        let summary = try XCTUnwrap(
            make("Error: Command failed (exit 2):\nfirst\nboom", isError: true))
        XCTAssertEqual(summary.outcome, .exited(2))
        XCTAssertEqual(summary.lineCount, 2)
        XCTAssertEqual(summary.tail, ["first", "boom"])
    }

    func testEmptySuccessPlaceholderHasNoLines() throws {
        let summary = try XCTUnwrap(make("(Command finished: success)"))
        XCTAssertEqual(summary.outcome, .exited(0))
        XCTAssertEqual(summary.lineCount, 0)
        XCTAssertEqual(summary.tail, [])
    }

    func testBenignExitNoteIsExitOneAndNotPartOfTheTail() throws {
        let summary = try XCTUnwrap(make("(Command finished: exit code 1)\n(No matches found)"))
        XCTAssertEqual(summary.outcome, .exited(1))
        XCTAssertEqual(summary.tail, [])

        let withOutput = try XCTUnwrap(make("x\ny\n(Files differ)"))
        XCTAssertEqual(withOutput.outcome, .exited(1))
        XCTAssertEqual(withOutput.lineCount, 2)
        XCTAssertEqual(withOutput.tail, ["x", "y"])
    }

    func testCwdNoteIsStrippedAndDoesNotChangeTheExitCode() throws {
        let summary = try XCTUnwrap(
            make("out\n(Shell cwd was reset to /p: a command may not leave the project)"))
        XCTAssertEqual(summary.outcome, .exited(0))
        XCTAssertEqual(summary.tail, ["out"])
        XCTAssertEqual(summary.lineCount, 1)
    }

    func testTimeoutIsItsOwnOutcomeWithThePartialTail() throws {
        let summary = try XCTUnwrap(
            make("Error: (Command timed out after 30s and was terminated.)\nstep 1\nstep 2",
                 isError: true))
        XCTAssertEqual(summary.outcome, .timedOut)
        XCTAssertEqual(summary.tail, ["step 1", "step 2"])
    }

    // MARK: - Refusals (nil when unsure)

    func testUnrecognizedErrorsReturnNil() {
        XCTAssertNil(make("Error: 'bash' is disabled.", isError: true))
        XCTAssertNil(make("Tool execution denied: blocked", isError: true))
    }

    func testBackgroundLaunchReturnsNilBecauseNothingFinished() {
        XCTAssertNil(make(
            "Command running in background with ID: b1.\nRetrieve its output with BashOutput"))
    }

    // MARK: - Text hygiene

    func testANSIEscapesAreStripped() throws {
        let summary = try XCTUnwrap(make("\u{1B}[32mok\u{1B}[0m\n\u{1B}[1mdone\u{1B}[0m"))
        XCTAssertEqual(summary.tail, ["ok", "done"])
    }

    func testCRLFAndProgressRedrawsKeepTheFinalFrame() throws {
        let summary = try XCTUnwrap(make("one\r\n10%\r50%\r100%\r\ntwo\r\n"))
        XCTAssertEqual(summary.tail, ["one", "100%", "two"])
        XCTAssertEqual(summary.lineCount, 3)
    }

    func testTrailingAndInterleavedBlankLinesAreNotTailed() throws {
        let summary = try XCTUnwrap(make("a\n\nb\n   \nc\n\n\n"))
        XCTAssertEqual(summary.tail, ["a", "b", "c"])
        // Interior blank lines are output; trailing ones are not.
        XCTAssertEqual(summary.lineCount, 5)
    }

    func testCompactedOutputHasNoLineCountAndDropsTheSpillTrailer() throws {
        let output = "head\n... [9000 chars truncated] ...\nlast\n"
            + "[Full output (40000 chars) saved to /s/x.log. Read it with read_file.]"
        let summary = try XCTUnwrap(make(output))
        XCTAssertNil(summary.lineCount)
        XCTAssertEqual(summary.tail.last, "last")
    }

    // MARK: - Duration

    func testDurationFormatting() {
        XCTAssertEqual(TerminalResultSummary.durationText(seconds: 0.04), "0.0 s")
        XCTAssertEqual(TerminalResultSummary.durationText(seconds: 12.345), "12.3 s")
        XCTAssertEqual(TerminalResultSummary.durationText(seconds: 125), "2 m 5 s")
        XCTAssertNil(TerminalResultSummary.durationText(seconds: 0))
        XCTAssertNil(TerminalResultSummary.durationText(seconds: .nan))
        XCTAssertNil(TerminalResultSummary.durationText(seconds: -1))
    }
}
