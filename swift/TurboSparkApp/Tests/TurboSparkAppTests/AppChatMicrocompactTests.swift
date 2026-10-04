import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

final class AppChatMicrocompactTests: XCTestCase {
    private let placeholder = "[Tool output omitted from model context.]"

    private func toolResult(_ output: String, isError: Bool = false) -> ChatMessage {
        let tag = isError ? "tool_error" : "tool_response"
        return ChatMessage.tool("<\(tag)>\n\(output)\n</\(tag)>")
    }

    private func projection(
        _ messages: [ChatMessage], sourceRows: [Int?], rowCount: Int
    ) -> AppChatHistoryProjection {
        AppChatHistoryProjection(
            messages: messages,
            sourceRowIndexByMessage: sourceRows,
            instructionPinBlockIndex: nil,
            sourceTranscriptRowCount: rowCount)
    }

    private func tokenCost(_ messages: [ChatMessage]) -> Int {
        messages.reduce(0) { $0 + $1.content.utf8.count }
    }

    func testCompactsOldToolResultsAndPreservesProtectedMessagesAndProjectionMetadata() async {
        let oldOutput = String(repeating: "old-result-", count: 30)
        let history = projection(
            [
                .system("SYSTEM-SENTINEL"),
                .user("<context_summary>SUMMARY-SENTINEL</context_summary>"),
                .assistant("reading both files"),
                toolResult(oldOutput),
                toolResult(oldOutput, isError: true),
                toolResult("recent result"),
                .user("current question"),
            ],
            sourceRows: [nil, nil, 0, 0, 0, 4, 5],
            rowCount: 6)

        let outcome = await AppChatMicrocompact.apply(
            history: history,
            keepRecent: 2,
            minimumSavingsTokens: 1,
            countTokens: { tokenCost($0) })

        XCTAssertEqual(outcome.rowsCleared, 1, "Two results from one row clear one source row.")
        XCTAssertGreaterThan(outcome.estimatedTokensSaved, 0)
        XCTAssertEqual(outcome.history.sourceRowIndexByMessage, history.sourceRowIndexByMessage)
        XCTAssertEqual(outcome.history.sourceTranscriptRowCount, history.sourceTranscriptRowCount)
        XCTAssertEqual(outcome.history.messages[0], history.messages[0])
        XCTAssertEqual(outcome.history.messages[1], history.messages[1])
        XCTAssertEqual(outcome.history.messages[2], history.messages[2], "Tool-call prose is preserved.")
        XCTAssertEqual(
            outcome.history.messages[3].content,
            "<tool_response>\n\(placeholder)\n</tool_response>")
        XCTAssertEqual(
            outcome.history.messages[4].content,
            "<tool_error>\n\(placeholder)\n</tool_error>")
        XCTAssertEqual(outcome.history.messages[5], history.messages[5], "Recent results are preserved.")
        XCTAssertEqual(outcome.history.messages[6], history.messages[6], "User rows are preserved.")
        XCTAssertEqual(history.messages[3], toolResult(oldOutput), "The input projection is unchanged.")
    }

    func testEmptyTranscriptRowsStillCountTowardTheRecentWindow() async {
        let history = projection(
            [
                toolResult(String(repeating: "first-", count: 20)),
                toolResult(String(repeating: "second-", count: 20)),
                .user("retained row"),
            ],
            sourceRows: [0, 1, 2],
            rowCount: 4)

        let outcome = await AppChatMicrocompact.apply(
            history: history,
            keepRecent: 2,
            minimumSavingsTokens: 1,
            countTokens: { tokenCost($0) })

        XCTAssertEqual(outcome.rowsCleared, 2)
        XCTAssertTrue(outcome.history.messages[0].content.contains(placeholder))
        XCTAssertTrue(outcome.history.messages[1].content.contains(placeholder))
        XCTAssertEqual(outcome.history.messages[2], history.messages[2])
    }

    func testMinimumSavingsIncludesPlaceholderCostAndUsesInclusiveThreshold() async {
        let history = projection(
            [toolResult(String(repeating: "long-output-", count: 60)), .user("recent")],
            sourceRows: [0, 1],
            rowCount: 2)
        let first = await AppChatMicrocompact.apply(
            history: history,
            keepRecent: 1,
            minimumSavingsTokens: 1,
            countTokens: { tokenCost($0) })
        let measuredSavings = tokenCost(history.messages) - tokenCost(first.history.messages)

        XCTAssertEqual(first.estimatedTokensSaved, measuredSavings)
        XCTAssertGreaterThan(measuredSavings, 1)

        let atThreshold = await AppChatMicrocompact.apply(
            history: history,
            keepRecent: 1,
            minimumSavingsTokens: measuredSavings,
            countTokens: { tokenCost($0) })
        XCTAssertEqual(atThreshold.rowsCleared, 1)
        XCTAssertEqual(atThreshold.estimatedTokensSaved, measuredSavings)

        let aboveThreshold = await AppChatMicrocompact.apply(
            history: history,
            keepRecent: 1,
            minimumSavingsTokens: measuredSavings + 1,
            countTokens: { tokenCost($0) })
        XCTAssertEqual(aboveThreshold.history.messages, history.messages)
        XCTAssertEqual(aboveThreshold.rowsCleared, 0)
        XCTAssertEqual(aboveThreshold.estimatedTokensSaved, 0)
    }

    func testNewestToolResultStaysProtectedWhenKeepRecentIsZero() async {
        let history = projection(
            [toolResult(String(repeating: "old-output-", count: 20)), toolResult("new output")],
            sourceRows: [0, 1],
            rowCount: 2)

        let outcome = await AppChatMicrocompact.apply(
            history: history,
            keepRecent: 0,
            minimumSavingsTokens: 1,
            countTokens: { tokenCost($0) })

        XCTAssertEqual(outcome.rowsCleared, 1)
        XCTAssertTrue(outcome.history.messages[0].content.contains(placeholder))
        XCTAssertEqual(outcome.history.messages[1], history.messages[1])
    }

    func testCountingFailuresAndMalformedProjectionFailOpen() async {
        let history = projection(
            [toolResult(String(repeating: "output-", count: 20))],
            sourceRows: [0],
            rowCount: 2)

        for failedCall in [1, 2] {
            let counter = CountingTokenCounter(failOnCall: failedCall)
            let outcome = await AppChatMicrocompact.apply(
                history: history,
                keepRecent: 1,
                minimumSavingsTokens: 1,
                countTokens: { try counter.count($0) })
            XCTAssertEqual(counter.calls, failedCall)
            XCTAssertEqual(outcome.history.messages, history.messages)
            XCTAssertEqual(outcome.history.sourceRowIndexByMessage, history.sourceRowIndexByMessage)
            XCTAssertEqual(outcome.history.sourceTranscriptRowCount, history.sourceTranscriptRowCount)
            XCTAssertEqual(outcome.rowsCleared, 0)
            XCTAssertEqual(outcome.estimatedTokensSaved, 0)
        }

        let malformed = projection(history.messages, sourceRows: [], rowCount: 2)
        let counter = CountingTokenCounter(failOnCall: nil)
        let outcome = await AppChatMicrocompact.apply(
            history: malformed,
            keepRecent: 1,
            minimumSavingsTokens: 1,
            countTokens: { try counter.count($0) })
        XCTAssertEqual(counter.calls, 0)
        XCTAssertEqual(outcome.history.messages, malformed.messages)
        XCTAssertEqual(outcome.rowsCleared, 0)
        XCTAssertEqual(outcome.estimatedTokensSaved, 0)
    }
}

private final class CountingTokenCounter {
    private let failOnCall: Int?
    private(set) var calls = 0

    init(failOnCall: Int?) {
        self.failOnCall = failOnCall
    }

    func count(_ messages: [ChatMessage]) throws -> Int {
        calls += 1
        if calls == failOnCall {
            throw CounterError.failed
        }
        return messages.reduce(0) { $0 + $1.content.utf8.count }
    }

    private enum CounterError: Error {
        case failed
    }
}
