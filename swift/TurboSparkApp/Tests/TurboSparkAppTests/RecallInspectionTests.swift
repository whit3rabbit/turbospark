import Foundation
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The retrieval inspector's pure half: the summary line, the score ranking,
/// the outcome the planner reports, and the bounded in-memory log.
final class RecallInspectionTests: XCTestCase {
    private typealias Planner = ChatRecallPlanner

    private func block(_ id: String, score: Double, condensed: Bool = false) -> RecallInspection.Block {
        RecallInspection.Block(
            id: id, source: "turn \(id)", breadcrumb: "", kind: "turn",
            score: score, condensed: condensed, tokens: 10)
    }

    private func retrieval(_ blocks: [RecallInspection.Block], candidates: Int = 7) -> RecallInspection.Retrieval {
        RecallInspection.Retrieval(
            blocks: blocks, candidatesConsidered: candidates, budget: 512, evidenceTokens: 120)
    }

    // MARK: Summary

    func testSummaryCountsRetrievedUsedAndTrimmedWhenInjected() {
        let inspection = RecallInspection(
            retrieval: retrieval([block("E1", score: 2), block("E2", score: 1, condensed: true)]),
            outcome: .injected)
        XCTAssertEqual(inspection.summary, .counts(retrieved: 7, used: 2, trimmed: 1))
    }

    func testSummaryNeverReportsFewerRetrievedThanUsed() {
        let inspection = RecallInspection(
            retrieval: retrieval([block("E1", score: 1), block("E2", score: 1)], candidates: 0),
            outcome: .injected)
        XCTAssertEqual(inspection.summary, .counts(retrieved: 2, used: 2, trimmed: 0))
    }

    func testDroppedInspectionSummarizesAsSkippedWithItsReason() {
        let inspection = RecallInspection(
            retrieval: retrieval([block("E1", score: 1)]), outcome: .dropped(.overBudget))
        XCTAssertEqual(inspection.summary, .skipped(.overBudget))
        XCTAssertTrue(inspection.rankedBlocks.isEmpty, "nothing was sent, so no row claims Included")
    }

    func testNilEvidenceWithoutARetrievalIsUnavailableNotNothingFound() {
        XCTAssertEqual(
            RecallInspection.make(outcome: .dropped(.nothingFound), retrieval: nil).outcome,
            .dropped(.unavailable))
        XCTAssertEqual(
            RecallInspection.make(outcome: .dropped(.nothingFound), retrieval: retrieval([])).outcome,
            .dropped(.nothingFound))
        XCTAssertEqual(
            RecallInspection.make(outcome: .dropped(.noRoom), retrieval: nil).outcome,
            .dropped(.noRoom))
    }

    // MARK: Ranking

    func testRankFollowsScoreDescendingWithPromptOrderBreakingTies() {
        let ranked = RecallInspection.rank([
            block("E1", score: 0.5), block("E2", score: 3.0), block("E3", score: 0.5), block("E4", score: 9),
        ])
        XCTAssertEqual(ranked.map(\.block.id), ["E4", "E2", "E1", "E3"])
        XCTAssertEqual(ranked.map(\.rank), [1, 2, 3, 4])
    }

    func testNonFiniteScoresRankLast() {
        let ranked = RecallInspection.rank([block("E1", score: .nan), block("E2", score: -1)])
        XCTAssertEqual(ranked.map(\.block.id), ["E2", "E1"])
    }

    // MARK: Log

    func testLogKeepsTheLatestPerMessageAndEvictsTheOldest() {
        var log = RecallInspectionLog(capacity: 2)
        let chat = UUID()
        let a = UUID(), b = UUID(), c = UUID()
        let injected = RecallInspection(retrieval: retrieval([]), outcome: .injected)
        let skipped = RecallInspection(retrieval: nil, outcome: .dropped(.noRoom))
        log.record(injected, messageID: a, chatID: chat)
        log.record(injected, messageID: b, chatID: chat)
        log.record(skipped, messageID: a, chatID: chat)
        XCTAssertEqual(log[a], skipped, "a later step of the same turn replaces the earlier one")
        log.record(injected, messageID: c, chatID: chat)
        XCTAssertNil(log[b], "b is now the oldest entry and is evicted")
        XCTAssertNotNil(log[a])
        XCTAssertNotNil(log[c])
    }

    func testLogForgetsAChat() {
        var log = RecallInspectionLog()
        let keep = UUID(), drop = UUID()
        let one = UUID(), two = UUID()
        let inspection = RecallInspection(retrieval: nil, outcome: .injected)
        log.record(inspection, messageID: one, chatID: keep)
        log.record(inspection, messageID: two, chatID: drop)
        log.removeAll(chatID: drop)
        XCTAssertNotNil(log[one])
        XCTAssertNil(log[two])
    }

    // MARK: Planner outcome

    private func projection() -> (messages: [ChatMessage], inputs: Planner.Inputs) {
        let messages = [
            ChatMessage(role: .system, content: "sys"),
            ChatMessage(role: .user, content: "q1"),
            ChatMessage(role: .assistant, content: "a1"),
            ChatMessage(role: .user, content: "q2"),
        ]
        let inputs = Planner.Inputs(
            projectionMessages: messages, sourceRows: [nil, 0, 1, 2], pinIndex: nil, boundary: 0,
            turns: [
                Planner.Turn(rowIndex: 0, role: .user, text: "q1"),
                Planner.Turn(rowIndex: 1, role: .assistant, text: "a1"),
                Planner.Turn(rowIndex: 2, role: .user, text: "q2"),
            ],
            question: messages[3], promptBudget: 4_000)
        return (messages, inputs)
    }

    /// The fit dropped q1 and a1, so recall is needed.
    private func trimmedFit(_ messages: [ChatMessage], measured: Int = 100) -> Planner.Fit {
        Planner.Fit(
            retained: [messages[0], messages[3]], measuredTokens: measured,
            removedTurnCount: 2, hasRoomForGeneration: true)
    }

    func testPlannerReportsInjected() async {
        let p = projection()
        var outcome: RecallInspection.Outcome?
        _ = await Planner.applyConversationRecall(
            fitted: trimmedFit(p.messages), inputs: p.inputs, refit: { _ in nil },
            countTokens: { _ in 200 }, evidence: { _, _ in "<evidence>x</evidence>" },
            policy: "POLICY", report: { outcome = $0 })
        XCTAssertEqual(outcome, .injected)
    }

    func testPlannerReportsOverBudgetAndNothingFoundAndNoRoom() async {
        let p = projection()
        var outcome: RecallInspection.Outcome?
        let fitted = trimmedFit(p.messages)
        let overBudget = await Planner.applyConversationRecall(
            fitted: fitted, inputs: p.inputs, refit: { _ in nil },
            countTokens: { _ in 4_000 }, evidence: { _, _ in "<evidence>x</evidence>" },
            policy: "POLICY", report: { outcome = $0 })
        XCTAssertEqual(outcome, .dropped(.overBudget))
        XCTAssertEqual(overBudget.retained, fitted.retained, "a dropped packet leaves the prompt unchanged")

        outcome = nil
        _ = await Planner.applyConversationRecall(
            fitted: fitted, inputs: p.inputs, refit: { _ in nil },
            countTokens: { _ in 200 }, evidence: { _, _ in "  " },
            policy: "POLICY", report: { outcome = $0 })
        XCTAssertEqual(outcome, .dropped(.nothingFound))

        outcome = nil
        // 3,990 of 4,000 used and the tighter refit fails.
        _ = await Planner.applyConversationRecall(
            fitted: trimmedFit(p.messages, measured: 3_990), inputs: p.inputs, refit: { _ in nil },
            countTokens: { _ in 200 },
            evidence: { _, _ in XCTFail("no evidence without room"); return nil },
            policy: "POLICY", report: { outcome = $0 })
        XCTAssertEqual(outcome, .dropped(.noRoom))
    }

    func testPlannerReportsNothingWhenRecallIsNotNeeded() async {
        let p = projection()
        var reported = false
        _ = await Planner.applyConversationRecall(
            fitted: Planner.Fit(
                retained: p.messages, measuredTokens: 100, removedTurnCount: 0,
                hasRoomForGeneration: true),
            inputs: p.inputs, refit: { _ in nil }, countTokens: { _ in 1 },
            evidence: { _, _ in nil }, policy: "POLICY", report: { _ in reported = true })
        XCTAssertFalse(reported, "every turn is in the prompt: no inspection to show")
    }
}
