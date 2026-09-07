import XCTest
@testable import TurboSparkApp

/// Covers the pure display rules behind the live task checklist panel
/// (`TaskChecklistPanelView`) and the counter text it shares with the
/// per-call tool card header (`TodoChecklistSummary`).
final class TodoChecklistViewsTests: XCTestCase {
    // MARK: - TodoChecklistSummary

    func testSummaryTextMixedList() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "completed"),
            TodoItem(content: "T3", status: "in_progress"),
            TodoItem(content: "T4", status: "pending")
        ]
        XCTAssertEqual(TodoChecklistSummary.text(todos), "2/4 done (1 in progress)")
    }

    func testSummaryTextAllCompleted() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "completed")
        ]
        XCTAssertEqual(TodoChecklistSummary.text(todos), "2/2 done")
    }

    func testSummaryTextNoProgressYet() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "pending"),
            TodoItem(content: "T3", status: "pending")
        ]
        XCTAssertEqual(TodoChecklistSummary.text(todos), "1/3 done")
    }

    func testSummaryTextCountsRealInProgressNumber() {
        // Two concurrent in-progress items must read "2", not the hardcoded
        // "1" the pre-panel card header used to print.
        let todos = [
            TodoItem(content: "T1", status: "in_progress"),
            TodoItem(content: "T2", status: "in_progress"),
            TodoItem(content: "T3", status: "pending")
        ]
        XCTAssertEqual(TodoChecklistSummary.text(todos), "0/3 done (2 in progress)")
    }

    func testSummaryTextCancelledIsNotCompleted() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "cancelled")
        ]
        XCTAssertEqual(TodoChecklistSummary.text(todos), "1/2 done")
    }

    func testSummaryTextEmptyList() {
        XCTAssertEqual(TodoChecklistSummary.text([]), "0/0 done")
    }

    // MARK: - TaskChecklistPanelModel.isAllSettled

    func testAllSettledIsEmptyListFalse() {
        XCTAssertFalse(TaskChecklistPanelModel.isAllSettled([]))
    }

    func testAllSettledAllCompletedTrue() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "completed")
        ]
        XCTAssertTrue(TaskChecklistPanelModel.isAllSettled(todos))
    }

    func testAllSettledMixedCompletedAndCancelledTrue() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "cancelled")
        ]
        XCTAssertTrue(TaskChecklistPanelModel.isAllSettled(todos))
    }

    func testAllSettledOnePendingFalse() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "pending")
        ]
        XCTAssertFalse(TaskChecklistPanelModel.isAllSettled(todos))
    }

    func testAllSettledOneInProgressFalse() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "in_progress")
        ]
        XCTAssertFalse(TaskChecklistPanelModel.isAllSettled(todos))
    }

    // MARK: - TaskChecklistPanelModel.displayItems

    func testDisplayItemsUnderLimitShowsEverything() {
        let todos = [
            TodoItem(content: "T1", status: "completed"),
            TodoItem(content: "T2", status: "in_progress"),
            TodoItem(content: "T3", status: "pending")
        ]
        let display = TaskChecklistPanelModel.displayItems(todos)
        XCTAssertEqual(display.shown.map(\.id), todos.map(\.id))
        XCTAssertEqual(display.foldedCompleted, 0)
    }

    func testDisplayItemsExactlyAtLimitShowsEverything() {
        let todos = (1...TaskChecklistPanelModel.settledRowLimit).map {
            TodoItem(content: "T\($0)", status: "completed")
        }
        let display = TaskChecklistPanelModel.displayItems(todos)
        XCTAssertEqual(display.shown.count, todos.count)
        XCTAssertEqual(display.foldedCompleted, 0)
    }

    func testDisplayItemsOverLimitFoldsOldestCompletedKeepsLive() {
        let oldest = TodoItem(content: "Old 1", status: "completed")
        let secondOldest = TodoItem(content: "Old 2", status: "completed")
        var todos = [oldest, secondOldest]
        todos.append(contentsOf: (3...8).map { TodoItem(content: "T\($0)", status: "completed") })
        todos.append(TodoItem(content: "Live", status: "in_progress"))
        todos.append(TodoItem(content: "Next", status: "pending"))

        let display = TaskChecklistPanelModel.displayItems(todos)

        // 8 completed with a limit of 6 folds exactly the two oldest; the
        // live rows always survive, and the original order is preserved.
        XCTAssertEqual(display.foldedCompleted, 2)
        XCTAssertEqual(display.shown.count, todos.count - 2)
        XCTAssertFalse(display.shown.contains(where: { $0.id == oldest.id }))
        XCTAssertFalse(display.shown.contains(where: { $0.id == secondOldest.id }))
        XCTAssertTrue(display.shown.contains(where: { $0.status == "in_progress" }))
        XCTAssertTrue(display.shown.contains(where: { $0.status == "pending" }))
        XCTAssertEqual(display.shown.map(\.id), todos.filter { shown in
            shown.status != "completed" || ![oldest.id, secondOldest.id].contains(shown.id)
        }.map(\.id))
    }

    func testDisplayItemsNeverFoldsCancelled() {
        // Folding counts COMPLETED rows only; a list of cancelled rows renders
        // in full even though they are settled for the collapse predicate.
        let todos = (1...9).map { TodoItem(content: "T\($0)", status: "cancelled") }
        let display = TaskChecklistPanelModel.displayItems(todos)
        XCTAssertEqual(display.shown.count, 9)
        XCTAssertEqual(display.foldedCompleted, 0)
    }

    func testDisplayItemsRespectsExplicitLimit() {
        let todos = (1...5).map { TodoItem(content: "T\($0)", status: "completed") }
        let display = TaskChecklistPanelModel.displayItems(todos, settledLimit: 2)
        XCTAssertEqual(display.foldedCompleted, 3)
        XCTAssertEqual(display.shown.count, 2)
        XCTAssertEqual(display.shown.map(\.content), ["T4", "T5"])
    }

    // MARK: - ToolCallDiffFormatter (shared helper refactor)

    func testFormatterTodoSummaryStillReportsSlashFormat() throws {
        let json = """
        [
            {"content": "T1", "status": "completed"},
            {"content": "T2", "status": "in_progress", "activeForm": "Working"},
            {"content": "T3", "status": "pending"}
        ]
        """
        let summary = ToolCallDiffFormatter.summarize(callName: "todowrite", arguments: ["todos": json])
        XCTAssertEqual(summary.action, "Tasks")
        XCTAssertEqual(summary.target, "1/3 done (1 in progress)")
    }
}
