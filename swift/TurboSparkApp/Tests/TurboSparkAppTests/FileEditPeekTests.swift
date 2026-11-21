import XCTest

@testable import TurboSparkApp

/// The collapsed file-edit card's short diff peek (component audit,
/// file-edit). Lines come from the same resolver the executor reads, so the
/// peek shows what runs, and its counts must equal the header's.
final class FileEditPeekTests: XCTestCase {

    private func lines(_ peek: FileEditPeek?) -> [String] {
        (peek?.lines ?? []).map { ($0.kind == .added ? "+" : "-") + $0.text }
    }

    // MARK: - Edit

    func testEditListsRemovedThenAddedLines() throws {
        let peek = try XCTUnwrap(
            FileEditPeek.make(.edit, arguments: [
                "path": "a.swift", "old_string": "a\nb", "new_string": "a\nc\nd",
            ]))
        XCTAssertEqual(lines(peek), ["-a", "-b", "+a", "+c", "+d"])
        XCTAssertEqual(peek.deletions, 2)
        XCTAssertEqual(peek.additions, 3)
        XCTAssertEqual(peek.hiddenLineCount, 0)
    }

    func testPureInsertionHasNoRemovedLines() throws {
        let peek = try XCTUnwrap(
            FileEditPeek.make(.edit, arguments: ["path": "a", "old_string": "", "new_string": "x\n"]))
        XCTAssertEqual(lines(peek), ["+x"])
        XCTAssertEqual(peek.deletions, 0)
        XCTAssertEqual(peek.additions, 1)
    }

    func testEditCountsMatchTheHeaderSummary() throws {
        let arguments = ["file_path": "m.rs", "old_string": "1\n2\n3\n", "new_string": "1\n2\n3\n4\n5\n6\n7"]
        let peek = try XCTUnwrap(FileEditPeek.make(.edit, arguments: arguments))
        let summary = ToolCallDiffFormatter.summarize(callName: "edit_file", arguments: arguments)
        XCTAssertEqual(peek.deletions, summary.deletions)
        XCTAssertEqual(peek.additions, summary.additions)
    }

    func testConflictingArgumentsGiveNoPeek() {
        // The call is refused; a peek of either spelling would show text
        // that never runs.
        XCTAssertNil(
            FileEditPeek.make(.edit, arguments: [
                "path": "a", "old_string": "x", "oldString": "y", "new_string": "z",
            ]))
    }

    func testEmptyEditGivesNoPeek() {
        XCTAssertNil(FileEditPeek.make(.edit, arguments: ["path": "a"]))
        XCTAssertNil(FileEditPeek.make(.edit, arguments: ["path": "a", "old_string": "", "new_string": ""]))
    }

    // MARK: - Write and the line limit

    func testWriteShowsTheFirstSixLinesAndCountsTheRest() throws {
        let content = (1...10).map { "line \($0)" }.joined(separator: "\n")
        let peek = try XCTUnwrap(FileEditPeek.make(.write, arguments: ["path": "f", "content": content]))
        XCTAssertEqual(lines(peek), (1...6).map { "+line \($0)" })
        XCTAssertEqual(peek.hiddenLineCount, 4)
        XCTAssertEqual(peek.additions, 10)
        XCTAssertEqual(peek.deletions, 0)
    }

    func testASingleHiddenLineIsShownInstead() throws {
        // "1 more line" costs the same row as the line itself.
        let content = (1...7).map(String.init).joined(separator: "\n")
        let peek = try XCTUnwrap(FileEditPeek.make(.write, arguments: ["path": "f", "content": content]))
        XCTAssertEqual(peek.lines.count, 7)
        XCTAssertEqual(peek.hiddenLineCount, 0)
    }

    func testTrailingNewlineAndCarriageReturnsAreNotLineContent() throws {
        let peek = try XCTUnwrap(FileEditPeek.make(.write, arguments: ["path": "f", "content": "a\r\nb\r\n"]))
        XCTAssertEqual(lines(peek), ["+a", "+b"])
        XCTAssertEqual(peek.additions, ToolCallDiffFormatter.countLines("a\r\nb\r\n"))
    }

    func testLargeWriteCountsEveryLine() throws {
        let content = String(repeating: "x\n", count: 100_000)
        let peek = try XCTUnwrap(FileEditPeek.make(.write, arguments: ["path": "f", "content": content]))
        XCTAssertEqual(peek.lines.count, FileEditPeek.visibleLineLimit)
        XCTAssertEqual(peek.additions, 100_000)
        XCTAssertEqual(peek.hiddenLineCount, 100_000 - FileEditPeek.visibleLineLimit)
    }

    func testEmptyWriteGivesNoPeek() {
        XCTAssertNil(FileEditPeek.make(.write, arguments: ["path": "f", "content": ""]))
        XCTAssertNil(FileEditPeek.make(.write, arguments: ["path": "f"]))
    }

    // MARK: - MultiEdit

    func testMultiEditJoinsEveryEditAndSumsTheCounts() throws {
        let edits = #"[{"file_path":"a","old_string":"x","new_string":"y\nz"},{"file_path":"b","old_string":"p\nq","new_string":"r"}]"#
        let peek = try XCTUnwrap(FileEditPeek.make(.multiEdit, arguments: ["edits": edits]))
        XCTAssertEqual(lines(peek), ["-x", "+y", "+z", "-p", "-q", "+r"])
        XCTAssertEqual(peek.additions, 3)
        XCTAssertEqual(peek.deletions, 3)
    }

    func testUnparseableMultiEditGivesNoPeek() {
        XCTAssertNil(FileEditPeek.make(.multiEdit, arguments: ["edits": "[{"]))
    }
}
