import XCTest
@testable import TurboSparkApp

/// "Compare with Current" in the artifact panel's history: a read-only line
/// diff of an earlier version against the file on disk now. Pure, so the
/// matching, numbering and folding are checked here rather than on screen.
final class ArtifactVersionDiffTests: XCTestCase {
    private typealias Diff = ArtifactVersionDiff

    func testSplitDropsTrailingNewlineAndCarriageReturns() {
        XCTAssertEqual(Diff.splitLines("a\r\nb\n"), ["a", "b"])
        XCTAssertEqual(Diff.splitLines("a\n\nb"), ["a", "", "b"])
        XCTAssertEqual(Diff.splitLines(""), [])
    }

    func testIdenticalTextHasNoChanges() throws {
        let lines = try XCTUnwrap(Diff.lines(old: "a\nb\n", new: "a\nb\n"))
        XCTAssertEqual(lines.map(\.kind), [.unchanged, .unchanged])
        XCTAssertFalse(Diff.hasChanges(lines))
    }

    /// A changed line is a removal then an addition, and the unchanged
    /// lines around it carry both numbers.
    func testReplacedLineIsRemovedThenAddedWithNumbers() throws {
        let lines = try XCTUnwrap(Diff.lines(old: "a\nb\nc", new: "a\nB\nc"))
        XCTAssertEqual(lines, [
            .init(kind: .unchanged, oldNumber: 1, newNumber: 1, text: "a"),
            .init(kind: .removed, oldNumber: 2, newNumber: nil, text: "b"),
            .init(kind: .added, oldNumber: nil, newNumber: 2, text: "B"),
            .init(kind: .unchanged, oldNumber: 3, newNumber: 3, text: "c"),
        ])
        XCTAssertTrue(Diff.hasChanges(lines))
    }

    func testInsertionShiftsLaterNewNumbers() throws {
        let lines = try XCTUnwrap(Diff.lines(old: "a\nc", new: "a\nb\nc"))
        XCTAssertEqual(lines.map(\.kind), [.unchanged, .added, .unchanged])
        XCTAssertEqual(lines.last?.oldNumber, 2)
        XCTAssertEqual(lines.last?.newNumber, 3)
    }

    func testFromEmptyEveryLineIsAdded() throws {
        let lines = try XCTUnwrap(Diff.lines(old: "", new: "x\ny"))
        XCTAssertEqual(lines.map(\.kind), [.added, .added])
    }

    /// Past the cap the panel says it cannot compare rather than freezing.
    func testOverLongTextIsRefused() {
        let long = Array(repeating: "x", count: Diff.maxLines + 1).joined(separator: "\n")
        XCTAssertNil(Diff.lines(old: long, new: "x"))
        XCTAssertNil(Diff.lines(old: "x", new: long))
    }

    /// Long unchanged runs fold to a count, keeping `context` lines beside
    /// each change; a run with nothing to hide is shown whole.
    func testRowsFoldLongUnchangedRunsAroundChanges() throws {
        let old = (1...20).map { "l\($0)" }.joined(separator: "\n")
        var newLines = (1...20).map { "l\($0)" }
        newLines[9] = "changed"
        let lines = try XCTUnwrap(Diff.lines(old: old, new: newLines.joined(separator: "\n")))
        let rows = Diff.rows(lines, context: 2)
        guard case .skipped(let head) = rows.first else { return XCTFail("leading run folds") }
        XCTAssertEqual(head, 7)
        guard case .skipped(let tail) = rows.last else { return XCTFail("trailing run folds") }
        XCTAssertEqual(tail, 8)
        let shown = rows.compactMap { row -> Diff.Line? in
            if case .line(let line) = row { return line }
            return nil
        }
        XCTAssertEqual(shown.map(\.text), ["l8", "l9", "l10", "changed", "l11", "l12"])
    }

    func testRowsNeverFoldASingleLine() throws {
        // Three unchanged lines between changes with context 1: hiding one
        // line would take the same row as the line itself.
        let lines = try XCTUnwrap(Diff.lines(old: "a\nx\ny\nz\nb", new: "A\nx\ny\nz\nB"))
        let rows = Diff.rows(lines, context: 1)
        XCTAssertFalse(rows.contains { if case .skipped = $0 { return true } else { return false } })
    }

    func testRowsOfIdenticalTextAreOneFold() throws {
        let lines = try XCTUnwrap(Diff.lines(old: "a\nb\nc", new: "a\nb\nc"))
        XCTAssertEqual(Diff.rows(lines, context: 3), [.skipped(3)])
    }

    /// A one-line identical file is shown whole, not folded, so "no
    /// changes" must be read from the rows' kinds, not from the fold.
    func testRowsHaveChangesReadsKindsNotFolds() throws {
        let same = try XCTUnwrap(Diff.lines(old: "a", new: "a"))
        XCTAssertEqual(Diff.rows(same), [.line(same[0])])
        XCTAssertFalse(Diff.hasChanges(Diff.rows(same)))
        let changed = try XCTUnwrap(Diff.lines(old: "a", new: "b"))
        XCTAssertTrue(Diff.hasChanges(Diff.rows(changed)))
    }
}
