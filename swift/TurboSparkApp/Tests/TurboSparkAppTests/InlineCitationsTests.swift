import XCTest

@testable import TurboSparkApp

/// Component audit, inline-citation: `[n]` in a finished reply becomes a
/// citation link only when n names one of the turn's listed sources, and
/// never inside code, links, reference labels or tables. A marker the
/// model made up (out of range) stays plain text.
final class InlineCitationsTests: XCTestCase {
    private func numbers(_ text: String, count: Int = 3) -> [Int] {
        InlineCitations.markers(in: text, sourceCount: count).map(\.number)
    }

    func testInRangeMarkersAreFound() {
        XCTAssertEqual(numbers("Swift is fast [1] and safe [2]."), [1, 2])
    }

    func testOutOfRangeAndMalformedNumbersAreIgnored() {
        XCTAssertEqual(numbers("[4] [0] [01] [1234] [-1] [1.5]", count: 3), [])
        XCTAssertEqual(numbers("[3]", count: 3), [3])
    }

    func testNoSourcesMeansNoMarkers() {
        XCTAssertEqual(numbers("A claim [1].", count: 0), [])
        XCTAssertNil(InlineCitations.apply(to: "A claim [1].", sourceCount: 0))
    }

    func testInlineCodeSpansAreUntouched() {
        XCTAssertEqual(numbers("Use `arr[1]` here [2]."), [2])
        XCTAssertEqual(numbers("``a ` [1]`` then [3]"), [3])
    }

    func testUnclosedBacktickIsLiteral() {
        XCTAssertEqual(numbers("a ` b [1]"), [1])
    }

    func testFencedCodeIsUntouched() {
        XCTAssertEqual(numbers("```swift\nlet x = a[1]\n```\nSee [2]."), [2])
        XCTAssertEqual(numbers("~~~\n[1]\n~~~\n[3]"), [3])
        // An unclosed fence runs to the end, as markdown renders it.
        XCTAssertEqual(numbers("```\n[1]\nstill code [2]"), [])
    }

    func testIndentedCodeAfterBlankLineIsUntouched() {
        XCTAssertEqual(numbers("Text.\n\n    code [1]\n\nMore [2]."), [2])
    }

    func testMarkdownLinksAreUntouched() {
        XCTAssertEqual(numbers("[1](https://example.com) and [2]"), [2])
        XCTAssertEqual(numbers("[see [1]](https://example.com) then [3]"), [3])
        XCTAssertEqual(numbers("![1](image.png)"), [])
        XCTAssertEqual(numbers("<https://example.com/[1]> [2]"), [2])
    }

    func testDefinedReferenceLabelsAreUntouched() {
        XCTAssertEqual(numbers("Shortcut [1] here.\n\n[1]: https://example.com"), [])
        XCTAssertEqual(numbers("Full [text][1] here [2].\n\n[1]: https://example.com"), [2])
        // Undefined, `[text][2]` renders literally, so the 2 is a citation.
        XCTAssertEqual(numbers("Full [text][2] here."), [2])
    }

    func testAdjacentMarkersBothCount() {
        XCTAssertEqual(numbers("Claim [1][2]."), [1, 2])
    }

    func testCheckboxesFootnotesAndEscapesAreUntouched() {
        XCTAssertEqual(numbers("- [x] done [1]\n- [ ] todo"), [1])
        XCTAssertEqual(numbers("Note[^1] and \\[2] and [3]"), [3])
    }

    func testTableRowsAreUntouched() {
        XCTAssertEqual(numbers("| a | b [1] |\n|---|---|\n| c | d |\n\nAfter [2]."), [2])
    }

    func testApplyRewritesOnlyMarkersAndListsCitedNumbersOnce() throws {
        let result = try XCTUnwrap(
            InlineCitations.apply(to: "B [2] then `x[1]` and A [1] and B [2].", sourceCount: 2))
        XCTAssertEqual(
            result.markdown,
            "B [\\[2\\]](turbospark-cite:2) then `x[1]` and A [\\[1\\]](turbospark-cite:1) "
                + "and B [\\[2\\]](turbospark-cite:2).")
        XCTAssertEqual(result.numbers, [2, 1])
    }

    func testApplyReturnsNilWhenNothingIsInRange() {
        XCTAssertNil(InlineCitations.apply(to: "Made up [9].", sourceCount: 2))
        XCTAssertNil(InlineCitations.apply(to: "", sourceCount: 2))
    }

    func testURLRoundTripIsBoundedBySourceCount() throws {
        let url = try XCTUnwrap(URL(string: "turbospark-cite:2"))
        XCTAssertEqual(InlineCitations.number(from: url, sourceCount: 2), 2)
        XCTAssertNil(InlineCitations.number(from: url, sourceCount: 1))
        XCTAssertNil(InlineCitations.number(from: try XCTUnwrap(URL(string: "https://example.com/2")), sourceCount: 5))
        XCTAssertNil(InlineCitations.number(from: try XCTUnwrap(URL(string: "turbospark-cite:x")), sourceCount: 5))
    }
}
