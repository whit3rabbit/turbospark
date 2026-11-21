import XCTest
@testable import TurboSparkApp

/// The streaming holdback trims only a dangling markdown marker at the very
/// end of the live text, never real text and never a closing marker.
final class StreamingMarkdownTailTests: XCTestCase {
    private func visible(_ text: String) -> String { StreamingMarkdownTail.visible(text) }

    func testHoldsDanglingOpeners() {
        XCTAssertEqual(visible("Hello **"), "Hello ")
        XCTAssertEqual(visible("Hello *"), "Hello ")
        XCTAssertEqual(visible("Hello _"), "Hello ")
        XCTAssertEqual(visible("Hello __"), "Hello ")
        XCTAssertEqual(visible("use `"), "use ")
        XCTAssertEqual(visible("Example:\n```"), "Example:\n")
        XCTAssertEqual(visible("see ["), "see ")
        XCTAssertEqual(visible("**"), "")
        XCTAssertEqual(visible("word**"), "word")
        XCTAssertEqual(visible("snake_"), "snake")
    }

    func testKeepsClosingMarkers() {
        XCTAssertEqual(visible("a **bold**"), "a **bold**")
        XCTAssertEqual(visible("an *em*"), "an *em*")
        XCTAssertEqual(visible("call `f()`"), "call `f()`")
        XCTAssertEqual(visible("_x_"), "_x_")
    }

    func testNeverHoldsRealText() {
        XCTAssertEqual(visible("Hello **wor"), "Hello **wor")
        XCTAssertEqual(visible("see [the link"), "see [the link")
        XCTAssertEqual(visible("plain text."), "plain text.")
        XCTAssertEqual(visible(""), "")
        XCTAssertEqual(visible("done\n"), "done\n")
    }

    func testOnlyTheCurrentLineCounts() {
        // An opener on an earlier line does not make this run a closer.
        XCTAssertEqual(visible("**a**\nnext**"), "**a**\nnext")
        XCTAssertEqual(visible("`a`\nuse `"), "`a`\nuse ")
    }

    func testLongLineScanIsBounded() {
        // A closer far beyond the scan window reads as no opener: the run is
        // held for one token at most, which is the safe direction.
        let long = "**" + String(repeating: "x", count: 5_000) + "**"
        XCTAssertEqual(visible(long), String(long.dropLast(2)))
    }
}
