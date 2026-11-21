import XCTest
@testable import TurboSparkApp

/// The one-line summary on a collapsed reasoning header. It quotes the
/// trace (first heading, else first sentence); it never invents steps.
final class ReasoningSummaryTests: XCTestCase {
    func testNilForBlankTrace() {
        XCTAssertNil(ReasoningSummary.headline(of: ""))
        XCTAssertNil(ReasoningSummary.headline(of: "  \n\n\t "))
        XCTAssertNil(ReasoningSummary.headline(of: "###\n**\n"))
    }

    func testPrefersFirstHeading() {
        let trace = "Okay, let me see.\n## Checking the cache size\nThe ring is 4096."
        XCTAssertEqual(ReasoningSummary.headline(of: trace), "Checking the cache size")
    }

    func testBoldLineCountsAsHeading() {
        let trace = "**Planning the answer**\n\nFirst I need to read the file."
        XCTAssertEqual(ReasoningSummary.headline(of: trace), "Planning the answer")
    }

    func testFallsBackToFirstSentence() {
        let trace = "\n  The user wants a sorted list. I should use sort(). Then print it."
        XCTAssertEqual(ReasoningSummary.headline(of: trace), "The user wants a sorted list.")
    }

    func testDecimalPointIsNotASentenceEnd() {
        XCTAssertEqual(
            ReasoningSummary.headline(of: "Version 1.82 is the MSRV here"),
            "Version 1.82 is the MSRV here")
    }

    func testFirstLineEndsTheSentence() {
        XCTAssertEqual(
            ReasoningSummary.headline(of: "Let me think\nabout this more"),
            "Let me think")
    }

    func testStripsInlineMarkup() {
        XCTAssertEqual(
            ReasoningSummary.headline(of: "I should call `read_file` on *main.rs*."),
            "I should call read_file on main.rs.")
    }

    func testTruncatesAtWordBoundary() {
        let long = String(repeating: "word ", count: 40)
        let headline = ReasoningSummary.headline(of: long, limit: 20)
        XCTAssertEqual(headline, "word word word word\u{2026}")
        XCTAssertLessThanOrEqual(headline!.count, 21)
    }

    func testTruncatesUnbrokenRun() {
        let run = String(repeating: "x", count: 50)
        XCTAssertEqual(ReasoningSummary.headline(of: run, limit: 10), String(repeating: "x", count: 10) + "\u{2026}")
    }
}
