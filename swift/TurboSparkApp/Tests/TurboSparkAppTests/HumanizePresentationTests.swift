import XCTest
@testable import TurboSparkApp

final class HumanizePresentationTests: XCTestCase {
    private func marked(_ original: String, _ rewrite: String) -> [String] {
        HumanizeDiff.compare(source: original, rewrite: rewrite).ranges.map {
            (rewrite as NSString).substring(with: $0)
        }
    }

    func testDiffPreservesUnchangedWordsAndFindsReplacementPhrases() {
        XCTAssertEqual(marked("Hello Maya, thanks for joining.", "Hello Maya, thank you for coming."),
                       ["thank you", "coming"])
        XCTAssertTrue(marked("Same words.", "Same words.").isEmpty)
        XCTAssertTrue(marked("Keep these extra words.", "Keep these words.").isEmpty)
    }

    func testWhitespaceDoesNotBecomeAChangeAndUnicodeRangesRemainValid() {
        XCTAssertTrue(marked("one two", "one\n\n  two").isEmpty)
        XCTAssertEqual(marked("Cafe\u{301} at 9.", "Cafe\u{301} at 10."), ["10"])
        XCTAssertEqual(marked("\u{4ECA}\u{5929}\u{597D}", "\u{4ECA}\u{5929}\u{5F88}\u{597D}"), ["\u{5F88}"])
        XCTAssertEqual(marked("", "New\nparagraph"), ["New", "paragraph"])
    }

    func testLongDocumentFallsBackToParagraphComparison() {
        let shared = String(repeating: "shared word ", count: 2500)
        XCTAssertEqual(marked(shared + "\nOld paragraph", shared + "\nNew paragraph"), ["New paragraph"])
    }

    func testContextAccountsForOutputAndPromptWithoutTruncating() {
        let small = HumanizeContextBudget(draftTokens: 100, outputLimit: nil)
        XCTAssertEqual(small.outputTokens, 256)
        XCTAssertEqual(small.estimatedTotal, 612)
        let boundary = HumanizeContextBudget(draftTokens: 5888, outputLimit: nil)
        XCTAssertTrue(boundary.fits)
        XCTAssertEqual(boundary.estimatedTotal, 8192)
        XCTAssertFalse(HumanizeContextBudget(draftTokens: 5889, outputLimit: nil).fits)
        XCTAssertEqual(HumanizeContextBudget(draftTokens: 6000, outputLimit: 4096).fraction, 1)
    }
}
