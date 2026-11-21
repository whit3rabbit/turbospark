import XCTest
@testable import TurboSparkApp

/// Per-change accept and reject in Humanize: the copied text is composed from
/// the draft, the rewrite and the user's choices. Nothing here changes what is
/// sent to the model.
final class HumanizeChangeReviewTests: XCTestCase {
    private func diff(_ source: String, _ rewrite: String) -> HumanizeDiff {
        HumanizeDiff.compare(source: source, rewrite: rewrite)
    }

    private func compose(_ source: String, _ rewrite: String, _ diff: HumanizeDiff, rejecting: Set<Int>) -> String? {
        HumanizeDiff.compose(source: source, rewrite: rewrite, changes: diff.changes, rejected: rejecting)
    }

    private func assertInvariants(_ source: String, _ rewrite: String, _ diff: HumanizeDiff,
                                  file: StaticString = #filePath, line: UInt = #line) {
        let all = Set(diff.changes.indices)
        XCTAssertEqual(compose(source, rewrite, diff, rejecting: []), rewrite,
                       "every change accepted is the rewrite", file: file, line: line)
        XCTAssertEqual(compose(source, rewrite, diff, rejecting: all), source,
                       "every change rejected is the draft", file: file, line: line)
    }

    private static let pairs: [(String, String)] = [
        ("Hello Maya, thanks for joining the demo on Friday.",
         "Hello Maya, thank you for joining the demo on Monday."),
        ("Same words.", "Same words."),
        ("", "New\nparagraph"),
        ("Old text", ""),
        ("one two", "one\n\n  two"),
        ("Keep  the words.", "Keep the words."),
        ("Hello, world.", "Hello world!"),
        ("It's fine.", "It\u{2019}s fine."),
        ("Cafe\u{301} at 9.", "Cafe\u{301} at 10."),
        ("\u{4ECA}\u{5929}\u{597D}", "\u{4ECA}\u{5929}\u{5F88}\u{597D}"),
        ("  Leading and trailing.\n", "Leading and trailing."),
        ("a\r\nb c", "a\nb d"),
        ("Hello Maya, thanks for joining.", "Hello Maya, thank you for coming."),
    ]

    func testEveryAcceptedIsTheRewriteAndEveryRejectedIsTheDraft() {
        for (source, rewrite) in Self.pairs {
            assertInvariants(source, rewrite, diff(source, rewrite))
        }
        XCTAssertTrue(diff("Same words.", "Same words.").changes.isEmpty)
    }

    func testRejectingOneChangeRestoresOnlyThatSpan() {
        let source = "Hello Maya, thanks for joining the demo on Friday."
        let rewrite = "Hello Maya, thank you for joining the demo on Monday."
        let result = diff(source, rewrite)
        XCTAssertEqual(result.changes.count, 2)
        XCTAssertEqual(compose(source, rewrite, result, rejecting: [0]),
                       "Hello Maya, thanks for joining the demo on Monday.")
        XCTAssertEqual(compose(source, rewrite, result, rejecting: [1]),
                       "Hello Maya, thank you for joining the demo on Friday.")
    }

    func testChangesAreTightAroundTheWordingNotTheSpacesAroundIt() {
        let source = "Hello Maya, thanks for joining the demo on Friday."
        let rewrite = "Hello Maya, thank you for joining the demo on Monday."
        let result = diff(source, rewrite)
        XCTAssertEqual(result.changes.map { (source as NSString).substring(with: $0.source) }, ["thanks", "Friday"])
        XCTAssertEqual(result.changes.map { (rewrite as NSString).substring(with: $0.rewrite) }, ["thank you", "Monday"])
    }

    func testAWhitespaceOnlyDifferenceIsItsOwnChangeSoRejectingAllIsExact() {
        let source = "Keep  the words.", rewrite = "Keep the words."
        let result = diff(source, rewrite)
        XCTAssertEqual(result.changes.count, 1)
        XCTAssertTrue(result.inserted.isEmpty, "spacing is still not highlighted")
        XCTAssertEqual(compose(source, rewrite, result, rejecting: [0]), source)
    }

    func testAdjacentChangesComposeIndependently() {
        // Two changes that touch, with no unchanged text between them.
        let changes = [
            HumanizeChange(source: NSRange(location: 0, length: 1), rewrite: NSRange(location: 0, length: 1)),
            HumanizeChange(source: NSRange(location: 1, length: 1), rewrite: NSRange(location: 1, length: 2)),
        ]
        XCTAssertEqual(HumanizeDiff.compose(source: "ab", rewrite: "xyz", changes: changes, rejected: []), "xyz")
        XCTAssertEqual(HumanizeDiff.compose(source: "ab", rewrite: "xyz", changes: changes, rejected: [0]), "ayz")
        XCTAssertEqual(HumanizeDiff.compose(source: "ab", rewrite: "xyz", changes: changes, rejected: [1]), "xb")
        XCTAssertEqual(HumanizeDiff.compose(source: "ab", rewrite: "xyz", changes: changes, rejected: [0, 1]), "ab")
    }

    func testMalformedChangesComposeToNilRatherThanAGuess() {
        let overlapping = [
            HumanizeChange(source: NSRange(location: 0, length: 2), rewrite: NSRange(location: 0, length: 2)),
            HumanizeChange(source: NSRange(location: 1, length: 1), rewrite: NSRange(location: 1, length: 1)),
        ]
        XCTAssertNil(HumanizeDiff.compose(source: "ab", rewrite: "xy", changes: overlapping, rejected: []))
        let outside = [HumanizeChange(source: NSRange(location: 0, length: 9), rewrite: NSRange(location: 0, length: 1))]
        XCTAssertNil(HumanizeDiff.compose(source: "ab", rewrite: "xy", changes: outside, rejected: [0]))
    }

    func testLongDocumentParagraphFallbackKeepsTheInvariants() {
        let shared = String(repeating: "shared word ", count: 2500)
        let source = shared + "\nOld paragraph", rewrite = shared + "\nNew paragraph"
        let result = diff(source, rewrite)
        XCTAssertEqual(result.changes.count, 1)
        assertInvariants(source, rewrite, result)
    }

    func testSegmentedRewriteKeepsTheInvariantsIncludingAStoppedTail() {
        let source = "Alpha beta.\n\nGamma delta."
        let rewrite = "Alpha BETA.\n\nGamma epsilon."
        let segments = [
            HumanizeSegment(source: NSRange(location: 0, length: 11), rewrite: NSRange(location: 0, length: 11)),
            HumanizeSegment(source: NSRange(location: 13, length: 12), rewrite: NSRange(location: 13, length: 14)),
        ]
        let full = HumanizeDiff.compare(source: source, rewrite: rewrite, segments: segments)
        XCTAssertEqual(full.changes.count, 2)
        assertInvariants(source, rewrite, full)

        // The unfinished tail is one change: rejecting it brings back the
        // draft the run never reached.
        let partial = "Alpha BETA.\n\nGam"
        let stopped = HumanizeDiff.compare(source: source, rewrite: partial, segments: [segments[0]])
        XCTAssertEqual(stopped.changes.count, 2)
        assertInvariants(source, partial, stopped)
        XCTAssertEqual(HumanizeDiff.compose(source: source, rewrite: partial, changes: stopped.changes, rejected: [0]),
                       "Alpha beta.\n\nGam")
    }

    func testDraftWhitespaceOutsideThePartsIsRestoredWhenRejected() {
        // The model's text is trimmed before it is recorded, so the draft's
        // outer whitespace is not in the rewrite.
        let source = "  Hi there.  ", rewrite = "Hello there."
        let segments = [HumanizeSegment(source: NSRange(location: 0, length: 13), rewrite: NSRange(location: 0, length: 12))]
        assertInvariants(source, rewrite, HumanizeDiff.compare(source: source, rewrite: rewrite, segments: segments))
    }
}
