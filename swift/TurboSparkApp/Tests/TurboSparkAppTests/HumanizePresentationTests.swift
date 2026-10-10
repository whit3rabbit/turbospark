import XCTest
@testable import TurboSparkApp

final class HumanizePresentationTests: XCTestCase {
    private func marked(_ original: String, _ rewrite: String) -> [String] {
        HumanizeDiff.compare(source: original, rewrite: rewrite).inserted.map {
            (rewrite as NSString).substring(with: $0)
        }
    }

    private func cut(_ original: String, _ rewrite: String) -> [String] {
        HumanizeDiff.compare(source: original, rewrite: rewrite).deleted.map {
            (original as NSString).substring(with: $0)
        }
    }

    func testDiffMarksNewWordingInTheRewriteAndCutWordingInTheDraft() {
        let draft = "Hello Maya, thanks for joining the demo on Friday."
        let rewrite = "Hello Maya, thank you for joining the demo on Monday."
        XCTAssertEqual(marked(draft, rewrite), ["thank you", "Monday"])
        XCTAssertEqual(cut(draft, rewrite), ["thanks", "Friday"])
        XCTAssertTrue(marked("Same words.", "Same words.").isEmpty)
        XCTAssertTrue(marked("Keep these extra words.", "Keep these words.").isEmpty)
        XCTAssertEqual(cut("Keep these extra words.", "Keep these words."), ["extra"])
    }

    func testAShortRunBetweenTwoChangesIsAbsorbedInsteadOfShreddingTheHighlight() {
        // "for" is shorter than half of the change on each side, so it reads
        // as part of one rewritten phrase rather than a lone survivor.
        XCTAssertEqual(marked("Hello Maya, thanks for joining.", "Hello Maya, thank you for coming."),
                       ["thank you for coming"])
        XCTAssertEqual(cut("Hello Maya, thanks for joining.", "Hello Maya, thank you for coming."),
                       ["thanks for joining"])
    }

    func testPunctuationOnlyAndQuoteStyleChangesAreNotMarked() {
        let diff = HumanizeDiff.compare(source: "Hello, world.", rewrite: "Hello world!")
        XCTAssertTrue(diff.inserted.isEmpty)
        XCTAssertTrue(diff.deleted.isEmpty)
        XCTAssertTrue(marked("It's fine.", "It\u{2019}s fine.").isEmpty)
    }

    func testWhitespaceDoesNotBecomeAChangeAndUnicodeRangesRemainValid() {
        XCTAssertTrue(marked("one two", "one\n\n  two").isEmpty)
        XCTAssertEqual(marked("Cafe\u{301} at 9.", "Cafe\u{301} at 10."), ["10"])
        XCTAssertEqual(marked("\u{4ECA}\u{5929}\u{597D}", "\u{4ECA}\u{5929}\u{5F88}\u{597D}"), ["\u{5F88}"])
        XCTAssertEqual(marked("", "New\nparagraph"), ["New", "paragraph"])
    }

    func testChangedShareIsTheWeightOfNewWordingInTheRewrite() {
        XCTAssertEqual(HumanizeDiff.compare(source: "Same words.", rewrite: "Same words.").changedShare, 0)
        XCTAssertEqual(HumanizeDiff.compare(source: "abc", rewrite: "xyz").changedShare, 1)
        XCTAssertEqual(HumanizeDiff.compare(source: "Hello there", rewrite: "Hello world").changedShare, 0.5)
    }

    func testLongDocumentFallsBackToParagraphComparison() {
        let shared = String(repeating: "shared word ", count: 2500)
        XCTAssertEqual(marked(shared + "\nOld paragraph", shared + "\nNew paragraph"), ["New paragraph"])
    }

    func testSegmentsCompareEachPartWithItsOwnDraftAndOffsetTheRanges() {
        let source = "Alpha beta.\n\nGamma delta."
        let rewrite = "Alpha BETA.\n\nGamma epsilon."
        let segments = [
            HumanizeSegment(source: NSRange(location: 0, length: 11), rewrite: NSRange(location: 0, length: 11)),
            HumanizeSegment(source: NSRange(location: 13, length: 12), rewrite: NSRange(location: 13, length: 14)),
        ]
        let diff = HumanizeDiff.compare(source: source, rewrite: rewrite, segments: segments)
        XCTAssertEqual(diff.inserted.map { (rewrite as NSString).substring(with: $0) }, ["BETA", "epsilon"])
        XCTAssertEqual(diff.deleted.map { (source as NSString).substring(with: $0) }, ["beta", "delta"])

        // A stopped run has no segment for its unfinished part: nothing there
        // is marked, rather than the whole unreached draft reading as cut.
        let stopped = HumanizeDiff.compare(source: source, rewrite: "Alpha BETA.\n\nGam",
                                           segments: [segments[0]])
        XCTAssertEqual(stopped.inserted.count, 1)
        XCTAssertEqual(stopped.deleted.map { (source as NSString).substring(with: $0) }, ["beta"])
    }

    // MARK: - Planning long drafts

    private func plan(_ draft: String, tokens: @escaping (String) -> Int = { $0.count }) async throws -> HumanizePlan {
        try await HumanizePlanner.plan(draft) { tokens($0) }
    }

    func testAShortDraftIsOnePartSentAsTyped() async throws {
        let plan = try await plan("  Hi there.  ")
        XCTAssertEqual(plan.parts, [HumanizePart(range: NSRange(location: 0, length: 13), tokens: 13)])
        XCTAssertTrue(plan.fits)
        let blank = try await self.plan(" \n ")
        XCTAssertTrue(blank.parts.isEmpty)
    }

    func testALongDraftSplitsAtParagraphBreaksAndJoinsNeighborsThatFit() async throws {
        let paragraphs = ["a", "b", "c"].map { String(repeating: $0, count: 300) }
        let draft = paragraphs.joined(separator: "\n\n")
        let plan = try await plan(draft)
        let text = draft as NSString
        XCTAssertEqual(plan.parts.map { text.substring(with: $0.range) },
                       [paragraphs[0] + "\n\n" + paragraphs[1], paragraphs[2]])
        XCTAssertEqual(plan.parts.map(\.tokens), [600, 300])
        XCTAssertTrue(plan.parts.allSatisfy { $0.tokens <= HumanizePlanner.partTokens })
    }

    func testAParagraphTooLongForOnePartSplitsAtSentenceEnds() async throws {
        let sentence = "This sentence is one of many, padded out with enough words to matter."
        let paragraph = Array(repeating: sentence, count: 20).joined(separator: " ")
        XCTAssertGreaterThan(paragraph.count, HumanizePlanner.partTokens)
        let plan = try await plan(paragraph)
        let text = paragraph as NSString
        let pieces = plan.parts.map { text.substring(with: $0.range) }
        XCTAssertGreaterThan(pieces.count, 1)
        XCTAssertEqual(pieces.joined(separator: " "), paragraph, "parts cover the paragraph in order")
        XCTAssertTrue(pieces.allSatisfy { $0.hasSuffix(".") }, "cuts land on sentence ends")
        XCTAssertTrue(plan.parts.allSatisfy { $0.tokens <= HumanizePlanner.partTokens })
    }

    func testOnlyAPassageTooLongForTheWindowFailsToFit() async throws {
        XCTAssertEqual(HumanizePlanner.outputTokens(for: 100), 256)
        XCTAssertEqual(HumanizePlanner.outputTokens(for: 800), 2000)
        XCTAssertEqual(HumanizePlanner.outputTokens(for: 1000), 2048)
        XCTAssertTrue(HumanizePlanner.fits(HumanizePart(range: NSRange(), tokens: 5888)))
        XCTAssertFalse(HumanizePlanner.fits(HumanizePart(range: NSRange(), tokens: 5889)))

        let huge = try await plan("one unbroken passage", tokens: { _ in 6000 })
        XCTAssertFalse(huge.fits)
        let long = try await plan(Array(repeating: String(repeating: "x", count: 700), count: 12)
            .joined(separator: "\n\n"))
        XCTAssertEqual(long.parts.count, 12)
        XCTAssertTrue(long.fits, "a long draft fits because each part does")
    }
}
