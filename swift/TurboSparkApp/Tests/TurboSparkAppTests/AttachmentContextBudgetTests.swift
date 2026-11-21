import XCTest

@testable import TurboSparkApp

/// The attachment tray's per-chip estimate and its context-budget reading.
final class AttachmentContextBudgetTests: XCTestCase {

    private func document(_ text: String, name: String = "notes.txt") -> AppPromptAttachment {
        AppPromptAttachment(
            fileName: name, formatLabel: "Text", extractedText: text,
            wasTruncatedDuringExtraction: false)
    }

    // MARK: Per-chip estimate

    func testDocumentEstimateIsCharactersOverFourRoundedUp() {
        XCTAssertEqual(AttachmentContextBudget.tokens(for: document(String(repeating: "a", count: 8))), 2)
        XCTAssertEqual(AttachmentContextBudget.tokens(for: document(String(repeating: "a", count: 9))), 3)
        XCTAssertEqual(AttachmentContextBudget.tokens(for: document("a")), 1)
    }

    /// The app knows no vision token cost, so a picture shows no figure
    /// rather than a guessed one (or a zero that reads as free).
    func testImageHasNoEstimate() {
        XCTAssertNil(AttachmentContextBudget.tokens(for: document("", name: "photo.png")))
    }

    func testEmptyDocumentHasNoEstimate() {
        XCTAssertNil(AttachmentContextBudget.tokens(for: document("")))
    }

    func testTotalSkipsImagesAndIsNilWithNothingToCount() {
        let attachments = [
            document(String(repeating: "a", count: 40)),
            document("", name: "photo.jpg"),
            document(String(repeating: "b", count: 4)),
        ]
        XCTAssertEqual(AttachmentContextBudget.totalTokens(attachments), 11)
        XCTAssertNil(AttachmentContextBudget.totalTokens([document("", name: "photo.jpg")]))
        XCTAssertNil(AttachmentContextBudget.totalTokens([]))
    }

    // MARK: Budget reading

    func testFitsWellUnderTheLimit() {
        let reading = AttachmentContextBudget.reading(
            attachmentTokens: 1_000, otherPromptTokens: 1_000,
            windowTokens: 10_000, reservedForReplyTokens: 0)
        XCTAssertEqual(reading?.level, .fits)
        XCTAssertEqual(reading?.attachmentPercent, 10)
        XCTAssertEqual(reading?.totalPercent, 20)
    }

    /// The reply reserve is not room the prompt can use, the same budget
    /// the importer warns against.
    func testReserveShrinksTheAvailableWindow() {
        let reading = AttachmentContextBudget.reading(
            attachmentTokens: 4_000, otherPromptTokens: 0,
            windowTokens: 10_000, reservedForReplyTokens: 5_000)
        XCTAssertEqual(reading?.attachmentPercent, 80)
        XCTAssertEqual(reading?.level, .fits)
    }

    func testNearLimitStartsAtEightyFivePercent() {
        let below = AttachmentContextBudget.reading(
            attachmentTokens: 8_499, otherPromptTokens: 0,
            windowTokens: 10_000, reservedForReplyTokens: 0)
        XCTAssertEqual(below?.level, .fits)
        let at = AttachmentContextBudget.reading(
            attachmentTokens: 8_000, otherPromptTokens: 500,
            windowTokens: 10_000, reservedForReplyTokens: 0)
        XCTAssertEqual(at?.level, .nearLimit)
    }

    func testExactlyFullIsNearAndOneMoreIsOver() {
        let full = AttachmentContextBudget.reading(
            attachmentTokens: 6_000, otherPromptTokens: 2_000,
            windowTokens: 10_000, reservedForReplyTokens: 2_000)
        XCTAssertEqual(full?.level, .nearLimit)
        let over = AttachmentContextBudget.reading(
            attachmentTokens: 6_001, otherPromptTokens: 2_000,
            windowTokens: 10_000, reservedForReplyTokens: 2_000)
        XCTAssertEqual(over?.level, .overLimit)
    }

    func testNilWhenThereIsNothingToJudge() {
        XCTAssertNil(AttachmentContextBudget.reading(
            attachmentTokens: 0, otherPromptTokens: 10,
            windowTokens: 10_000, reservedForReplyTokens: 0))
        XCTAssertNil(AttachmentContextBudget.reading(
            attachmentTokens: 10, otherPromptTokens: 0,
            windowTokens: 1_000, reservedForReplyTokens: 1_000))
        XCTAssertNil(AttachmentContextBudget.reading(
            attachmentTokens: 10, otherPromptTokens: -1,
            windowTokens: 1_000, reservedForReplyTokens: 0))
    }
}
