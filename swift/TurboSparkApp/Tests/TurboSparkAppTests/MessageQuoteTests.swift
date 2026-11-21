import XCTest
@testable import TurboSparkApp

/// "Quote in reply" (component audit, selection-toolbar): a message becomes
/// a markdown blockquote appended to the composer draft, never sent.
final class MessageQuoteTests: XCTestCase {
    func testBlankTextHasNoQuote() {
        XCTAssertNil(MessageQuote.blockquote(""))
        XCTAssertNil(MessageQuote.blockquote(" \n\t\n"))
    }

    func testSingleLine() {
        XCTAssertEqual(MessageQuote.blockquote("Hello there"), "> Hello there")
    }

    func testMultiLineKeepsBlankLinesAsBareMarkers() {
        XCTAssertEqual(
            MessageQuote.blockquote("First\n\nSecond\nThird"),
            "> First\n>\n> Second\n> Third")
    }

    func testCRLFIsNormalized() {
        XCTAssertEqual(MessageQuote.blockquote("a\r\nb"), "> a\n> b")
    }

    func testTrailingWhitespaceAndOuterBlankLinesAreStripped() {
        XCTAssertEqual(
            MessageQuote.blockquote("\n\n  indented   \nnext\t\n\n\n"),
            "> " + "  indented\n> next")
    }

    func testExistingQuoteLinesNest() {
        XCTAssertEqual(
            MessageQuote.blockquote("> earlier\nreply"),
            "> > earlier\n> reply")
    }

    func testCodeFenceIsQuotedLineByLine() {
        let text = "Run:\n```sh\nmake test\n```\nDone."
        XCTAssertEqual(
            MessageQuote.blockquote(text),
            "> Run:\n> ```sh\n> make test\n> ```\n> Done.")
    }

    func testCapAddsEllipsis() throws {
        let text = String(repeating: "a", count: 50)
        let quote = try XCTUnwrap(MessageQuote.blockquote(text, maxCharacters: 10))
        XCTAssertEqual(quote, "> aaaaaaaaaa\u{2026}")
    }

    func testCapAtExactLengthAddsNoEllipsis() {
        XCTAssertEqual(MessageQuote.blockquote("abcde", maxCharacters: 5), "> abcde")
    }

    func testCapInsideCodeFenceClosesTheFenceBeforeEllipsis() throws {
        let text = "Intro\n````swift\nlet a = 1\nlet b = 2\n````"
        // The cut lands inside the fence: an unclosed fence would swallow
        // the rest of the draft as code.
        let quote = try XCTUnwrap(MessageQuote.blockquote(text, maxCharacters: 20))
        XCTAssertEqual(quote, "> Intro\n> ````swift\n> let\n> ````\n> \u{2026}")
    }

    func testCapDoesNotSplitAGraphemeCluster() throws {
        let text = String(repeating: "\u{1F680}", count: 30)
        let quote = try XCTUnwrap(MessageQuote.blockquote(text, maxCharacters: 4))
        XCTAssertEqual(quote, "> " + String(repeating: "\u{1F680}", count: 4) + "\u{2026}")
    }

    func testAppendToEmptyDraft() {
        XCTAssertEqual(MessageQuote.appending(quote: "> a", to: ""), "> a\n\n")
        XCTAssertEqual(MessageQuote.appending(quote: "> a", to: "  \n"), "> a\n\n")
    }

    func testAppendKeepsExistingDraftFirst() {
        XCTAssertEqual(
            MessageQuote.appending(quote: "> a", to: "My question\n\n"),
            "My question\n\n> a\n\n")
    }

    // MARK: - AppModel

    @MainActor
    func testQuoteInReplyAppendsToDraftAndDoesNotSend() {
        let model = AppModel()
        model.writePromptTextDirectly("Draft")
        let before = model.selectedChat.messages.count

        XCTAssertTrue(model.quoteInReply("Line one\nLine two"))

        XCTAssertEqual(model.promptText, "Draft\n\n> Line one\n> Line two\n\n")
        XCTAssertEqual(model.selectedChat.messages.count, before)
        XCTAssertFalse(model.isRunning)
    }

    @MainActor
    func testQuoteInReplyIgnoresBlankText() {
        let model = AppModel()
        model.writePromptTextDirectly("Draft")
        XCTAssertFalse(model.quoteInReply("   "))
        XCTAssertEqual(model.promptText, "Draft")
    }
}
