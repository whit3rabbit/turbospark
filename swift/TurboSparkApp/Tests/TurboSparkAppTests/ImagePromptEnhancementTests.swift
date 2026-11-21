import XCTest
import TurboSpark
@testable import TurboSparkApp

/// Prompt construction, reply cleanup, gating and the suggestion state
/// machine for the Images composer's Enhance action. No model is loaded:
/// the completion is injected, as in `AppTitleGenerationTests`.
final class ImagePromptEnhancementTests: XCTestCase {
    // MARK: - Prompt construction

    func testMessagesAskForOneParagraphAndCarryTheUserPrompt() {
        let messages = ImagePromptEnhancement.messages(for: "  a red fox in snow  ")
        XCTAssertEqual(messages.map(\.role), [.system, .user])
        XCTAssertTrue(messages[0].content.localizedCaseInsensitiveContains("one paragraph"))
        XCTAssertTrue(messages[0].content.localizedCaseInsensitiveContains("only the prompt"))
        XCTAssertEqual(messages[1].content, "a red fox in snow")
    }

    func testLongInputIsTruncatedBeforePrefill() {
        let long = String(repeating: "x", count: ImagePromptEnhancement.promptCharacterLimit + 50)
        let messages = ImagePromptEnhancement.messages(for: long)
        XCTAssertEqual(messages[1].content.count, ImagePromptEnhancement.promptCharacterLimit)
    }

    func testOptionsAreBoundedPlainTextWithoutReasoning() {
        let options = ImagePromptEnhancement.options()
        XCTAssertEqual(options.reasoning, .off)
        XCTAssertEqual(options.maxNewTokens, ImagePromptEnhancement.maxNewTokens)
        XCTAssertNotNil(options.temperature)
    }

    // MARK: - Reply cleanup

    func testPlainReplyIsKept() {
        XCTAssertEqual(
            ImagePromptEnhancement.cleanedSuggestion(
                reply: "  A red fox standing in fresh snow at dawn, soft light.  ",
                original: "red fox"),
            "A red fox standing in fresh snow at dawn, soft light.")
    }

    func testThinkBlockIsRemoved() {
        XCTAssertEqual(
            ImagePromptEnhancement.cleanedSuggestion(
                reply: "<think>The user wants a fox.</think>\nA red fox in deep snow.",
                original: "fox"),
            "A red fox in deep snow.")
    }

    func testDanglingCloseTagKeepsTextAfterIt() {
        // Templates that open the think block themselves leave only the close tag.
        XCTAssertEqual(
            ImagePromptEnhancement.cleanedSuggestion(
                reply: "planning the scene...</think>A red fox in deep snow.",
                original: "fox"),
            "A red fox in deep snow.")
    }

    func testUnclosedThinkIsNil() {
        XCTAssertNil(ImagePromptEnhancement.cleanedSuggestion(
            reply: "<think>still planning the scene when tokens ran out", original: "fox"))
    }

    func testPromptLabelsAndQuotesAreStripped() {
        let cases = [
            "Prompt: A red fox in deep snow.",
            "Enhanced prompt: \"A red fox in deep snow.\"",
            "**Image Prompt:** A red fox in deep snow.",
            "\"A red fox in deep snow.\"",
            "\u{201C}A red fox in deep snow.\u{201D}",
            "`A red fox in deep snow.`",
            "```\nA red fox in deep snow.\n```",
            "Here is the improved prompt: A red fox in deep snow.",
        ]
        for reply in cases {
            XCTAssertEqual(
                ImagePromptEnhancement.cleanedSuggestion(reply: reply, original: "fox"),
                "A red fox in deep snow.", reply)
        }
    }

    func testPrefaceLineAndTrailingNotesAreDropped() {
        XCTAssertEqual(
            ImagePromptEnhancement.cleanedSuggestion(
                reply: "Sure! Here's a richer version:\n\nA red fox in deep snow.\n\nThis adds lighting and setting.",
                original: "fox"),
            "A red fox in deep snow.")
    }

    func testInnerQuotesAreNotTreatedAsWrapping() {
        XCTAssertEqual(
            ImagePromptEnhancement.cleanedSuggestion(
                reply: "\"Open\" sign over a cafe door, \"Closed\" sign beside it",
                original: "cafe signs"),
            "\"Open\" sign over a cafe door, \"Closed\" sign beside it")
    }

    func testWhitespaceIsCollapsed() {
        XCTAssertEqual(
            ImagePromptEnhancement.cleanedSuggestion(
                reply: "A red   fox\nin deep\tsnow.", original: "fox"),
            "A red fox in deep snow.")
    }

    func testEmptyReplyIsNil() {
        XCTAssertNil(ImagePromptEnhancement.cleanedSuggestion(reply: " \n\t ", original: "fox"))
        XCTAssertNil(ImagePromptEnhancement.cleanedSuggestion(reply: "Prompt: \"\"", original: "fox"))
    }

    func testEchoOfOriginalIsNil() {
        XCTAssertNil(ImagePromptEnhancement.cleanedSuggestion(
            reply: "\"A Red  Fox\"", original: "a red fox"))
    }

    func testLongReplyIsCappedAtAWordBoundary() {
        let reply = String(repeating: "snowy ", count: 400)
        let cleaned = try? XCTUnwrap(ImagePromptEnhancement.cleanedSuggestion(reply: reply, original: "fox"))
        XCTAssertNotNil(cleaned)
        XCTAssertLessThanOrEqual(cleaned!.count, ImagePromptEnhancement.suggestionCharacterLimit)
        XCTAssertTrue(cleaned!.hasSuffix("snowy"))
    }

    // MARK: - Suggest

    func testSuggestUsesInjectedCompletionAndCleansTheReply() async {
        let suggestion = await ImagePromptEnhancement.suggest(prompt: "fox") { messages, options in
            XCTAssertEqual(messages.last?.content, "fox")
            XCTAssertEqual(options.reasoning, .off)
            return "Prompt: A red fox in deep snow."
        }
        XCTAssertEqual(suggestion, "A red fox in deep snow.")
    }

    func testSuggestReturnsNilOnFailure() async {
        struct Boom: Error {}
        let suggestion = await ImagePromptEnhancement.suggest(prompt: "fox") { _, _ in throw Boom() }
        XCTAssertNil(suggestion)
    }

    // MARK: - Gate

    func testAvailabilityRequiresAnIdleLocalTextModelAndAPrompt() {
        func gate(
            prompt: String = "fox", loaded: Bool = true, busy: Bool = false,
            image: Bool = false, ghost: Bool = false
        ) -> ImagePromptEnhancement.Availability {
            ImagePromptEnhancement.availability(
                prompt: prompt, textModelLoaded: loaded, textModelBusy: busy,
                imageGenerating: image, temporaryChat: ghost)
        }
        XCTAssertEqual(gate(), .ready)
        XCTAssertEqual(gate(prompt: "  \n"), .emptyPrompt)
        XCTAssertEqual(gate(loaded: false), .noTextModel)
        XCTAssertEqual(gate(busy: true), .textModelBusy)
        XCTAssertEqual(gate(image: true), .imageGenerating)
        XCTAssertEqual(gate(ghost: true), .temporaryChat)
        // No model outranks a busy flag: loading one is the only fix.
        XCTAssertEqual(gate(loaded: false, busy: true), .noTextModel)
    }

    // MARK: - Suggestion state

    @MainActor
    func testSuggestionAcceptAndUndo() async {
        let enhancer = ImagePromptEnhancer()
        enhancer.start(prompt: "fox") { _, _ in "A red fox in deep snow." }
        await enhancer.waitForRun()
        XCTAssertEqual(enhancer.phase, .suggestion(source: "fox", text: "A red fox in deep snow."))

        XCTAssertEqual(enhancer.accept(), "A red fox in deep snow.")
        XCTAssertEqual(enhancer.phase, .idle)
        // The field now holds the accepted text; that change keeps Undo.
        enhancer.promptChanged(to: "A red fox in deep snow.")
        XCTAssertEqual(enhancer.undoPrompt, "fox")
        XCTAssertEqual(enhancer.undoAccept(), "fox")
        XCTAssertNil(enhancer.undoPrompt)
    }

    @MainActor
    func testEditingAfterAcceptDropsUndo() async {
        let enhancer = ImagePromptEnhancer()
        enhancer.start(prompt: "fox") { _, _ in "A red fox in deep snow." }
        await enhancer.waitForRun()
        _ = enhancer.accept()
        enhancer.promptChanged(to: "A red fox in deep snow.")
        enhancer.promptChanged(to: "A red fox in deep snow, at night")
        XCTAssertNil(enhancer.undoPrompt)
        XCTAssertNil(enhancer.undoAccept())
    }

    @MainActor
    func testFieldChangeCancelsRunAndDiscardsLateResult() async {
        let enhancer = ImagePromptEnhancer()
        let gate = EnhanceReplyGate()
        enhancer.start(prompt: "fox") { _, _ in
            await gate.wait()
            return "A red fox in deep snow."
        }
        XCTAssertEqual(enhancer.phase, .running(source: "fox"))
        enhancer.promptChanged(to: "fox!")
        XCTAssertEqual(enhancer.phase, .idle)
        await gate.open()
        await enhancer.waitForRun()
        XCTAssertEqual(enhancer.phase, .idle)
    }

    @MainActor
    func testUnchangedFieldKeepsSuggestion() async {
        let enhancer = ImagePromptEnhancer()
        enhancer.start(prompt: "fox") { _, _ in "A red fox in deep snow." }
        await enhancer.waitForRun()
        enhancer.promptChanged(to: "fox")
        XCTAssertEqual(enhancer.phase, .suggestion(source: "fox", text: "A red fox in deep snow."))
    }

    @MainActor
    func testDismissDropsSuggestion() async {
        let enhancer = ImagePromptEnhancer()
        enhancer.start(prompt: "fox") { _, _ in "A red fox in deep snow." }
        await enhancer.waitForRun()
        enhancer.dismiss()
        XCTAssertEqual(enhancer.phase, .idle)
        XCTAssertNil(enhancer.accept())
    }

    @MainActor
    func testUnusableReplyIsAFailureNotASuggestion() async {
        let enhancer = ImagePromptEnhancer()
        enhancer.start(prompt: "fox") { _, _ in "  " }
        await enhancer.waitForRun()
        XCTAssertEqual(enhancer.phase, .failed(source: "fox"))
        XCTAssertNil(enhancer.accept())
    }
}

/// Holds a fake completion open until the test releases it.
private actor EnhanceReplyGate {
    private var isOpen = false
    private var waiters: [CheckedContinuation<Void, Never>] = []

    func wait() async {
        if isOpen { return }
        await withCheckedContinuation { waiters.append($0) }
    }

    func open() {
        isOpen = true
        waiters.forEach { $0.resume() }
        waiters = []
    }
}
