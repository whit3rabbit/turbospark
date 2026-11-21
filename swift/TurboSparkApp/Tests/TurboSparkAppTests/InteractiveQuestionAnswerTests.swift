import XCTest
@testable import TurboSparkApp

/// The "Other" free-text answer on an AskUserQuestion card (component audit,
/// question). The typed text travels in the same answers map as a picked
/// label, so the waiter and the tool result need no new shape.
final class InteractiveQuestionAnswerTests: XCTestCase {
    private typealias Answer = InteractiveQuestionAnswer

    private let options = [
        UserQuestionOption(label: "Rust", description: "r"),
        UserQuestionOption(label: "Swift", description: "s"),
    ]

    func testFreeTextIsTrimmedAndBlankIsNil() {
        XCTAssertEqual(Answer.freeText("  Go  \n"), "Go")
        XCTAssertNil(Answer.freeText(""))
        XCTAssertNil(Answer.freeText(" \n\t "))
        XCTAssertNil(Answer.freeText(nil))
    }

    func testSingleSelectFreeTextIsTheAnswer() {
        XCTAssertEqual(
            Answer.compose(selected: [], freeText: " Zig ", options: options, multiSelect: false),
            "Zig")
    }

    func testSingleSelectFixedPickIsUnchanged() {
        XCTAssertEqual(
            Answer.compose(selected: ["Swift"], freeText: nil, options: options, multiSelect: false),
            "Swift")
    }

    func testMultiSelectAddsFreeTextAfterSortedLabels() {
        XCTAssertEqual(
            Answer.compose(selected: ["Swift", "Rust"], freeText: "Zig", options: options, multiSelect: true),
            "Rust, Swift, Zig")
        XCTAssertEqual(
            Answer.compose(selected: [], freeText: "Zig", options: options, multiSelect: true),
            "Zig")
    }

    func testMultiSelectWithoutFreeTextMatchesTheOldJoin() {
        XCTAssertEqual(
            Answer.compose(selected: ["Swift", "Rust"], freeText: "   ", options: options, multiSelect: true),
            "Rust, Swift")
    }

    func testBlankEverywhereIsNoAnswer() {
        XCTAssertNil(Answer.compose(selected: [], freeText: "  ", options: options, multiSelect: false))
        XCTAssertNil(Answer.compose(selected: [], freeText: nil, options: options, multiSelect: true))
    }

    func testTextThatNamesAnOptionIsThatOption() {
        // Typing an option's label (any case) is the same answer as picking
        // it, spelled the way the model wrote it.
        XCTAssertEqual(
            Answer.compose(selected: [], freeText: " swift ", options: options, multiSelect: false),
            "Swift")
        // In a multi-select it joins the picks once, not twice.
        XCTAssertEqual(
            Answer.compose(selected: ["Swift"], freeText: "SWIFT", options: options, multiSelect: true),
            "Swift")
        XCTAssertEqual(
            Answer.compose(selected: ["Swift"], freeText: "rust", options: options, multiSelect: true),
            "Rust, Swift")
    }

    // MARK: - Through the real waiter

    @MainActor
    func testFreeTextReachesTheToolResult() async throws {
        let previous = AskUserQuestionExecutor.answerWaiter
        defer { AskUserQuestionExecutor.answerWaiter = previous }
        // AppModel's init installs the app-side waiter the registry uses.
        let model = AppModel()
        let chat = UUID()
        let questions = """
            [{"question":"Which language?","header":"Lang","multiSelect":false,\
            "options":[{"label":"Rust","description":"r"},{"label":"Swift","description":"s"}]}]
            """
        let call = Task { @MainActor in
            try await AskUserQuestionExecutor.executeAwaitingAnswer(
                arguments: ["questions": questions], chatID: chat)
        }
        var spins = 0
        while model.pendingUserQuestions == nil, spins < 10_000 {
            await Task.yield()
            spins += 1
        }
        let pending = try XCTUnwrap(model.pendingUserQuestions, "the waiter never published the card")
        await Task.yield()
        let item = try XCTUnwrap(pending.items.first)
        let typed = try XCTUnwrap(Answer.compose(
            selected: [], freeText: "  Zig, for the build tool ", options: item.options,
            multiSelect: item.multiSelect))
        model.submitUserQuestionAnswers([item.header: typed])

        let result = try await call.value
        XCTAssertTrue(result.contains("Lang: Zig, for the build tool"), result)
        XCTAssertNil(model.pendingUserQuestions)
    }
}
