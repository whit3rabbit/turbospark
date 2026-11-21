import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The context ring popover's action row and the context-full stop notice
/// (`ContextMeterActions`, `AppModel.compactKeepingDraft`). The buttons only
/// reach the existing `/compact` and `createChat` paths, so these pin when
/// each is refused and that a button press does not eat the composer draft
/// the way the typed `/compact` command clears its own text.
final class ContextMeterActionsTests: XCTestCase {
    func testCompactIsAvailableOnlyWithAnIdleLoadedModel() {
        XCTAssertNil(ContextMeterActions.compactBlocker(
            hasSession: true, isCompacting: false, generating: false,
            submitting: false, awaitingApproval: false))
        XCTAssertEqual(ContextMeterActions.compactBlocker(
            hasSession: false, isCompacting: false, generating: false,
            submitting: false, awaitingApproval: false), .noModel)
    }

    func testCompactIsRefusedWhileAnyTurnWorkIsInFlight() {
        // The same three conditions `handleCompactCommand` silently returns on.
        for (generating, submitting, approval) in [(true, false, false), (false, true, false), (false, false, true)] {
            XCTAssertEqual(ContextMeterActions.compactBlocker(
                hasSession: true, isCompacting: false, generating: generating,
                submitting: submitting, awaitingApproval: approval), .turnRunning)
        }
    }

    func testCompactingWinsOverTheSubmittingItRunsUnder() {
        // A manual compaction holds `submitting`; the reason shown must say
        // compaction is running, not that some turn is.
        XCTAssertEqual(ContextMeterActions.compactBlocker(
            hasSession: true, isCompacting: true, generating: false,
            submitting: true, awaitingApproval: false), .compacting)
    }

    func testNewChatIsRefusedOnlyWhileGenerating() {
        XCTAssertNil(ContextMeterActions.newChatBlocker(generating: false))
        XCTAssertEqual(ContextMeterActions.newChatBlocker(generating: true), .turnRunning)
    }

    @MainActor
    func testCompactButtonKeepsTheComposerDraftAndCompacts() async throws {
        let chat = AppChat(messages: [
            AppChatMessage(role: .user, content: "First question."),
            AppChatMessage(role: .assistant, content: "First answer."),
            AppChatMessage(role: .user, content: "Second question."),
            AppChatMessage(role: .assistant, content: "Second answer."),
            AppChatMessage(role: .user, content: "Recent question."),
        ])
        let model = AppModel()
        model.stopCronScheduler()
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.compactionKeepRecentTurns = 2
        model.compactionSummaryOverride = { _, _ in "Button summary." }
        model.compactionTokenCountOverride = { messages, _ in messages.count > 1 ? 90 : 10 }
        model.promptText = "half-typed draft"

        model.compactKeepingDraft()
        XCTAssertEqual(model.promptText, "half-typed draft")
        let task = try XCTUnwrap(model.submissionTask)
        await task.value

        XCTAssertEqual(model.compactionState(chatID: chat.id).summary, "Button summary.")
        XCTAssertEqual(model.promptText, "half-typed draft")
    }
}
