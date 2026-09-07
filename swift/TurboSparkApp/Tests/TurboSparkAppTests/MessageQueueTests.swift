import XCTest

@testable import TurboSparkApp

/// The message queue: the admission predicate, the enqueue/drain cycle, and
/// the cleanup contract. The PREDICATE is the load-bearing pure piece (the
/// live combination it gates on cannot be reached without a running turn),
/// so its term matrix is spelled out here the way `canRunTerms`' is.
@MainActor
final class MessageQueueTests: XCTestCase {
    var appModel: AppModel!

    override func setUp() {
        super.setUp()
        appModel = AppModel()
    }

    override func tearDown() {
        appModel = nil
        super.tearDown()
    }

    private func makeChat() -> UUID {
        let chat = AppChat(title: "Queue")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        return chat.id
    }

    // MARK: - canQueueTerms, the pure predicate

    func testTheBusyTermsAreADisjunctionOverASessionAndAnInput() {
        // Busy by generation, by an awaited hook, or by an approval card --
        // each one alone admits the queue.
        XCTAssertTrue(AppModel.canQueueTerms(
            generating: true, submitting: false, hasPendingCall: false,
            hasSession: true, hasInput: true))
        XCTAssertTrue(AppModel.canQueueTerms(
            generating: false, submitting: true, hasPendingCall: false,
            hasSession: true, hasInput: true))
        XCTAssertTrue(AppModel.canQueueTerms(
            generating: false, submitting: false, hasPendingCall: true,
            hasSession: true, hasInput: true))
        // Nothing busy: the queue is not where an idle send goes.
        XCTAssertFalse(AppModel.canQueueTerms(
            generating: false, submitting: false, hasPendingCall: false,
            hasSession: true, hasInput: true))
        // No session: nothing to drain against (an open in flight lands
        // here too, deliberately -- nothing drains at the END of an open).
        XCTAssertFalse(AppModel.canQueueTerms(
            generating: true, submitting: false, hasPendingCall: false,
            hasSession: false, hasInput: true))
        // Nothing to queue.
        XCTAssertFalse(AppModel.canQueueTerms(
            generating: true, submitting: false, hasPendingCall: false,
            hasSession: true, hasInput: false))
    }

    func testCanRunOrQueueAgreesWithItsTwoTerms() {
        appModel.generating = false
        appModel.promptText = "hi"
        _ = makeChat()
        // No session: neither term holds.
        XCTAssertFalse(appModel.canRun)
        XCTAssertFalse(appModel.canQueue)
        XCTAssertFalse(appModel.canRunOrQueue)
    }

    // MARK: - enqueue and restore

    func testEnqueueMovesTheDraftAndAttachmentsOutOfTheComposer() {
        let chatID = makeChat()
        appModel.promptText = "  hello there  "
        appModel.chats[0].draftAttachments = []

        appModel.enqueueCurrentDraft(chatID: chatID)

        let queued = appModel.pendingUserMessages[chatID] ?? []
        XCTAssertEqual(queued.count, 1)
        XCTAssertEqual(queued.first?.text, "hello there", "Trimmed the way run() trims.")
        XCTAssertEqual(appModel.promptText, "", "The composer is free for the next draft.")
        XCTAssertEqual(appModel.chats[0].draft, "", "The row's draft went with it.")
    }

    func testEnqueueWithAnEmptyDraftIsANoOp() {
        let chatID = makeChat()
        appModel.promptText = "   "
        appModel.enqueueCurrentDraft(chatID: chatID)
        XCTAssertNil(appModel.pendingUserMessages[chatID])
    }

    func testRestorePutsTheEntryBackIntoTheDraftAndRemovesIt() {
        let chatID = makeChat()
        appModel.promptText = "first message"
        appModel.enqueueCurrentDraft(chatID: chatID)
        appModel.promptText = "second draft"
        let id = appModel.queuedMessages(for: chatID)[0].id

        appModel.restoreQueuedMessage(id: id)

        XCTAssertTrue(appModel.queuedMessages(for: chatID).isEmpty)
        // The restored text APPENDS to whatever the user has typed since,
        // rather than clobbering it.
        XCTAssertEqual(appModel.promptText, "second draft\n\nfirst message")
    }

    // MARK: - drain

    func testDrainSendsTheFirstEntryAndKeepsTheRestParked() {
        let chatID = makeChat()
        appModel.promptText = "first"
        appModel.enqueueCurrentDraft(chatID: chatID)
        appModel.promptText = "second"
        appModel.enqueueCurrentDraft(chatID: chatID)

        appModel.drainPendingUserMessagesIfIdle(chatID: chatID)

        // The first entry is back in the composer and handed to run();
        // without a session run() refuses at its guard, which this test's
        // absence of a session exercises for free. The SECOND entry waits.
        XCTAssertEqual(appModel.promptText, "first")
        XCTAssertEqual(appModel.queuedMessages(for: chatID).count, 1)
        XCTAssertEqual(appModel.queuedMessages(for: chatID).first?.text, "second")
    }

    func testDrainOfAnotherChatDoesNothing() {
        let chatID = makeChat()
        appModel.promptText = "mine"
        appModel.enqueueCurrentDraft(chatID: chatID)

        appModel.drainPendingUserMessagesIfIdle(chatID: UUID())

        XCTAssertEqual(appModel.queuedMessages(for: chatID).count, 1)
        XCTAssertEqual(appModel.promptText, "")
    }

    // MARK: - deletion

    func testDeletingTheChatDiscardsItsQueue() {
        let chatID = makeChat()
        appModel.promptText = "doomed"
        appModel.enqueueCurrentDraft(chatID: chatID)
        XCTAssertEqual(appModel.queuedMessages(for: chatID).count, 1)

        appModel.deleteChat(id: chatID)

        XCTAssertTrue(appModel.queuedMessages(for: chatID).isEmpty,
                      "A deleted chat's parked drafts are discarded with it.")
    }
}
