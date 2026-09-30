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

    func testTailDrainCarriesUserSteerPresentationAndAttachmentsIntoSubmission() {
        let chatID = makeChat()
        let attachment = AppPromptAttachment(
            fileName: "notes.txt", formatLabel: "TXT",
            extractedText: "the payload", wasTruncatedDuringExtraction: false)
        appModel.promptText = "first"
        appModel.chats[0].draftAttachments = [attachment]
        appModel.enqueueCurrentDraft(chatID: chatID)

        let submission = appModel.drainPendingUserMessagesIfIdle(chatID: chatID)

        XCTAssertEqual(submission?.text, "first")
        XCTAssertEqual(submission?.attachments, [attachment])
        XCTAssertEqual(submission?.presentation, .userSteer)
        XCTAssertTrue(appModel.queuedMessages(for: chatID).isEmpty)
        XCTAssertEqual(appModel.promptText, "first")
        XCTAssertEqual(appModel.chats[0].draftAttachments, [attachment])
    }

    func testSelectingChatDrainsQueuedUserPromptBeforeTaskNotification() {
        let currentChat = AppChat(title: "Current")
        let targetChat = AppChat(title: "Target")
        appModel.chats = [currentChat, targetChat]
        appModel.selectedChatID = currentChat.id
        appModel.pendingUserMessages[targetChat.id] = [QueuedUserPrompt(text: "user first")]
        let note = "background result"
        appModel.pendingTaskNotifications[targetChat.id] = [
            PendingTaskNotification(kind: .taskNotification, note: note)
        ]

        appModel.selectChat(id: targetChat.id)

        XCTAssertTrue(appModel.queuedMessages(for: targetChat.id).isEmpty)
        XCTAssertEqual(appModel.promptText, "user first")
        XCTAssertEqual(appModel.pendingTaskNotifications[targetChat.id]?.count, 1)
        XCTAssertTrue(appModel.chats[1].messages.isEmpty)

        appModel.promptText = ""
        appModel.drainPendingTaskNotificationsIfIdle(chatID: targetChat.id)

        XCTAssertNil(appModel.pendingTaskNotifications[targetChat.id])
        XCTAssertEqual(appModel.chats[1].messages.count, 1)
        XCTAssertEqual(appModel.chats[1].messages[0].role, .user)
        XCTAssertEqual(
            appModel.chats[1].messages[0].content,
            MidTurnInputPresentation.taskNotification.wrap(note))
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

    // MARK: - steer delivery at a step boundary

    /// The boundary drain appends MID-TURN (opencode v2's `steer` mode):
    /// every entry reaches the transcript in queue order, the queue
    /// empties, and the composer is untouched -- a steer is not a draft
    /// restore.
    func testABoundaryDeliveryAppendsEveryEntryInOrderAndEmptiesTheQueue() async {
        let chatID = makeChat()
        appModel.promptText = "first"
        appModel.enqueueCurrentDraft(chatID: chatID)
        appModel.promptText = "second"
        appModel.enqueueCurrentDraft(chatID: chatID)

        let delivered = await appModel.deliverSteersAtBoundary(
            chatID: chatID, project: nil, visionAvailable: false)

        XCTAssertTrue(delivered)
        XCTAssertTrue(appModel.queuedMessages(for: chatID).isEmpty)
        let messages = appModel.chats[0].messages
        XCTAssertEqual(messages.count, 2)
        XCTAssertEqual(messages[0].role, .user)
        XCTAssertEqual(messages[0].content, MidTurnInputPresentation.userSteer.wrap("first"))
        XCTAssertEqual(messages[0].presentationLabel, MidTurnInputPresentation.userSteer.label)
        XCTAssertEqual(messages[1].role, .user)
        XCTAssertEqual(messages[1].content, MidTurnInputPresentation.userSteer.wrap("second"))
        XCTAssertEqual(messages[1].presentationLabel, MidTurnInputPresentation.userSteer.label)
        XCTAssertEqual(appModel.promptText, "")
    }

    func testImageOnlySteerKeepsEmptyTextAndCarriesItsPresentationLabel() {
        let message = MidTurnInputPresentation.userSteer.makeUserMessage(
            "", imagePaths: ["/tmp/page.png"])

        XCTAssertEqual(message.role, .user)
        XCTAssertEqual(message.content, MidTurnInputPresentation.userSteer.wrap(""))
        XCTAssertEqual(message.imagePaths, ["/tmp/page.png"])
        XCTAssertEqual(message.presentationLabel, MidTurnInputPresentation.userSteer.label)
    }

    func testABoundaryDeliveryWithNothingQueuedReturnsFalse() async {
        let chatID = makeChat()

        let delivered = await appModel.deliverSteersAtBoundary(
            chatID: chatID, project: nil, visionAvailable: false)

        XCTAssertFalse(delivered)
        XCTAssertTrue(appModel.chats[0].messages.isEmpty)
    }

    func testABoundaryDeliveryWaitsWhileAnApprovalCardIsUp() async {
        let chatID = makeChat()
        appModel.promptText = "steered"
        appModel.enqueueCurrentDraft(chatID: chatID)
        appModel.pendingToolCall = AppToolCall(name: "run_command")

        let delivered = await appModel.deliverSteersAtBoundary(
            chatID: chatID, project: nil, visionAvailable: false)

        XCTAssertFalse(delivered, "An approval card means the loop is parked, not at a boundary.")
        XCTAssertEqual(appModel.queuedMessages(for: chatID).count, 1,
                       "The entry stays parked for the boundary the approval lands on.")
        XCTAssertTrue(appModel.chats[0].messages.isEmpty)
    }

    func testABlockedBoundarySteerIsReparkedAheadOfLaterEntries() async {
        let store = AppHookStore.shared
        store.refresh(projectDirectory: nil)
        let hook = AppHookCommand(
            name: "Block queued boundary steer",
            event: .userPromptSubmit,
            type: .command,
            command: "echo 'boundary steer blocked' >&2; exit 2",
            sourceType: .custom)
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let chatID = makeChat()
        appModel.promptText = "blocked first"
        appModel.enqueueCurrentDraft(chatID: chatID)
        appModel.promptText = "later second"
        appModel.enqueueCurrentDraft(chatID: chatID)

        let delivered = await appModel.deliverSteersAtBoundary(
            chatID: chatID, project: nil, visionAvailable: false)

        XCTAssertFalse(delivered)
        XCTAssertTrue(appModel.chats[0].messages.isEmpty)
        XCTAssertEqual(
            appModel.queuedMessages(for: chatID).map(\.text),
            ["blocked first", "later second"])
    }

    func testABoundaryDeliveryDoesNotRunAgainstACancelledTurn() async {
        let chatID = makeChat()
        appModel.promptText = "too late"
        appModel.enqueueCurrentDraft(chatID: chatID)
        appModel.isCancellationPending = true

        let delivered = await appModel.deliverSteersAtBoundary(
            chatID: chatID, project: nil, visionAvailable: false)

        XCTAssertFalse(delivered, "A steer must not outrun the Stop the user just pressed.")
        XCTAssertEqual(appModel.queuedMessages(for: chatID).count, 1)
    }

    func testASteerInlinesTextAttachmentsTheWayRunDoes() async {
        let chatID = makeChat()
        appModel.promptText = "read this"
        appModel.chats[0].draftAttachments = [
            AppPromptAttachment(
                fileName: "notes.txt", formatLabel: "TXT",
                extractedText: "the payload", wasTruncatedDuringExtraction: false)
        ]
        appModel.enqueueCurrentDraft(chatID: chatID)

        let delivered = await appModel.deliverSteersAtBoundary(
            chatID: chatID, project: nil, visionAvailable: false)

        XCTAssertTrue(delivered)
        let content = appModel.chats[0].messages[0].content
        XCTAssertTrue(content.contains("read this"))
        XCTAssertTrue(content.contains("--- Attachment: notes.txt (TXT) ---"),
                      "The queued attachment rides the steer, not just the text.")
        XCTAssertTrue(content.contains("the payload"))
        // The moved-out attachments are consumed, not left on the row.
        XCTAssertTrue(appModel.chats[0].draftAttachments.isEmpty)
    }

    func testAnImageOnlyQueuedSteerKeepsItsWrapperAndImagePath() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("queued-image-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let imageURL = directory.appendingPathComponent("page.png")
        try Data([0x89, 0x50, 0x4E, 0x47]).write(to: imageURL)

        let chatID = makeChat()
        appModel.chats[0].draftAttachments = [
            AppPromptAttachment(
                fileName: "page.png", formatLabel: "Image",
                extractedText: "", wasTruncatedDuringExtraction: false,
                sourcePath: imageURL.path)
        ]
        appModel.enqueueCurrentDraft(chatID: chatID)

        let delivered = await appModel.deliverSteersAtBoundary(
            chatID: chatID, project: nil, visionAvailable: true)

        XCTAssertTrue(delivered)
        let message = try XCTUnwrap(appModel.chats[0].messages.first)
        XCTAssertEqual(message.role, .user)
        XCTAssertEqual(message.content, MidTurnInputPresentation.userSteer.wrap(""))
        XCTAssertEqual(message.imagePaths, [imageURL.path])
        XCTAssertEqual(message.presentationLabel, MidTurnInputPresentation.userSteer.label)
    }
}
