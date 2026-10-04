import XCTest
@testable import TurboSparkApp

@MainActor
final class AppTitleGenerationLifecycleTests: XCTestCase {
    private func makeModel(chats: [AppChat] = []) -> AppModel {
        AppChatFileStore.save(.empty())
        let model = AppModel()
        model.stopCronScheduler()
        model.chats = chats
        if !chats.isEmpty {
            model.persistChats()
        }
        return model
    }

    func testOnlyFirstNonGhostUserRowClaimsAnAttempt() {
        let existing = AppChat(
            title: "New Chat",
            messages: [AppChatMessage(role: .user, content: "Earlier prompt")],
            titleProvenance: .unclaimed,
            titleGenerationAttempted: false)
        let fresh = AppChat(
            title: "New Chat",
            titleProvenance: .unclaimed,
            titleGenerationAttempted: false)
        let model = makeModel(chats: [existing, fresh])

        model.recordTitleGenerationAttempt(chatID: existing.id, firstUserMessage: "Later prompt")
        model.recordTitleGenerationAttempt(chatID: fresh.id, firstUserMessage: "First prompt")
        model.recordTitleGenerationAttempt(chatID: fresh.id, firstUserMessage: "Duplicate prompt")

        XCTAssertFalse(model.chats[0].titleGenerationAttempted)
        XCTAssertTrue(model.chats[1].titleGenerationAttempted)
        XCTAssertEqual(model.pendingTitleGenerations.map(\.chatID), [fresh.id])
        XCTAssertEqual(model.pendingTitleGenerations.first?.firstUserMessage, "First prompt")
    }

    func testAttemptMarkerSurvivesFailureAndRelaunchWithoutRetry() async {
        let chat = AppChat(
            title: "New Chat",
            titleProvenance: .unclaimed,
            titleGenerationAttempted: false)
        let model = makeModel(chats: [chat])
        model.titleGenerationEnqueueOverride = { _ in nil }
        model.recordTitleGenerationAttempt(chatID: chat.id, firstUserMessage: "First prompt")
        model.mutateTurnMessages(for: chat.id) {
            $0.append(AppChatMessage(role: .user, content: "First prompt"))
        }

        model.drainPendingTitleGenerationIfIdle()
        let titleTask = model.titleGenerationTask
        await titleTask?.value

        XCTAssertEqual(model.chats[0].title, "New Chat")
        XCTAssertEqual(model.chats[0].titleProvenance, .unclaimed)
        XCTAssertTrue(model.chats[0].titleGenerationAttempted)
        XCTAssertEqual(AppChatFileStore.load().chats.first?.titleGenerationAttempted, true)

        let relaunched = AppModel()
        relaunched.stopCronScheduler()
        relaunched.recordTitleGenerationAttempt(
            chatID: chat.id, firstUserMessage: "Do not retry")

        XCTAssertTrue(relaunched.pendingTitleGenerations.isEmpty)
        XCTAssertTrue(relaunched.chats.first?.titleGenerationAttempted == true)
    }

    func testTitleWorkWaitsForActiveTurnWithoutBlockingCaller() async {
        let chat = AppChat(
            title: "New Chat",
            titleProvenance: .unclaimed,
            titleGenerationAttempted: false)
        let model = makeModel(chats: [chat])
        var enqueuedMessages: [String] = []
        model.titleGenerationEnqueueOverride = { message in
            enqueuedMessages.append(message)
            try? await Task.sleep(nanoseconds: 50_000_000)
            return nil
        }
        model.recordTitleGenerationAttempt(chatID: chat.id, firstUserMessage: "First prompt")

        model.generating = true
        model.drainPendingTitleGenerationIfIdle()
        XCTAssertNil(model.titleGenerationTask)
        XCTAssertTrue(enqueuedMessages.isEmpty)

        model.generating = false
        model.drainPendingTitleGenerationIfIdle()

        XCTAssertNotNil(model.titleGenerationTask)
        XCTAssertEqual(model.chats[0].title, "New Chat")
        let titleTask = model.titleGenerationTask
        await titleTask?.value
        XCTAssertEqual(enqueuedMessages, ["First prompt"])
    }

    func testUnrelatedParkedQueuesDoNotStarveTitleWork() async {
        let titleChat = AppChat(
            title: "New Chat",
            titleProvenance: .unclaimed,
            titleGenerationAttempted: false)
        let unrelatedChat = AppChat(title: "Unrelated")
        let model = makeModel(chats: [titleChat, unrelatedChat])
        var enqueuedMessages: [String] = []
        model.titleGenerationEnqueueOverride = { message in
            enqueuedMessages.append(message)
            return "Generated title"
        }
        model.recordTitleGenerationAttempt(
            chatID: titleChat.id, firstUserMessage: "First prompt")
        model.mutateTurnMessages(for: titleChat.id) {
            $0.append(AppChatMessage(role: .user, content: "First prompt"))
        }
        model.pendingUserMessages[unrelatedChat.id] = [
            QueuedUserPrompt(text: "Parked user prompt")
        ]
        model.pendingTaskNotifications[unrelatedChat.id] = [
            PendingTaskNotification(kind: .taskNotification, note: "Parked task result")
        ]

        model.drainPendingTitleGenerationIfIdle()
        let titleTask = model.titleGenerationTask
        await titleTask?.value

        XCTAssertEqual(enqueuedMessages, ["First prompt"])
        XCTAssertEqual(model.chats.first(where: { $0.id == titleChat.id })?.title, "Generated title")
    }

    func testCompletedTitleUpdatesChatRowAndPersistedArchive() async {
        let chat = AppChat(
            title: "New Chat",
            titleProvenance: .unclaimed,
            titleGenerationAttempted: false)
        let model = makeModel(chats: [chat])
        model.selectedChatID = chat.id
        model.titleGenerationEnqueueOverride = { _ in "A generated title" }
        model.recordTitleGenerationAttempt(chatID: chat.id, firstUserMessage: "First prompt")
        model.mutateTurnMessages(for: chat.id) {
            $0.append(AppChatMessage(role: .user, content: "First prompt"))
        }
        model.drainPendingTitleGenerationIfIdle()
        let titleTask = model.titleGenerationTask
        await titleTask?.value

        XCTAssertEqual(model.chats[0].title, "A generated title")
        XCTAssertEqual(model.selectedChat.title, "A generated title")
        XCTAssertEqual(model.chats[0].titleProvenance, .generated)
        XCTAssertTrue(model.chats[0].titleGenerationAttempted)
        XCTAssertEqual(AppChatFileStore.load().chats.first?.title, "A generated title")
        XCTAssertEqual(AppChatFileStore.load().chats.first?.titleProvenance, .generated)
    }

    func testForegroundWorkCancelsAndDrainsInFlightTitleWithoutApplyingItsResult() async {
        let chat = AppChat(title: "New Chat", titleProvenance: .unclaimed, titleGenerationAttempted: false)
        let model = makeModel(chats: [chat])
        let started = expectation(description: "Title work started")
        var titleFinished = false
        model.titleGenerationEnqueueOverride = { _ in
            started.fulfill()
            do {
                try await Task.sleep(nanoseconds: 30_000_000_000)
            } catch {
                titleFinished = true
            }
            return "Cancelled title"
        }
        model.recordTitleGenerationAttempt(chatID: chat.id, firstUserMessage: "First prompt")
        model.drainPendingTitleGenerationIfIdle()
        await fulfillment(of: [started], timeout: 1)
        model.generating = true

        await model.interruptTitleGenerationForForeground()

        XCTAssertTrue(titleFinished, "Foreground model work must wait until title consumption ends.")
        XCTAssertNil(model.titleGenerationTask)
        XCTAssertEqual(model.chats[0].title, "New Chat")
        XCTAssertNil(model.pendingTitleGenerationTokens[chat.id])
    }

    func testLateTitleCannotReplaceManualRenameToDefaultTitle() throws {
        let chat = AppChat(
            title: "New Chat",
            titleProvenance: .unclaimed,
            titleGenerationAttempted: false)
        let model = makeModel(chats: [chat])
        model.recordTitleGenerationAttempt(chatID: chat.id, firstUserMessage: "First prompt")
        let attemptID = try XCTUnwrap(model.pendingTitleGenerationTokens[chat.id])

        model.renameChat(id: chat.id, title: "New Chat")
        model.applyTitleIfUngoverned(
            chatID: chat.id, candidate: "Late generated title", attemptID: attemptID)

        XCTAssertEqual(model.chats[0].title, "New Chat")
        XCTAssertEqual(model.chats[0].titleProvenance, .user)
        XCTAssertTrue(model.chats[0].titleGenerationAttempted)
        XCTAssertNil(model.pendingTitleGenerationTokens[chat.id])
        XCTAssertTrue(model.pendingTitleGenerations.isEmpty)
    }

    func testGhostMessageIsNeverQueuedForTitleGeneration() {
        let ghost = AppChat(
            title: "Temporary Chat",
            isGhost: true,
            titleProvenance: .unclaimed,
            titleGenerationAttempted: false)
        let model = makeModel(chats: [ghost])
        var enqueuedMessages: [String] = []
        model.titleGenerationEnqueueOverride = { message in
            enqueuedMessages.append(message)
            return nil
        }

        model.recordTitleGenerationAttempt(
            chatID: ghost.id, firstUserMessage: "Private ghost prompt")
        model.drainPendingTitleGenerationIfIdle()
        model.renameChat(id: ghost.id, title: "Changed ghost title")

        XCTAssertTrue(model.pendingTitleGenerations.isEmpty)
        XCTAssertFalse(model.chats[0].titleGenerationAttempted)
        XCTAssertEqual(model.chats[0].title, "Temporary Chat")
        XCTAssertTrue(enqueuedMessages.isEmpty)
    }
}
