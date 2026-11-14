import XCTest

@testable import TurboSparkApp
@testable import TurboSpark

/// ZP-23: the single-lane scheduler and the per-chat guards that replaced
/// the blanket "anything running blocks everything" guard.
///
/// Two halves. The pure coordinator is exercised cell by cell (the live
/// combinations need a running turn, which needs a model). The AppModel half
/// drives the real guards and the background tool deferral with the flags a
/// running turn would have set, the way `MessageQueueTests` does.
@MainActor
final class ChatGenerationCoordinatorTests: XCTestCase {
    private var appModel: AppModel!

    override func setUp() {
        super.setUp()
        appModel = AppModel()
    }

    override func tearDown() {
        appModel = nil
        super.tearDown()
    }

    private func result() -> GenerationResult {
        let json = """
        {"promptTokens": 0, "newTokens": 0, "reusedPrefixTokens": 0,
         "prefillSeconds": 0, "decodeSeconds": 0, "stopReason": "toolCalls",
         "tokensPerSecond": null, "content": "reply"}
        """
        return try! JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }

    /// Chats A (running) and B (selected), as a user who left A generating.
    private func leaveAGenerating() -> (a: UUID, b: UUID) {
        let a = AppChat(title: "Alpha")
        let b = AppChat(title: "Beta")
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id
        appModel.generating = true
        appModel.claimLane(for: a.id)
        appModel.outputText = "partial reply"
        return (a.id, b.id)
    }

    // MARK: - The lease (pure)

    func testTheCapIsOneLaneAndASecondRequestQueuesInsteadOfRefusing() {
        XCTAssertEqual(ChatGenerationCoordinator.laneCapacity, 1)
        var c = ChatGenerationCoordinator()
        let a = UUID(), b = UUID(), d = UUID()
        XCTAssertEqual(c.requestLane(for: a), .granted)
        XCTAssertEqual(c.requestLane(for: a), .granted, "Idempotent for the holder.")
        XCTAssertEqual(c.requestLane(for: b), .queued(position: 1))
        XCTAssertEqual(c.requestLane(for: d), .queued(position: 2))
        XCTAssertEqual(c.laneHolder, a)
    }

    func testOnlyTheHolderCanReleaseTheLane() {
        var c = ChatGenerationCoordinator()
        let a = UUID(), stale = UUID()
        c.requestLane(for: a)
        XCTAssertFalse(c.releaseLane(for: stale), "A stale tail must not free a newer turn's lane.")
        XCTAssertEqual(c.laneHolder, a)
        XCTAssertTrue(c.releaseLane(for: a))
        XCTAssertNil(c.laneHolder)
    }

    func testTheForegroundChatIsAdmittedBeforeOlderWaiters() {
        var c = ChatGenerationCoordinator()
        let running = UUID(), old = UUID(), foreground = UUID()
        c.requestLane(for: running)
        c.enqueue(old)
        c.enqueue(foreground)
        XCTAssertNil(c.nextAdmission(foreground: foreground), "The lane is held.")
        c.releaseHolder()
        XCTAssertEqual(c.nextAdmission(foreground: foreground), foreground)
        XCTAssertEqual(c.nextAdmission(foreground: UUID()), old, "Otherwise oldest first.")
    }

    func testEnqueueNeverGrantsTheLane() {
        var c = ChatGenerationCoordinator()
        let chat = UUID()
        c.enqueue(chat)
        XCTAssertNil(c.laneHolder, "A chat with parked input is not running anything.")
        XCTAssertEqual(c.state(of: chat), .queued)
    }

    func testAwaitingAToolReleasesTheLaneAndAwaitingAQuestionDoesNot() {
        var c = ChatGenerationCoordinator()
        let tool = UUID(), question = UUID()
        c.requestLane(for: tool)
        c.park(tool, as: .awaitingTool)
        XCTAssertNil(c.laneHolder)
        XCTAssertEqual(c.state(of: tool), .awaitingTool)

        c.requestLane(for: question)
        c.park(question, as: .awaitingQuestion)
        XCTAssertEqual(c.laneHolder, question, "The question blocks inside the running tool.")
        XCTAssertEqual(c.state(of: question), .generating, "The holder reads as generating.")
        c.unpark(tool)
        XCTAssertEqual(c.state(of: tool), .idle)
    }

    func testForgettingAChatDropsItsLaneQueueAndParkedState() {
        var c = ChatGenerationCoordinator()
        let chat = UUID()
        c.requestLane(for: chat)
        c.forget(chat)
        XCTAssertNil(c.laneHolder)
        c.enqueue(chat)
        c.park(chat, as: .awaitingTool)
        c.forget(chat)
        XCTAssertEqual(c.state(of: chat), .idle)
        XCTAssertNil(c.nextAdmission(foreground: chat))
    }

    // MARK: - The guard matrix (pure)

    func testSelectionIsAllowedWhileAChatGeneratesAndRefusedOnlyForTheNamedCases() {
        let a = UUID(), b = UUID()
        func v(_ s: ChatLaneSnapshot, target: UUID? = b) -> ChatLaneVerdict {
            ChatGenerationCoordinator.verdict(.select(target: target), s)
        }
        XCTAssertEqual(v(.init()), .allowed)
        XCTAssertEqual(
            v(.init(generating: true, generatingChatID: a)), .allowed,
            "A generating chat no longer pins the user to it.")
        XCTAssertEqual(
            v(.init(generating: true, generatingChatID: nil)), .refused(blockingChat: nil),
            "Work that is not a chat turn keeps the blanket refusal.")
        XCTAssertEqual(v(.init(submitting: true)), .refused(blockingChat: nil))
        XCTAssertEqual(
            v(.init(approvalChatID: a)), .refused(blockingChat: a),
            "An approval card pins the user to its chat.")
        XCTAssertEqual(v(.init(approvalChatID: b)), .allowed)
        XCTAssertEqual(
            v(.init(approvalChatID: a), target: nil), .refused(blockingChat: a),
            "Creating a chat is a selection.")
        XCTAssertEqual(
            v(.init(hasUnattributedApproval: true)), .refused(blockingChat: nil))
    }

    func testDeletionBlocksOnlyTheChatThatOwnsTheRunningTurn() {
        let a = UUID(), b = UUID()
        func v(_ s: ChatLaneSnapshot, chat: UUID) -> ChatLaneVerdict {
            ChatGenerationCoordinator.verdict(.delete(chat: chat), s)
        }
        let running = ChatLaneSnapshot(generating: true, generatingChatID: a)
        XCTAssertEqual(v(running, chat: a), .refused(blockingChat: a))
        XCTAssertEqual(v(running, chat: b), .allowed)
        XCTAssertEqual(
            v(.init(generating: true, generatingChatID: nil), chat: b), .refused(blockingChat: nil))
        XCTAssertEqual(v(.init(submitting: true), chat: b), .refused(blockingChat: nil))
        XCTAssertEqual(v(.init(), chat: a), .allowed)
    }

    func testTheModelCannotBeSwitchedWhileAnyChatGeneratesAndTheRefusalNamesIt() {
        let a = UUID()
        XCTAssertEqual(
            ChatGenerationCoordinator.verdict(
                .switchModel, .init(generating: true, generatingChatID: a)),
            .refused(blockingChat: a))
        XCTAssertEqual(
            ChatGenerationCoordinator.verdict(.switchModel, .init(submitting: true)),
            .refused(blockingChat: nil))
        XCTAssertEqual(ChatGenerationCoordinator.verdict(.switchModel, .init()), .allowed)
    }

    // MARK: - Guards on the real model

    func testSwitchingAwayFromAGeneratingChatIsAllowedAndKeepsItsLiveBuffer() {
        let (a, b) = leaveAGenerating()

        appModel.selectChat(id: b)

        XCTAssertEqual(appModel.selectedChatID, b, "The switch is no longer blocked.")
        XCTAssertTrue(appModel.generating, "The turn keeps running.")
        XCTAssertEqual(appModel.generatingChatID, a)
        XCTAssertEqual(
            appModel.outputText, "partial reply",
            "The buffer belongs to the running turn, not to the selection.")
        XCTAssertFalse(appModel.liveOutputBelongsToSelectedChat)
        XCTAssertFalse(appModel.isRunningInSelectedChat)
        XCTAssertTrue(appModel.isRunning, "The app-wide flag still says a turn runs.")
        XCTAssertEqual(appModel.chatTurnState(for: a), .generating)
        XCTAssertEqual(appModel.chatTurnState(for: b), .idle)
    }

    func testSwitchingBackToTheGeneratingChatShowsItsBufferAgain() {
        let (a, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        appModel.selectChat(id: a)
        XCTAssertTrue(appModel.isRunningInSelectedChat)
        XCTAssertEqual(appModel.outputText, "partial reply")
    }

    func testUnattributedBusyWorkStillBlocksSwitchingLikeBefore() {
        let a = AppChat(title: "A"), b = AppChat(title: "B")
        appModel.chats = [a, b]
        appModel.selectedChatID = a.id
        appModel.generating = true  // no `claimLane`: not a chat turn
        appModel.selectChat(id: b.id)
        XCTAssertEqual(appModel.selectedChatID, a.id)
        XCTAssertFalse(appModel.canSelectChat(id: b.id))
    }

    func testCreatingAChatWhileAnotherGeneratesLeavesTheStreamAlone() {
        let (a, _) = leaveAGenerating()
        // A chat with content, so `createChat` does not reuse it as the empty one.
        appModel.mutateTurnMessages(for: a) {
            $0.append(AppChatMessage(role: .user, content: "question"))
        }
        let before = appModel.chats.count

        let created = appModel.createChat()

        XCTAssertEqual(appModel.chats.count, before + 1)
        XCTAssertNotEqual(created, a)
        XCTAssertEqual(appModel.selectedChatID, created)
        XCTAssertEqual(appModel.outputText, "partial reply")
        XCTAssertTrue(appModel.canCreateChat)
    }

    func testOnlyTheGeneratingChatCannotBeDeleted() {
        let (a, b) = leaveAGenerating()

        appModel.deleteChat(id: a)
        XCTAssertTrue(appModel.chats.contains { $0.id == a }, "Its own turn pins it.")

        appModel.deleteChat(id: b)
        XCTAssertFalse(appModel.chats.contains { $0.id == b }, "Another chat's turn does not.")
        XCTAssertEqual(appModel.outputText, "partial reply")
    }

    func testTheFlagDroppingReleasesTheLaneAndTheOwner() {
        let (a, _) = leaveAGenerating()
        XCTAssertEqual(appModel.turnCoordinator.laneHolder, a)
        appModel.generating = false
        XCTAssertNil(appModel.generatingChatID)
        XCTAssertNil(appModel.turnCoordinator.laneHolder)
    }

    func testModelSwitchRefusalNamesTheGeneratingChat() {
        let (a, _) = leaveAGenerating()
        XCTAssertEqual(appModel.laneVerdict(.switchModel), .refused(blockingChat: a))
        XCTAssertEqual(appModel.laneChatTitle(a), "Alpha")
    }

    // MARK: - Background chats do not run tools

    private func call() -> AppToolCall {
        AppToolCall(name: "list_directory", arguments: ["path": "/tmp"], category: .fileRead)
    }

    func testABackgroundTurnsToolProposalParksInsteadOfRunning() async {
        let (a, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        let messagesBefore = appModel.chats.first { $0.id == a }!.messages.count

        await appModel.handleExtractedToolCalls(
            [call()], fullContent: "reply", reasoning: "", result: result(),
            currentStep: 0, chatID: a, project: nil)

        XCTAssertNotNil(appModel.deferredToolProposals[a], "The proposal parks whole.")
        XCTAssertNil(appModel.pendingToolCall, "No card is raised for a chat nobody is looking at.")
        XCTAssertEqual(
            appModel.chats.first { $0.id == a }!.messages.count, messagesBefore,
            "Nothing ran and nothing was recorded.")
        XCTAssertEqual(appModel.chatTurnState(for: a), .awaitingTool)
        XCTAssertEqual(appModel.turnCoordinator.state(of: a), .awaitingTool)
        XCTAssertNil(
            appModel.turnCoordinator.laneHolder,
            "Awaiting the user releases the lane.")
    }

    func testTheForegroundChatsProposalIsNotDeferred() async {
        let (a, _) = leaveAGenerating()
        XCTAssertFalse(
            appModel.shouldDeferToolProposal(chatID: a),
            "The selected chat takes the ordinary path, card and all.")
    }

    func testSelectingTheParkedChatReplaysItsProposalThroughTheOrdinaryPath() async {
        let (a, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        await appModel.handleExtractedToolCalls(
            [call()], fullContent: "reply", reasoning: "", result: result(),
            currentStep: 0, chatID: a, project: nil)
        // The turn's tail: the flag drops.
        appModel.generating = false
        appModel.outputText = ""

        appModel.selectChat(id: a)

        XCTAssertNil(appModel.deferredToolProposals[a], "Replayed exactly once.")
        XCTAssertTrue(appModel.generating, "The replay raises the flag like the approve path.")
        XCTAssertEqual(appModel.generatingChatID, a)
        let replay = appModel.toolExecutionTask
        await replay?.value
        XCTAssertFalse(appModel.generating)
        XCTAssertFalse(
            appModel.chats.first { $0.id == a }!.messages.isEmpty,
            "The ordinary path recorded the assistant turn.")
    }

    func testAProposalIsNotReplayedWhileAnotherTurnHoldsTheLane() async {
        let (a, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        await appModel.handleExtractedToolCalls(
            [call()], fullContent: "reply", reasoning: "", result: result(),
            currentStep: 0, chatID: a, project: nil)
        appModel.generating = false
        // Chat B now runs; the user goes back to A.
        appModel.generating = true
        appModel.claimLane(for: b)

        appModel.selectChat(id: a)

        XCTAssertNotNil(appModel.deferredToolProposals[a], "It waits for the lane.")
        XCTAssertEqual(appModel.generatingChatID, b)
    }

    func testAChatWithAParkedProposalTakesNoInjectedTurn() async {
        let (a, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        await appModel.handleExtractedToolCalls(
            [call()], fullContent: "reply", reasoning: "", result: result(),
            currentStep: 0, chatID: a, project: nil)
        appModel.generating = false
        XCTAssertFalse(
            appModel.canStartTurn(inChat: a),
            "A turn started on top of the parked proposal would orphan it.")
        XCTAssertTrue(appModel.canStartTurn(inChat: b))
    }

    func testDeletingTheParkedChatDropsItsProposalAndLaneState() async {
        let (a, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        await appModel.handleExtractedToolCalls(
            [call()], fullContent: "reply", reasoning: "", result: result(),
            currentStep: 0, chatID: a, project: nil)
        appModel.generating = false

        appModel.deleteChat(id: a)

        XCTAssertNil(appModel.deferredToolProposals[a])
        XCTAssertEqual(appModel.turnCoordinator.state(of: a), .idle)
    }

    // MARK: - Admission when the lane frees up

    func testAScheduledPromptForABackgroundChatStartsWhenTheLaneFrees() {
        let (_, b) = leaveAGenerating()
        let c = AppChat(title: "Gamma")
        appModel.chats.append(c)
        appModel.selectChat(id: b)
        appModel.scheduledDeliveryModelReadyOverride = true
        appModel.pendingUserMessages[c.id] = [QueuedUserPrompt(text: "nightly", isScheduled: true)]
        appModel.noteParkedWork(chatID: c.id)
        XCTAssertEqual(appModel.chatTurnState(for: c.id), .queued)

        // The running turn is A's own, so its tail finds nothing for A and the
        // lane frees. Run the admission the tail runs.
        appModel.generating = false
        appModel.admitParkedWork()

        XCTAssertNil(appModel.pendingUserMessages[c.id], "The parked prompt was admitted.")
        XCTAssertEqual(
            appModel.chats.first { $0.id == c.id }!.messages.last?.content, "nightly",
            "It was delivered to ITS chat, not the selected one.")
        XCTAssertTrue(appModel.chats.first { $0.id == b }!.messages.isEmpty)
    }

    func testAnAdmittedBackgroundChatIsDequeuedSoTheLoopAdvances() {
        let (_, b) = leaveAGenerating()
        let c = AppChat(title: "Gamma")
        appModel.chats.append(c)
        appModel.selectChat(id: b)
        appModel.pendingUserMessages[c.id] = [QueuedUserPrompt(text: "typed")]
        appModel.noteParkedWork(chatID: c.id)
        appModel.generating = false

        appModel.admitParkedWork()

        XCTAssertNil(
            appModel.turnCoordinator.nextAdmission(foreground: nil),
            "A chat that cannot start (typed prompt, needs the composer) is not retried forever.")
    }

    func testATypedPromptParkedInABackgroundChatWaitsForSelection() {
        let (_, b) = leaveAGenerating()
        let c = AppChat(title: "Gamma")
        appModel.chats.append(c)
        appModel.selectChat(id: b)
        appModel.pendingUserMessages[c.id] = [QueuedUserPrompt(text: "typed")]
        appModel.noteParkedWork(chatID: c.id)
        appModel.generating = false

        appModel.admitParkedWork()

        XCTAssertEqual(
            appModel.pendingUserMessages[c.id]?.count, 1,
            "A typed prompt goes through the composer, which describes only the selected chat.")
        XCTAssertEqual(appModel.chatTurnState(for: c.id), .queued)
    }

    func testTheSelectedChatsQueuedPromptStartsWhenAnotherChatsTurnEnds() {
        let (_, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        appModel.pendingUserMessages[b] = [QueuedUserPrompt(text: "mine")]
        appModel.noteParkedWork(chatID: b)
        appModel.generating = false

        appModel.admitParkedWork()

        // No session, so `run()` refuses at its guard, which leaves the entry
        // restored into the composer: proof the drain reached the SELECTED
        // chat although the finishing turn belonged to another.
        XCTAssertEqual(appModel.promptText, "mine")
        XCTAssertNil(appModel.pendingUserMessages[b])
    }

    func testNothingIsAdmittedWhileAnApprovalCardIsUp() {
        let (_, b) = leaveAGenerating()
        let c = AppChat(title: "Gamma")
        appModel.chats.append(c)
        appModel.selectChat(id: b)
        appModel.scheduledDeliveryModelReadyOverride = true
        appModel.pendingUserMessages[c.id] = [QueuedUserPrompt(text: "nightly", isScheduled: true)]
        appModel.noteParkedWork(chatID: c.id)
        appModel.generating = false
        appModel.pendingToolCall = call()
        appModel.pendingToolCallChatID = b

        appModel.admitParkedWork()

        XCTAssertEqual(
            appModel.pendingUserMessages[c.id]?.count, 1,
            "A background turn must not start beside an approval the user is deciding.")
    }

    // MARK: - Quit

    func testAQuitFlushesTheRunningTurnIntoItsOwnChatNotTheSelectedOne() {
        let (a, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        appModel.mutateTurnMessages(for: a) {
            $0.append(AppChatMessage(role: .user, content: "question"))
        }

        XCTAssertEqual(appModel.shutdownFlushChatID, a)
        appModel.finishCancelled(chatID: appModel.shutdownFlushChatID, reason: "cancelled")

        let rowsA = appModel.chats.first { $0.id == a }!.messages
        XCTAssertEqual(rowsA.last?.role, .assistant)
        XCTAssertEqual(rowsA.last?.content, "partial reply")
        XCTAssertTrue(
            appModel.chats.first { $0.id == b }!.messages.isEmpty,
            "The reply must not land in the chat the user happened to be looking at.")
    }

    func testAQuitFlushesEveryParkedProposalIntoAnInterruptedRow() async {
        let (a, b) = leaveAGenerating()
        appModel.selectChat(id: b)
        appModel.mutateTurnMessages(for: a) {
            $0.append(AppChatMessage(role: .user, content: "question"))
        }
        await appModel.handleExtractedToolCalls(
            [call()], fullContent: "the prose before the call", reasoning: "", result: result(),
            currentStep: 0, chatID: a, project: nil)
        appModel.generating = false

        appModel.flushDeferredProposalsForShutdown()

        let rows = appModel.chats.first { $0.id == a }!.messages
        XCTAssertEqual(rows.last?.role, .assistant)
        XCTAssertTrue(rows.last?.content.contains("the prose before the call") == true)
        XCTAssertTrue(appModel.deferredToolProposals.isEmpty)
        XCTAssertNotNil(appModel.chats.first { $0.id == a }!.recoveryAnchor)
    }
}
