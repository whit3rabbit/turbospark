import XCTest

@testable import TurboSparkApp
@testable import TurboSpark

/// ZP-23 remainder: two chats each holding their own approval, the lane
/// held while an AskUserQuestion card is open, a typed prompt started in a
/// background chat, and the cross-project rules for parked work.
///
/// Every test drives the real `AppModel` with the flags a live turn would
/// set (no model is loaded), the way `ChatGenerationCoordinatorTests` does.
@MainActor
final class ConcurrentApprovalsTests: XCTestCase {
    private var model: AppModel!

    override func setUp() {
        super.setUp()
        model = AppModel()
        model.interactionMode = .projects
    }

    override func tearDown() {
        model = nil
        super.tearDown()
    }

    // MARK: - Fixtures

    private func result() -> GenerationResult {
        let json = """
        {"promptTokens": 0, "newTokens": 0, "reusedPrefixTokens": 0,
         "prefillSeconds": 0, "decodeSeconds": 0, "stopReason": "toolCalls",
         "tokensPerSecond": null, "content": "reply"}
        """
        return try! JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }

    private func makeProject(
        _ name: String, permissions: AppProjectPermissions = .alwaysAsk
    ) -> AppProject {
        let project = AppProject(
            name: name, rootDirectoryPath: "/tmp", permissions: permissions)
        model.projects.append(project)
        return project
    }

    /// Edits the live project by id: `AppModel()` may start with projects
    /// another test left in the shared store, so index 0 is not ours.
    private func edit(_ project: AppProject, _ change: (inout AppProject) -> Void) {
        let index = model.projects.firstIndex { $0.id == project.id }!
        change(&model.projects[index])
    }

    private func makeChat(_ title: String, in project: AppProject?) -> AppChat {
        let chat = AppChat(projectID: project?.id, title: title)
        model.chats.append(chat)
        return chat
    }

    private func listCall() -> AppToolCall {
        AppToolCall(name: "list_directory", arguments: ["path": "/tmp"], category: .fileRead)
    }

    /// A call the always-ask project stops at a card. Harmless if approved.
    private func askCall() -> AppToolCall {
        AppToolCall(
            name: "run_command", arguments: ["command": "echo hi"], category: .terminal)
    }

    private func shellCall() -> AppToolCall {
        AppToolCall(
            name: "run_command", arguments: ["command": "touch x"], category: .terminal)
    }

    /// Proposes `call` in `chat` as a foreground turn would, leaving a card.
    private func propose(_ call: AppToolCall, in chat: AppChat, project: AppProject?) async {
        await model.handleExtractedToolCall(
            call, fullContent: "reply", reasoning: "", result: result(),
            currentStep: 0, chatID: chat.id, project: project)
    }

    private func toolCall(_ id: UUID, in chat: AppChat) -> AppToolCall? {
        model.chats.first { $0.id == chat.id }?.messages
            .flatMap(\.toolCalls).last { $0.id == id }
    }

    private func results(in chat: AppChat) -> [AppToolResult] {
        model.chats.first { $0.id == chat.id }?.messages.flatMap(\.toolResults) ?? []
    }

    // MARK: - CUJ 2: two chats, two approvals

    func testTwoChatsEachKeepTheirOwnApprovalAndAnswerInEitherOrder() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        let callA = askCall(), callB = askCall()

        await propose(callA, in: a, project: p)
        XCTAssertEqual(model.pendingToolCall?.id, callA.id)

        model.selectChat(id: b.id)
        XCTAssertNil(model.pendingToolCall, "Leaving A empties the slot.")
        XCTAssertNotNil(model.parkedApprovals[a.id])
        XCTAssertEqual(model.chatTurnState(for: a.id), .awaitingTool)

        await propose(callB, in: b, project: p)
        XCTAssertEqual(model.pendingToolCall?.id, callB.id)
        XCTAssertEqual(model.pendingToolCallChatID, b.id)

        // A's card id cannot answer B's slot: nothing starts, nothing moves.
        model.approvePendingToolCall(id: callA.id)
        XCTAssertFalse(model.generating)
        XCTAssertEqual(model.pendingToolCallChatID, b.id)
        XCTAssertEqual(toolCall(callA.id, in: a)?.status, .pendingApproval)

        // Back to A: B's card parks, A's returns.
        model.selectChat(id: a.id)
        XCTAssertEqual(model.pendingToolCall?.id, callA.id)
        XCTAssertNotNil(model.parkedApprovals[b.id])
        XCTAssertNil(model.parkedApprovals[a.id])

        model.approvePendingToolCall(id: callA.id)
        await model.toolExecutionTask?.value
        XCTAssertTrue(results(in: a).contains { $0.callID == callA.id }, "A ran its own call.")
        XCTAssertTrue(results(in: b).isEmpty, "A's approval never touched B.")
        XCTAssertEqual(toolCall(callB.id, in: b)?.status, .pendingApproval)
        XCTAssertNotNil(model.parkedApprovals[b.id], "B still waits.")

        // B is answered later, in its own chat, and only it is affected.
        model.selectChat(id: b.id)
        XCTAssertEqual(model.pendingToolCall?.id, callB.id)
        model.denyPendingToolCall(id: callB.id)
        await model.toolExecutionTask?.value
        XCTAssertEqual(toolCall(callB.id, in: b)?.status, .denied)
        XCTAssertNotEqual(toolCall(callA.id, in: a)?.status, .denied)
        XCTAssertTrue(model.parkedApprovals.isEmpty)
    }

    func testAnApprovalCannotBeAnsweredFromTheOtherChatsSlotOrWhileParked() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        let callA = askCall()
        await propose(callA, in: a, project: p)
        model.selectChat(id: b.id)

        model.approvePendingToolCall(id: callA.id)
        model.denyPendingToolCall(id: callA.id)

        XCTAssertFalse(model.generating, "A parked card has no slot to answer from.")
        XCTAssertEqual(toolCall(callA.id, in: a)?.status, .pendingApproval)
        XCTAssertNotNil(model.parkedApprovals[a.id])
    }

    func testTheCardIsNotRestoredWhileAnotherChatsTurnHoldsTheLane() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        await propose(askCall(), in: a, project: p)
        model.selectChat(id: b.id)
        model.generating = true
        model.claimLane(for: b.id)

        model.selectChat(id: a.id)
        XCTAssertNil(model.pendingToolCall, "Approve would start a second turn beside B's.")
        XCTAssertNotNil(model.parkedApprovals[a.id])

        // B's turn ends: the tail's admission puts the foreground card back.
        model.generating = false
        model.admitParkedWork()
        XCTAssertNotNil(model.pendingToolCall)
        XCTAssertNil(model.parkedApprovals[a.id])
    }

    func testDeletingAChatDropsItsParkedApproval() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        await propose(askCall(), in: a, project: p)
        model.selectChat(id: b.id)

        model.deleteChat(id: a.id)

        XCTAssertNil(model.parkedApprovals[a.id])
        XCTAssertEqual(model.turnCoordinator.state(of: a.id), .idle)
    }

    func testAnApprovalWithNoChatStillPinsTheUser() async {
        let a = makeChat("A", in: nil), b = makeChat("B", in: nil)
        model.selectedChatID = a.id
        model.pendingToolCall = askCall()
        model.pendingToolCallChatID = nil

        model.selectChat(id: b.id)

        XCTAssertEqual(model.selectedChatID, a.id, "It cannot be parked under a chat key.")
    }

    // MARK: - Replay re-validation

    func testAParkedApprovalTheNewPolicyRefusesIsDeniedNotShown() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        let call = shellCall()
        await propose(call, in: a, project: p)
        model.selectChat(id: b.id)

        // The user tightens the project while A's card waits.
        edit(p) { $0.permissions = AppProjectPermissions(mode: .ask, terminal: .deny) }
        model.selectChat(id: a.id)
        await model.toolExecutionTask?.value

        XCTAssertNil(model.pendingToolCall, "A refused call gets no Approve button.")
        XCTAssertEqual(toolCall(call.id, in: a)?.status, .denied)
        let output = results(in: a).first { $0.callID == call.id }?.output ?? ""
        XCTAssertTrue(output.hasPrefix("Tool execution denied:"), output)
        XCTAssertNotEqual(output, TOOL_REJECTED_MESSAGE, "The user did not say no.")
    }

    func testAParkedApprovalThePolicyNowAllowsStillWaitsForTheUser() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        let call = askCall()
        await propose(call, in: a, project: p)
        model.selectChat(id: b.id)

        edit(p) { $0.permissions = .fullAccess }
        model.selectChat(id: a.id)

        XCTAssertEqual(model.pendingToolCall?.id, call.id, "Relaxed policy never auto-approves.")
        XCTAssertFalse(model.generating)
        XCTAssertEqual(toolCall(call.id, in: a)?.status, .pendingApproval)
    }

    func testAParkedApprovalForADeletedProjectIsDenied() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        let call = askCall()
        await propose(call, in: a, project: p)
        model.selectChat(id: b.id)

        model.projects.removeAll()
        model.selectChat(id: a.id)
        await model.toolExecutionTask?.value

        XCTAssertEqual(toolCall(call.id, in: a)?.status, .denied)
        XCTAssertTrue(
            (results(in: a).first { $0.callID == call.id }?.output ?? "")
                .contains("no longer exists"))
    }

    func testAParkedApprovalRunsUnderTheLiveProjectNotTheSnapshot() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        await propose(askCall(), in: a, project: p)
        model.selectChat(id: b.id)
        edit(p) { $0.maxAutonomousSteps = 9 }

        model.selectChat(id: a.id)

        XCTAssertEqual(model.pendingToolCallProject?.maxAutonomousSteps, 9)
    }

    func testADeferredProposalIsReplayedAgainstTheLiveProject() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        model.generating = true
        model.claimLane(for: a.id)
        model.selectChat(id: b.id)
        let call = listCall()
        await model.handleExtractedToolCalls(
            [call], fullContent: "reply", reasoning: "", result: result(),
            currentStep: 0, chatID: a.id, project: p)
        model.generating = false
        XCTAssertNotNil(model.deferredToolProposals[a.id])

        edit(p) { $0.permissions = AppProjectPermissions(mode: .ask, fileRead: .deny) }
        model.selectChat(id: a.id)
        await model.toolExecutionTask?.value

        XCTAssertEqual(
            toolCall(call.id, in: a)?.status, .denied,
            "The stale snapshot would have shown a card for a call the policy now refuses.")
        XCTAssertNil(model.pendingToolCall)
    }

    func testADeferredProposalForADeletedProjectIsDroppedNotRun() async {
        let p = makeProject("alpha")
        let a = makeChat("A", in: p), b = makeChat("B", in: p)
        model.selectedProjectID = p.id
        model.selectedChatID = a.id
        model.generating = true
        model.claimLane(for: a.id)
        model.selectChat(id: b.id)
        await model.handleExtractedToolCalls(
            [listCall()], fullContent: "the prose", reasoning: "", result: result(),
            currentStep: 0, chatID: a.id, project: p)
        model.generating = false

        model.projects.removeAll()
        model.selectChat(id: a.id)

        XCTAssertFalse(model.generating, "Nothing ran.")
        XCTAssertNil(model.deferredToolProposals[a.id])
        XCTAssertTrue(
            model.chats.first { $0.id == a.id }!.messages.last?.content.contains("the prose")
                == true, "The prose survives as an interrupted row.")
    }

    // MARK: - Cross-project

    func testOpeningAChatInAnotherProjectParksTheCardAndRestoresItInItsOwnWorkspace() async {
        let p1 = makeProject("one"), p2 = makeProject("two")
        let a = makeChat("A", in: p1), b = makeChat("B", in: p2)
        model.selectedProjectID = p1.id
        model.selectedChatID = a.id
        let call = askCall()
        await propose(call, in: a, project: p1)
        XCTAssertTrue(model.canOpenChat(id: b.id), "The chat's own card does not block the move.")

        model.openChat(id: b.id)

        XCTAssertEqual(model.selectedChatID, b.id)
        XCTAssertEqual(model.selectedProjectID, p2.id)
        XCTAssertNil(model.pendingToolCall)
        XCTAssertEqual(model.parkedApprovals[a.id]?.project?.id, p1.id)

        model.openChat(id: a.id)
        XCTAssertEqual(model.selectedProjectID, p1.id)
        XCTAssertEqual(model.pendingToolCall?.id, call.id)
        XCTAssertEqual(model.pendingToolCallProject?.id, p1.id)
    }

    func testAProjectSwitchWhileATurnRunsLeavesTheCardInItsChat() async {
        let p1 = makeProject("one"), p2 = makeProject("two")
        let a = makeChat("A", in: p1), b = makeChat("B", in: p2)
        model.selectedProjectID = p1.id
        model.selectedChatID = a.id
        await propose(askCall(), in: a, project: p1)
        model.generating = true
        model.claimLane(for: UUID())

        model.openChat(id: b.id)

        XCTAssertEqual(model.selectedChatID, a.id, "A refused move must not strand the card.")
        XCTAssertNotNil(model.pendingToolCall)
        XCTAssertNil(model.parkedApprovals[a.id])
    }

    func testParkedWorkOfAnotherProjectWaitsForAQuietWorkspace() async {
        let p1 = makeProject("one"), p2 = makeProject("two")
        let a = makeChat("A", in: p1), b = makeChat("B", in: p2)
        model.selectedProjectID = p1.id
        model.selectedChatID = a.id
        let call = askCall()
        await propose(call, in: a, project: p1)
        model.selectChat(id: b.id)
        model.selectedProjectID = p2.id
        model.generating = true
        model.claimLane(for: b.id)

        // Selected without the project move (a notification click, say).
        model.selectChat(id: a.id)
        XCTAssertNil(model.pendingToolCall, "The workspace is p2 and a turn runs in it.")
        XCTAssertNotNil(model.parkedApprovals[a.id])
        XCTAssertEqual(model.selectedProjectID, p2.id, "The workspace did not move under a turn.")

        model.generating = false
        model.admitParkedWork()
        XCTAssertEqual(model.selectedProjectID, p1.id)
        XCTAssertEqual(model.pendingToolCall?.id, call.id)
    }

    // MARK: - AskUserQuestion holds the lane

    func testAnOpenQuestionHoldsTheLaneAndTheRowSaysSo() async throws {
        let a = makeChat("A", in: nil), b = makeChat("B", in: nil)
        let c = makeChat("C", in: nil)
        model.selectedChatID = a.id
        model.generating = true
        model.claimLane(for: a.id)
        let items = [
            UserQuestionItem(
                question: "Which?", header: "Pick",
                options: [UserQuestionOption(label: "One", description: "1")])
        ]
        let waiting = Task { @MainActor in
            await self.model.waitForUserAnswers(chatID: a.id, items: items, toolCallID: nil)
        }
        while model.pendingUserQuestions == nil { await Task.yield() }
        try await Task.sleep(nanoseconds: 50_000_000)

        XCTAssertEqual(model.chatTurnState(for: a.id), .awaitingQuestion)
        XCTAssertEqual(model.turnCoordinator.laneHolder, a.id, "The question holds the lane.")
        XCTAssertEqual(
            model.turnCoordinator.state(of: a.id), .awaitingQuestion,
            "The scheduler knows what the holder waits for.")
        XCTAssertFalse(model.canStartTurn(inChat: b.id), "No second turn beside it.")

        // The user wanders off to B; C has scheduled work. Nothing starts.
        model.selectChat(id: b.id)
        XCTAssertEqual(model.selectedChatID, b.id, "Looking elsewhere is allowed.")
        model.scheduledDeliveryModelReadyOverride = true
        model.pendingUserMessages[c.id] = [QueuedUserPrompt(text: "nightly", isScheduled: true)]
        model.noteParkedWork(chatID: c.id)
        model.admitParkedWork()
        XCTAssertEqual(model.pendingUserMessages[c.id]?.count, 1)
        XCTAssertEqual(model.chatTurnState(for: a.id), .awaitingQuestion)

        // The answer goes to A's question, whichever chat is on screen.
        model.submitUserQuestionAnswers(["Pick": "One"])
        let answer = await waiting.value
        XCTAssertTrue(answer.contains("One"))
        XCTAssertNil(model.pendingUserQuestions)
        XCTAssertEqual(model.chatTurnState(for: a.id), .generating, "Answered: A runs again.")
        XCTAssertEqual(model.turnCoordinator.state(of: a.id), .generating)
        XCTAssertEqual(model.turnCoordinator.laneHolder, a.id, "Still A's turn, still A's lane.")

        // The turn ends; only now does C get the lane.
        model.generating = false
        model.admitParkedWork()
        XCTAssertNil(model.pendingUserMessages[c.id])
    }

    func testStoppingAnOpenQuestionReleasesTheLaneWithTheTurn() async throws {
        let a = makeChat("A", in: nil)
        model.selectedChatID = a.id
        model.generating = true
        model.claimLane(for: a.id)
        let items = [
            UserQuestionItem(
                question: "Which?", header: "Pick",
                options: [UserQuestionOption(label: "One", description: "1")])
        ]
        let waiting = Task { @MainActor in
            await self.model.waitForUserAnswers(chatID: a.id, items: items, toolCallID: nil)
        }
        while model.pendingUserQuestions == nil { await Task.yield() }
        try await Task.sleep(nanoseconds: 50_000_000)

        model.dismissUserQuestions()
        _ = await waiting.value
        model.generating = false

        XCTAssertNil(model.turnCoordinator.laneHolder)
        XCTAssertNil(model.turnCoordinator.parked[a.id], "No stale awaitingQuestion mark.")
        XCTAssertEqual(model.chatTurnState(for: a.id), .idle)
    }

    // MARK: - A typed prompt in a background chat

    func testATypedPromptParkedInABackgroundChatStartsWithoutTheComposer() async {
        let a = makeChat("A", in: nil), b = makeChat("B", in: nil)
        let c = makeChat("C", in: nil)
        model.selectedChatID = a.id
        model.generating = true
        model.claimLane(for: a.id)
        model.selectChat(id: b.id)
        model.scheduledDeliveryModelReadyOverride = true
        model.promptText = "half-typed draft in B"
        model.pendingUserMessages[c.id] = [
            QueuedUserPrompt(text: "first"), QueuedUserPrompt(text: "second"),
        ]
        model.noteParkedWork(chatID: c.id)

        model.generating = false
        model.admitParkedWork()
        XCTAssertTrue(model.submitting, "The hook window is covered like run()'s.")
        await model.submissionTask?.value

        let rows = model.chats.first { $0.id == c.id }!.messages
        XCTAssertEqual(rows.count, 1, "Only the first entry starts.")
        XCTAssertTrue(rows.first?.content.hasSuffix("first") == true)
        XCTAssertEqual(model.pendingUserMessages[c.id]?.map(\.text), ["second"])
        XCTAssertEqual(model.promptText, "half-typed draft in B", "B's composer is untouched.")
        XCTAssertTrue(model.chats.first { $0.id == b.id }!.messages.isEmpty)
    }

    func testBackgroundTypedPromptsThatAreCommandsStillWait() {
        XCTAssertFalse(AppModel.startsWithoutComposer("/compact"))
        XCTAssertFalse(AppModel.startsWithoutComposer("!ls"))
        XCTAssertFalse(AppModel.startsWithoutComposer("# remember this"))
        XCTAssertFalse(AppModel.startsWithoutComposer("   "))
        XCTAssertTrue(AppModel.startsWithoutComposer("explain this"))
    }
}
