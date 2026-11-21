import XCTest

@testable import TurboSparkApp

/// The blocking ExitPlanMode gate: the tool result waits for the plan card's
/// Start or Request changes, Stop resolves it as cancelled, and no other tool
/// changes behavior. Driven through the real `AppModel` waiter (installed by
/// its init) and `AppToolRegistry.execute`, the path a live turn takes.
private final class ResultBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: AppToolResult?
    var value: AppToolResult? {
        get { lock.withLock { stored } }
        set { lock.withLock { stored = newValue } }
    }
}

@MainActor
final class PlanGateTests: XCTestCase {
    private var model: AppModel!
    private var dir: URL!
    private var project: AppProject!

    override func setUp() async throws {
        try await super.setUp()
        model = AppModel()
        // Questions from other suites must not park here.
        AskUserQuestionExecutor.answerWaiter = nil
        dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        project = AppProject(name: "plan-gate", rootDirectoryPath: dir.path)
    }

    override func tearDown() async throws {
        // AppModel's init installed the gate as static state; leave none.
        PlanModeExecutor.decisionWaiter = nil
        model = nil
        if let dir { try? FileManager.default.removeItem(at: dir) }
        try await super.tearDown()
    }

    private func makeChat() -> UUID {
        let chat = AppChat(projectID: nil, title: "Plan")
        model.chats.append(chat)
        return chat.id
    }

    private func exitCall() -> AppToolCall {
        AppToolCall(name: "ExitPlanMode", arguments: ["plan": "1. Read\n2. Edit"], category: .automation)
    }

    /// Enters plan mode for `chat`, then starts ExitPlanMode and waits until
    /// its card is published (the call is parked).
    private func startGate(chat: UUID) async -> (AppToolCall, Task<AppToolResult, Never>) {
        _ = await AppToolRegistry.execute(
            call: AppToolCall(name: "EnterPlanMode", arguments: [:], category: .automation),
            in: project, chatID: chat)
        XCTAssertTrue(PlanModeExecutor.isPlanModeActive(for: chat))
        let call = exitCall()
        let project = self.project
        let task = Task { await AppToolRegistry.execute(call: call, in: project, chatID: chat) }
        while model.pendingPlanApproval == nil { await Task.yield() }
        return (call, task)
    }

    func testTheToolWaitsAndStartApprovesAndEndsPlanMode() async throws {
        let chat = makeChat()
        let (call, task) = await startGate(chat: chat)

        XCTAssertEqual(model.pendingPlanApproval?.toolCallID, call.id)
        XCTAssertEqual(model.pendingPlanApproval?.plan, "1. Read\n2. Edit")
        XCTAssertEqual(model.chatTurnState(for: chat), .awaitingQuestion, "The row says it waits on the user.")
        // Still parked after a while: nothing auto-approves.
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertNotNil(model.pendingPlanApproval)
        XCTAssertTrue(PlanModeExecutor.isPlanModeActive(for: chat), "Plan mode lasts until a decision.")

        model.approvePendingPlan()
        let result = await task.value
        XCTAssertFalse(result.isError, result.output)
        XCTAssertEqual(result.output, PlanModeExecutor.approvedResult)
        XCTAssertFalse(PlanModeExecutor.isPlanModeActive(for: chat), "Approval ends plan mode.")
        XCTAssertNil(model.pendingPlanApproval)
    }

    func testRequestChangesReturnsTheTextAndKeepsPlanMode() async throws {
        let chat = UUID()
        let (_, task) = await startGate(chat: chat)

        // Blank text is ignored: the card stays up.
        model.requestChangesToPendingPlan("   ")
        XCTAssertNotNil(model.pendingPlanApproval)

        model.requestChangesToPendingPlan("  Split step 2 into two.  ")
        let result = await task.value
        XCTAssertEqual(result.output, "The user requested changes: Split step 2 into two.")
        XCTAssertTrue(PlanModeExecutor.isPlanModeActive(for: chat), "Changes keep plan mode on.")
        XCTAssertNil(model.pendingPlanApproval)
    }

    func testStopResolvesTheGateAsCancelledAndKeepsPlanMode() async throws {
        let chat = makeChat()
        model.selectedChatID = chat
        _ = await AppToolRegistry.execute(
            call: AppToolCall(name: "EnterPlanMode", arguments: [:], category: .automation),
            in: project, chatID: chat)
        let call = exitCall()
        let project = self.project
        let box = ResultBox()
        let done = expectation(description: "gate returns after Stop")
        // The live shape: the call runs inside the turn's `toolExecutionTask`
        // and Stop is `cancel()`, which cancels that task.
        model.generating = true
        model.claimLane(for: chat)
        model.toolExecutionTask = Task {
            box.value = await AppToolRegistry.execute(call: call, in: project, chatID: chat)
            done.fulfill()
        }
        while model.pendingPlanApproval == nil { await Task.yield() }

        model.cancel()
        let waited = await XCTWaiter.fulfillment(of: [done], timeout: 5)
        guard waited == .completed else {
            return XCTFail("Stop left the plan gate parked")
        }
        XCTAssertEqual(box.value?.output, PlanModeExecutor.cancelledResult)
        XCTAssertTrue(PlanModeExecutor.isPlanModeActive(for: chat), "No decision: plan mode stays.")
        XCTAssertNil(model.pendingPlanApproval, "The waiting card is gone.")
        XCTAssertNil(model.turnCoordinator.parked[chat], "No stale parked mark.")
        // A late click after Stop resumes nothing and does not crash.
        model.approvePendingPlan()
        PlanModeExecutor.submitDecision(chatID: chat, toolCallID: call.id, decision: .approved)
        XCTAssertTrue(PlanModeExecutor.isPlanModeActive(for: chat))
        model.generating = false
    }

    func testCancelBeforeTheWaiterParksStillCancels() async throws {
        // The waiter parks only after the cancel landed: the early-cancel
        // record must resolve it rather than leak the continuation.
        let chat = UUID()
        let call = exitCall()
        PlanModeExecutor.decisionWaiter = { chatID, _, toolCallID in
            try? await Task.sleep(nanoseconds: 300_000_000)
            return await withCheckedContinuation { continuation in
                PlanModeExecutor.waitForDecision(
                    chatID: chatID, toolCallID: toolCallID, continuation: continuation)
            }
        }
        let done = expectation(description: "returns after early cancel")
        let task = Task { () -> String in
            let out = await PlanModeExecutor.exitAwaitingDecision(
                arguments: call.arguments, chatID: chat, toolCallID: call.id)
            done.fulfill()
            return out
        }
        try await Task.sleep(nanoseconds: 50_000_000)
        task.cancel()
        let waited = await XCTWaiter.fulfillment(of: [done], timeout: 5)
        guard waited == .completed else {
            return XCTFail("an early cancel left the plan gate parked")
        }
        let out = await task.value
        XCTAssertEqual(out, PlanModeExecutor.cancelledResult)
    }

    func testDoubleSubmitResumesOnceAndAnotherCallsIDDoesNotResume() async throws {
        let chat = UUID()
        let (call, task) = await startGate(chat: chat)

        // A decision for a different tool call is not this gate's.
        PlanModeExecutor.submitDecision(chatID: chat, toolCallID: UUID(), decision: .approved)
        try await Task.sleep(nanoseconds: 50_000_000)
        XCTAssertNotNil(model.pendingPlanApproval, "Another call's decision must not resume this one.")

        model.approvePendingPlan()
        model.approvePendingPlan()
        model.requestChangesToPendingPlan("late")
        PlanModeExecutor.submitDecision(
            chatID: chat, toolCallID: call.id, decision: .changesRequested("racing"))
        let result = await task.value
        XCTAssertEqual(result.output, PlanModeExecutor.approvedResult, "Only the first submit counts.")
        XCTAssertFalse(PlanModeExecutor.isPlanModeActive(for: chat))
    }

    func testOtherToolsDoNotParkOnTheGate() async throws {
        let chat = UUID()
        let invoked = expectation(description: "gate invoked")
        invoked.isInverted = true
        PlanModeExecutor.decisionWaiter = { _, _, _ in
            invoked.fulfill()
            return .approved
        }
        let enter = await AppToolRegistry.execute(
            call: AppToolCall(name: "EnterPlanMode", arguments: [:], category: .automation),
            in: project, chatID: chat)
        XCTAssertFalse(enter.isError)
        let question = await AppToolRegistry.execute(
            call: AppToolCall(name: "AskUserQuestion", arguments: ["question": "Go?"], category: .automation),
            in: project, chatID: chat)
        XCTAssertFalse(question.isError, question.output)
        XCTAssertTrue(question.output.contains("Presented"), "Questions keep their own path.")
        let todo = await AppToolRegistry.execute(
            call: AppToolCall(
                name: "TodoWrite",
                arguments: ["todos": #"[{"content":"a","status":"pending","activeForm":"a"}]"#],
                category: .automation),
            in: project, chatID: chat)
        XCTAssertFalse(todo.isError, todo.output)
        await fulfillment(of: [invoked], timeout: 0.2)
        XCTAssertNil(model.pendingPlanApproval)
    }

    func testWithoutAGateExitStaysNonBlocking() async {
        PlanModeExecutor.decisionWaiter = nil
        let chat = UUID()
        _ = await AppToolRegistry.execute(
            call: AppToolCall(name: "EnterPlanMode", arguments: [:], category: .automation),
            in: project, chatID: chat)
        let result = await AppToolRegistry.execute(call: exitCall(), in: project, chatID: chat)
        XCTAssertTrue(result.output.hasPrefix("Exited plan mode."), result.output)
        XCTAssertFalse(PlanModeExecutor.isPlanModeActive(for: chat))
    }

    // MARK: - Post-turn offer

    private func planMessages(output: String) -> (UUID, [AppChatMessage]) {
        let call = exitCall()
        let result = AppToolResult(callID: call.id, output: output, isError: false)
        return (call.id, [
            AppChatMessage(role: .user, content: "plan it"),
            AppChatMessage(role: .assistant, content: "", toolCalls: [call], toolResults: [result]),
        ])
    }

    func testPostTurnOfferHidesWhileTheGateWaitsOnThatCall() {
        let (id, messages) = planMessages(output: "Exited plan mode.")
        XCTAssertTrue(PlanCardActions.isOffered(callID: id, in: messages, isIdle: true))
        XCTAssertFalse(PlanCardActions.isOffered(callID: id, in: messages, isIdle: true, awaitingCallID: id))
        XCTAssertTrue(
            PlanCardActions.isOffered(callID: id, in: messages, isIdle: true, awaitingCallID: UUID()),
            "Only the parked call's offer is hidden.")
    }

    func testPostTurnOfferHidesOnceTheGateDecidedButNotAfterACancel() {
        let approved = planMessages(output: PlanModeExecutor.approvedResult)
        XCTAssertFalse(PlanCardActions.isOffered(callID: approved.0, in: approved.1, isIdle: true))
        let changes = planMessages(output: PlanModeExecutor.changesRequestedPrefix + "x")
        XCTAssertFalse(PlanCardActions.isOffered(callID: changes.0, in: changes.1, isIdle: true))
        // A stopped gate decided nothing: the non-blocking card is the way on.
        let cancelled = planMessages(output: PlanModeExecutor.cancelledResult)
        XCTAssertTrue(PlanCardActions.isOffered(callID: cancelled.0, in: cancelled.1, isIdle: true))
    }
}
