import TurboSpark
import XCTest

@testable import TurboSparkApp

/// OC-02: the four emit points on the real `AppModel`, against a fake
/// center. Each test names the lifecycle point it covers.
@MainActor
final class AttentionEventHooksTests: XCTestCase {
    private var model: AppModel!
    private var backend: FakeAttentionBackend!
    private var chat: AppChat!

    override func setUp() async throws {
        model = AppModel()
        model.stopCronScheduler()
        backend = FakeAttentionBackend()
        backend.status = .authorized
        model.attentionCenter = AppNotificationCenter(
            backend: backend, isActive: { false }, playSound: { _ in })
        await model.attentionCenter.refreshAuthorization()
        chat = AppChat(title: "Release notes")
        model.chats = [chat]
        model.selectedChatID = chat.id
        for event in AttentionEvent.allCases {
            model.attentionPreferences[event] = AttentionEventPreference(enabled: true)
        }
    }

    override func tearDown() {
        model = nil
        backend = nil
        super.tearDown()
    }

    private static func stubResult() -> GenerationResult {
        let json = #"{"promptTokens":1,"newTokens":1,"prefillSeconds":0,"decodeSeconds":0,"stopReason":"toolCalls","content":""}"#
        return try! JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }

    /// Lets the cue tasks the hooks spawn run.
    private func settle() async {
        for _ in 0..<20 { await Task.yield() }
    }

    func testApprovalCardRaisesNeedsPermissionNamingTheToolOnly() async {
        let call = AppToolCall(name: "run_command", arguments: ["command": "rm -rf secret"], category: .terminal)
        model.pendingToolCall = call
        // The real sites assign the chat id after the call.
        model.pendingToolCallChatID = chat.id
        await settle()
        XCTAssertEqual(backend.posted.count, 1)
        XCTAssertEqual(backend.posted[0].chatID, chat.id)
        XCTAssertEqual(backend.posted[0].title, "Release notes")
        XCTAssertTrue(backend.posted[0].body.contains("run_command"))
        XCTAssertFalse(backend.posted[0].body.contains("secret"), "arguments never reach a banner")
    }

    func testReplacingAnOpenCardDoesNotRaiseASecondCue() async {
        model.pendingToolCallChatID = chat.id
        model.pendingToolCall = AppToolCall(name: "a", arguments: [:], category: .terminal)
        await settle()
        model.pendingToolCall = AppToolCall(name: "b", arguments: [:], category: .terminal)
        await settle()
        XCTAssertEqual(backend.posted.count, 1)
    }

    func testQuestionRaisesNeedsAnswer() async {
        let waiter = Task { @MainActor in
            await self.model.waitForUserAnswers(
                chatID: self.chat.id,
                items: [UserQuestionItem(question: "Which?", header: "H", options: [], multiSelect: false)],
                toolCallID: nil)
        }
        await settle()
        XCTAssertEqual(backend.posted.map(\.body), [AppModel.attentionBody(.needsAnswer, tool: nil)])
        model.dismissUserQuestions()
        _ = await waiter.value
    }

    func testBackgroundChatParkingAProposalRaisesNeedsPermission() async {
        let other = AppChat(title: "Background")
        model.chats.append(other)
        let proposal = DeferredToolProposal(
            calls: [AppToolCall(name: "write_file", arguments: [:], category: .terminal)],
            fullContent: "", reasoning: "", result: Self.stubResult(), step: 1, project: nil)
        model.deferToolProposal(proposal, chatID: other.id)
        await settle()
        XCTAssertEqual(backend.posted.first?.chatID, other.id)
        XCTAssertTrue(backend.posted.first?.body.contains("write_file") ?? false)
    }

    func testIdleTurnTailRaisesTurnFinished() async {
        model.generating = true
        model.generationEpoch = 3
        model.finishTurnTail(myEpoch: 3, turnChatID: chat.id, session: nil)
        await settle()
        XCTAssertEqual(backend.posted.map(\.body), [AppModel.attentionBody(.turnFinished, tool: nil)])
    }

    func testACancelledTurnDoesNotRaiseTurnFinished() async {
        model.generating = true
        model.isCancellationPending = true
        model.generationEpoch = 3
        model.finishTurnTail(myEpoch: 3, turnChatID: chat.id, session: nil)
        await settle()
        XCTAssertTrue(backend.posted.isEmpty)
    }

    func testAStaleTailRaisesNothing() async {
        model.generating = true
        model.generationEpoch = 4
        model.finishTurnTail(myEpoch: 3, turnChatID: chat.id, session: nil)
        await settle()
        XCTAssertTrue(backend.posted.isEmpty)
    }

    func testATurnThatEndedInAnErrorRaisesErrorNotFinished() async {
        model.noteTurnStopped(chatID: chat.id, reason: "error")
        model.generating = true
        model.generationEpoch = 3
        model.finishTurnTail(myEpoch: 3, turnChatID: chat.id, session: nil)
        await settle()
        XCTAssertEqual(backend.posted.map(\.body), [AppModel.attentionBody(.error, tool: nil)])
    }

    func testContextOverflowCountsAsAnErrorButAUserStopDoesNot() async {
        model.noteTurnStopped(chatID: chat.id, reason: "cancelled")
        await settle()
        XCTAssertTrue(backend.posted.isEmpty)
        model.noteTurnStopped(chatID: chat.id, reason: "context_overflow")
        await settle()
        XCTAssertEqual(backend.posted.count, 1)
    }

    func testAChatParkedOnAnApprovalIsNotFinished() async {
        // Parked directly so only the finish path is under test.
        model.deferredToolProposals[chat.id] = DeferredToolProposal(
            calls: [AppToolCall(name: "write_file", arguments: [:], category: .terminal)],
            fullContent: "", reasoning: "", result: Self.stubResult(), step: 1, project: nil)
        model.generating = true
        model.generationEpoch = 3
        model.finishTurnTail(myEpoch: 3, turnChatID: chat.id, session: nil)
        await settle()
        // The admission step may replay the parked proposal for the selected
        // chat (which can raise its own cues); the tail itself must not call
        // the turn finished while the chat is still parked or running it.
        XCTAssertFalse(
            backend.posted.contains { $0.body == AppModel.attentionBody(.turnFinished, tool: nil) })
    }

    func testAToggledOffEventRaisesNothing() async {
        model.attentionPreferences[.turnFinished].enabled = false
        model.generating = true
        model.generationEpoch = 3
        model.finishTurnTail(myEpoch: 3, turnChatID: chat.id, session: nil)
        await settle()
        XCTAssertTrue(backend.posted.isEmpty)
    }

    func testAGhostChatNeverRaisesABanner() async {
        let ghost = AppChat(title: "Secret plans", isGhost: true)
        model.chats.append(ghost)
        await model.raiseAttention(.needsPermission, chatID: ghost.id, tool: "x").value
        XCTAssertTrue(backend.posted.isEmpty)
    }

    func testClickSelectsAnExistingChatAndIgnoresADeletedOne() {
        let other = AppChat(title: "Other")
        model.chats.append(other)
        XCTAssertTrue(model.selectChatFromNotification(other.id))
        XCTAssertEqual(model.selectedChatID, other.id)
        XCTAssertFalse(model.selectChatFromNotification(UUID()))
        XCTAssertEqual(model.selectedChatID, other.id)
    }

    func testTogglesPersistThroughTheSettingsFile() {
        model.attentionPreferences = AttentionPreferences()
        model.setAttentionEnabled(.needsAnswer, true)
        model.setAttentionSound(.needsAnswer, .ping)
        let reloaded = AppModel()
        reloaded.stopCronScheduler()
        XCTAssertTrue(reloaded.attentionPreferences[.needsAnswer].enabled)
        XCTAssertEqual(reloaded.attentionPreferences[.needsAnswer].soundChoice, .ping)
        XCTAssertFalse(reloaded.attentionPreferences[.turnFinished].enabled)
        // Leave the temp profile as the other tests expect it.
        reloaded.setAttentionEnabled(.needsAnswer, false)
        reloaded.setAttentionSound(.needsAnswer, .none)
    }
}
