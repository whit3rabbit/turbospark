import XCTest
@testable import TurboSparkApp

/// Regression tests for the agent-loop turn-routing cluster (state#6, #7,
/// #9): every append (`appendToolExecutionTurn`, `approvePendingToolCall`,
/// `denyPendingToolCall`) used to resolve its target chat from whatever
/// `selectedChatID` was AT THE MOMENT OF APPENDING rather than from the chat
/// the turn actually belongs to, and `continueAgentLoop()`'s default `step: 1`
/// reset `maxAutonomousSteps` on every approval. `executeGenerationTurn`
/// itself needs a live model session to exercise (state#8's hook-step fix,
/// state#10's epoch guard), so those two are verified by inspection/mutation
/// reasoning rather than here.
@MainActor
final class AgentLoopRoutingTests: XCTestCase {
    private func makeTwoChats(_ appModel: AppModel) -> (a: UUID, b: UUID) {
        let chatA = AppChat(title: "Chat A")
        let chatB = AppChat(title: "Chat B")
        appModel.chats = [chatA, chatB]
        appModel.selectedChatID = chatA.id
        return (chatA.id, chatB.id)
    }

    // MARK: - state#9: appendToolExecutionTurn routes by explicit chatID

    func testAppendToolExecutionTurnTargetsTheGivenChatNotTheSelectedOne() {
        let appModel = AppModel()
        let (chatA, chatB) = makeTwoChats(appModel)
        appModel.selectedChatID = chatB // user has switched away from chat A

        let call = AppToolCall(name: "list_directory", arguments: ["path": "."], category: .fileRead)
        let result = AppToolResult(callID: call.id, output: "ok")
        appModel.appendToolExecutionTurn(call: call, result: result, chatID: chatA)

        let indexA = appModel.chats.firstIndex(where: { $0.id == chatA })!
        let indexB = appModel.chats.firstIndex(where: { $0.id == chatB })!
        XCTAssertEqual(appModel.chats[indexA].messages.count, 1, "The tool turn must land in the ORIGINATING chat.")
        XCTAssertEqual(appModel.chats[indexB].messages.count, 0, "The chat the user switched TO must be untouched.")
    }

    // MARK: - state#6 / state#9: denyPendingToolCall resumes at the right chat and step

    func testDenyPendingToolCallAppendsToTheOriginatingChatEvenAfterSwitchingAway() {
        let appModel = AppModel()
        let (chatA, chatB) = makeTwoChats(appModel)

        let call = AppToolCall(name: "run_command", arguments: ["command": "rm -rf /"], category: .terminal)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chatA
        appModel.pendingToolCallStep = 3

        // User switches to chat B before responding. Under the OLD code,
        // `appendToolExecutionTurn` had no `chatID` parameter and always
        // resolved `selectedChatIndex` live, so this deny would have landed
        // in chat B instead of chat A.
        appModel.selectedChatID = chatB

        appModel.denyPendingToolCall(id: call.id)

        let indexA = appModel.chats.firstIndex(where: { $0.id == chatA })!
        let indexB = appModel.chats.firstIndex(where: { $0.id == chatB })!
        XCTAssertEqual(appModel.chats[indexA].messages.count, 1, "The denial must land in the chat the call was PROPOSED in.")
        XCTAssertEqual(appModel.chats[indexB].messages.count, 0)
        XCTAssertNil(appModel.pendingToolCall)
        XCTAssertNil(appModel.pendingToolCallChatID)
    }

    func testApprovePendingToolCallResumesInTheOriginatingChat() async throws {
        let appModel = AppModel()
        let (chatA, chatB) = makeTwoChats(appModel)

        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let project = AppProject(name: "test", rootDirectoryPath: dir.path)

        let call = AppToolCall(name: "list_directory", arguments: ["path": "."], category: .fileRead)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chatA
        appModel.pendingToolCallStep = 2
        appModel.selectedProjectID = project.id
        appModel.projects = [project]

        appModel.selectedChatID = chatB // switch away before approving

        appModel.approvePendingToolCall(id: call.id)

        // approvePendingToolCall dispatches its work in a Task; give it a
        // short, bounded window to complete (the tool itself is a fast
        // local directory listing).
        let deadline = Date().addingTimeInterval(5)
        while (appModel.chats.first(where: { $0.id == chatA })?.messages.isEmpty ?? true) && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        let indexA = appModel.chats.firstIndex(where: { $0.id == chatA })!
        let indexB = appModel.chats.firstIndex(where: { $0.id == chatB })!
        XCTAssertEqual(appModel.chats[indexA].messages.count, 1, "The approved call's result must land in the chat it was proposed in.")
        XCTAssertEqual(appModel.chats[indexB].messages.count, 0)
    }

    // MARK: - state#9 (UX half): switching chats is blocked mid-approval

    func testSelectChatIsBlockedWhileATooCallIsPendingApproval() {
        let appModel = AppModel()
        let (chatA, chatB) = makeTwoChats(appModel)
        appModel.pendingToolCall = AppToolCall(name: "run_command", arguments: ["command": "ls"], category: .terminal)

        appModel.selectChat(id: chatB)
        XCTAssertEqual(appModel.selectedChatID, chatA, "Switching chats must be refused while a tool call awaits approval.")

        appModel.pendingToolCall = nil
        appModel.selectChat(id: chatB)
        XCTAssertEqual(appModel.selectedChatID, chatB, "Switching must work normally once nothing is pending.")
    }

    // MARK: - state#7: selectedChat is a pure getter

    func testSelectedChatDoesNotMutateSelectedChatIDWhenItDoesNotMatchAnyChat() {
        let appModel = AppModel()
        let chat = AppChat(title: "Only Chat")
        appModel.chats = [chat]
        let driftedID = UUID() // does not match `chat.id`
        appModel.selectedChatID = driftedID

        // Reading `selectedChat` repeatedly must never mutate `selectedChatID`
        // as a side effect (state#7: SwiftUI forbids publishing changes from
        // within a view update, and this getter is read from view bodies).
        _ = appModel.selectedChat
        _ = appModel.selectedChat
        _ = appModel.selectedChat
        XCTAssertEqual(appModel.selectedChatID, driftedID, "The getter must not silently repoint selectedChatID to chats.first.")
    }

    func testSelectedChatReturnsTheMatchingChatWithoutMutation() {
        let appModel = AppModel()
        let (chatA, chatB) = makeTwoChats(appModel)
        appModel.selectedChatID = chatB

        let result = appModel.selectedChat
        XCTAssertEqual(result.id, chatB)
        XCTAssertEqual(appModel.selectedChatID, chatB, "Reading selectedChat must not change selectedChatID.")
        _ = chatA
    }
}
