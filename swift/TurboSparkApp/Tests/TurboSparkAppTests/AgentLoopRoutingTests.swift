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

    // MARK: - Always Allow on a shell call grants the command, not the shell

    func testAlwaysAllowOnAShellCallGrantsOnlyThatCommand() async throws {
        let appModel = AppModel()
        let (chatA, _) = makeTwoChats(appModel)
        let sessionID = chatA.uuidString
        await SessionApprovalStore.shared.clear(sessionID: sessionID)
        defer { Task { await SessionApprovalStore.shared.clear(sessionID: sessionID) } }

        // No project: if it runs, the rootless refusal stops it harmlessly.
        let call = AppToolCall(name: "run_command", arguments: ["command": "ls"], category: .terminal)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chatA

        appModel.approvePendingToolCall(id: call.id, alwaysAllowSession: true)

        let deadline = Date().addingTimeInterval(5)
        while appModel.toolExecutionTask != nil && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        let store = SessionApprovalStore.shared
        let sameCommand = await store.isApproved(sessionID: sessionID, toolName: "run_command", command: "ls")
        let withArgs = await store.isApproved(sessionID: sessionID, toolName: "run_command", command: "ls -la")
        let otherCommand = await store.isApproved(sessionID: sessionID, toolName: "run_command", command: "git push")
        XCTAssertTrue(sameCommand)
        XCTAssertTrue(withArgs, "The prefix grant still covers the same command with more arguments.")
        XCTAssertFalse(
            otherCommand,
            "A grant on the shell tool's NAME covered every allowlisted command, `git push` included.")
    }

    /// A grant given under a looser policy must not survive a mode change,
    /// while a rule added from an approval card leaves the others alone.
    func testChangingTheProjectPolicyDropsSessionGrants() async throws {
        let appModel = AppModel()
        let project = AppProject(name: "grants", permissions: .standard)
        appModel.projects = [project]
        var chat = AppChat(title: "grant chat")
        chat.projectID = project.id
        appModel.chats = [chat]
        let sessionID = chat.id.uuidString
        let store = SessionApprovalStore.shared
        defer { Task { await store.clear(sessionID: sessionID) } }

        await store.allowTool(sessionID: sessionID, toolName: "write_file")

        var withRule = project
        withRule.permissions.mcpAllowRules = ["mcp__docs__search"]
        appModel.updateProject(withRule)
        try await Task.sleep(nanoseconds: 100_000_000)
        let keptAfterRule = await store.isApproved(sessionID: sessionID, toolName: "write_file")
        XCTAssertTrue(keptAfterRule, "Adding an allow rule is not a policy change.")

        var stricter = withRule
        stricter.permissions.mode = .ask
        appModel.updateProject(stricter)
        let deadline = Date().addingTimeInterval(2)
        var stillApproved = true
        while stillApproved && Date() < deadline {
            stillApproved = await store.isApproved(sessionID: sessionID, toolName: "write_file")
            if stillApproved { try await Task.sleep(nanoseconds: 20_000_000) }
        }
        XCTAssertFalse(stillApproved, "A grant from the Auto policy kept answering under Ask.")
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
