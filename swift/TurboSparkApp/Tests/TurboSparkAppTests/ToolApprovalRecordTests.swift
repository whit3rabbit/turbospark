import XCTest

@testable import TurboSparkApp
@testable import TurboSpark

/// The resolved approval record (component audit, tool-approval): what the
/// card stores once the user answers, how old archives decode without it,
/// and the cwd and reason lines on a pending card. Display only: none of
/// this decides what a choice permits.
@MainActor
final class ToolApprovalRecordTests: XCTestCase {

    // MARK: - Persistence

    func testAnArchiveWrittenBeforeTheRecordDecodesWithoutOne() throws {
        let json = """
        {"id": "6F9619FF-8B86-D011-B42D-00C04FC964FF", "approvalID": "abc",
         "name": "run_command", "arguments": {"command": "ls"},
         "rawInvocation": "", "status": "completed", "category": "terminal",
         "createdAt": 700000000}
        """
        let call = try JSONDecoder().decode(AppToolCall.self, from: Data(json.utf8))
        XCTAssertNil(call.approvalRecord)
        XCTAssertEqual(call.status, .completed)
    }

    func testTheRecordRoundTrips() throws {
        var call = AppToolCall(name: "run_command", arguments: ["command": "ls"], category: .terminal)
        call.approvalRecord = ToolApprovalRecord(
            decision: .allowedForChat, actor: .user, decidedAt: Date(timeIntervalSince1970: 1_000))
        let data = try JSONEncoder().encode(call)
        let back = try JSONDecoder().decode(AppToolCall.self, from: data)
        XCTAssertEqual(back.approvalRecord, call.approvalRecord)
    }

    func testAnUnknownDecisionDropsTheRecordNotTheChat() throws {
        // A newer build may write a decision this one does not know. The
        // call (and so the whole chat) must still decode.
        let json = """
        {"name": "run_command", "status": "completed", "category": "terminal",
         "approvalRecord": {"decision": "allowedForever", "actor": "user", "decidedAt": 1}}
        """
        let call = try JSONDecoder().decode(AppToolCall.self, from: Data(json.utf8))
        XCTAssertNil(call.approvalRecord)
        XCTAssertEqual(call.name, "run_command")
    }

    // MARK: - Presentation

    func testEachDecisionHasItsOwnWordsAndSign() {
        func line(_ d: ToolApprovalDecision, _ a: ToolApprovalActor) -> ToolApprovalRecordLine {
            ToolApprovalRecordLine(ToolApprovalRecord(decision: d, actor: a, decidedAt: Date()))
        }
        XCTAssertEqual(line(.allowedOnce, .user).labelKey, "Allowed once by you")
        XCTAssertEqual(line(.allowedForChat, .user).labelKey, "Allowed for this chat by you")
        XCTAssertEqual(line(.allowedAlways, .user).labelKey, "Always allowed by you")
        XCTAssertEqual(line(.denied, .user).labelKey, "Denied by you")
        XCTAssertEqual(line(.denied, .policy).labelKey, "Denied by permission settings")
        XCTAssertTrue(line(.allowedAlways, .user).isAllowed)
        XCTAssertFalse(line(.denied, .user).isAllowed)
    }

    // MARK: - Pending card context

    func testTheReasonIsTheShellCommandsDescription() {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "rm -rf build", "description": "  Removes the build folder.\n"],
            category: .terminal)
        XCTAssertEqual(ToolApprovalContext.reason(for: call), "Removes the build folder.")
    }

    func testABlankReasonIsNoReason() {
        let call = AppToolCall(
            name: "run_command", arguments: ["command": "ls", "description": "  "], category: .terminal)
        XCTAssertNil(ToolApprovalContext.reason(for: call))
    }

    func testAnAgentsDescriptionIsNotAReason() {
        // The agent preview already shows its task description; repeating it
        // as "the model's reason" would mislabel the task as a justification.
        let call = AppToolCall(
            name: "agent", arguments: ["description": "Explore the repo"], category: .automation)
        XCTAssertNil(ToolApprovalContext.reason(for: call))
    }

    func testOtherToolsUseAnExplicitReasonArgument() {
        let call = AppToolCall(
            name: "write_file", arguments: ["path": "a.txt", "reason": "Saves the notes"],
            category: .fileWrite)
        XCTAssertEqual(ToolApprovalContext.reason(for: call), "Saves the notes")
    }

    func testAVeryLongReasonIsCapped() {
        let long = String(repeating: "a", count: 500)
        let call = AppToolCall(
            name: "run_command", arguments: ["command": "ls", "description": long], category: .terminal)
        let reason = ToolApprovalContext.reason(for: call)
        XCTAssertNotNil(reason)
        XCTAssertLessThanOrEqual(reason?.count ?? 0, ToolApprovalContext.maxReasonLength + 3)
        XCTAssertTrue(reason?.hasSuffix("...") ?? false)
    }

    func testTheDirectoryUnderHomeReadsWithATilde() {
        let home = URL(fileURLWithPath: "/Users/someone")
        let dir = URL(fileURLWithPath: "/Users/someone/code/app")
        XCTAssertEqual(ToolApprovalContext.displayPath(dir, home: home), "~/code/app")
        XCTAssertEqual(
            ToolApprovalContext.displayPath(URL(fileURLWithPath: "/opt/x"), home: home), "/opt/x")
        XCTAssertEqual(ToolApprovalContext.displayPath(home, home: home), "~")
    }

    // MARK: - Recording through the real approval path

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

    private func result() -> GenerationResult {
        let json = """
        {"promptTokens": 0, "newTokens": 0, "reusedPrefixTokens": 0,
         "prefillSeconds": 0, "decodeSeconds": 0, "stopReason": "toolCalls",
         "tokensPerSecond": null, "content": "reply"}
        """
        return try! JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }

    /// Proposes a harmless shell call in an always-ask project, leaving a card.
    private func proposeCard() async -> (AppChat, AppToolCall) {
        let project = AppProject(name: "p", rootDirectoryPath: "/tmp", permissions: .alwaysAsk)
        model.projects.append(project)
        let chat = AppChat(projectID: project.id, title: "c")
        model.chats.append(chat)
        model.selectedProjectID = project.id
        model.selectedChatID = chat.id
        let call = AppToolCall(
            name: "run_command", arguments: ["command": "echo hi"], category: .terminal)
        await model.handleExtractedToolCall(
            call, fullContent: "reply", reasoning: "", result: result(),
            currentStep: 0, chatID: chat.id, project: project)
        XCTAssertEqual(model.pendingToolCall?.id, call.id, "The call must stop at a card.")
        return (chat, call)
    }

    private func stored(_ id: UUID, in chat: AppChat) -> AppToolCall? {
        model.chats.first { $0.id == chat.id }?.messages.flatMap(\.toolCalls).last { $0.id == id }
    }

    func testAllowOnceIsRecordedAsTheUsersChoice() async {
        let (chat, call) = await proposeCard()
        XCTAssertNil(stored(call.id, in: chat)?.approvalRecord, "A pending card has no record.")
        model.approvePendingToolCall(id: call.id)
        await model.toolExecutionTask?.value
        let record = stored(call.id, in: chat)?.approvalRecord
        XCTAssertEqual(record?.decision, .allowedOnce)
        XCTAssertEqual(record?.actor, .user)
    }

    func testAllowForTheChatIsRecordedAsSuch() async {
        let (chat, call) = await proposeCard()
        model.approvePendingToolCall(id: call.id, alwaysAllowSession: true)
        await model.toolExecutionTask?.value
        XCTAssertEqual(stored(call.id, in: chat)?.approvalRecord?.decision, .allowedForChat)
    }

    func testAPersistedRuleIsRecordedAsAlwaysWithoutWideningTheApproval() async {
        let (chat, call) = await proposeCard()
        model.approvePendingToolCall(id: call.id, alwaysAllowSession: false, recordAs: .allowedAlways)
        await model.toolExecutionTask?.value
        XCTAssertEqual(stored(call.id, in: chat)?.approvalRecord?.decision, .allowedAlways)
    }

    func testDenyIsRecordedAndAPolicyDenyIsNotBlamedOnTheUser() async {
        let (chat, call) = await proposeCard()
        model.denyPendingToolCall(id: call.id)
        await model.toolExecutionTask?.value
        XCTAssertEqual(stored(call.id, in: chat)?.approvalRecord?.decision, .denied)
        XCTAssertEqual(stored(call.id, in: chat)?.approvalRecord?.actor, .user)

        let (chat2, call2) = await proposeCard()
        model.denyPendingToolCall(id: call2.id, policyReason: "terminal is denied")
        await model.toolExecutionTask?.value
        XCTAssertEqual(stored(call2.id, in: chat2)?.approvalRecord?.actor, .policy)
    }
}
