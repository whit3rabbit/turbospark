import XCTest
import TurboSpark

@testable import TurboSparkApp

/// The batch router (`AppModel+AgentLoop.handleExtractedToolCalls`): an
/// all-agent batch executes every call and records them on ONE message, a
/// mixed batch keeps the one-call-per-turn rule (state#37), and a batch
/// under an asking permission parks AS ONE card whose approval runs (or
/// whose denial refuses) the whole batch. No live session is needed: a
/// subagent run without a session fails honestly, which is exactly the
/// result the recording has to carry.
@MainActor
final class SubagentBatchRoutingTests: XCTestCase {
    private func makeChat(_ appModel: AppModel) -> UUID {
        let chat = AppChat(title: "Batch")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id
        return chat.id
    }

    private func agentCall(_ prompt: String) -> AppToolCall {
        AppToolCall(name: "agent", arguments: ["prompt": prompt], category: .automation)
    }

    private var lastMessage: AppChatMessage? {
        appModel.chats.first?.messages.last
    }

    var appModel: AppModel!

    override func setUp() {
        super.setUp()
        appModel = AppModel()
    }

    override func tearDown() {
        appModel = nil
        super.tearDown()
    }

    /// `GenerationResult` is Decodable with no public memberwise init, so a
    /// fixture decodes one -- `stopReason: toolCalls` is what a reply that
    /// proposed calls carries.
    private func generationResult() -> GenerationResult {
        let json = """
        {"promptTokens": 0, "newTokens": 0, "reusedPrefixTokens": 0,
         "prefillSeconds": 0, "decodeSeconds": 0, "stopReason": "toolCalls",
         "tokensPerSecond": null, "content": "reply"}
        """
        return try! JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }

    func testTheAgentFamilyNamesAreExactlyTheThreeTheRegistryAnswers() {
        for name in ["agent", "subagent", "task"] {
            XCTAssertTrue(
                AppModel.isAgentFamilyToolName(name),
                "'\(name)' is an agent-family name.")
            XCTAssertTrue(
                AppModel.isAgentFamilyToolName(name.uppercased()),
                "The registry lowercases before it matches; so must the router.")
        }
        for name in ["list_directory", "run_command", "stop_agent", "agentx", ""] {
            XCTAssertFalse(
                AppModel.isAgentFamilyToolName(name),
                "'\(name)' is not an agent-family name.")
        }
    }

    /// A permissive project: the standard `.auto` preset asks for the
    /// `.automation` category, which is the parked-card path, not this one.
    private func permissiveProject() -> AppProject {
        AppProject(
            name: "p", rootDirectoryPath: "/tmp",
            permissions: AppProjectPermissions.preset(for: .permissive))
    }

    func testAnAllAgentBatchRunsEveryCallAndRecordsThemOnOneMessage() async {
        let chatID = makeChat(appModel)
        let first = agentCall("first task")
        let second = agentCall("second task")

        await appModel.handleExtractedToolCalls(
            [first, second], fullContent: "reply", reasoning: "",
            result: generationResult(),
            currentStep: 0, chatID: chatID, project: permissiveProject())

        XCTAssertEqual(appModel.chats[0].messages.count, 1, "The batch is ONE assistant turn.")
        let message = lastMessage
        XCTAssertEqual(message?.toolCalls.count, 2, "Both calls are recorded.")
        XCTAssertEqual(message?.toolResults.count, 2, "Both results are recorded.")
        // No session: each run fails honestly rather than silently.
        for result in message?.toolResults ?? [] {
            XCTAssertTrue(result.isError)
            XCTAssertTrue(
                result.output.contains("No active model session"),
                "A run that could not start says so. Got: \(result.output)")
        }
        for call in message?.toolCalls ?? [] {
            XCTAssertEqual(call.status, .failed)
        }
    }

    func testAMixedBatchStillRunsOnlyTheFirstCall() async {
        let chatID = makeChat(appModel)
        let fileCall = AppToolCall(
            name: "list_directory", arguments: ["path": "."], category: .fileRead)
        let first = agentCall("only agent in the mix")

        await appModel.handleExtractedToolCalls(
            [first, fileCall], fullContent: "reply", reasoning: "",
            result: generationResult(),
            currentStep: 0, chatID: chatID, project: permissiveProject())

        let message = lastMessage
        XCTAssertEqual(message?.toolCalls.count, 2, "The first call plus the deferred record.")
        XCTAssertEqual(message?.toolCalls.first?.name, "agent")
        XCTAssertEqual(
            message?.toolResults.count, 2,
            "The agent result plus the refusal for the second call.")
        XCTAssertTrue(
            message?.toolResults.last?.output.contains(TOOL_DEFERRED_MESSAGE) ?? false,
            "The non-agent call must carry the one-per-turn refusal, unchanged from state#37.")
    }

    func testABatchWhereEveryCallWouldAskParksTheWholeBatchUnderOneCard() async {
        let chatID = makeChat(appModel)
        // `agent` is category `.automation`; the standard preset asks for it,
        // so a batch must PARK AS ONE rather than fall back to single-call
        // (which would make the parallel path unreachable under default
        // permissions).
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(
            name: "p", rootDirectoryPath: dir.path,
            permissions: AppProjectPermissions(mode: .auto, automation: .ask))

        let first = agentCall("first task")
        let second = agentCall("second task")
        await appModel.handleExtractedToolCalls(
            [first, second], fullContent: "reply", reasoning: "",
            result: generationResult(),
            currentStep: 0, chatID: chatID, project: project)

        XCTAssertEqual(appModel.pendingToolCall?.id, first.id,
                       "The card renders the first call's verdict.")
        XCTAssertEqual(appModel.pendingBatchCalls?.count, 2,
                       "Approval acts on the WHOLE batch.")
        let message = lastMessage
        XCTAssertEqual(message?.toolCalls.count, 2)
        XCTAssertTrue(message?.toolResults.isEmpty ?? false,
                      "Nothing ran; the parked calls carry no results yet.")
        for call in message?.toolCalls ?? [] {
            XCTAssertEqual(call.status, .pendingApproval)
        }
    }

    func testApprovingAParkedBatchRunsEveryCallAndUpdatesInPlace() async throws {
        let chatID = makeChat(appModel)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(
            name: "p", rootDirectoryPath: dir.path,
            permissions: AppProjectPermissions(mode: .auto, automation: .ask))

        let first = agentCall("first task")
        let second = agentCall("second task")
        await appModel.handleExtractedToolCalls(
            [first, second], fullContent: "reply", reasoning: "",
            result: generationResult(),
            currentStep: 0, chatID: chatID, project: project)
        guard let parkedID = appModel.pendingToolCall?.id else {
            return XCTFail("The batch must park before it can be approved.")
        }

        appModel.approvePendingToolCall(id: parkedID)

        // approvePendingToolCall dispatches its work in a Task; give it a
        // short, bounded window. Both runs fail honestly (no session).
        let deadline = Date().addingTimeInterval(10)
        while ((lastMessage?.toolResults.count ?? 0) < 2) && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }

        XCTAssertEqual(appModel.chats[0].messages.count, 1,
                       "Approval UPDATES the parked message; it must not append a second turn.")
        XCTAssertEqual(lastMessage?.toolCalls.count, 2)
        XCTAssertEqual(lastMessage?.toolResults.count, 2)
        for result in lastMessage?.toolResults ?? [] {
            XCTAssertTrue(result.output.contains("No active model session"),
                          "Got: \(result.output)")
        }
        for call in lastMessage?.toolCalls ?? [] {
            XCTAssertEqual(call.status, .failed)
        }
    }

    func testDenyingAParkedBatchRefusesEveryCall() async {
        let chatID = makeChat(appModel)
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(
            name: "p", rootDirectoryPath: dir.path,
            permissions: AppProjectPermissions(mode: .auto, automation: .ask))

        let first = agentCall("first task")
        let second = agentCall("second task")
        await appModel.handleExtractedToolCalls(
            [first, second], fullContent: "reply", reasoning: "",
            result: generationResult(),
            currentStep: 0, chatID: chatID, project: project)
        guard let parkedID = appModel.pendingToolCall?.id else {
            return XCTFail("The batch must park before it can be denied.")
        }

        appModel.denyPendingToolCall(id: parkedID)
        // denyPendingToolCall appends its records synchronously before
        // spawning its notification task.
        XCTAssertEqual(appModel.chats[0].messages.count, 1)
        XCTAssertEqual(lastMessage?.toolCalls.count, 2)
        for call in lastMessage?.toolCalls ?? [] {
            XCTAssertEqual(call.status, .denied)
        }
        for result in lastMessage?.toolResults ?? [] {
            XCTAssertTrue(result.isError)
            XCTAssertTrue(result.output.contains(TOOL_REJECTED_MESSAGE),
                          "Got: \(result.output)")
        }
        XCTAssertNil(appModel.pendingBatchCalls)
    }
}
