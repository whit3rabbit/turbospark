import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

/// Edit-before-run (component audit, edit-before-run): an edited command is
/// a NEW call that goes through the whole proposal ladder, never runs on the
/// edit itself, is approved once only, and tells the model what changed.
@MainActor
final class ToolCallCommandEditTests: XCTestCase {

    // MARK: - Pure: building the edited call

    private func shellCall(_ arguments: [String: String], name: String = "run_command") -> AppToolCall {
        AppToolCall(name: name, arguments: arguments, rawInvocation: "<raw/>", category: .terminal)
    }

    func testTheEditIsANewCallWithEveryOtherArgumentCopied() throws {
        var original = shellCall(["command": "ls -la", "timeout": "5000", "description": "List"])
        original.autoApprovedBy = "classifier"
        original.approvalRecord = ToolApprovalRecord(decision: .allowedOnce, actor: .user)
        let edited = try XCTUnwrap(ToolCallCommandEdit.editedCall(from: original, text: "  ls -l src\n"))

        XCTAssertNotEqual(edited.id, original.id)
        XCTAssertNotEqual(edited.approvalID, original.approvalID)
        XCTAssertEqual(edited.name, original.name)
        XCTAssertEqual(edited.category, .terminal)
        XCTAssertEqual(edited.arguments["command"], "ls -l src", "The edit is trimmed.")
        XCTAssertEqual(edited.arguments["timeout"], "5000")
        XCTAssertEqual(edited.arguments["description"], "List")
        XCTAssertEqual(edited.status, .pendingApproval)
        XCTAssertEqual(edited.editedFromCommand, "ls -la")
        XCTAssertNil(edited.riskAssessment, "The model's rating must not carry over.")
        XCTAssertNil(edited.autoApprovedBy)
        XCTAssertNil(edited.approvalRecord)
        XCTAssertEqual(edited.rawInvocation, "")
    }

    func testTheEditReplacesTheAliasTheExecutorReads() throws {
        let original = shellCall(["cmd": "make"], name: "bash")
        let edited = try XCTUnwrap(ToolCallCommandEdit.editedCall(from: original, text: "make test"))
        XCTAssertEqual(edited.shellCommand, "make test")
        XCTAssertNil(edited.arguments["command"], "No second key may shadow or bypass the alias.")
    }

    func testAnUnchangedOrEmptyEditIsACancel() {
        let original = shellCall(["command": "echo hi"])
        XCTAssertNil(ToolCallCommandEdit.editedCall(from: original, text: "echo hi"))
        XCTAssertNil(ToolCallCommandEdit.editedCall(from: original, text: "  echo hi \n"))
        XCTAssertNil(ToolCallCommandEdit.editedCall(from: original, text: " \n "))
    }

    func testOnlyShellAliasesAreEditable() {
        XCTAssertTrue(ToolCallCommandEdit.isEditableTool(shellCall(["command": "ls"], name: "Bash")))
        XCTAssertFalse(ToolCallCommandEdit.isEditableTool(
            AppToolCall(name: "write_file", arguments: ["path": "a", "command": "x"], category: .fileWrite)))
        XCTAssertFalse(ToolCallCommandEdit.isEditableTool(shellCall(["description": "no command"])))
        XCTAssertNil(ToolCallCommandEdit.editedCall(
            from: AppToolCall(name: "webfetch", arguments: ["command": "x"], category: .web), text: "y"))
    }

    func testANativeCallGetsItsOwnIdAndJSONThatSaysWhatRan() throws {
        var original = shellCall(["command": "ls", "timeout": "5000"])
        original.nativeCallID = "call_1"
        original.nativeArgumentsJSON = #"{"command":"ls","timeout":5000}"#
        let edited = try XCTUnwrap(ToolCallCommandEdit.editedCall(from: original, text: "ls src"))
        XCTAssertNotNil(edited.nativeCallID)
        XCTAssertNotEqual(edited.nativeCallID, "call_1")
        let json = try XCTUnwrap(edited.nativeArgumentsJSON)
        let object = try XCTUnwrap(
            JSONSerialization.jsonObject(with: Data(json.utf8)) as? [String: Any])
        XCTAssertEqual(object["command"] as? String, "ls src")
        XCTAssertEqual(object["timeout"] as? Int, 5000, "Other values keep their JSON type.")

        let textCall = try XCTUnwrap(
            ToolCallCommandEdit.editedCall(from: shellCall(["command": "ls"]), text: "ls src"))
        XCTAssertNil(textCall.nativeCallID)
        XCTAssertNil(textCall.nativeArgumentsJSON)
    }

    func testAnEditedCallAllowsNoStandingGrant() throws {
        let original = shellCall(["command": "ls"])
        XCTAssertTrue(ToolCallCommandEdit.allowsStandingGrant(original))
        let edited = try XCTUnwrap(ToolCallCommandEdit.editedCall(from: original, text: "ls src"))
        XCTAssertFalse(ToolCallCommandEdit.allowsStandingGrant(edited))
    }

    func testABatchOrCompoundCardOffersNoEdit() {
        XCTAssertEqual(ToolCallCommandEdit.unavailableReason(batchCount: 2, hasValidationLeg: false), .batch)
        XCTAssertEqual(ToolCallCommandEdit.unavailableReason(batchCount: nil, hasValidationLeg: true), .compound)
        XCTAssertNil(ToolCallCommandEdit.unavailableReason(batchCount: nil, hasValidationLeg: false))
        XCTAssertNil(ToolCallCommandEdit.unavailableReason(batchCount: 1, hasValidationLeg: false))
    }

    func testTheResultNamesBothCommands() throws {
        let original = shellCall(["command": "ls"])
        let edited = try XCTUnwrap(ToolCallCommandEdit.editedCall(from: original, text: "ls src"))
        let raw = AppToolResult(callID: edited.id, output: "a.swift", isError: false, durationSeconds: 0.1)
        let annotated = ToolCallCommandEdit.annotate(raw, for: edited)
        XCTAssertTrue(annotated.output.contains("Your command: ls\n"))
        XCTAssertTrue(annotated.output.contains("ls src"))
        XCTAssertTrue(annotated.output.hasSuffix("a.swift"))
        XCTAssertEqual(ToolCallCommandEdit.annotate(raw, for: original), raw, "An unedited result is untouched.")

        let superseded = ToolCallCommandEdit.supersededResult(for: original, editedCommand: "ls src")
        XCTAssertEqual(superseded.callID, original.id)
        XCTAssertTrue(superseded.output.hasPrefix("Not run"))
        XCTAssertTrue(superseded.output.contains("ls src"))
    }

    // MARK: - Persistence

    func testOldArchivesDecodeAndTheNewFieldsRoundTrip() throws {
        let old = """
        {"name": "run_command", "arguments": {"command": "ls"}, "status": "denied", "category": "terminal",
         "approvalRecord": {"decision": "denied", "actor": "user", "decidedAt": 1}}
        """
        let decoded = try JSONDecoder().decode(AppToolCall.self, from: Data(old.utf8))
        XCTAssertNil(decoded.editedFromCommand)
        XCTAssertEqual(decoded.approvalRecord?.decision, .denied)

        var call = shellCall(["command": "ls src"])
        call.editedFromCommand = "ls"
        call.approvalRecord = ToolApprovalRecord(
            decision: .editedByUser, actor: .user, decidedAt: Date(timeIntervalSince1970: 5))
        let back = try JSONDecoder().decode(AppToolCall.self, from: JSONEncoder().encode(call))
        XCTAssertEqual(back.editedFromCommand, "ls")
        XCTAssertEqual(back.approvalRecord?.decision, .editedByUser)
        let line = ToolApprovalRecordLine(try XCTUnwrap(back.approvalRecord))
        XCTAssertEqual(line.labelKey, "Edited by you, not run")
        XCTAssertTrue(line.isEdited)
        XCTAssertFalse(line.isAllowed)
    }

    // MARK: - Through the real proposal and approval paths

    private var model: AppModel!
    private var root: URL!

    override func setUp() {
        super.setUp()
        model = AppModel()
        model.interactionMode = .projects
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("edit-before-run-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let root { try? FileManager.default.removeItem(at: root) }
        model = nil
        root = nil
        super.tearDown()
    }

    private func generationResult() -> GenerationResult {
        let json = """
        {"promptTokens": 0, "newTokens": 0, "reusedPrefixTokens": 0,
         "prefillSeconds": 0, "decodeSeconds": 0, "stopReason": "toolCalls",
         "tokensPerSecond": null, "content": "reply"}
        """
        return try! JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }

    private func makeProject(_ permissions: AppProjectPermissions) -> (AppProject, AppChat) {
        let project = AppProject(name: "p", rootDirectoryPath: root.path, permissions: permissions)
        model.projects.append(project)
        let chat = AppChat(projectID: project.id, title: "c")
        model.chats.append(chat)
        model.selectedProjectID = project.id
        model.selectedChatID = chat.id
        return (project, chat)
    }

    /// A model proposal in an always-ask project, stopped at its card.
    private func proposeCard(_ command: String) async -> (AppProject, AppChat, AppToolCall) {
        let (project, chat) = makeProject(.alwaysAsk)
        let call = AppToolCall(name: "run_command", arguments: ["command": command], category: .terminal)
        await model.handleExtractedToolCall(
            call, fullContent: "reply", reasoning: "", result: generationResult(),
            currentStep: 0, chatID: chat.id, project: project)
        XCTAssertEqual(model.pendingToolCall?.id, call.id, "The proposal must stop at a card.")
        return (project, chat, call)
    }

    private func stored(_ id: UUID, in chat: AppChat) -> AppToolCall? {
        model.chats.first { $0.id == chat.id }?.messages.flatMap(\.toolCalls).last { $0.id == id }
    }

    private func storedResult(_ id: UUID, in chat: AppChat) -> AppToolResult? {
        model.chats.first { $0.id == chat.id }?.messages.flatMap(\.toolResults).last { $0.callID == id }
    }

    private func exists(_ name: String) -> Bool {
        FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path)
    }

    func testAnEditCreatesANewCardAndRunsNothing() async throws {
        let (_, chat, original) = await proposeCard("touch original_ran")
        XCTAssertTrue(model.submitEditedPendingToolCall(id: original.id, command: "touch edited_ran"))
        await model.toolExecutionTask?.value

        let pending = try XCTUnwrap(model.pendingToolCall)
        XCTAssertNotEqual(pending.id, original.id)
        XCTAssertEqual(pending.shellCommand, "touch edited_ran")
        XCTAssertEqual(pending.status, .pendingApproval)
        XCTAssertNotNil(stored(pending.id, in: chat), "The new card is on the transcript.")
        XCTAssertFalse(exists("edited_ran"), "An edit must not run the edited command.")
        XCTAssertFalse(exists("original_ran"))

        let superseded = try XCTUnwrap(stored(original.id, in: chat))
        XCTAssertEqual(superseded.status, .denied)
        XCTAssertEqual(superseded.approvalRecord?.decision, .editedByUser)
        XCTAssertTrue(storedResult(original.id, in: chat)?.output.hasPrefix("Not run") ?? false)
    }

    func testApprovingRunsTheEditedCommandOnceAndNeverTheOriginal() async throws {
        let (_, chat, original) = await proposeCard("touch original_ran")
        model.submitEditedPendingToolCall(id: original.id, command: "touch edited_ran")
        await model.toolExecutionTask?.value
        let edited = try XCTUnwrap(model.pendingToolCall)
        let sessionID = chat.id.uuidString
        await SessionApprovalStore.shared.clear(sessionID: sessionID)

        // A stray "for this chat" request must still approve once only.
        model.approvePendingToolCall(id: edited.id, alwaysAllowSession: true, recordAs: .allowedAlways)
        await model.toolExecutionTask?.value

        XCTAssertTrue(exists("edited_ran"), "Approve runs the EDITED command.")
        XCTAssertFalse(exists("original_ran"), "The original proposal never runs.")
        XCTAssertEqual(stored(edited.id, in: chat)?.approvalRecord?.decision, .allowedOnce)
        let output = try XCTUnwrap(storedResult(edited.id, in: chat)?.output)
        XCTAssertTrue(output.contains("Your command: touch original_ran"))
        XCTAssertTrue(output.contains("touch edited_ran"))
        let granted = await SessionApprovalStore.shared.isApproved(
            sessionID: sessionID, toolName: "run_command", command: "touch edited_ran")
        XCTAssertFalse(granted, "An edited command gets no session grant.")
    }

    func testAHighRiskEditIsClassifiedHigh() async throws {
        let (_, _, original) = await proposeCard("echo hi")
        model.submitEditedPendingToolCall(id: original.id, command: "rm -rf build")
        await model.toolExecutionTask?.value
        let pending = try XCTUnwrap(model.pendingToolCall)
        XCTAssertEqual(pending.riskAssessment?.level, .high)
    }

    func testAPolicyDeniedEditIsDenied() async throws {
        let (project, chat, original) = await proposeCard("echo hi")
        // The policy tightened while the card waited: the edit is a new
        // decision point and must meet the live policy.
        if let index = model.projects.firstIndex(where: { $0.id == project.id }) {
            model.projects[index].permissions = .readOnly
        }
        model.submitEditedPendingToolCall(id: original.id, command: "touch edited_ran")
        await model.toolExecutionTask?.value

        XCTAssertNil(model.pendingToolCall, "A refused edit shows no card.")
        let edited = try XCTUnwrap(
            model.chats.first { $0.id == chat.id }?.messages.flatMap(\.toolCalls)
                .last { $0.editedFromCommand != nil })
        XCTAssertEqual(edited.status, .denied)
        XCTAssertEqual(edited.approvalRecord?.actor, .policy)
        let output = try XCTUnwrap(storedResult(edited.id, in: chat)?.output)
        XCTAssertTrue(output.contains("Tool execution denied"))
        XCTAssertTrue(output.contains("Your command: echo hi"))
        XCTAssertFalse(exists("edited_ran"))
    }

    /// The no-auto-run guard: an engine `.allow` still parks an edit.
    func testAnEditTheEngineWouldAllowStillWaitsForApproval() async throws {
        let (project, chat) = makeProject(.fullAccess)
        let original = AppToolCall(
            name: "run_command", arguments: ["command": "echo hi"], category: .terminal)
        model.mutateTurnMessages(for: chat.id) {
            $0.append(AppChatMessage(
                role: .assistant, content: "reply", reasoning: "", stopReason: "tool_use",
                toolCalls: [original], toolResults: []))
        }
        model.pendingToolCall = original
        model.pendingToolCallChatID = chat.id
        model.pendingToolCallProject = project

        XCTAssertTrue(model.submitEditedPendingToolCall(id: original.id, command: "touch edited_ran"))
        await model.toolExecutionTask?.value

        XCTAssertFalse(exists("edited_ran"), "Full access must not auto-run a user-edited command.")
        XCTAssertEqual(model.pendingToolCall?.shellCommand, "touch edited_ran")
    }

    func testAnUnchangedEditOrABatchChangesNothing() async throws {
        let (_, _, original) = await proposeCard("echo hi")
        XCTAssertFalse(model.submitEditedPendingToolCall(id: original.id, command: "  echo hi\n"))
        XCTAssertEqual(model.pendingToolCall?.id, original.id, "Unchanged behaves like Cancel.")
        XCTAssertNil(model.toolExecutionTask)

        let other = AppToolCall(name: "run_command", arguments: ["command": "ls"], category: .terminal)
        model.pendingBatchCalls = [original, other]
        XCTAssertEqual(model.pendingCommandEditUnavailableReason(for: original), .batch)
        XCTAssertFalse(model.submitEditedPendingToolCall(id: original.id, command: "echo bye"))
        XCTAssertEqual(model.pendingToolCall?.id, original.id)
    }
}
