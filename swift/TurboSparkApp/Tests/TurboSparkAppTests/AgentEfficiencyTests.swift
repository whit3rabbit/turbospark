import XCTest

@testable import TurboSpark
@testable import TurboSparkApp

@MainActor
final class AgentEfficiencyTests: XCTestCase {
    private func temporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("agent-efficiency-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func generationResult() throws -> GenerationResult {
        try JSONDecoder().decode(GenerationResult.self, from: Data(
            "{\"promptTokens\":1,\"newTokens\":1,\"prefillSeconds\":0,\"decodeSeconds\":0,\"stopReason\":\"toolCalls\",\"content\":\"\"}".utf8))
    }

    func testValidationUsesFlatArgumentsAndRecognizesOnlyMutationTools() {
        let write = AppToolCall(
            name: "write_file",
            arguments: ["path": "a.txt", "content": "new", "validate_command": "swift test", "validate_timeout_ms": "9000"],
            category: .fileWrite)
        XCTAssertEqual(ToolActionFusion.validation(for: write)?.command, "swift test")
        XCTAssertEqual(ToolActionFusion.validation(for: write)?.timeoutMs, 9000)
        XCTAssertNil(ToolActionFusion.validation(for: AppToolCall(
            name: "read_file", arguments: ["validate_command": "false"], category: .fileRead)))
    }

    func testPatchTargetsAndFingerprintRulesCoverAddUpdateAndDelete() {
        let patch = """
        *** Add File: added.txt
        +new
        *** Update File: updated.txt
        @@
        *** Delete File: removed.txt
        """
        let call = AppToolCall(name: "apply_patch", arguments: ["patch_text": patch], category: .fileWrite)
        XCTAssertEqual(ToolActionFusion.targetPaths(for: call), ["added.txt", "updated.txt", "removed.txt"])
        let before = [
            ToolActionFusion.Fingerprint(path: "added.txt", exists: false, hash: nil),
            ToolActionFusion.Fingerprint(path: "updated.txt", exists: true, hash: "old"),
            ToolActionFusion.Fingerprint(path: "removed.txt", exists: true, hash: "old"),
        ]
        let after = [
            ToolActionFusion.Fingerprint(path: "added.txt", exists: true, hash: "new"),
            ToolActionFusion.Fingerprint(path: "updated.txt", exists: true, hash: "new"),
            ToolActionFusion.Fingerprint(path: "removed.txt", exists: false, hash: nil),
        ]
        XCTAssertTrue(ToolActionFusion.postMutationVerified(before: before, after: after, for: call))
        XCTAssertFalse(ToolActionFusion.postMutationVerified(
            before: before, after: [after[0], before[1], after[2]], for: call))
    }

    func testFusedWriteMutatesBeforeItsValidationCommand() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel()
        let chat = AppChat(title: "fusion")
        model.chats = [chat]
        model.selectedChatID = chat.id
        let project = AppProject(name: "fusion", rootDirectoryPath: root.path, permissions: .fullAccess)
        let call = AppToolCall(
            name: "write_file",
            arguments: [
                "path": "created.txt", "content": "ready",
                "validate_command": "test -f created.txt", "validate_timeout_ms": "10000",
            ], category: .fileWrite)
        let fingerprints = ToolActionFusion.fingerprints(for: call, rootURL: root)
        let executed = await model.executeApprovedTool(
            call, project: project, chatID: chat.id, preMutationFingerprints: fingerprints)
        XCTAssertFalse(executed.result.isError, executed.result.output)
        XCTAssertEqual(try String(contentsOf: root.appendingPathComponent("created.txt")), "ready")
        XCTAssertEqual(executed.result.efficiency?.validation, "passed")
        XCTAssertTrue(executed.result.output.contains("fused_validation"))
    }

    func testValidationPreflightDenialPreventsTheMutation() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel()
        let chat = AppChat(title: "preflight")
        model.chats = [chat]
        model.selectedChatID = chat.id
        let permissions = AppProjectPermissions(mode: .auto, fileWrite: .allow, terminal: .deny)
        let project = AppProject(name: "preflight", rootDirectoryPath: root.path, permissions: permissions)
        let call = AppToolCall(
            name: "write_file",
            arguments: ["path": "blocked.txt", "content": "never", "validate_command": "true"],
            category: .fileWrite)
        await model.handleExtractedToolCall(
            call, fullContent: "write", reasoning: "", result: try generationResult(),
            currentStep: 0, chatID: chat.id, project: project)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("blocked.txt").path))
        XCTAssertEqual(model.chats[0].messages.last?.toolCalls.first?.status, .denied)
    }

    func testValidationPreToolUseHookRunsBeforeTheMutation() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let hook = AppHookCommand(
            name: "block validation", event: .preToolUse, type: .command,
            command: "echo blocked >&2; exit 2", matcher: "run_command", sourceType: .custom)
        AppHookStore.shared.addCustomHook(hook)
        defer { AppHookStore.shared.deleteCustomHook(id: hook.id) }

        let model = AppModel()
        let chat = AppChat(title: "hook preflight")
        model.chats = [chat]
        model.selectedChatID = chat.id
        let permissions = AppProjectPermissions(mode: .auto, fileWrite: .allow, terminal: .allow)
        let project = AppProject(name: "hook preflight", rootDirectoryPath: root.path, permissions: permissions)
        let call = AppToolCall(
            name: "write_file",
            arguments: ["path": "hooked.txt", "content": "never", "validate_command": "true"],
            category: .fileWrite)
        await model.handleExtractedToolCall(
            call, fullContent: "write", reasoning: "", result: try generationResult(),
            currentStep: 0, chatID: chat.id, project: project)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("hooked.txt").path))
        XCTAssertEqual(model.chats[0].messages.last?.toolCalls.first?.status, .denied)
    }

    func testValidationAndMutationParkBehindOneCompoundApproval() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let model = AppModel()
        let chat = AppChat(title: "compound")
        model.chats = [chat]
        model.selectedChatID = chat.id
        let permissions = AppProjectPermissions(mode: .auto, fileWrite: .allow, terminal: .ask)
        let project = AppProject(name: "compound", rootDirectoryPath: root.path, permissions: permissions)
        let call = AppToolCall(
            name: "write_file",
            arguments: ["path": "waiting.txt", "content": "not yet", "validate_command": "true"],
            category: .fileWrite)
        await model.handleExtractedToolCall(
            call, fullContent: "write", reasoning: "", result: try generationResult(),
            currentStep: 0, chatID: chat.id, project: project)
        XCTAssertNotNil(model.pendingToolCall)
        XCTAssertNotNil(model.pendingValidationCall)
        XCTAssertTrue(model.pendingToolCall?.riskAssessment?.reasons.contains(where: {
            $0.contains("Fused validation")
        }) ?? false)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("waiting.txt").path))
    }

    func testMutationHookAllowCannotAuthorizeValidationHookAsk() async throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let validationHook = AppHookCommand(
            name: "ask for validation", event: .permissionRequest, type: .command,
            command: "echo '{\"permissionDecision\":\"ask\"}'",
            matcher: "run_command", sourceType: .custom)
        let mutationHook = AppHookCommand(
            name: "allow writes", event: .permissionRequest, type: .command,
            command: "echo '{\"permissionDecision\":\"allow\"}'",
            matcher: "write_file", sourceType: .custom)
        AppHookStore.shared.addCustomHook(validationHook)
        AppHookStore.shared.addCustomHook(mutationHook)
        defer {
            AppHookStore.shared.deleteCustomHook(id: validationHook.id)
            AppHookStore.shared.deleteCustomHook(id: mutationHook.id)
        }

        let model = AppModel()
        let chat = AppChat(title: "independent permissions")
        model.chats = [chat]
        model.selectedChatID = chat.id
        let permissions = AppProjectPermissions(mode: .auto, fileWrite: .allow, terminal: .ask)
        let project = AppProject(
            name: "independent permissions", rootDirectoryPath: root.path,
            permissions: permissions)
        let call = AppToolCall(
            name: "write_file",
            arguments: [
                "path": "waiting.txt", "content": "not yet", "validate_command": "true",
            ], category: .fileWrite)

        await model.handleExtractedToolCall(
            call, fullContent: "write", reasoning: "", result: try generationResult(),
            currentStep: 0, chatID: chat.id, project: project)

        XCTAssertNotNil(model.pendingToolCall)
        XCTAssertNotNil(model.pendingValidationCall)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: root.appendingPathComponent("waiting.txt").path))
    }

    func testObservationStoreReplaysExactBytePagesAndCleansUp() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ToolObservationStore(rootURL: root)
        let chatID = UUID()
        let source = Data("abcdef012345".utf8)
        let reference = try store.archive(source, chatID: chatID)
        XCTAssertEqual(try store.recall(reference, chatID: chatID, offsetBytes: 3, maxBytes: 4), "def0")
        XCTAssertThrowsError(try store.recall(reference, chatID: chatID, offsetBytes: 99, maxBytes: 4))
        XCTAssertEqual(try ToolObservationStore(rootURL: root).load(reference, chatID: chatID), source)
        store.delete(chatID: chatID)
        XCTAssertThrowsError(try store.load(reference, chatID: chatID))
    }

    func testObservationStoreSeparatesChats() throws {
        let root = try temporaryDirectory()
        defer { try? FileManager.default.removeItem(at: root) }
        let store = ToolObservationStore(rootURL: root)
        let first = UUID()
        let second = UUID()
        let reference = try store.archive(Data("private".utf8), chatID: first)
        XCTAssertThrowsError(try store.load(reference, chatID: second))
    }

    func testGhostObservationStaysInTheEncryptedPayload() {
        let model = AppModel()
        var ghost = AppChat(title: "Ghost")
        ghost.isGhost = true
        model.chats = [ghost]
        model.selectedChatID = ghost.id
        let result = AppToolResult(
            callID: UUID(), output: "shown", archivalOutput: String(repeating: "x", count: 12_000))
        let archived = model.archiveToolObservation(result, chatID: ghost.id)
        XCTAssertNotNil(archived.observation)
        XCTAssertEqual(model.ghostPayload(for: ghost.id).toolObservations.count, 1)
    }

    func testProjectionBecomesStableHeadTailAfterTwoPromptSends() {
        let model = AppModel()
        var chat = AppChat(title: "normal")
        let call = AppToolCall(name: "run_command", arguments: ["command": "build"], category: .terminal)
        let original = AppToolResult(
            callID: call.id, output: "shown", archivalOutput: String(repeating: "a", count: 12_000))
        let archived = model.archiveToolObservation(original, chatID: chat.id)
        chat.messages = [AppChatMessage(
            role: .assistant, content: "", stopReason: "tool_use", toolCalls: [call], toolResults: [archived])]
        model.chats = [chat]
        model.selectedChatID = chat.id

        model.prepareToolOutputProjectionsForPrompt(chatID: chat.id)
        model.prepareToolOutputProjectionsForPrompt(chatID: chat.id)
        model.prepareToolOutputProjectionsForPrompt(chatID: chat.id)
        let result = model.chats[0].messages[0].toolResults[0]
        XCTAssertEqual(result.fullPromptSendCount, ToolOutputProjection.fullSendLimit)
        XCTAssertTrue(result.modelOutput.contains("Archived tool output"))
        let history = model.buildAppendOnlyHistory(chatIndex: 0, project: nil)
        let estimate = model.buildEstimateParts().history
        XCTAssertEqual(history.filter { $0.role == .tool }.map(\.content), estimate.filter { $0.role == .tool }.map(\.content))
    }

    func testRecallToolOutputUsesTheRegistryAndTheCallingChat() async throws {
        let model = AppModel()
        var chat = AppChat(title: "recall")
        let original = AppToolResult(
            callID: UUID(), output: "shown", archivalOutput: String(repeating: "z", count: 12_000))
        let archived = model.archiveToolObservation(original, chatID: chat.id)
        chat.messages = [AppChatMessage(role: .assistant, content: "", toolResults: [archived])]
        model.chats = [chat]
        model.selectedChatID = chat.id
        let call = AppToolCall(
            name: "recall_tool_output",
            arguments: [
                "observation_id": try XCTUnwrap(archived.observation?.id.uuidString),
                "offset_bytes": "9", "max_bytes": "5",
            ], category: .fileRead)
        let response = await AppToolRegistry.execute(call: call, in: nil, chatID: chat.id)
        XCTAssertFalse(response.isError, response.output)
        XCTAssertEqual(response.output, "zzzzz")
    }

    func testReducerRequiresExactHashStatusAndQuotes() throws {
        let source = "compile error at line 4\nmissing semicolon"
        let hash = ToolObservationStore.hash(Data(source.utf8))
        let receipt = ToolOutputReducer.Receipt(
            source_hash: hash, status: "error", summary: "A semicolon is missing.",
            quotes: ["missing semicolon"])
        let data = try JSONEncoder().encode(receipt)
        let raw = try XCTUnwrap(String(data: data, encoding: .utf8))
        XCTAssertNotNil(ToolOutputReducer.validReceipt(raw, source: source, sourceHash: hash, status: "error"))
        XCTAssertNil(ToolOutputReducer.validReceipt(raw, source: source, sourceHash: hash, status: "success"))
        XCTAssertNil(ToolOutputReducer.validReceipt("not json", source: source, sourceHash: hash, status: "error"))
    }

    func testTodoBoundaryRequiresCompletionUnfinishedWorkAndNetBenefit() {
        let previous = [
            TodoItem(id: "one", content: "first", status: "in_progress", activeForm: "first"),
            TodoItem(id: "two", content: "second", status: "pending", activeForm: "second"),
        ]
        let completed = [
            TodoItem(id: "one", content: "first", status: "completed", activeForm: "first"),
            TodoItem(id: "two", content: "second", status: "pending", activeForm: "second"),
        ]
        XCTAssertTrue(TodoBoundaryCompactionPolicy.shouldCompact(
            previous: previous, current: completed, messageCount: 5, boundary: 0, keepRecent: 2))
        XCTAssertFalse(TodoBoundaryCompactionPolicy.shouldCompact(
            previous: completed, current: completed, messageCount: 5, boundary: 0, keepRecent: 2))
        XCTAssertFalse(TodoBoundaryCompactionPolicy.shouldCompact(
            previous: previous, current: completed, messageCount: 3, boundary: 0, keepRecent: 2))
    }

    func testSettingsDefaultAllEfficiencyFeaturesOn() {
        let settings = MacAppSettings()
        XCTAssertTrue(settings.actionFusion)
        XCTAssertTrue(settings.observationPack)
        XCTAssertTrue(settings.evidenceReducer)
        XCTAssertTrue(settings.todoBoundaryCompaction)
    }
}
