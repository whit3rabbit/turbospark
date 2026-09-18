import TurboSpark
import XCTest
@testable import TurboSparkApp

/// Regression tests for the hook-decision cluster (T10, T11, T17).
final class HookDecisionRoutingTests: XCTestCase {
    private func makeGenerationResult(content: String = "") throws -> GenerationResult {
        let json = """
        {"promptTokens":1,"newTokens":1,"prefillSeconds":0,"decodeSeconds":0,"stopReason":"toolCalls","content":"\(content)"}
        """
        return try JSONDecoder().decode(GenerationResult.self, from: Data(json.utf8))
    }

    // MARK: - T11: an async PreToolUse hook must still be able to deny

    @MainActor
    func testAsyncPreToolUseHookDecisionIsNotDiscarded() async {
        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Async Guardrail",
            event: .preToolUse,
            type: .command,
            command: "echo 'blocked by async hook' >&2; exit 2",
            matcher: "terminal",
            isAsync: true, // this used to be discarded via Task.detached
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer {
            Task { store.deleteCustomHook(id: hook.id) }
        }

        let decision = await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: UUID().uuidString,
            toolName: "terminal",
            toolArguments: ["command": "rm -rf /"]
        )

        XCTAssertEqual(decision.behavior, .deny, "An async PreToolUse hook's decision must still gate execution, not be silently discarded.")
    }

    // MARK: - T17: matcher is exact, ifCondition is anchored + whitespace-normalized

    func testMatcherIsExactNotBidirectionalSubstring() {
        let engine = AppHookExecutionEngine.shared
        let hook = AppHookCommand(name: "h", event: .preToolUse, type: .command, command: "true", matcher: "Bash")
        // A tool name that CONTAINS the matcher as a substring must not match.
        XCTAssertFalse(engine.matchesCondition(hook: hook, toolName: "BashExecutorTool", toolArguments: nil))
        // A matcher that CONTAINS the tool name as a substring must not match either.
        let hook2 = AppHookCommand(name: "h2", event: .preToolUse, type: .command, command: "true", matcher: "WriteFileTool")
        XCTAssertFalse(engine.matchesCondition(hook: hook2, toolName: "Write", toolArguments: nil))
        // Exact (case-insensitive) match still works.
        let hook3 = AppHookCommand(name: "h3", event: .preToolUse, type: .command, command: "true", matcher: "Bash")
        XCTAssertTrue(engine.matchesCondition(hook: hook3, toolName: "bash", toolArguments: nil))
    }

    func testCommandGlobIsAnchoredAndWhitespaceNormalized() {
        let engine = AppHookExecutionEngine.shared
        // Under-match fixed: irregular whitespace still matches a prefix pattern.
        XCTAssertTrue(engine.matchesCommandGlob("git *", against: "git  push"))
        // Over-match fixed: the pattern text merely APPEARING in an unrelated
        // command must not match a prefix pattern.
        XCTAssertFalse(engine.matchesCommandGlob("git *", against: "echo \"git push \""))
        // Exact (non-wildcard) pattern still requires the whole command to match.
        XCTAssertTrue(engine.matchesCommandGlob("git push", against: "git   push"))
        XCTAssertFalse(engine.matchesCommandGlob("git push", against: "git push origin main"))
    }

    // MARK: - state#78: a hook's `ask` must not lift a project's category deny

    @MainActor
    func testAHookAskingForConfirmationCannotOverrideAReadOnlyProject() async throws {
        let appModel = AppModel()
        let chat = AppChat(title: "deny chat")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Ask First",
            event: .preToolUse,
            type: .command,
            command:
                "echo '{\"permissionDecision\":\"ask\","
                + "\"permissionDecisionReason\":\"needs human review\"}'",
            matcher: "run_command",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        // A project that refuses the terminal category outright.
        let project = AppProject(
            name: "read only",
            rootDirectoryPath: "/tmp",
            permissions: AppProjectPermissions(mode: .ask, terminal: .deny))
        let call = AppToolCall(
            name: "run_command", arguments: ["command": "cargo check"], category: .terminal)

        await appModel.handleExtractedToolCall(
            call, fullContent: "running a command", reasoning: "",
            result: try makeGenerationResult(), currentStep: 0, chatID: chat.id, project: project)

        XCTAssertNil(
            appModel.pendingToolCall,
            "The hook returned before `evaluate` ever ran, so a Strict Read-Only project got an "
                + "Approve button for a shell command -- and `approvePendingToolCall` "
                + "re-evaluates nothing.")
        let recorded = appModel.chats[0].messages.last
        XCTAssertEqual(
            recorded?.toolCalls.first?.status, .denied,
            "It must be recorded as refused, so the model is told rather than left waiting.")
    }

    // MARK: - T10: a hook's `ask` decision must reach the approval UI, not fall through to allow

    @MainActor
    func testHookAskDecisionRoutesToApprovalRatherThanSilentlyAllowing() async throws {
        let appModel = AppModel()
        let chat = AppChat(title: "T10 chat")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Ask First",
            event: .preToolUse,
            type: .command,
            command: "echo '{\"permissionDecision\":\"ask\",\"permissionDecisionReason\":\"needs human review\"}'",
            matcher: "run_command",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer {
            Task { store.deleteCustomHook(id: hook.id) }
        }

        // A benign, low-risk command that `AppToolPermissionEngine` would
        // otherwise auto-allow under `.standard`/`.auto` -- if the hook's
        // `ask` verdict is dropped, this runs immediately with no prompt.
        let call = AppToolCall(name: "run_command", arguments: ["command": "cargo check"], category: .terminal)
        let result = try makeGenerationResult()

        // Awaited rather than polled: `handleExtractedToolCall` is `async`
        // now, so the turn's own lifecycle stays honest across the call
        // (`generating` no longer drops while a tool runs).
        await appModel.handleExtractedToolCall(call, fullContent: "running a command", reasoning: "", result: result, currentStep: 0, chatID: chat.id, project: nil)

        XCTAssertNotNil(appModel.pendingToolCall, "A hook's `ask` decision must surface a pending approval, not silently allow the call.")
        XCTAssertEqual(appModel.pendingToolCallChatID, chat.id)
        XCTAssertTrue(
            appModel.pendingToolCall?.riskAssessment?.reasons.contains(where: { $0.contains("needs human review") }) ?? false,
            "The hook's reason should be visible in the pending call's risk assessment."
        )
        // The tool must NOT have executed.
        XCTAssertEqual(appModel.chats.first(where: { $0.id == chat.id })?.messages.first?.toolResults, [])
    }
}
