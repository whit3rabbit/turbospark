import XCTest
@testable import TurboSparkApp

/// Coverage for the Claude Code hook contract gaps this app's hook system
/// closed: matcher aliasing, the per-event exit-code-2 table, decision
/// aggregation across several hooks, the stdin payload shape, and the
/// UserPromptSubmit/Stop/PostToolUse wiring in `AppModel`.
/// See https://code.claude.com/docs/en/hooks.
final class HookClaudeCodeContractTests: XCTestCase {

    // MARK: - Matcher aliasing and dispatch order

    /// Tests that Claude Code tool names like "Bash" match the app's canonical tool names like "run_command".
    func testClaudeCodeToolNameMatchesAppToolName() {
        let engine = AppHookExecutionEngine.shared
        let hook = AppHookCommand(name: "h", event: .preToolUse, type: .command, command: "true", matcher: "Bash")
        XCTAssertTrue(engine.matchesCondition(hook: hook, toolName: "run_command", toolArguments: nil))
    }

    /// Tests that app tool names like "run_command" match hooks configured with Claude Code matcher names like "Bash".
    func testAppToolNameMatchesClaudeCodeStyleMatcher() {
        let engine = AppHookExecutionEngine.shared
        let hook = AppHookCommand(name: "h", event: .preToolUse, type: .command, command: "true", matcher: "run_command")
        XCTAssertTrue(engine.matchesCondition(hook: hook, toolName: "Bash", toolArguments: nil))
    }

    /// Tests that regex matchers like "mcp__.*" only match MCP tools and ignore non-matching built-in tools.
    func testRegexMatcherFiresOnMcpToolNamesOnly() {
        let engine = AppHookExecutionEngine.shared
        let hook = AppHookCommand(name: "h", event: .preToolUse, type: .command, command: "true", matcher: "mcp__.*")
        XCTAssertTrue(engine.matchesCondition(hook: hook, toolName: "mcp__memory__create_entities", toolArguments: nil))
        XCTAssertFalse(engine.matchesCondition(hook: hook, toolName: "run_command", toolArguments: nil))
    }

    /// Tests that comma- and pipe-separated matcher lists match any of the specified tool names.
    func testCommaAndPipeSeparatedMatcherLists() {
        let engine = AppHookExecutionEngine.shared
        let hook = AppHookCommand(name: "h", event: .preToolUse, type: .command, command: "true", matcher: "Edit, Write")
        XCTAssertTrue(engine.matchesCondition(hook: hook, toolName: "write_file", toolArguments: nil))
        XCTAssertTrue(engine.matchesCondition(hook: hook, toolName: "edit_file", toolArguments: nil))
        XCTAssertFalse(engine.matchesCondition(hook: hook, toolName: "read_file", toolArguments: nil))
    }

    /// Tests that wildcard matchers like "*" match any tool name unconditionally.
    func testWildcardMatcherFiresOnEverything() {
        let engine = AppHookExecutionEngine.shared
        let hook = AppHookCommand(name: "h", event: .preToolUse, type: .command, command: "true", matcher: "*")
        XCTAssertTrue(engine.matchesCondition(hook: hook, toolName: "anything_at_all", toolArguments: nil))
    }

    // MARK: - Exit-code-2 table (event-dependent)

    /// Tests that an exit code of 2 blocks execution for PreToolUse, UserPromptSubmit, and Stop events.
    func testExitTwoBlocksPreToolUseUserPromptSubmitAndStop() {
        for event in [AppHookEvent.preToolUse, .userPromptSubmit, .stop] {
            let outcome = AppHookResponseParser.parseHookOutput(stdout: "", stderr: "nope", exitCode: 2, event: event)
            guard case .blocked(let reason) = outcome else {
                return XCTFail("\(event) should block on exit 2")
            }
            XCTAssertEqual(reason, "nope")
        }
    }

    /// Tests that an exit code of 2 produces non-blocking feedback for PostToolUse and PermissionRequest events.
    func testExitTwoDoesNotBlockPostToolUseOrPermissionRequest() {
        for event in [AppHookEvent.postToolUse, .postToolUseFailure, .permissionRequest] {
            let outcome = AppHookResponseParser.parseHookOutput(stdout: "", stderr: "feedback", exitCode: 2, event: event)
            guard case .nonBlockingError(let message) = outcome else {
                return XCTFail("\(event) must not block on exit 2 -- the tool already ran (or permission resolves via JSON)")
            }
            XCTAssertEqual(message, "feedback")
        }
    }

    /// Tests that exit code 0 with valid JSON stdout is parsed into structured hook responses.
    func testExitZeroWithJsonParsesStructuredOutput() {
        let outcome = AppHookResponseParser.parseHookOutput(
            stdout: #"{"continue":true,"additionalContext":"extra","hookSpecificOutput":{"permissionDecision":"allow"}}"#,
            stderr: "",
            exitCode: 0,
            event: .preToolUse
        )
        guard case .structured(let response) = outcome else { return XCTFail("expected structured outcome") }
        XCTAssertEqual(response.additionalContext, "extra")
        XCTAssertEqual(response.hookSpecificOutput?.permissionDecision, "allow")
    }

    /// Tests that exit code 0 with non-JSON plain text is treated as advisory output.
    func testExitZeroWithPlainTextIsAdvisoryOnly() {
        let outcome = AppHookResponseParser.parseHookOutput(stdout: "just some text", stderr: "", exitCode: 0, event: .postToolUse)
        guard case .plainText(let text) = outcome else { return XCTFail("expected plain text outcome") }
        XCTAssertEqual(text, "just some text")
    }

    // MARK: - Decision aggregation across several hooks

    /// Helper to construct mock hook execution results for decision aggregation tests.
    private func resultWith(_ outcome: AppHookOutcome, event: AppHookEvent = .preToolUse) -> AppHookExecutionResult {
        AppHookExecutionResult(hookID: UUID(), hookName: "h", event: event, exitCode: 0, stdout: "", stderr: "", durationSeconds: 0, outcome: outcome)
    }

    /// Tests that decision aggregation strictly prioritizes deny over ask, and ask over allow.
    func testAggregatorDenyBeatsAskBeatsAllow() {
        let results = [
            resultWith(.structured(AppHookResponse(hookSpecificOutput: AppHookSpecificOutput(permissionDecision: "allow")))),
            resultWith(.structured(AppHookResponse(hookSpecificOutput: AppHookSpecificOutput(permissionDecision: "ask")))),
            resultWith(.structured(AppHookResponse(hookSpecificOutput: AppHookSpecificOutput(permissionDecision: "deny", permissionDecisionReason: "no"))))
        ]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .preToolUse)
        XCTAssertEqual(verdict.permissionDecision, .deny)
        XCTAssertEqual(verdict.permissionReason, "no")
    }

    /// Tests that the first blocking outcome takes precedence for Stop events.
    func testAggregatorFirstBlockWinsForStop() {
        let results = [
            resultWith(.blocked(reason: "first"), event: .stop),
            resultWith(.blocked(reason: "second"), event: .stop)
        ]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .stop)
        XCTAssertTrue(verdict.isBlocked)
        XCTAssertEqual(verdict.blockReason, "first")
    }

    /// Tests that additional context strings from multiple hooks are joined with double newlines.
    func testAggregatorConcatenatesAdditionalContextAcrossHooks() {
        let results = [
            resultWith(.structured(AppHookResponse(additionalContext: "one")), event: .userPromptSubmit),
            resultWith(.structured(AppHookResponse(additionalContext: "two")), event: .userPromptSubmit)
        ]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .userPromptSubmit)
        XCTAssertEqual(verdict.additionalContext, "one\n\ntwo")
    }

    /// Tests that later hooks override updated input fields when multiple hooks modify tool inputs.
    func testAggregatorUpdatedInputLastWriteWins() {
        let results = [
            resultWith(.structured(AppHookResponse(updatedInput: ["command": "first"]))),
            resultWith(.structured(AppHookResponse(updatedInput: ["command": "second"])))
        ]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .preToolUse)
        XCTAssertEqual(verdict.updatedInput?["command"], "second")
    }

    /// Tests that errors in PostToolUse hooks surface as feedback messages rather than execution blocks.
    func testAggregatorPostToolUseFeedbackIsNeverABlock() {
        let results = [resultWith(.nonBlockingError("stderr text"), event: .postToolUse)]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .postToolUse)
        XCTAssertFalse(verdict.isBlocked)
        XCTAssertEqual(verdict.feedbackMessage, "stderr text")
    }

    // MARK: - Stdin payload shape

    /// Tests that the stdin payload for PreToolUse events contains tool name and input dictionary.
    func testStdinPayloadCarriesToolFieldsForPreToolUse() {
        let payload = AppHookStdinPayload.build(
            event: .preToolUse,
            sessionID: "s1",
            transcriptPath: "/tmp/t.json",
            cwd: "/tmp",
            toolName: "run_command",
            toolArguments: ["command": "ls"]
        )
        XCTAssertEqual(payload["hook_event_name"] as? String, "PreToolUse")
        XCTAssertEqual(payload["tool_name"] as? String, "run_command")
        XCTAssertEqual((payload["tool_input"] as? [String: String])?["command"], "ls")
        XCTAssertNil(payload["prompt"])
    }

    /// Tests that the stdin payload for UserPromptSubmit events includes the user prompt.
    func testStdinPayloadCarriesPromptForUserPromptSubmit() {
        let payload = AppHookStdinPayload.build(
            event: .userPromptSubmit, sessionID: "s1", transcriptPath: "/tmp/t.json", cwd: "/tmp", prompt: "hello"
        )
        XCTAssertEqual(payload["prompt"] as? String, "hello")
        XCTAssertNil(payload["tool_name"])
    }

    /// Tests that the stdin payload for Stop events sets the stop_hook_active boolean flag.
    func testStdinPayloadCarriesStopHookActiveForStop() {
        let payload = AppHookStdinPayload.build(
            event: .stop, sessionID: "s1", transcriptPath: "/tmp/t.json", cwd: "/tmp", stopHookActive: true
        )
        XCTAssertEqual(payload["stop_hook_active"] as? Bool, true)
    }

    // MARK: - AppModel wiring: UserPromptSubmit

    /// Tests that a blocking UserPromptSubmit hook correctly surfaces its block reason in the UI.
    @MainActor
    func testUserPromptSubmitHookBlockSurfacesReason() async {
        let store = await AppHookStore.shared
        let hook = AppHookCommand(
            name: "Block Prompt",
            event: .userPromptSubmit,
            type: .command,
            command: "echo 'no secrets please' >&2; exit 2",
            sourceType: .custom
        )
        await store.addCustomHook(hook)
        defer { Task { await store.deleteCustomHook(id: hook.id) } }

        let appModel = AppModel()
        let verdict = await appModel.evaluateUserPromptSubmit(
            prompt: "leak my api key", chatID: appModel.selectedChatID, project: nil)
        XCTAssertTrue(verdict.isBlocked)
        XCTAssertEqual(verdict.blockReason, "no secrets please")
    }

    // MARK: - AppModel wiring: PreToolUse updatedInput / additionalContext

    /// Tests that PreToolUse hooks can update input arguments and attach additional context.
    func testPreToolUseUpdatedInputAndAdditionalContextRoundTrip() async {
        let store = await AppHookStore.shared
        let hook = AppHookCommand(
            name: "Rewrite Command",
            event: .preToolUse,
            type: .command,
            command: "echo '{\"hookSpecificOutput\":{\"permissionDecision\":\"allow\",\"updatedInput\":{\"command\":\"echo safe\"},\"additionalContext\":\"rewrote unsafe command\"}}'",
            matcher: "run_command",
            sourceType: .custom
        )
        await store.addCustomHook(hook)
        defer { Task { await store.deleteCustomHook(id: hook.id) } }

        let decision = await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: UUID().uuidString, toolName: "run_command", toolArguments: ["command": "rm -rf /"]
        )
        XCTAssertEqual(decision.behavior, .allow)
        XCTAssertEqual(decision.updatedInput?["command"], "echo safe")
        XCTAssertEqual(decision.additionalContext, "rewrote unsafe command")
    }

    // MARK: - AppModel wiring: PostToolUse feedback

    /// Tests that exit code 2 from a PostToolUse hook generates advisory feedback without blocking.
    @MainActor
    func testPostToolUseExitTwoSurfacesAsFeedbackNotABlock() async {
        let store = await AppHookStore.shared
        let hook = AppHookCommand(
            name: "Post Feedback",
            event: .postToolUse,
            type: .command,
            command: "echo 'consider re-reading the file' >&2; exit 2",
            sourceType: .custom
        )
        await store.addCustomHook(hook)
        defer { Task { await store.deleteCustomHook(id: hook.id) } }

        let appModel = AppModel()
        let verdict = await appModel.dispatchPostToolUseVerdict(
            toolName: "read_file",
            toolArguments: ["path": "README.md"],
            toolOutput: "file contents",
            toolDurationSeconds: 0.01,
            isError: false,
            chatID: appModel.selectedChatID,
            project: nil
        )
        XCTAssertFalse(verdict.isBlocked, "PostToolUse cannot block -- the tool already ran")
        XCTAssertEqual(verdict.feedbackMessage, "consider re-reading the file")
    }

    // MARK: - AppModel wiring: Stop block-and-continue cap

    /// Tests that Stop hook blocking re-enters the agent loop at most 8 times before stopping.
    @MainActor
    func testStopHookBlockReentersUpToEightTimesThenStops() async {
        let store = await AppHookStore.shared
        let hook = AppHookCommand(
            name: "Always Continue",
            event: .stop,
            type: .command,
            command: "exit 2",
            sourceType: .custom
        )
        await store.addCustomHook(hook)
        defer { Task { await store.deleteCustomHook(id: hook.id) } }

        let appModel = AppModel()
        let chat = AppChat(title: "Stop cap test")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        var reentries = 0
        for _ in 0..<10 {
            let blocked = await appModel.dispatchStopAndContinueIfBlocked(chatID: chat.id, resumeStep: 0, project: nil)
            if blocked {
                reentries += 1
            } else {
                break
            }
        }
        XCTAssertEqual(reentries, 8, "must cap at 8 consecutive re-entries, like Claude Code's own Stop hook cap")
    }

    // MARK: - Discovery: settings.local.json, missing `type`, diagnostics

    /// Tests hook discovery defaults missing type fields to command and diagnoses invalid entries.
    @MainActor
    func testDiscoveryDefaultsMissingTypeToCommandAndFlagsUnparseableEntries() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let claudeDir = tempDir.appendingPathComponent(".claude")
        try FileManager.default.createDirectory(at: claudeDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let settings: [String: Any] = [
            "hooks": [
                "PreToolUse": [
                    [
                        "matcher": "Bash",
                        "hooks": [
                            ["command": "echo ok"],  // missing `type`: should default to .command
                            ["type": "command"]       // missing `command`: should be dropped and diagnosed
                        ]
                    ]
                ]
            ]
        ]
        let data = try JSONSerialization.data(withJSONObject: settings)
        try data.write(to: claudeDir.appendingPathComponent("settings.json"))

        let localSettings: [String: Any] = [
            "hooks": [
                "SessionEnd": [
                    ["matcher": "", "hooks": [["type": "command", "command": "echo bye"]]]
                ]
            ]
        ]
        let localData = try JSONSerialization.data(withJSONObject: localSettings)
        try localData.write(to: claudeDir.appendingPathComponent("settings.local.json"))

        let store = await AppHookStore.shared
        store.refresh(projectDirectory: tempDir.path)
        defer { store.refresh(projectDirectory: nil) }

        let projectHooks = store.hooks.filter { $0.sourcePath?.hasPrefix(tempDir.path) == true }
        XCTAssertTrue(projectHooks.contains { $0.command == "echo ok" && $0.type == .command })
        XCTAssertTrue(projectHooks.contains { $0.command == "echo bye" && $0.sourceType == .localConfig })

        XCTAssertTrue(store.discoveryDiagnostics.contains { $0.contains("PreToolUse") })
    }

    // MARK: - Source group ordering

    /// Tests that source groups are ordered with local config preceding custom hooks.
    @MainActor
    func testSourceGroupOrderingPlacesLocalConfigBeforeCustom() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let claudeDir = tempDir.appendingPathComponent(".claude")
        try FileManager.default.createDirectory(at: claudeDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let store = await AppHookStore.shared
        store.refresh(projectDirectory: tempDir.path)
        defer { store.refresh(projectDirectory: nil) }

        let expectedIDsInOrder = ["user_config", "project_config", "local_config", "custom"]
        let actualOrder = store.sourceGroups.map { $0.id }.filter { expectedIDsInOrder.contains($0) }
        XCTAssertEqual(actualOrder, expectedIDsInOrder)
    }
}
