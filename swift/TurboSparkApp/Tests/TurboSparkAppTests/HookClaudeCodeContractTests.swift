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
        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Block Prompt",
            event: .userPromptSubmit,
            type: .command,
            command: "echo 'no secrets please' >&2; exit 2",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let appModel = AppModel()
        let verdict = await appModel.evaluateUserPromptSubmit(
            prompt: "leak my api key", chatID: appModel.selectedChatID, project: nil)
        XCTAssertTrue(verdict.isBlocked)
        XCTAssertEqual(verdict.blockReason, "no secrets please")
    }

    // MARK: - AppModel wiring: PreToolUse updatedInput / additionalContext

    /// Tests that PreToolUse hooks can update input arguments and attach additional context.
    @MainActor
    func testPreToolUseUpdatedInputAndAdditionalContextRoundTrip() async {
        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Rewrite Command",
            event: .preToolUse,
            type: .command,
            command: "echo '{\"hookSpecificOutput\":{\"permissionDecision\":\"allow\",\"updatedInput\":{\"command\":\"echo safe\"},\"additionalContext\":\"rewrote unsafe command\"}}'",
            matcher: "run_command",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

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
        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Post Feedback",
            event: .postToolUse,
            type: .command,
            command: "echo 'consider re-reading the file' >&2; exit 2",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

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
        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Always Continue",
            event: .stop,
            type: .command,
            command: "exit 2",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

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

        let store = AppHookStore.shared
        store.refresh(projectDirectory: tempDir.path)
        defer { store.refresh(projectDirectory: nil) }

        let projectHooks = store.hooks.filter { $0.sourcePath?.hasPrefix(tempDir.path) == true }
        XCTAssertTrue(projectHooks.contains { $0.command == "echo ok" && $0.type == .command })
        XCTAssertTrue(projectHooks.contains { $0.command == "echo bye" && $0.sourceType == .localConfig })

        XCTAssertTrue(store.discoveryDiagnostics.contains { $0.contains("PreToolUse") })
    }

    // MARK: - continue: false (prevent continuation)

    /// Tests that `continue: false` in a hook's JSON output aggregates into
    /// `preventContinuation`, carrying the FIRST hook's stopReason, and is
    /// not a block (whose reason would go to the model rather than the user).
    func testAggregatorContinueFalseSetsPreventContinuationWithFirstStopReason() {
        let results = [
            resultWith(.structured(AppHookResponse(continueGeneration: false, stopReason: "enough")), event: .stop),
            resultWith(.structured(AppHookResponse(continueGeneration: false, stopReason: "second")), event: .stop)
        ]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .stop)
        XCTAssertTrue(verdict.preventContinuation)
        XCTAssertEqual(verdict.continuationStopReason, "enough")
        XCTAssertFalse(verdict.isBlocked, "continue:false ends the turn for the user; it does not feed the model a block reason")
    }

    /// Tests that an absent or true `continue` field never sets preventContinuation.
    func testAggregatorContinueTrueOrAbsentKeepsPreventContinuationFalse() {
        for response in [AppHookResponse(), AppHookResponse(continueGeneration: true, stopReason: "unused")] {
            let verdict = AppHookDecisionAggregator.aggregate(
                [resultWith(.structured(response), event: .userPromptSubmit)], event: .userPromptSubmit)
            XCTAssertFalse(verdict.preventContinuation)
            XCTAssertNil(verdict.continuationStopReason)
        }
    }

    /// Tests that a Stop hook sending `continue: false` ends the turn
    /// WITHOUT re-entering it: no user turn appended, no re-entry consumed.
    @MainActor
    func testStopHookContinueFalseEndsTurnWithoutReentry() async {
        let store = AppHookStore.shared
        // `saveCustomHooks` is skipped before the first refresh, so make the
        // hook durable (and visible to AppModel.init's own refresh) even
        // when this test runs without the rest of the suite.
        store.refresh(projectDirectory: nil)
        let hook = AppHookCommand(
            name: "Hard Stop",
            event: .stop,
            type: .command,
            command: #"echo '{"continue":false,"stopReason":"work is done"}'"#,
            sourceType: .custom)
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let appModel = AppModel()
        let chat = AppChat(title: "Stop continue false")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let reentered = await appModel.dispatchStopAndContinueIfBlocked(
            chatID: chat.id, resumeStep: 0, project: nil)
        XCTAssertFalse(reentered, "continue:false must end the turn, not re-enter it")
        XCTAssertTrue(appModel.chats[0].messages.isEmpty, "no user turn may be appended by a prevent-continuation Stop hook")
        // The stopReason is shown to the USER (Claude Code's contract). This
        // is what distinguishes the prevent-continuation branch from merely
        // "not blocked": both return false, only one surfaces the reason.
        XCTAssertEqual(appModel.activeToast?.message, "Turn stopped by hook: work is done")
    }

    /// Tests that a UserPromptSubmit hook's `continue: false` carries the
    /// stopReason through the app-level evaluation.
    @MainActor
    func testUserPromptSubmitContinueFalseCarriesStopReason() async {
        let store = AppHookStore.shared
        // `saveCustomHooks` is skipped before the first refresh, so make the
        // hook durable (and visible to AppModel.init's own refresh) even
        // when this test runs without the rest of the suite.
        store.refresh(projectDirectory: nil)
        let hook = AppHookCommand(
            name: "Refuse Prompt",
            event: .userPromptSubmit,
            type: .command,
            command: #"echo '{"continue":false,"stopReason":"prompt refused"}'"#,
            sourceType: .custom)
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let appModel = AppModel()
        let verdict = await appModel.evaluateUserPromptSubmit(
            prompt: "hi", chatID: appModel.selectedChatID, project: nil)
        XCTAssertTrue(verdict.preventContinuation)
        XCTAssertEqual(verdict.continuationStopReason, "prompt refused")
    }

    /// Tests that a PreToolUse hook's `continue: false` reaches the
    /// PreToolUse decision the agent loop acts on.
    @MainActor
    func testPreToolUseContinueFalseCarriesPreventContinuation() async {
        let store = AppHookStore.shared
        // `saveCustomHooks` is skipped before the first refresh, so make the
        // hook durable (and visible to AppModel.init's own refresh) even
        // when this test runs without the rest of the suite.
        store.refresh(projectDirectory: nil)
        let hook = AppHookCommand(
            name: "Nope",
            event: .preToolUse,
            type: .command,
            command: #"echo '{"continue":false,"stopReason":"not this"}'"#,
            matcher: "read_file",
            sourceType: .custom)
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let decision = await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: UUID().uuidString, toolName: "read_file", toolArguments: ["path": "x"])
        XCTAssertTrue(decision.preventContinuation)
        XCTAssertEqual(decision.continuationStopReason, "not this")
    }

    // MARK: - Legacy decision field on permission events

    /// Tests that the legacy `decision: "approve"` maps onto allow for
    /// PreToolUse, with its reason.
    func testAggregatorLegacyApproveDecisionAllowsOnPermissionEvents() {
        let results = [resultWith(.structured(AppHookResponse(decision: "approve", reason: "safe here")))]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .preToolUse)
        XCTAssertEqual(verdict.permissionDecision, .allow)
        XCTAssertEqual(verdict.permissionReason, "safe here")
    }

    /// Tests that the legacy `decision: "block"` DENIES on a permission
    /// event (it used to be ignored there) rather than blocking a turn.
    func testAggregatorLegacyBlockDecisionDeniesOnPermissionEvents() {
        let results = [resultWith(.structured(AppHookResponse(decision: "block", reason: "not safe")))]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .preToolUse)
        XCTAssertEqual(verdict.permissionDecision, .deny)
        XCTAssertEqual(verdict.permissionReason, "not safe")
        XCTAssertFalse(verdict.isBlocked, "a permission event denies the call; it has no turn to block")
    }

    /// Tests that `decision: "block"` still blocks the turn on events that
    /// carry no permission decision (Stop).
    func testAggregatorLegacyBlockDecisionStillBlocksNonPermissionEvents() {
        let results = [resultWith(.structured(AppHookResponse(decision: "block", reason: "keep going")), event: .stop)]
        let verdict = AppHookDecisionAggregator.aggregate(results, event: .stop)
        XCTAssertTrue(verdict.isBlocked)
        XCTAssertEqual(verdict.blockReason, "keep going")
    }

    // MARK: - PermissionDenied event

    /// Tests that the PermissionDenied stdin payload carries the tool
    /// fields plus the denial reason.
    func testStdinPayloadCarriesToolFieldsAndReasonForPermissionDenied() {
        let payload = AppHookStdinPayload.build(
            event: .permissionDenied,
            sessionID: "s1",
            transcriptPath: "/tmp/t.json",
            cwd: "/tmp",
            toolName: "run_command",
            toolArguments: ["command": "rm -rf /"],
            reason: "Denied by the user"
        )
        XCTAssertEqual(payload["hook_event_name"] as? String, "PermissionDenied")
        XCTAssertEqual(payload["tool_name"] as? String, "run_command")
        XCTAssertEqual(payload["reason"] as? String, "Denied by the user")
        XCTAssertNil(payload["prompt"])
    }

    /// Tests that a configured PermissionDenied hook actually runs when a
    /// denial is dispatched.
    @MainActor
    func testPermissionDeniedDispatchReachesConfiguredHooks() async {
        let store = AppHookStore.shared
        // `saveCustomHooks` is skipped before the first refresh, so make the
        // hook durable (and visible to AppModel.init's own refresh) even
        // when this test runs without the rest of the suite.
        store.refresh(projectDirectory: nil)
        let hook = AppHookCommand(
            name: "Deny Auditor",
            event: .permissionDenied,
            type: .command,
            command: #"echo '{"systemMessage":"denial recorded"}'"#,
            sourceType: .custom)
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let appModel = AppModel()
        let results = await appModel.dispatchPermissionDenied(
            toolName: "run_command", toolArguments: ["command": "ls"], reason: "test",
            chatID: appModel.selectedChatID, project: nil)
        XCTAssertEqual(results.count, 1)
        guard case .structured(let response) = results[0].outcome else {
            return XCTFail("expected a structured outcome from the PermissionDenied hook")
        }
        XCTAssertEqual(response.systemMessage, "denial recorded")
    }

    // MARK: - SubagentStart / SubagentStop

    /// Tests that the subagent lifecycle payloads carry agent_id/agent_type,
    /// and SubagentStop adds stop_hook_active and the transcript path.
    func testStdinPayloadCarriesAgentFieldsForSubagentEvents() {
        let start = AppHookStdinPayload.build(
            event: .subagentStart, sessionID: "s1", transcriptPath: "/tmp/t.json", cwd: "/tmp",
            agentID: "run-1", agentType: "explore")
        XCTAssertEqual(start["hook_event_name"] as? String, "SubagentStart")
        XCTAssertEqual(start["agent_id"] as? String, "run-1")
        XCTAssertEqual(start["agent_type"] as? String, "explore")

        let stop = AppHookStdinPayload.build(
            event: .subagentStop, sessionID: "s1", transcriptPath: "/tmp/t.json", cwd: "/tmp",
            stopHookActive: false, agentID: "run-1", agentType: "explore")
        XCTAssertEqual(stop["hook_event_name"] as? String, "SubagentStop")
        XCTAssertEqual(stop["stop_hook_active"] as? Bool, false)
        XCTAssertEqual(stop["agent_id"] as? String, "run-1")
        XCTAssertEqual(stop["agent_transcript_path"] as? String, "/tmp/t.json")
    }

    // MARK: - Prompt hooks fail loudly

    /// Tests that a discovered prompt hook is diagnosed at discovery, and
    /// Claude Code's `agent` type maps to prompt (never to command, which
    /// would run the prompt text as a shell command).
    @MainActor
    func testPromptHookDiscoveryIsDiagnosedAndAgentTypeMapsToPrompt() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let claudeDir = tempDir.appendingPathComponent(".claude")
        try FileManager.default.createDirectory(at: claudeDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let settings: [String: Any] = [
            "hooks": [
                "Stop": [
                    ["hooks": [
                        ["type": "prompt", "prompt": "check the diff"],
                        ["type": "agent", "prompt": "verify the build"]
                    ]]
                ]
            ]
        ]
        try JSONSerialization.data(withJSONObject: settings)
            .write(to: claudeDir.appendingPathComponent("settings.json"))

        let store = AppHookStore.shared
        store.refresh(projectDirectory: tempDir.path)
        defer { store.refresh(projectDirectory: nil) }

        let projectHooks = store.hooks.filter { $0.sourcePath?.hasPrefix(tempDir.path) == true }
        XCTAssertEqual(projectHooks.filter { $0.type == .prompt }.count, 2,
                       "prompt and agent both load as the unevaluated prompt type")
        XCTAssertFalse(projectHooks.contains { $0.type == .command },
                       "an agent hook must never fall through to a shell command")
        XCTAssertTrue(store.discoveryDiagnostics.contains { $0.contains("does not evaluate") })
    }

    /// Tests that running a prompt hook produces a visible non-blocking
    /// outcome rather than the anonymous no-op it used to be.
    @MainActor
    func testPromptHookRunSurfacesNonBlockingOutcome() async {
        let store = AppHookStore.shared
        // `saveCustomHooks` is skipped before the first refresh, so make the
        // hook durable (and visible to AppModel.init's own refresh) even
        // when this test runs without the rest of the suite.
        store.refresh(projectDirectory: nil)
        let hook = AppHookCommand(
            name: "LLM Check",
            event: .postToolUse,
            type: .prompt,
            command: "verify the output",
            sourceType: .custom)
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let results = await AppHookExecutionEngine.shared.dispatch(
            event: .postToolUse, sessionID: UUID().uuidString, toolName: "read_file",
            toolArguments: ["path": "x"], toolOutput: "y", toolDurationSeconds: 0.01, isError: false)
        XCTAssertEqual(results.count, 1)
        guard case .nonBlockingError(let message) = results[0].outcome else {
            return XCTFail("expected a visible non-blocking outcome for an unevaluated prompt hook")
        }
        XCTAssertTrue(message.contains("prompt hook"), "the outcome must name why nothing ran")
    }

    // MARK: - Source group ordering

    /// Tests that source groups are ordered with local config preceding custom hooks.
    @MainActor
    func testSourceGroupOrderingPlacesLocalConfigBeforeCustom() async throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let claudeDir = tempDir.appendingPathComponent(".claude")
        try FileManager.default.createDirectory(at: claudeDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let store = AppHookStore.shared
        store.refresh(projectDirectory: tempDir.path)
        defer { store.refresh(projectDirectory: nil) }

        let expectedIDsInOrder = ["user_config", "project_config", "local_config", "custom"]
        let actualOrder = store.sourceGroups.map { $0.id }.filter { expectedIDsInOrder.contains($0) }
        XCTAssertEqual(actualOrder, expectedIDsInOrder)
    }
}
