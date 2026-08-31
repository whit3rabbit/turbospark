import XCTest
@testable import TurboSparkApp

final class HookSystemTests: XCTestCase {

    func testHookContentHashDeterminism() {
        let hook1 = AppHookCommand(
            name: "Lint",
            event: .preToolUse,
            type: .command,
            command: "cargo fmt --check",
            ifCondition: "Bash(cargo *)",
            matcher: "Bash",
            shell: .zsh
        )

        let hook2 = AppHookCommand(
            name: "Different Name Lint",
            event: .preToolUse,
            type: .command,
            command: "cargo fmt --check",
            ifCondition: "Bash(cargo *)",
            matcher: "Bash",
            shell: .zsh
        )

        // Same executable parameters yield identical content hash
        XCTAssertEqual(hook1.contentHash, hook2.contentHash)

        let hook3 = AppHookCommand(
            name: "Lint",
            event: .preToolUse,
            type: .command,
            command: "cargo clippy",
            ifCondition: "Bash(cargo *)",
            matcher: "Bash",
            shell: .zsh
        )

        // Different command yields distinct content hash
        XCTAssertNotEqual(hook1.contentHash, hook3.contentHash)
    }

    func testHandAuthoredHookDefaultTimeoutIsThirtySecondsByDesign() {
        let hook = AppHookCommand(name: "test", event: .preToolUse, command: "echo hi")
        XCTAssertEqual(
            hook.timeoutSeconds, 30.0,
            "Editor/struct default intentionally differs from the 600s Claude-Code-parity fallback used only for discovered hooks; see AppHookModels.swift doc comment before changing this."
        )
    }

    func testTrustManagement() async {
        let store = await AppHookStore.shared

        let externalHook = AppHookCommand(
            name: "External Plugin Hook",
            event: .sessionStart,
            type: .command,
            command: "echo 'session init'",
            sourceType: .plugin,
            pluginName: "test-plugin"
        )

        // Should not be trusted initially
        await store.untrustHook(externalHook)
        let isTrustedInitially = await store.isHookTrusted(externalHook)
        XCTAssertFalse(isTrustedInitially)

        // Trust the hook
        await store.trustHook(externalHook)
        let isTrustedAfter = await store.isHookTrusted(externalHook)
        XCTAssertTrue(isTrustedAfter)

        // Custom hooks are always trusted by default
        let customHook = AppHookCommand(
            name: "My Custom Hook",
            event: .postToolUse,
            type: .command,
            command: "echo 'done'",
            sourceType: .custom
        )
        let isCustomTrusted = await store.isHookTrusted(customHook)
        XCTAssertTrue(isCustomTrusted)
    }

    func testOptionValuesPersistence() async {
        let store = await AppHookStore.shared
        let sourceID = "plugin_forge-rs"

        await store.updateOptionValue(sourceID: sourceID, key: "api_key", value: "secret-key-12345")
        let retrieved = await store.getOptionValue(sourceID: sourceID, key: "api_key")
        XCTAssertEqual(retrieved, "secret-key-12345")

        let fallback = await store.getOptionValue(sourceID: sourceID, key: "missing_key", defaultVal: "default-val")
        XCTAssertEqual(fallback, "default-val")
    }

    func testExecutionEngineEvaluatesPreToolUseExitCode2AsBlocking() async {
        let store = await AppHookStore.shared

        // Add a hook that blocks with exit 2
        let blockingHook = AppHookCommand(
            name: "Security Guardrail",
            event: .preToolUse,
            type: .command,
            command: "echo 'Destructive rm command blocked by hook' >&2; exit 2",
            matcher: "terminal",
            sourceType: .custom
        )
        await store.addCustomHook(blockingHook)

        let decision = await AppHookExecutionEngine.shared.evaluatePreToolUse(
            sessionID: UUID().uuidString,
            toolName: "terminal",
            toolArguments: ["command": "rm -rf /"]
        )

        XCTAssertEqual(decision.behavior, .deny)
        XCTAssertTrue(decision.reason?.contains("Destructive rm command blocked") ?? false)

        // Cleanup
        await store.deleteCustomHook(id: blockingHook.id)
    }

    func testExecutionEnginePassesBenignHooks() async {
        let store = await AppHookStore.shared

        let benignHook = AppHookCommand(
            name: "Auditor",
            event: .postToolUse,
            type: .command,
            command: "exit 0",
            sourceType: .custom
        )
        await store.addCustomHook(benignHook)

        let results = await AppHookExecutionEngine.shared.dispatch(
            event: .postToolUse,
            sessionID: UUID().uuidString,
            toolName: "read_file",
            toolArguments: ["rel_path": "README.md"]
        )

        XCTAssertFalse(results.isEmpty)
        if let first = results.first(where: { $0.hookID == benignHook.id }) {
            XCTAssertTrue(first.isSuccess)
            XCTAssertEqual(first.exitCode, 0)
        }

        // Cleanup
        await store.deleteCustomHook(id: benignHook.id)
    }

    func testHookEnvironmentDoesNotInheritParentSecrets() async {
        let store = await AppHookStore.shared

        // Set a marker secret in the parent process environment
        setenv("TURBOSPARK_PARENT_SECRET_MARKER", "top_secret_token_12345", 1)
        defer { unsetenv("TURBOSPARK_PARENT_SECRET_MARKER") }

        let envInspectionHook = AppHookCommand(
            name: "Env Inspector",
            event: .postToolUse,
            type: .command,
            command: "echo \"MARKER=$TURBOSPARK_PARENT_SECRET_MARKER\"",
            sourceType: .custom
        )
        await store.addCustomHook(envInspectionHook)

        let results = await AppHookExecutionEngine.shared.dispatch(
            event: .postToolUse,
            sessionID: UUID().uuidString,
            toolName: "read_file",
            toolArguments: ["rel_path": "README.md"]
        )

        let hookResult = results.first(where: { $0.hookID == envInspectionHook.id })
        XCTAssertNotNil(hookResult)
        XCTAssertFalse(hookResult?.stdout.contains("top_secret_token_12345") ?? true, "Hook child process must not inherit arbitrary parent environment secrets.")

        // Cleanup
        await store.deleteCustomHook(id: envInspectionHook.id)
    }

    func testHookEnvironmentExcludesSensitiveOptions() async {
        let store = await AppHookStore.shared
        let pluginName = "forge-rs"
        let sourceID = "plugin_\(pluginName)"

        // Update sensitive api_key option
        await store.updateOptionValue(sourceID: sourceID, key: "api_key", value: "sensitive_api_token_abc")
        await store.updateOptionValue(sourceID: sourceID, key: "strict_mode", value: "true")

        let inspectOptionsHook = AppHookCommand(
            name: "Plugin Options Inspector",
            event: .postToolUse,
            type: .command,
            command: "echo \"API_KEY=$TURBOSPARK_OPTION_API_KEY;STRICT=$TURBOSPARK_OPTION_STRICT_MODE\"",
            sourceType: .plugin,
            pluginName: pluginName
        )
        await store.trustHook(inspectOptionsHook)

        let results = await AppHookExecutionEngine.shared.dispatch(
            event: .postToolUse,
            sessionID: UUID().uuidString,
            toolName: "read_file",
            toolArguments: ["rel_path": "README.md"]
        )

        if let hookResult = results.first(where: { $0.hookID == inspectOptionsHook.id }) {
            XCTAssertFalse(hookResult.stdout.contains("sensitive_api_token_abc"), "Sensitive option must not be exported into environment variables.")
        }
    }
}
