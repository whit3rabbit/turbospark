import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Regression tests for the "Contract drift" review findings: places where a
/// surface promised something its implementation did not deliver.
final class ContractDriftFixTests: XCTestCase {
    private func tempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("contract-drift-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        addTeardownBlock { try? FileManager.default.removeItem(at: dir) }
        return dir
    }

    // MARK: - MCP import honours profile isolation

    func testMcpImportScanReturnsNothingOutsideTheDefaultProfile() {
        let isolated = AgentContentImportSheet.scanMcpServers(
            existingNames: [], allowSharedRoots: false)
        XCTAssertTrue(isolated.isEmpty, "a non-default profile must not read other agents' configs")
    }

    // MARK: - Hook editor

    func testHookEditorDoesNotOfferTheNeverEvaluatedPromptType() {
        XCTAssertFalse(HookEditorSheet.selectableTypes(existing: nil).contains(.prompt))
        XCTAssertFalse(HookEditorSheet.selectableTypes(existing: .command).contains(.prompt))
        XCTAssertTrue(HookEditorSheet.selectableTypes(existing: .command).contains(.http))
        // A discovered prompt hook must still be displayable when edited.
        XCTAssertTrue(HookEditorSheet.selectableTypes(existing: .prompt).contains(.prompt))
    }

    // MARK: - Inspector ladder key

    func testLadderKeyChangesWithCacheSlotsAndLoadGuard() {
        let base = ContextLadderPicker.ladderKey(path: "/m", slots: .auto, loadGuard: .relaxed)
        XCTAssertNotEqual(
            base, ContextLadderPicker.ladderKey(path: "/m", slots: .fixed(32), loadGuard: .relaxed))
        XCTAssertNotEqual(
            base, ContextLadderPicker.ladderKey(path: "/m", slots: .auto, loadGuard: .off))
        XCTAssertNotEqual(
            base, ContextLadderPicker.ladderKey(path: "/other", slots: .auto, loadGuard: .relaxed))
        XCTAssertEqual(
            base, ContextLadderPicker.ladderKey(path: "/m", slots: .auto, loadGuard: .relaxed))
    }

    // MARK: - Timestamps follow the passed locale

    func testTimestampsUseTheGivenLocaleNotTheSystemOne() {
        let german = Locale(identifier: "de_DE")
        let now = Date(timeIntervalSince1970: 1_710_000_000)
        let relative = MessageTimestampFormatter.relativeString(
            for: now.addingTimeInterval(-300), relativeTo: now, locale: german)
        XCTAssertTrue(relative.contains("Minuten"), "got: \(relative)")
        XCTAssertFalse(relative.contains("ago"))

        let exact = MessageTimestampFormatter.exactString(for: now, locale: german)
        XCTAssertTrue(exact.contains("M\u{E4}rz"), "got: \(exact)")
        XCTAssertFalse(exact.contains(" at "))
        XCTAssertFalse(exact.contains("AM") || exact.contains("PM"))

        // Yesterday no longer hardcodes the English "Yesterday at".
        let yesterday = MessageTimestampFormatter.relativeString(
            for: now.addingTimeInterval(-30 * 3600), relativeTo: now, locale: german)
        XCTAssertFalse(yesterday.contains("Yesterday"), "got: \(yesterday)")
    }

    // MARK: - /clear drops the goal and queued prompts

    @MainActor
    func testClearOutputClearsGoalAndQueuedPrompts() {
        let model = AppModel()
        let chat = AppChat(title: "clear")
        model.chats = [chat]
        model.selectedChatID = chat.id
        model.setGoal(condition: "tests pass", chatID: chat.id)
        model.pendingUserMessages[chat.id] = [QueuedUserPrompt(text: "later")]
        XCTAssertNotNil(model.activeGoals[chat.id])

        model.clearOutput()

        XCTAssertNil(model.activeGoals[chat.id], "the cleared conversation must not keep its goal")
        XCTAssertNil(model.chats.first?.goal)
        XCTAssertNil(model.pendingUserMessages[chat.id], "queued prompts must not drain into the cleared chat")
        XCTAssertNil(model.goalEvalMessageCounts[chat.id])
    }

    // MARK: - Goal judge slice

    func testGoalJudgeSliceIsExactlyTheRowsSinceTheBaseline() {
        let rows = (0..<30).map { AppChatMessage(role: .assistant, content: "row \($0)") }
        let slice = AppModel.goalJudgeSlice(rows, sliceStart: 25)
        XCTAssertEqual(slice.map(\.content), (25..<30).map { "row \($0)" })

        // A round longer than the old fixed tail is no longer cut short.
        let long = AppModel.goalJudgeSlice(rows, sliceStart: 2)
        XCTAssertEqual(long.count, 28)

        // Empty slice falls back to the tail rather than judging nothing.
        let empty = AppModel.goalJudgeSlice(rows, sliceStart: 30)
        XCTAssertEqual(empty.count, min(30, GoalPolicy.evaluatorTailMessages))
    }

    // MARK: - Marketplace entry components

    func testNonStrictEntryComponentsAreWrittenIntoTheStagedManifest() throws {
        let staged = try tempDir()
        let entry = PluginManifestParser.MarketplaceEntry(
            name: "docs", sourceValue: "./", strict: false,
            raw: ["name": "docs", "strict": false, "skills": ["./document-skills/xlsx"]])

        try PluginMarketplaceManager.applyEntryComponents(entry: entry, stagedRoot: staged)

        let data = try Data(contentsOf: staged.appendingPathComponent(".claude-plugin/plugin.json"))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        XCTAssertEqual(json["skills"] as? [String], ["./document-skills/xlsx"])
        XCTAssertEqual(json["name"] as? String, "docs")
    }

    func testStrictEntryFillsGapsButConflictingComponentsThrow() throws {
        let staged = try tempDir()
        let manifestDir = staged.appendingPathComponent(".claude-plugin")
        try FileManager.default.createDirectory(at: manifestDir, withIntermediateDirectories: true)
        try Data(#"{"name":"p","skills":["./own"]}"#.utf8).write(
            to: manifestDir.appendingPathComponent("plugin.json"))

        let filling = PluginManifestParser.MarketplaceEntry(
            name: "p", sourceValue: "./", strict: true,
            raw: ["name": "p", "agents": ["./agents/a.md"]])
        try PluginMarketplaceManager.applyEntryComponents(entry: filling, stagedRoot: staged)
        let merged = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(contentsOf: manifestDir.appendingPathComponent("plugin.json"))) as? [String: Any])
        XCTAssertEqual(merged["skills"] as? [String], ["./own"], "the plugin's own value wins")
        XCTAssertEqual(merged["agents"] as? [String], ["./agents/a.md"])

        let conflicting = PluginManifestParser.MarketplaceEntry(
            name: "p", sourceValue: "./", strict: true,
            raw: ["name": "p", "skills": ["./other"]])
        XCTAssertThrowsError(
            try PluginMarketplaceManager.applyEntryComponents(entry: conflicting, stagedRoot: staged))
    }

    func testStrictEntryWithComponentsButNoManifestThrowsAndPlainEntryIsUntouched() throws {
        let staged = try tempDir()
        let strict = PluginManifestParser.MarketplaceEntry(
            name: "p", sourceValue: "./", strict: true, raw: ["name": "p", "skills": ["./s"]])
        XCTAssertThrowsError(
            try PluginMarketplaceManager.applyEntryComponents(entry: strict, stagedRoot: staged))

        // No components declared: convention-directory plugins keep working.
        let plain = PluginManifestParser.MarketplaceEntry(
            name: "p", sourceValue: "./", strict: true, raw: ["name": "p"])
        try PluginMarketplaceManager.applyEntryComponents(entry: plain, stagedRoot: staged)
        XCTAssertFalse(FileManager.default.fileExists(
            atPath: staged.appendingPathComponent(".claude-plugin/plugin.json").path))
    }

    // MARK: - Config set

    func testConfigSetRefusesInsteadOfReportingFakeSuccess() {
        XCTAssertThrowsError(
            try ConfigToolExecutor.execute(
                arguments: ["action": "set", "key": "guardrails.mode", "value": "off"], project: nil)
        ) { error in
            XCTAssertTrue((error as NSError).localizedDescription.contains("not supported"))
        }
        XCTAssertNoThrow(try ConfigToolExecutor.execute(arguments: ["action": "list"], project: nil))
    }

    // MARK: - apply_patch format agreement

    func testFusionTargetsAndKindsUnderstandUnifiedDiffs() {
        let patch = """
        --- /dev/null
        +++ b/added.txt
        @@ -0,0 +1 @@
        +new
        --- a/updated.txt
        +++ b/updated.txt
        @@ -1 +1 @@
        -old
        +new
        --- a/removed.txt
        +++ /dev/null
        @@ -1 +0,0 @@
        -gone
        """
        let call = AppToolCall(name: "apply_patch", arguments: ["patch_text": patch], category: .fileWrite)
        XCTAssertEqual(ToolActionFusion.targetPaths(for: call), ["added.txt", "updated.txt", "removed.txt"])
        let kinds = ToolActionFusion.patchEntries(call).map(\.kind)
        XCTAssertEqual(kinds, ["add", "update", "delete"])
    }

    func testApplyPatchWithNoRecognizedOperationsFailsInsteadOfReportingSuccess() async throws {
        let root = try tempDir()
        let beginPatch = """
        *** Begin Patch
        *** Add File: hello.txt
        +hi
        *** End Patch
        """
        do {
            _ = try await ApplyPatchExecutor.apply(patchText: beginPatch, rootURL: root)
            XCTFail("an unrecognized patch format must not report success")
        } catch {
            XCTAssertTrue((error as NSError).localizedDescription.contains("NOTHING was applied"))
        }
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("hello.txt").path))
    }

    // MARK: - Fabricated MCP resource / goal / feedback / worktree results

    func testMcpResourceToolsAdmitTheyDoNothing() async throws {
        let dir = try tempDir()
        let server = McpServerConfig(
            name: "docs", transport: .stdio(command: "/usr/bin/true"))
        let project = AppProject(
            name: "p", rootDirectoryPath: dir.path, mcpServers: [server])

        do {
            _ = try await McpResourceExecutor.readResource(
                arguments: ["server": "docs", "uri": "docs://guide"], project: project, rootURL: dir)
            XCTFail("ReadMcpResource must not fabricate a read")
        } catch {
            let text = (error as NSError).localizedDescription
            XCTAssertTrue(text.contains("NOT read"), text)
        }
        do {
            _ = try await McpResourceExecutor.listResources(
                arguments: ["server": "docs"], project: project, rootURL: dir)
            XCTFail("ListMcpResources must not claim a connection")
        } catch {
            XCTAssertTrue((error as NSError).localizedDescription.contains("not implemented"))
        }
    }

    func testProposeGoalAndSendFeedbackAdmitNothingWasDone() {
        XCTAssertThrowsError(try ProposeGoalExecutor.execute(arguments: ["condition": "ship"]))
        XCTAssertThrowsError(try SendFeedbackExecutor.execute(arguments: ["title": "bug"]))
    }

    // MARK: - Workflow world read accepts script numerals

    func testWorldReadAcceptsTheInterpretersNumberRepresentation() async throws {
        let root = try tempDir()
        try Data("hello world".utf8).write(to: root.appendingPathComponent("README.md"))
        let world = WorkflowWorld(workspaceRoot: root, executor: WorkflowWorldTestExecutor())
        let site = WorkflowSiteKey(lane: "main", siteIndex: 0, ordinal: 0)
        let context = WorkflowAttemptContext(cancellation: WorkflowCancellationToken(), deadline: nil)

        func operation(_ limit: WorkflowCanonicalValue) -> WorkflowInterpreterOperation {
            .worldRead(operation: .object([
                "kind": .string("read"),
                "arguments": .array([.string("README.md"), limit]),
            ]))
        }

        let result = try await world.perform(operation(.number(4096.0)), at: site, context: context)
        XCTAssertNotNil(result.value)

        for bad in [WorkflowCanonicalValue.number(1.5), .number(.infinity), .number(.nan)] {
            do {
                _ = try await world.perform(operation(bad), at: site, context: context)
                XCTFail("a non-whole byte limit must be refused")
            } catch let error as WorkflowError {
                XCTAssertEqual(error.kind, .sandboxRefusal)
            }
        }
    }

    // MARK: - Workflow checker mirrors the interpreter's scoping

    private func rules(_ body: String) -> [WorkflowScriptDiagnosticRule] {
        WorkflowScriptChecker.parse("async function workflow() {\n\(body)\n}").diagnostics.map(\.rule)
    }

    func testCheckerReportsUndeclaredIdentifiers() {
        XCTAssertTrue(rules("""
        agent("planner", "p");
        const plan = await ask("planner", "p", {});
        await report(pln);
        """).contains(.undeclaredIdentifier))
        // The declaration is only visible AFTER its own value.
        XCTAssertTrue(rules("const x = x;").contains(.undeclaredIdentifier))
        // A binding from an inner block does not leak out.
        XCTAssertTrue(rules("""
        if (args.a) { const inner = 1; }
        await report(inner);
        """).contains(.undeclaredIdentifier))
        // Valid scripts stay valid.
        XCTAssertEqual(rules("""
        const xs = ["a", "b"];
        for (const x of xs) { await report(x); }
        await report(xs);
        """), [])
    }

    func testCheckerReportsDuplicateAndReservedBindings() {
        XCTAssertTrue(rules("""
        const xs = ["a"];
        for (const x of xs) { const x = 1; }
        """).contains(.duplicateBinding))
        XCTAssertTrue(rules("const a = 1; const a = 2;").contains(.duplicateBinding))
        XCTAssertTrue(rules("const args = 1;").contains(.reservedBindingName))
        // Shadowing in a NESTED block is allowed by the interpreter.
        XCTAssertEqual(rules("""
        const a = 1;
        if (args.a) { const a = 2; await report(a); }
        """), [])
    }

    func testCheckerReportsDuplicateCommandSlots() {
        XCTAssertTrue(rules("""
        command("lint", { executable: "/usr/bin/true", workingDirectory: "workspace", argv: [{ name: "path", kind: "workspaceInputPath" }, { name: "path", kind: "workspaceInputPath" }] });
        """).contains(.duplicateCommandSlot))
    }
}
