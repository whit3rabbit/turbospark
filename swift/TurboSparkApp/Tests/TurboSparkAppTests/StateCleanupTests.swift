import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Tier 3 of the third state-layer review: the small ones. Mostly fields with
/// no writer, keys that were not unique, and parsers that were right about
/// the common case.
@MainActor
final class StateCleanupTests: XCTestCase {
    private func model(alias: String, path: String, family: String = "gemma4") -> InstalledModel {
        InstalledModel(alias: alias, repo: "x/y", path: path, family: family)
    }

    // MARK: - state#96

    func testAReasoningDefaultIsRememberedAgainstThePathNotTheAlias() {
        let appModel = AppModel()
        let shared = "shared-alias"
        appModel.selected = model(alias: shared, path: "/models/a.gturbo")
        appModel.setReasoning(.low)
        appModel.selected = model(alias: shared, path: "/models/b.gturbo")
        appModel.setReasoning(.medium)

        XCTAssertEqual(
            appModel.modelReasoningDefaults["/models/a.gturbo"], GenerateOptions.Reasoning.low.rawValue,
            "An alias is not unique once a scanned row exists, so two installs sharing a name "
                + "shared one remembered level.")
        XCTAssertEqual(
            appModel.modelReasoningDefaults["/models/b.gturbo"],
            GenerateOptions.Reasoning.medium.rawValue)
        XCTAssertNil(
            appModel.modelReasoningDefaults[shared],
            "The alias arm of the old `??` was reached first, so the path arm was dead code.")
    }

    // MARK: - state#97

    func testToolCallingIsNotAnsweredForSomeOtherInstalledModel() {
        let appModel = AppModel()
        appModel.selected = nil
        appModel.installed = [model(alias: "odd", path: "/models/odd.gturbo", family: "notafamily")]
        XCTAssertTrue(
            appModel.isToolCallingSupported,
            "With nothing selected there is no answer; falling back to `installed.first` made "
                + "an unrelated install decide the guardrails default.")

        // **THE SECOND HALF OF THIS CASE USED TO ASSERT A FALSEHOOD, and it
        // is corrected here rather than deleted** (`swift/CLAUDE.md`
        // Gotcha 42: a red pre-existing test after a state fix is as likely
        // to be the defect reproducing as a regression).
        //
        // It read "the family list must track the engine's own (museGlimmer,
        // qwen4exp and deepseekV4Flash were missing)" and selected a
        // museGlimmer install to prove it. museGlimmer is NOT natively
        // tool-calling: its checkpoint frames calls as an
        // `<atem:function_calls>` block this engine has no parser for, so
        // `StructuredAssistantDecoder` reports them as reasoning and no call
        // is ever handed over. The list was wrong in the direction the case
        // was pinning.
        //
        // There is no family list at all now -- the answer comes from
        // `info.toolCalling.native`, which is a property of the DIALECT and
        // therefore unanswerable from an install row. So selecting ANY
        // family with no session open must still give the permissive default,
        // which is what state#97 established and is all this can check
        // offline.
        for family in ["museGlimmer", "gemma4", "notafamily"] {
            appModel.selected = model(
                alias: "m", path: "/models/m.gturbo", family: family)
            XCTAssertTrue(
                appModel.isToolCallingSupported,
                "\(family): with no session there is no answer, and the family must not be "
                    + "consulted -- it cannot know the checkpoint's dialect.")
        }
    }

    // MARK: - state#99

    func testStopDuringAnApprovedToolDoesNotLatchTheCancelFlag() async throws {
        let appModel = AppModel()
        let chat = AppChat(title: "one")
        appModel.chats = [chat]
        appModel.selectedChatID = chat.id

        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("cancel_latch_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let project = AppProject(
            name: "P", rootDirectoryPath: dir.path,
            permissions: AppProjectPermissions(mode: .auto, fileRead: .allow))
        appModel.projects = [project]

        let call = AppToolCall(
            name: "list_directory", arguments: ["path": "."], category: .fileRead)
        appModel.pendingToolCall = call
        appModel.pendingToolCallChatID = chat.id
        appModel.pendingToolCallProject = project

        appModel.approvePendingToolCall(id: call.id)
        appModel.cancel()
        XCTAssertTrue(appModel.isCancellationPending, "Precondition: Stop was pressed.")

        let deadline = Date().addingTimeInterval(5)
        while appModel.generating && Date() < deadline {
            try await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTAssertFalse(
            appModel.isCancellationPending,
            "Nothing else runs a tail after a cancelled tool, so a latched flag left `canCancel` "
                + "false and `continueAgentLoop` refusing every later turn for the life of the "
                + "process.")
    }

    // MARK: - state#102

    func testTheInstallETAIsSilentUntilItHasSomethingToSay() {
        XCTAssertNil(
            AppModel.installETA(done: 1_000, total: 1_000_000, elapsed: 0.2),
            "A rate measured over the first instant of a multi-gigabyte stream is wrong by an "
                + "order of magnitude and then visibly corrects itself.")
        XCTAssertNil(
            AppModel.installETA(done: 0, total: 1_000_000, elapsed: 30),
            "Nothing downloaded is no rate at all.")
        XCTAssertNil(
            AppModel.installETA(done: 1_000_000, total: 1_000_000, elapsed: 30),
            "And a finished download has nothing remaining.")

        // 100 MB of 1 GB in 10 s is 10 MB/s, so 900 MB left is 90 s.
        XCTAssertEqual(
            AppModel.installETA(done: 100_000_000, total: 1_000_000_000, elapsed: 10.0),
            "about 1m remaining")
        // 900 of 1,000 in 10 s is 90/s, so the last 100 is just over a second.
        XCTAssertEqual(
            AppModel.installETA(done: 900, total: 1_000, elapsed: 10.0),
            "about 1s remaining")
        // 1 GB of 100 GB in 10 s is 100 MB/s, so 99 GB left is about 16m.
        XCTAssertEqual(
            AppModel.installETA(done: 1_000_000_000, total: 100_000_000_000, elapsed: 10.0),
            "about 16m remaining")
    }

    // MARK: - state#106

    func testABlankNameInAJSONAgentFileFallsBackToTheFileName() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("agent_json_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let fileURL = dir.appendingPathComponent("reviewer-two.json")
        try #"{"name": "  ", "description": "d", "prompt": "p"}"#
            .write(to: fileURL, atomically: true, encoding: .utf8)

        let parsed = try AgentParser.parseFile(at: fileURL, scope: .userGlobal, sourceAgent: .custom)
        XCTAssertEqual(
            parsed.name, "reviewer-two",
            "An agent keyed on the empty string takes the effective map's \"\" slot, so two such "
                + "files collapse into one and neither is reachable by name.")
    }

    // MARK: - state#109

    func testABlockScalarBodyIndentedByOneSpaceIsStillTheDescription() {
        let text = """
        ---
        name: deploy
        description: |
         Ships the thing.

         name: not-a-key
        ---
        Body.
        """
        let parsed = SkillParser.parseContent(rawText: text)
        XCTAssertEqual(
            parsed.name, "deploy",
            "Ending the block at a one-space line reparses the paragraph as keys, and a `name:` "
                + "in that prose renames the skill.")
        XCTAssertTrue(parsed.skillDescription.contains("not-a-key"))
    }

    func testABlockScalarMarkerOnAListKeyIsNotAOneElementList() {
        let text = """
        ---
        name: deploy
        description: d
        allowed-tools: |
          - read_file
          - run_command
        ---
        Body.
        """
        let parsed = SkillParser.parseContent(rawText: text)
        XCTAssertEqual(
            parsed.manifest.allowedTools ?? [], ["read_file", "run_command"],
            "The marker fell to the scalar arm and produced a list whose one entry is the "
                + "literal \"|\", which matches no tool.")
    }

    func testACommaInsideAQuotedArrayElementDoesNotEndIt() {
        XCTAssertEqual(
            SkillParser.parseInlineArray("[\"Bash(git commit -m 'a, b')\", \"read_file\"]"),
            ["Bash(git commit -m 'a, b')", "read_file"],
            "Splitting on every comma cut an allowlist entry in half and produced two that name "
                + "no tool at all.")
        XCTAssertEqual(
            SkillParser.parseInlineArray("[a, b, c]"), ["a", "b", "c"],
            "And the ordinary form is unchanged.")
    }
}
