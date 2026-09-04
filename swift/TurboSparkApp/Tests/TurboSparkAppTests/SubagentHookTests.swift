import TurboSpark
import XCTest

@testable import TurboSparkApp

/// state#68 and state#74: the two things the subagent loop did differently
/// from the main one, both of them silent.
///
/// `SubagentRunner.observation` gated on `isToolAllowed`, `permissionRefusal`
/// and the depth counter and then executed. The main loop additionally runs
/// `PreToolUse` and `PostToolUse` -- so a deny hook, which state#40 made fail
/// CLOSED specifically so it can be relied on, was bypassed on the one path
/// that runs unattended for `maxTurns` turns. And every observation it fed
/// back was a `.system` message, which is state#32's defect: three of the
/// five fallback renderers refuse a mid-history system message outright.
@MainActor
final class SubagentHookTests: XCTestCase {
    private func makeProject(fileWrite: AppToolPermission = .allow) throws -> (AppProject, () -> Void) {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("subagent_hooks_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(
            name: "P",
            rootDirectoryPath: dir.path,
            permissions: AppProjectPermissions(mode: .auto, fileWrite: fileWrite))
        return (project, { try? FileManager.default.removeItem(at: dir) })
    }

    // MARK: - state#68

    func testAPreToolUseDenyHookRefusesASubagentToolCall() async throws {
        let (project, cleanup) = try makeProject()
        defer { cleanup() }

        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Deny Writes",
            event: .preToolUse,
            type: .command,
            command:
                "echo '{\"hookSpecificOutput\":{\"permissionDecision\":\"deny\","
                + "\"permissionDecisionReason\":\"writes are blocked here\"}}'",
            matcher: "write_file",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let call = AppToolCall(
            name: "write_file",
            arguments: ["path": "escaped.txt", "content": "hello"],
            category: .fileWrite)
        // `general-purpose` by name, not `builtInAgents[0]`: that is
        // `explore`, which disallows every write and would refuse this
        // call before any hook ran.
        let agent = try XCTUnwrap(AgentManager.shared.findAgent(name: "general-purpose"))

        let observation = await SubagentRunner.observation(
            for: call, agent: agent, project: project)

        XCTAssertTrue(
            observation.content.contains("writes are blocked here"),
            "The hook's own reason is what the subagent must be told. Got: \(observation.content)")
        // **THE DISCRIMINATING ASSERTION.** A refusal message alone could
        // come from any gate; what separates "the hook ran" from "the hook
        // was skipped" is whether the file exists. The project permits
        // `.fileWrite`, so nothing else in this path would refuse it.
        let target = URL(fileURLWithPath: project.rootDirectoryPath!)
            .appendingPathComponent("escaped.txt")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: target.path),
            "A denied call must not have written anything.")
    }

    func testAPostToolUseHookFoldsItsFeedbackIntoTheSubagentObservation() async throws {
        let (project, cleanup) = try makeProject()
        defer { cleanup() }

        let store = AppHookStore.shared
        let hook = AppHookCommand(
            name: "Post Note",
            event: .postToolUse,
            type: .command,
            command: "echo 'remember to re-read it' >&2; exit 2",
            matcher: "write_file",
            sourceType: .custom
        )
        store.addCustomHook(hook)
        defer { store.deleteCustomHook(id: hook.id) }

        let call = AppToolCall(
            name: "write_file",
            arguments: ["path": "written.txt", "content": "hello"],
            category: .fileWrite)
        // `general-purpose` by name, not `builtInAgents[0]`: that is
        // `explore`, which disallows every write and would refuse this
        // call before any hook ran.
        let agent = try XCTUnwrap(AgentManager.shared.findAgent(name: "general-purpose"))

        let observation = await SubagentRunner.observation(
            for: call, agent: agent, project: project)

        XCTAssertTrue(
            observation.content.contains("remember to re-read it"),
            "PostToolUse feedback must reach the subagent. Got: \(observation.content)")
        XCTAssertTrue(
            observation.content.contains("<hook_feedback>"),
            "Folded in under the same tag the main loop uses.")
    }

    // MARK: - state#74

    func testEveryObservationGoesBackAsAToolMessageAndNotASystemOne() async throws {
        let (project, cleanup) = try makeProject()
        defer { cleanup() }
        // `general-purpose` by name, not `builtInAgents[0]`: that is
        // `explore`, which disallows every write and would refuse this
        // call before any hook ran.
        let agent = try XCTUnwrap(AgentManager.shared.findAgent(name: "general-purpose"))

        // The success arm.
        let ran = await SubagentRunner.observation(
            for: AppToolCall(
                name: "write_file", arguments: ["path": "a.txt", "content": "x"],
                category: .fileWrite),
            agent: agent, project: project)
        XCTAssertEqual(
            ran.role, .tool,
            "A tool result is a `.tool` message. A mid-history `.system` one is refused by three "
                + "of the five fallback renderers and priced at `u64::MAX` by fit_window.")

        // And a refusal arm, which is the same message shape.
        let refused = SubagentRunner.errorObservation("nope")
        XCTAssertEqual(refused.role, .tool, "A refusal is fed back the same way a result is.")
    }
}
