import XCTest
@testable import TurboSparkApp

/// Regression tests for T5: `execute(call:in:)`'s default case used to
/// return `"Executed \(call.name) successfully."` with `isError: false` for
/// any tool name it had no real handler for, so the model was told a call
/// succeeded when nothing ran. `AppToolCatalog` also advertised those same
/// unimplemented tools in its system-prompt definitions.
final class FabricatedToolSuccessTests: XCTestCase {
    func testUnimplementedToolReturnsAnErrorRatherThanFabricatedSuccess() async {
        let call = AppToolCall(name: "repl", arguments: ["command": "swift"], category: .terminal)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError, "An unimplemented tool must report isError, not a fabricated success.")
        XCTAssertFalse(result.output.lowercased().contains("successfully"))
    }

    /// The three arms that returned a canned string with no side effect
    /// ("Task created: ...", "Task list: No active blocking tasks.",
    /// "Question submitted to user.") were the same T5 class as the old
    /// `default` arm, one level up: nothing in the UI read the call, so the
    /// model was told a task existed or a user had been asked when neither
    /// happened. Every spelling now falls to the honest not-implemented
    /// error, and is filtered out of the advertised list by `isImplemented`.
    func testCannedNoOpToolArmsAreGoneFromExecute() async {
        for name in ["repl", "workflow", "croncreate", "schedulewakeup"] {
            let call = AppToolCall(name: name, arguments: ["subject": "x"], category: .automation)
            let result = await AppToolRegistry.execute(call: call, in: nil)
            XCTAssertTrue(result.isError, "'\(name)' had no effect and must not report success: \(result.output)")
            XCTAssertTrue(
                result.output.contains("not implemented by this client"),
                "'\(name)' must fall to the honest error, got: \(result.output)")
            XCTAssertFalse(AppToolRegistry.isImplemented(name), "'\(name)' must not be advertised as implemented")
        }
    }

    func testMalformedMcpToolNameReturnsAnErrorRatherThanFabricatedSuccess() async {
        // "mcp__" prefix with too few "__"-separated parts to name a server + tool.
        let call = AppToolCall(name: "mcp__onlyserver", arguments: [:], category: .mcp)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
    }

    func testImplementedToolsAreUnaffected() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        try "hello".write(to: dir.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        let project = AppProject(name: "t", rootDirectoryPath: dir.path)

        let call = AppToolCall(name: "read_file", arguments: ["path": "a.txt"], category: .fileRead)
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("hello"))
    }

    // MARK: - Unimplemented tools are not advertised to the model

    func testUnimplementedToolNamesAreExcludedFromAllTools() {
        let advertisedNames = Set(AppToolCatalog.allTools.map { $0.function.name.lowercased() })
        for unimplemented in ["repl", "workflow", "croncreate", "crondelete", "cronlist", "schedulewakeup"] {
            XCTAssertFalse(
                advertisedNames.contains(unimplemented),
                "'\(unimplemented)' has no executor and must not be advertised to the model."
            )
        }
    }

    func testImplementedToolNamesAreStillAdvertised() {
        let advertisedNames = Set(AppToolCatalog.allTools.map { $0.function.name.lowercased() })
        for implemented in [
            "bash", "fileread", "filewrite", "fileedit", "grep", "glob", "apply_patch",
            "skill", "agent", "todowrite", "webfetch", "websearch",
            "askuserquestion", "enterplanmode", "exitplanmode", "reportfindings",
            "proposeskills", "proposegoal", "sendfeedback", "notebookedit",
            "snip", "senduserfile", "taskcreate", "taskget", "tasklist",
            "taskupdate", "taskstop", "taskoutput", "sleep", "pushnotification",
            "config", "ctxinspect", "enterworktree", "exitworktree", "listmcpresources", "readmcpresource"
        ] {
            XCTAssertTrue(
                advertisedNames.contains(implemented.lowercased()),
                "'\(implemented)' has a real executor and should remain advertised."
            )
        }
    }

    func testToolsForEveryAgentTypeExcludeUnimplementedNames() {
        for agentType in AppAgentType.allCases {
            let names = Set(AppToolCatalog.tools(for: agentType).map { $0.function.name.lowercased() })
            XCTAssertTrue(names.contains("webfetch"), "\(agentType) tool list must include implemented WebFetch by default.")
            XCTAssertTrue(names.contains("websearch"), "\(agentType) tool list must include implemented WebSearch by default.")
            XCTAssertFalse(names.contains("repl"), "\(agentType) tool list must not include unimplemented REPL.")
        }
    }

    func testIsImplementedRecognizesDynamicMcpToolNames() {
        XCTAssertTrue(AppToolRegistry.isImplemented("mcp__filesystem__read_file"))
        XCTAssertFalse(AppToolRegistry.isImplemented("SomeRandomTool"))
    }
}
