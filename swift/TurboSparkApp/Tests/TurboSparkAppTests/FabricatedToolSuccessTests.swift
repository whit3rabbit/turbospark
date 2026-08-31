import XCTest
@testable import TurboSparkApp

/// Regression tests for T5: `execute(call:in:)`'s default case used to
/// return `"Executed \(call.name) successfully."` with `isError: false` for
/// any tool name it had no real handler for, so the model was told a call
/// succeeded when nothing ran. `AppToolCatalog` also advertised those same
/// unimplemented tools in its system-prompt definitions.
final class FabricatedToolSuccessTests: XCTestCase {
    func testUnimplementedToolReturnsAnErrorRatherThanFabricatedSuccess() async {
        let call = AppToolCall(name: "WebSearch", arguments: ["query": "swift"], category: .web)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError, "An unimplemented tool must report isError, not a fabricated success.")
        XCTAssertFalse(result.output.lowercased().contains("successfully"))
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
        for unimplemented in ["websearch", "repl", "notebookedit", "croncreate", "schedulewakeup", "taskstop", "listmcpresources"] {
            XCTAssertFalse(
                advertisedNames.contains(unimplemented),
                "'\(unimplemented)' has no executor and must not be advertised to the model."
            )
        }
    }

    func testImplementedToolNamesAreStillAdvertised() {
        let advertisedNames = Set(AppToolCatalog.allTools.map { $0.function.name.lowercased() })
        for implemented in ["bash", "fileread", "filewrite", "fileedit", "grep", "glob", "apply_patch", "skill", "agent", "taskcreate", "tasklist", "askuserquestion", "webfetch"] {
            XCTAssertTrue(
                advertisedNames.contains(implemented),
                "'\(implemented)' has a real executor and should remain advertised."
            )
        }
    }

    func testToolsForEveryAgentTypeExcludeUnimplementedNames() {
        for agentType in AppAgentType.allCases {
            let names = Set(AppToolCatalog.tools(for: agentType).map { $0.function.name.lowercased() })
            XCTAssertTrue(names.contains("webfetch"), "\(agentType) tool list must include implemented WebFetch by default.")
            XCTAssertFalse(names.contains("websearch"), "\(agentType) tool list must not include unimplemented WebSearch.")
            XCTAssertFalse(names.contains("repl"), "\(agentType) tool list must not include unimplemented REPL.")
        }
    }

    func testIsImplementedRecognizesDynamicMcpToolNames() {
        XCTAssertTrue(AppToolRegistry.isImplemented("mcp__filesystem__read_file"))
        XCTAssertFalse(AppToolRegistry.isImplemented("SomeRandomTool"))
    }
}
