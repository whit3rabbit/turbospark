import Foundation
import XCTest

@testable import TurboSparkApp

/// A model that calls a tool in the Claude Code, opencode or Pi vocabulary is
/// understood when this turn offered that tool, and never when it did not.
final class ToolCallNormalizationTests: XCTestCase {
    private func offered(
        profile: AppToolProfile = .default, webTools: Bool = true
    ) -> TurnAvailableTools {
        AppToolCatalog.captureTurnAvailableTools(
            for: AppProject(name: "fixture", agentType: .coder), chatToolsEnabled: true,
            globalMcpServers: [], contextTokens: nil, webToolsEnabled: webTools, profile: profile)
    }

    private func validated(
        _ tools: TurnAvailableTools, _ name: String, _ json: String,
        file: StaticString = #filePath, line: UInt = #line
    ) -> (tool: String, arguments: [String: ToolCallJSONValue])? {
        guard case .validated(let tool, let arguments) = tools.validate(
            toolName: name, argumentsJSON: json)
        else {
            XCTFail("\(name) \(json) was refused", file: file, line: line)
            return nil
        }
        return (tool.function.name, arguments)
    }

    // MARK: Names

    func testOtherSpellingsOfAnOfferedToolResolveToIt() {
        let tools = offered()
        for name in ["FileRead", "read", "Read", "read_file", "READ"] {
            XCTAssertEqual(tools.resolvedDefinition(for: name)?.function.name, "FileRead", name)
        }
        for name in ["Bash", "bash", "shell", "run_command"] {
            XCTAssertEqual(tools.resolvedDefinition(for: name)?.function.name, "Bash", name)
        }
        for name in ["Agent", "task", "Task", "subagent"] {
            XCTAssertEqual(tools.resolvedDefinition(for: name)?.function.name, "Agent", name)
        }
        XCTAssertEqual(tools.resolvedDefinition(for: "todo_write")?.function.name, "TodoWrite")
    }

    func testAnExactNameStillWinsOverAFamilyMatch() {
        let tools = offered()
        // `grep_search` and `Grep` are different tools with different schemas.
        XCTAssertEqual(tools.resolvedDefinition(for: "grep_search")?.function.name, "grep_search")
        XCTAssertEqual(tools.resolvedDefinition(for: "Grep")?.function.name, "Grep")
    }

    func testNormalizationNeverReachesAToolTheTurnDidNotOffer() {
        let readOnly = offered(profile: .readOnly)
        XCTAssertNil(readOnly.resolvedDefinition(for: "bash"))
        XCTAssertNil(readOnly.resolvedDefinition(for: "shell"))
        XCTAssertNil(readOnly.resolvedDefinition(for: "task"))
        XCTAssertNil(readOnly.resolvedDefinition(for: "edit"))
        guard case .refused(.unavailableTool) = readOnly.validate(
            toolName: "bash", argumentsJSON: #"{"command":"rm -rf x"}"#)
        else { return XCTFail("a removed tool must stay refused under any spelling") }
    }

    func testGrepAndGrepSearchDoNotResolveToEachOther() {
        let onlyGrepSearch = AppToolProfile(
            name: "t", allowedTools: ["grep_search"], includesMcpAndCustom: false)
        XCTAssertNil(offered(profile: onlyGrepSearch).resolvedDefinition(for: "Grep"))
        let onlyGrep = AppToolProfile(name: "t", allowedTools: ["grep"], includesMcpAndCustom: false)
        XCTAssertNil(offered(profile: onlyGrep).resolvedDefinition(for: "grep_search"))
    }

    // MARK: Arguments

    func testOpencodeAndPiFileArgumentsReachTheDeclaredNames() throws {
        let tools = offered()
        let pi = try XCTUnwrap(validated(tools, "read", #"{"path":"a.txt","offset":3}"#))
        XCTAssertEqual(pi.tool, "FileRead")
        XCTAssertEqual(pi.arguments["file_path"], .string("a.txt"))
        XCTAssertNil(pi.arguments["path"])

        let opencode = try XCTUnwrap(validated(tools, "edit", """
            {"filePath":"a.txt","oldString":"x","newString":"y","replaceAll":true}
            """))
        XCTAssertEqual(opencode.tool, "FileEdit")
        XCTAssertEqual(opencode.arguments["file_path"], .string("a.txt"))
        XCTAssertEqual(opencode.arguments["old_string"], .string("x"))
        XCTAssertEqual(opencode.arguments["new_string"], .string("y"))
        XCTAssertEqual(opencode.arguments["replace_all"], .boolean(true))

        let write = try XCTUnwrap(validated(tools, "write", #"{"filePath":"a","content":"hi"}"#))
        XCTAssertEqual(write.arguments["file_path"], .string("a"))
    }

    func testOtherToolsFoldTheirOwnSpellings() throws {
        let tools = offered()
        let bash = try XCTUnwrap(validated(tools, "bash", #"{"command":"ls","background":true}"#))
        XCTAssertEqual(bash.arguments["run_in_background"], .boolean(true))

        let grep = try XCTUnwrap(validated(
            tools, "Grep", #"{"pattern":"x","include":"*.swift","ignoreCase":true}"#))
        XCTAssertEqual(grep.arguments["glob"], .string("*.swift"))
        XCTAssertEqual(grep.arguments["-i"], .boolean(true))

        let search = try XCTUnwrap(validated(tools, "websearch", #"{"query":"q","numResults":4}"#))
        XCTAssertEqual(search.arguments["num_results"], .number(4))
    }

    func testACanonicalValueIsNeverOverwrittenByASynonym() throws {
        let result = try XCTUnwrap(validated(
            offered(), "read", #"{"file_path":"real.txt","path":"other.txt"}"#))
        XCTAssertEqual(result.arguments["file_path"], .string("real.txt"))
    }

    func testTodoWriteDoesNotRequireActiveForm() throws {
        let todo = try XCTUnwrap(validated(offered(), "todowrite", """
            {"todos":[{"content":"Write tests","status":"in_progress"}]}
            """))
        XCTAssertEqual(todo.tool, "TodoWrite")
    }

    func testAMissingRequiredArgumentIsStillRefused() {
        guard case .refused(.invalidArguments) = offered().validate(
            toolName: "read", argumentsJSON: #"{"offset":3}"#)
        else { return XCTFail("no spelling of the path means the call is still incomplete") }
    }

    // MARK: Gate

    func testTheGateDispatchesTheOfferedNameAndCanonicalArguments() throws {
        let content = """
            <tool_call>
            <name>read</name>
            <arguments>{"path": "notes.md"}</arguments>
            </tool_call>
            """
        let result = ToolCallDispatchGate.evaluate(
            content: content, streamState: .completed, availableTools: offered(),
            forgeGuardrailsEnabled: false)
        XCTAssertTrue(result.refusals.isEmpty, "\(result.refusals)")
        let call = try XCTUnwrap(result.dispatchableCalls.first)
        XCTAssertEqual(call.name, "FileRead")
        XCTAssertEqual(call.arguments["file_path"], "notes.md")
    }
}
