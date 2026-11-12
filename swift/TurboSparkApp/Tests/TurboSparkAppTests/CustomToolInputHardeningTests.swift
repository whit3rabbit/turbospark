import XCTest
@testable import TurboSparkApp

final class CustomToolInputHardeningTests: XCTestCase {
    private func properties(_ names: String...) -> JSONSchema {
        JSONSchema.object(
            properties: Dictionary(uniqueKeysWithValues: names.map { ($0, JSONSchemaProperty(type: "string")) }))
    }

    func testScriptArgumentValueIsNotResubstituted() async throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("s4_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        // printf '%s' echoes argv[1] verbatim; the value names the other marker.
        let tool = CustomToolDefinition(
            name: "pf", toolDescription: "d", parameters: properties("a", "secret"),
            execution: CustomToolExecution(
                type: .script, scriptContent: "printf '%s' \"$1\"", arguments: ["{{a}}"]))
        for _ in 0..<20 {  // dictionary order varies per process run, not per call; loop anyway
            let out = try await CustomToolExecutor.execute(
                tool: tool, arguments: ["a": "{{secret}}${secret}", "secret": "LEAKED"],
                projectRootURL: dir)
            XCTAssertEqual(out, "{{secret}}${secret}")
        }
    }

    func testSubstituteLiteralIsSinglePass() {
        let out = CustomToolExecutor.substituteLiteral(
            into: "{{a}} ${b} {{c}}", arguments: ["a": "${b}", "b": "{{a}}"])
        XCTAssertEqual(out, "${b} {{a}} {{c}}")
    }

    func testUnsafeToolNamesAreRejected() {
        for bad in ["../evil", "a/b", "", "x y", ".hidden", "a\u{0}b", String(repeating: "a", count: 65)] {
            XCTAssertFalse(CustomToolDefinition.isSafeName(bad), bad)
        }
        for good in ["build", "run-tests", "run_tests", "Tool2"] {
            XCTAssertTrue(CustomToolDefinition.isSafeName(good), good)
        }
    }

    func testSaveToolRejectsTraversalName() {
        let tool = CustomToolDefinition(
            name: "../../escape", toolDescription: "d",
            execution: CustomToolExecution(type: .command, command: "true"))
        XCTAssertThrowsError(try CustomToolManager.shared.saveTool(tool))
    }

    func testScriptExecutionRejectsTraversalName() async {
        let tool = CustomToolDefinition(
            name: "../../escape", toolDescription: "d",
            execution: CustomToolExecution(type: .script, scriptContent: "true"))
        do {
            _ = try await CustomToolExecutor.execute(
                tool: tool, arguments: [:], projectRootURL: FileManager.default.temporaryDirectory)
            XCTFail("unsafe name must throw")
        } catch {}
    }

    func testParserRejectsUnsafeName() throws {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("s4p_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("t.json")
        try Data(#"{"name":"../x","description":"d","command":"true"}"#.utf8).write(to: file)
        XCTAssertThrowsError(try CustomToolParser.parse(fileURL: file, scope: .userGlobal))
    }
}
