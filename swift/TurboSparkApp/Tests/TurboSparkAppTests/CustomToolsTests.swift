import XCTest
@testable import TurboSparkApp

final class CustomToolsTests: XCTestCase {
    private func makeTempDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("custom_tools_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    func testCustomToolDefinitionToOpenAITool() {
        let execution = CustomToolExecution(
            type: .command,
            command: "echo 'hello {{name}}'"
        )
        let schema = JSONSchema.object(
            properties: [
                "name": JSONSchemaProperty.string(description: "Target user name")
            ],
            required: ["name"]
        )
        let tool = CustomToolDefinition(
            name: "greet_user",
            toolDescription: "Greets a user by name",
            category: .terminal,
            parameters: schema,
            execution: execution
        )

        let openAI = tool.openAITool
        XCTAssertEqual(openAI.function.name, "greet_user")
        XCTAssertEqual(openAI.function.description, "Greets a user by name")
        XCTAssertEqual(openAI.function.parameters.required, ["name"])
    }

    func testCustomToolParserFromJSON() throws {
        let json = """
        {
            "name": "calc_checksum",
            "displayName": "Calculate Checksum",
            "toolDescription": "Computes SHA-256 for a file",
            "category": "terminal",
            "execution": {
                "type": "command",
                "command": "shasum -a 256 {{path}}"
            },
            "parameters": {
                "type": "object",
                "properties": {
                    "path": { "type": "string", "description": "Target file path" }
                },
                "required": ["path"]
            }
        }
        """
        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let fileURL = tempDir.appendingPathComponent("calc_checksum.json")
        try json.write(to: fileURL, atomically: true, encoding: .utf8)

        let parsed = try CustomToolParser.parse(fileURL: fileURL, scope: .userGlobal)
        XCTAssertEqual(parsed.name, "calc_checksum")
        XCTAssertEqual(parsed.displayName, "Calculate Checksum")
        XCTAssertEqual(parsed.execution.type, .command)
        XCTAssertEqual(parsed.execution.command, "shasum -a 256 {{path}}")
    }

    func testCustomToolExecutorCommandInterpolation() async throws {
        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let execution = CustomToolExecution(
            type: .command,
            command: "echo 'Result: {{greeting}} {{target}}'"
        )
        let tool = CustomToolDefinition(
            name: "echo_greeting",
            toolDescription: "Echo greeting",
            execution: execution
        )

        let output = try await CustomToolExecutor.execute(
            tool: tool,
            arguments: ["greeting": "Hello", "target": "TurboSpark"],
            projectRootURL: tempDir
        )

        XCTAssertTrue(output.contains("Result: Hello TurboSpark"))
    }

    func testCustomToolExecutorScriptExecution() async throws {
        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let script = """
        #!/bin/zsh
        echo "Script Arg 1: $1"
        """
        let execution = CustomToolExecution(
            type: .script,
            scriptContent: script,
            arguments: ["{{val}}"]
        )
        let tool = CustomToolDefinition(
            name: "run_script",
            toolDescription: "Runs script",
            execution: execution
        )

        let output = try await CustomToolExecutor.execute(
            tool: tool,
            arguments: ["val": "42"],
            projectRootURL: tempDir
        )

        XCTAssertTrue(output.contains("Script Arg 1: 42"))
    }

    func testAppToolExecutionWithCustomTool() async throws {
        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let projectToolsDir = tempDir.appendingPathComponent(".turbospark/tools", isDirectory: true)
        try FileManager.default.createDirectory(at: projectToolsDir, withIntermediateDirectories: true)

        let json = """
        {
            "name": "custom_echo",
            "toolDescription": "Custom echo tool",
            "category": "terminal",
            "execution": {
                "type": "command",
                "command": "echo 'CustomEcho: {{msg}}'"
            }
        }
        """
        let toolURL = projectToolsDir.appendingPathComponent("custom_echo.json")
        try json.write(to: toolURL, atomically: true, encoding: .utf8)

        let project = AppProject(name: "TestProj", rootDirectoryPath: tempDir.path)

        let call = AppToolCall(
            name: "custom_echo",
            arguments: ["msg": "WorksSuccessfully"]
        )

        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "Execution should succeed: \(result.output)")
        XCTAssertTrue(result.output.contains("CustomEcho: WorksSuccessfully"))
    }

    func testEditFileWithCurlyQuotesAndLinePrefixes() async throws {
        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let source = """
        let quote = "Hello World";
        let message = 'Welcome';
        """
        let fileURL = tempDir.appendingPathComponent("code.swift")
        try source.write(to: fileURL, atomically: true, encoding: .utf8)

        let project = AppProject(name: "TestProj", rootDirectoryPath: tempDir.path)

        // Model passes line number prefix '  1 | ' and curly quotes
        let call = AppToolCall(
            name: "edit_file",
            arguments: [
                "path": "code.swift",
                "old_string": "  1 | let quote = “Hello World”;",
                "new_string": "let quote = \"Hello TurboSpark\";"
            ],
            category: .fileWrite
        )

        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "edit_file should succeed with line prefix and curly quotes: \(result.output)")

        let updated = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertTrue(updated.contains("Hello TurboSpark"))
    }

    func testEditFileAmbiguousMatchError() async throws {
        let tempDir = try makeTempDirectory()
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let source = """
        let x = 1;
        let x = 1;
        """
        let fileURL = tempDir.appendingPathComponent("ambig.swift")
        try source.write(to: fileURL, atomically: true, encoding: .utf8)

        let project = AppProject(name: "TestProj", rootDirectoryPath: tempDir.path)

        let call = AppToolCall(
            name: "edit_file",
            arguments: [
                "path": "ambig.swift",
                "old_string": "let x = 1;",
                "new_string": "let x = 2;"
            ],
            category: .fileWrite
        )

        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("appears 2 times"))
    }

    func testCompactOutputHeadAndTailPreservation() {
        let lines = (1...200).map { "Line \($0): status log entry" }.joined(separator: "\n")
        let compacted = AppToolRegistry.compactOutput(lines, maxLines: 50)

        XCTAssertTrue(compacted.contains("Line 1:"))
        XCTAssertTrue(compacted.contains("Line 25:"))
        XCTAssertTrue(compacted.contains("lines truncated"))
        XCTAssertTrue(compacted.contains("Line 200:"))
    }
}
