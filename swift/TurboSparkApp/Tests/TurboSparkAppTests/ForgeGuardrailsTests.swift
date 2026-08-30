import XCTest
@testable import TurboSparkApp

final class ForgeGuardrailsTests: XCTestCase {

    // MARK: - Global Settings Tests

    func testGuardrailsModePropertiesAndEncoding() throws {
        XCTAssertEqual(AppGuardrailsMode.alwaysOn.shortLabel, "Always On")
        XCTAssertEqual(AppGuardrailsMode.alwaysOff.shortLabel, "Always Off")
        XCTAssertEqual(AppGuardrailsMode.select.shortLabel, "Select")

        let settings = MacAppSettings(guardrailsMode: "alwaysOn")
        let data = try JSONEncoder().encode(settings)
        let decoded = try JSONDecoder().decode(MacAppSettings.self, from: data)
        XCTAssertEqual(decoded.guardrailsMode, "alwaysOn")

        let defaultSettings = MacAppSettings()
        XCTAssertEqual(defaultSettings.guardrailsMode, "select")
    }

    // MARK: - Project Persistence Tests

    func testProjectGuardrailsPersistence() throws {
        var project = AppProject(
            name: "Guardrail Project",
            forgeGuardrailsEnabled: true
        )
        XCTAssertEqual(project.forgeGuardrailsEnabled, true)

        let data = try JSONEncoder().encode(project)
        let decoded = try JSONDecoder().decode(AppProject.self, from: data)
        XCTAssertEqual(decoded.forgeGuardrailsEnabled, true)

        project.forgeGuardrailsEnabled = false
        let dataFalse = try JSONEncoder().encode(project)
        let decodedFalse = try JSONDecoder().decode(AppProject.self, from: dataFalse)
        XCTAssertEqual(decodedFalse.forgeGuardrailsEnabled, false)

        project.forgeGuardrailsEnabled = nil
        let dataNil = try JSONEncoder().encode(project)
        let decodedNil = try JSONDecoder().decode(AppProject.self, from: dataNil)
        XCTAssertNil(decodedNil.forgeGuardrailsEnabled)
    }

    // MARK: - Resolution Tests

    @MainActor
    func testEffectiveResolutionModes() {
        let model = AppModel()

        // 1. Always On
        model.guardrailsMode = .alwaysOn
        XCTAssertTrue(model.effectiveForgeGuardrailsEnabled)

        // 2. Always Off
        model.guardrailsMode = .alwaysOff
        XCTAssertFalse(model.effectiveForgeGuardrailsEnabled)

        // 3. Select mode with composer override
        model.guardrailsMode = .select
        model.composerGuardrailsOverride = true
        XCTAssertTrue(model.effectiveForgeGuardrailsEnabled)

        model.composerGuardrailsOverride = false
        XCTAssertFalse(model.effectiveForgeGuardrailsEnabled)

        // 4. Select mode with project override
        model.composerGuardrailsOverride = nil
        let proj = model.createProject(name: "Test", forgeGuardrailsEnabled: true)
        XCTAssertTrue(model.effectiveForgeGuardrailsEnabled)

        model.updateProject(AppProject(id: proj.id, name: "Test", forgeGuardrailsEnabled: false))
        XCTAssertFalse(model.effectiveForgeGuardrailsEnabled)

        // Clean up
        model.deleteProject(id: proj.id)
    }

    // MARK: - Dialect Rescue Tests

    func testRescueXmlFunctionDialect() {
        let text = "<function=read_file>\n{\"path\": \"src/main.rs\"}\n</function>"
        let available: Set<String> = ["read_file", "write_file", "bash"]
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: available)

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "read_file")
        XCTAssertEqual(rescued[0].arguments["path"], "src/main.rs")
    }

    func testRescueMistralToolCallsDialect() {
        let text = "[TOOL_CALLS] [{\"name\": \"bash\", \"arguments\": {\"command\": \"cargo test\"}}]"
        let available: Set<String> = ["bash", "read_file"]
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: available)

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "bash")
        XCTAssertEqual(rescued[0].arguments["command"], "cargo test")
    }

    func testRescueMarkdownToolCallBlock() {
        let text = "Here is the code:\n```tool_call\n{\n  \"name\": \"grep_search\",\n  \"arguments\": {\"query\": \"guardrails\", \"path\": \"crates/server\"}\n}\n```"
        let available: Set<String> = ["grep_search", "read_file"]
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: available)

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "grep_search")
        XCTAssertEqual(rescued[0].arguments["query"], "guardrails")
        XCTAssertEqual(rescued[0].arguments["path"], "crates/server")
    }

    // MARK: - Allowlist Enforcement (T6)

    func testRescueXmlFunctionDialectRejectsAToolOutsideTheAllowlist() {
        // Previously `availableToolNames.contains(name) || !name.isEmpty`
        // accepted ANY non-empty name, so a rescued call could manufacture
        // a real AppToolCall for a tool the current agent type was never
        // granted.
        let text = "<function=delete_everything>\n{}\n</function>"
        let available: Set<String> = ["read_file", "write_file", "bash"]
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: available)
        XCTAssertTrue(rescued.isEmpty, "A tool name outside the granted set must never be rescued into a real call.")
    }

    func testRescueJsonDialectsRejectAToolOutsideTheAllowlist() {
        let mistral = "[TOOL_CALLS] [{\"name\": \"delete_everything\", \"arguments\": {}}]"
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(from: mistral, availableToolNames: ["bash"]).isEmpty)

        let markdown = "```tool_call\n{\"name\": \"delete_everything\", \"arguments\": {}}\n```"
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(from: markdown, availableToolNames: ["bash"]).isEmpty)

        let bare = "{\"name\": \"delete_everything\", \"arguments\": {}}"
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(from: bare, availableToolNames: ["bash"]).isEmpty)
    }

    func testAllowlistCheckIsCaseInsensitive() {
        XCTAssertTrue(ForgeGuardrailsEngine.isToolNameAllowed("Read_File", in: ["read_file"]))
        XCTAssertFalse(ForgeGuardrailsEngine.isToolNameAllowed("", in: ["read_file"]))
        XCTAssertFalse(ForgeGuardrailsEngine.isToolNameAllowed("anything", in: []))
    }

    func testRescueBareJsonObject() {
        let text = "{\"name\": \"read_file\", \"arguments\": {\"path\": \"README.md\"}}"
        let available: Set<String> = ["read_file"]
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: available)

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "read_file")
        XCTAssertEqual(rescued[0].arguments["path"], "README.md")
    }

    // MARK: - Schema Validation & Retry Tests

    func testSchemaValidationValidCall() {
        let tool = OpenAITool.function(
            name: "read_file",
            description: "Read file contents",
            parameters: .object(
                properties: ["path": .string(description: "Path to file")],
                required: ["path"]
            )
        )

        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "Package.swift"],
            rawInvocation: "",
            status: .pendingApproval,
            category: .fileRead,
            riskAssessment: ToolRiskAssessment(level: .low, category: .fileRead, reasons: ["Read"])
        )

        let verdict = ForgeGuardrailsEngine.inspect(
            text: "Reading file",
            parsedCalls: [call],
            availableTools: [tool]
        )

        XCTAssertEqual(verdict, ForgeGuardrailVerdict.accept)
    }

    func testSchemaValidationMissingRequiredParameter() {
        let tool = OpenAITool.function(
            name: "read_file",
            description: "Read file contents",
            parameters: .object(
                properties: ["path": .string(description: "Path to file")],
                required: ["path"]
            )
        )

        let call = AppToolCall(
            name: "read_file",
            arguments: [:],
            rawInvocation: "",
            status: .pendingApproval,
            category: .fileRead,
            riskAssessment: ToolRiskAssessment(level: .low, category: .fileRead, reasons: ["Read"])
        )

        let verdict = ForgeGuardrailsEngine.inspect(
            text: "Reading file",
            parsedCalls: [call],
            availableTools: [tool]
        )

        switch verdict {
        case .retry(let nudge):
            XCTAssertTrue(nudge.contains("missing required parameter `path`"))
            XCTAssertTrue(nudge.contains("Call the tool again with corrected arguments"))
        default:
            XCTFail("Expected retry verdict for missing required parameter")
        }
    }

    func testSchemaValidationTypeMismatch() {
        let tool = OpenAITool.function(
            name: "count_lines",
            description: "Count lines",
            parameters: .object(
                properties: [
                    "limit": .integer(description: "Max lines"),
                    "verbose": .boolean(description: "Verbose mode")
                ],
                required: ["limit"]
            )
        )

        let call = AppToolCall(
            name: "count_lines",
            arguments: ["limit": "not-a-number", "verbose": "invalid-bool"],
            rawInvocation: "",
            status: .pendingApproval,
            category: .fileRead,
            riskAssessment: ToolRiskAssessment(level: .low, category: .fileRead, reasons: ["Read"])
        )

        let verdict = ForgeGuardrailsEngine.inspect(
            text: "Counting lines",
            parsedCalls: [call],
            availableTools: [tool]
        )

        switch verdict {
        case .retry(let nudge):
            XCTAssertTrue(nudge.contains("parameter `limit` must be an integer"))
            XCTAssertTrue(nudge.contains("parameter `verbose` must be a boolean"))
        default:
            XCTFail("Expected retry verdict for type mismatch")
        }
    }

    func testProseSanitization() {
        let raw = "<function=read_file>\n{\"path\": \"src/main.rs\"}\n</function>"
        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "src/main.rs"],
            rawInvocation: raw,
            status: .pendingApproval,
            category: .fileRead,
            riskAssessment: ToolRiskAssessment(level: .low, category: .fileRead, reasons: ["Read"])
        )

        let sanitized = ForgeGuardrailsEngine.sanitizeProse(text: raw, rescuedCalls: [call])
        XCTAssertEqual(sanitized, "")

        let mixed = "I will read the file now:\n\(raw)\nPlease wait."
        let sanitizedMixed = ForgeGuardrailsEngine.sanitizeProse(text: mixed, rescuedCalls: [call])
        XCTAssertEqual(sanitizedMixed, "I will read the file now:\n\nPlease wait.")
    }
}
