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

    // MARK: - Extra Dialect Rescue (GLM / MiniMax / Kimi K2 / Longcat)
    //
    // Mirrors of the Rust fixtures in crates/server/src/guardrails/tests.rs,
    // kept side by side so the two implementations can be checked for parity
    // by eye. None of these four families is installed on this machine;
    // every fixture is the markup the family's own published chat template
    // teaches.

    func testRescueGlmArgKeyPairs() {
        let text = "<tool_call>get_weather\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n<arg_key>days</arg_key>\n<arg_value>3</arg_value>\n</tool_call>"
        let available: Set<String> = ["get_weather", "read_file"]
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: available)

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "get_weather")
        XCTAssertEqual(rescued[0].arguments["city"], "Oslo")
        XCTAssertEqual(rescued[0].arguments["days"], "3")
    }

    func testRescueGlmPairsWithWrapperStripped() {
        // The shape that reaches the rescue when a GLM vocabulary loads
        // under a dialect whose decoder swallows the `<tool_call>` special
        // tokens: name and pairs only.
        let text = "get_weather\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n"
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: ["get_weather"])

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "get_weather")
        XCTAssertEqual(rescued[0].arguments["city"], "Oslo")
    }

    func testRescueGlmRejectsAToolOutsideTheAllowlist() {
        let text = "<tool_call>delete_everything\n<arg_key>city</arg_key>\n<arg_value>Oslo</arg_value>\n</tool_call>"
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: ["get_weather"]).isEmpty)
    }

    func testRescueMinimaxInvokeBlock() {
        let text = "<minimax:tool_call>\n<invoke name=\"get_weather\">\n<parameter name=\"city\">Oslo</parameter>\n</invoke>\n</minimax:tool_call>"
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: ["get_weather"])

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "get_weather")
        XCTAssertEqual(rescued[0].arguments["city"], "Oslo")
    }

    func testRescueKimiK2SectionId() {
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>functions.get_weather:0<|tool_call_argument_begin|>{\"city\": \"Oslo\"}<|tool_call_end|><|tool_calls_section_end|>"
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: ["get_weather"])

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "get_weather")
        XCTAssertEqual(rescued[0].arguments["city"], "Oslo")
    }

    func testRescueKimiAnomalyIdRescuesNothing() {
        // Moonshot's documented anomaly: an opaque `call_...` id carries no
        // name, so nothing may be rescued from it.
        let text = "<|tool_calls_section_begin|><|tool_call_begin|>call_59adf5614cfe4f4b8a71be54<|tool_call_argument_begin|>{\"city\": \"Oslo\"}<|tool_call_end|><|tool_calls_section_end|>"
        XCTAssertTrue(ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: ["get_weather"]).isEmpty)
    }

    func testRescueLongcatTaggedJson() {
        let text = "<longcat_tool_call>\n{\"name\": \"get_weather\", \"arguments\": {\"city\": \"Oslo\"}}\n</longcat_tool_call>"
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: ["get_weather"])

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "get_weather")
        XCTAssertEqual(rescued[0].arguments["city"], "Oslo")
    }

    func testRescueGemmaDsl() {
        let text = "call:get_weather{city: \"Oslo\", days: 3}"
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: ["get_weather"])

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "get_weather")
        XCTAssertEqual(rescued[0].arguments["city"], "Oslo")
        XCTAssertEqual(rescued[0].arguments["days"], "3")
    }

    func testRescueQwenXmlParameters() {
        let text = "<function=get_weather>\n<parameter=city>Oslo</parameter>\n<parameter=days>3</parameter>\n</function>"
        let rescued = ForgeGuardrailsEngine.rescueToolCalls(from: text, availableToolNames: ["get_weather"])

        XCTAssertEqual(rescued.count, 1)
        XCTAssertEqual(rescued[0].name, "get_weather")
        XCTAssertEqual(rescued[0].arguments["city"], "Oslo")
        XCTAssertEqual(rescued[0].arguments["days"], "3")
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

    func testProseSanitizationWithWrappers() {
        let text = "<|tool_calls_section_begin|><function=read_file>{\"path\": \"src/main.rs\"}</function><|tool_calls_section_end|>"
        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "src/main.rs"],
            rawInvocation: "<function=read_file>{\"path\": \"src/main.rs\"}</function>",
            status: .pendingApproval,
            category: .fileRead,
            riskAssessment: ToolRiskAssessment(level: .low, category: .fileRead, reasons: ["Read"])
        )
        let sanitized = ForgeGuardrailsEngine.sanitizeProse(text: text, rescuedCalls: [call])
        XCTAssertEqual(sanitized, "")
    }
}
