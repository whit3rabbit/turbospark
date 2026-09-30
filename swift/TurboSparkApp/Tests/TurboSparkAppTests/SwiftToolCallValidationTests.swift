import XCTest

@testable import TurboSparkApp

final class SwiftToolCallValidationTests: XCTestCase {
    private func tool(_ schema: JSONSchema, name: String = "fixture") -> OpenAITool {
        OpenAITool.function(name: name, description: "fixture tool", parameters: schema)
    }

    private func schema() -> JSONSchema {
        .object(
            properties: [
                "mode": .string(enumValues: ["fast", "safe"]),
                "count": .integer(),
                "ratio": .number(),
                "enabled": .boolean(),
                "empty": JSONSchemaProperty(type: "null"),
                "nested": .object(
                    properties: ["label": .string(), "active": .boolean()],
                    required: ["label"],
                    additionalProperties: false),
                "items": .array(
                    items: .object(
                        properties: ["value": .number()],
                        required: ["value"],
                        additionalProperties: false))
            ],
            required: ["mode", "count", "nested", "items"],
            additionalProperties: false)
    }

    func testSnapshotRetainsExactDefinitionsAndAcceptsRecursiveObjectArguments() throws {
        let offered = [tool(schema())]
        let available = TurnAvailableTools(definitions: offered)

        XCTAssertEqual(available.definitions, offered)
        let result = available.validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":3,"ratio":1.25,"enabled":true,"empty":null,"nested":{"label":"x","active":false},"items":[{"value":0.5}]}"#)

        guard case .validated(let definition, let arguments) = result else {
            return XCTFail("Expected the complete object to validate, got \(result)")
        }
        XCTAssertEqual(definition, offered[0])
        XCTAssertEqual(arguments["mode"], .string("safe"))
        XCTAssertEqual(arguments["items"], .array([.object(["value": .number(0.5)])]))
    }

    func testRequiredKeysAreCheckedAtTheRoot() {
        let result = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","nested":{"label":"x"},"items":[]}"#)

        XCTAssertEqual(result.refusal, .invalidArguments(path: "$.count", reason: "required"))
    }

    func testScalarTypesAreCheckedWithoutStringCoercion() {
        let result = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":"3","nested":{"label":"x"},"items":[]}"#)

        XCTAssertEqual(result.refusal, .invalidArguments(path: "$.count", reason: "type"))

        let fractionalInteger = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":3.5,"nested":{"label":"x"},"items":[]}"#)
        XCTAssertEqual(fractionalInteger.refusal, .invalidArguments(path: "$.count", reason: "type"))
    }

    func testArrayAndNullTypesAreChecked() {
        let wrongArray = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":3,"nested":{"label":"x"},"items":{}}"#)
        XCTAssertEqual(wrongArray.refusal, .invalidArguments(path: "$.items", reason: "type"))

        let wrongNull = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":3,"empty":false,"nested":{"label":"x"},"items":[]}"#)
        XCTAssertEqual(wrongNull.refusal, .invalidArguments(path: "$.empty", reason: "type"))
    }

    func testEnumsAreChecked() {
        let result = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"turbo","count":3,"nested":{"label":"x"},"items":[]}"#)

        XCTAssertEqual(result.refusal, .invalidArguments(path: "$.mode", reason: "enum"))
    }

    func testNestedRequiredKeysAndArrayItemsAreCheckedRecursively() {
        let missingNested = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":3,"nested":{},"items":[]}"#)
        XCTAssertEqual(missingNested.refusal, .invalidArguments(path: "$.nested.label", reason: "required"))

        let wrongArrayItem = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":3,"nested":{"label":"x"},"items":[{"value":"bad"}]}"#)
        XCTAssertEqual(wrongArrayItem.refusal, .invalidArguments(path: "$.items[0].value", reason: "type"))
    }

    func testDeclaredAdditionalPropertiesFalseRejectsUnknownKeysRecursively() {
        let root = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":3,"nested":{"label":"x"},"items":[],"extra":1}"#)
        XCTAssertEqual(root.refusal, .invalidArguments(path: "$.extra", reason: "additionalProperty"))

        let nested = TurnAvailableTools(definitions: [tool(schema())]).validate(
            toolName: "fixture",
            argumentsJSON: #"{"mode":"safe","count":3,"nested":{"label":"x","extra":1},"items":[]}"#)
        XCTAssertEqual(nested.refusal, .invalidArguments(path: "$.nested.extra", reason: "additionalProperty"))
    }

    func testDeclaredAdditionalPropertiesTrueAllowsUnknownKeys() {
        let schema = JSONSchema.object(
            properties: ["known": .string()],
            required: ["known"],
            additionalProperties: true)
        let result = TurnAvailableTools(definitions: [tool(schema)]).validate(
            toolName: "fixture", argumentsJSON: #"{"known":"ok","other":{"anything":true}}"#)

        XCTAssertNil(result.refusal)
    }

    func testUnavailableMalformedAndNonObjectCallsHaveTypedRefusals() {
        let available = TurnAvailableTools(definitions: [tool(schema())])
        XCTAssertEqual(
            available.validate(toolName: "missing", argumentsJSON: "{}" ).refusal,
            .unavailableTool(name: "missing"))
        XCTAssertEqual(
            available.validate(toolName: "fixture", argumentsJSON: "{" ).refusal,
            .malformedJSON)
        XCTAssertEqual(
            available.validate(toolName: "fixture", argumentsJSON: "[]" ).refusal,
            .nonObjectArguments)
    }

    func testUnsupportedSchemaFormsAreRefused() {
        let unsupported = JSONSchema(type: "oneOf")
        let result = TurnAvailableTools(definitions: [tool(unsupported)]).validate(
            toolName: "fixture", argumentsJSON: "{}")

        guard case .unsupportedSchema = result.refusal else {
            return XCTFail("Expected unsupported-schema refusal, got \(result)")
        }
    }

    func testPromptUsesCapturedDefinitionsWithoutRebuildingCatalog() {
        let offered = tool(.emptyObject(), name: "captured_only")
        let available = TurnAvailableTools(definitions: [offered])
        let prompt = AppToolCatalog.systemPromptAddendum(
            for: .coder, availableTools: available)

        XCTAssertTrue(prompt.contains("`captured_only`: fixture tool"))
        XCTAssertFalse(prompt.contains("`Bash`:"))
    }

    func testTurnSnapshotPreservesPromptVocabularyAndRuntimeFilters() {
        let project = AppProject(name: "fixture", agentType: .researcher)
        let contextTokens = 4096
        let expected = AppToolCatalog.tools(
            for: project.agentType,
            projectURL: project.rootDirectoryURL,
            contextTokens: contextTokens,
            webToolsEnabled: false)
        let available = AppToolCatalog.captureTurnAvailableTools(
            for: project,
            globalMcpServers: [],
            contextTokens: contextTokens,
            webToolsEnabled: false)

        XCTAssertEqual(available.promptDefinitions, expected)
        XCTAssertFalse(available.promptDefinitions.contains { $0.function.name == "websearch" })

        let priorPrompt = AppToolCatalog.systemPromptAddendum(
            for: project.agentType,
            projectURL: project.rootDirectoryURL,
            contextTokens: contextTokens,
            webToolsEnabled: false)
        let capturedPrompt = AppToolCatalog.systemPromptAddendum(
            for: project.agentType,
            projectURL: project.rootDirectoryURL,
            contextTokens: contextTokens,
            webToolsEnabled: false,
            availableTools: available)
        XCTAssertEqual(capturedPrompt, priorPrompt)
    }

    @MainActor
    func testTurnSnapshotAppliesProjectMcpEnableAndDenyFilters() {
        McpToolCatalogCache.shared.removeAll()
        defer { McpToolCatalogCache.shared.removeAll() }
        let server = McpServerConfig(name: "snapshot_fixture", transport: .stdio(command: "/bin/echo"))
        McpToolCatalogCache.shared.setTools(
            [McpDiscoveredTool(
                name: "read", description: "Read a record", inputSchemaJSON: #"{"type":"object"}"#,
                serverName: server.name)],
            for: server)

        var project = AppProject(name: "fixture")
        project.enabledMcpServers[server.id.uuidString] = false
        let disabled = AppToolCatalog.captureTurnAvailableTools(
            for: project, globalMcpServers: [server], contextTokens: nil, webToolsEnabled: true)
        XCTAssertTrue(disabled.deferredMcpTools.isEmpty)

        project.enabledMcpServers[server.id.uuidString] = true
        project.permissions.mcpDenyRules = ["mcp__snapshot_fixture__read"]
        let denied = AppToolCatalog.captureTurnAvailableTools(
            for: project, globalMcpServers: [server], contextTokens: nil, webToolsEnabled: true)
        XCTAssertTrue(denied.deferredMcpTools.isEmpty)
        XCTAssertFalse(denied.definitions.contains { $0.function.name == "mcp__snapshot_fixture__read" })
    }

    @MainActor
    func testMcpSnapshotRetainsSupportedSchemaAndRawDeferredDefinition() {
        McpToolCatalogCache.shared.removeAll()
        defer { McpToolCatalogCache.shared.removeAll() }
        let server = McpServerConfig(name: "schema_fixture", transport: .stdio(command: "/bin/echo"))
        let rawSchema = #"{"type":"object","properties":{"mode":{"type":"string","enum":["read","write"]},"records":{"type":"array","items":{"type":"object","properties":{"id":{"type":"integer"}},"required":["id"],"additionalProperties":false}},"open":{"type":"object","additionalProperties":true}},"required":["mode"],"additionalProperties":false}"#
        McpToolCatalogCache.shared.setTools(
            [McpDiscoveredTool(
                name: "apply", description: "Apply records", inputSchemaJSON: rawSchema,
                serverName: server.name)],
            for: server)

        let snapshot = AppToolCatalogMcp.catalogSnapshot(servers: [server], permissions: nil)
        XCTAssertEqual(snapshot.definitions.map(\.function.name), ["mcp__schema_fixture__apply"])
        XCTAssertEqual(snapshot.deferredDescriptors.first?.inputSchemaJSON, rawSchema)
        let parameters = snapshot.definitions[0].function.parameters
        XCTAssertEqual(parameters.required, ["mode"])
        XCTAssertEqual(parameters.additionalProperties, false)
        XCTAssertEqual(parameters.properties?["mode"]?.enumValues, ["read", "write"])
        XCTAssertEqual(parameters.properties?["records"]?.items?.value.required, ["id"])
        XCTAssertEqual(parameters.properties?["records"]?.items?.value.additionalProperties, false)
        XCTAssertEqual(parameters.properties?["open"]?.additionalProperties, true)

        let result = TurnAvailableTools(definitions: snapshot.definitions).validate(
            toolName: "mcp__schema_fixture__apply",
            argumentsJSON: #"{"mode":"read","records":[{"id":7}],"open":{"arbitrary":3}}"#)
        guard case .validated = result else {
            return XCTFail("Expected the parsed MCP schema to validate, got \(result)")
        }
    }

    @MainActor
    func testMcpUnsupportedValidationKeywordProducesTypedRefusal() {
        McpToolCatalogCache.shared.removeAll()
        defer { McpToolCatalogCache.shared.removeAll() }
        let server = McpServerConfig(name: "schema_fixture", transport: .stdio(command: "/bin/echo"))
        McpToolCatalogCache.shared.setTools(
            [McpDiscoveredTool(
                name: "bounded", description: "Bounded value",
                inputSchemaJSON: #"{"type":"object","properties":{"value":{"type":"number","minimum":0}}}"#,
                serverName: server.name)],
            for: server)

        let definition = AppToolCatalogMcp.toolDefinitions(servers: [server], permissions: nil)[0]
        let result = TurnAvailableTools(definitions: [definition]).validate(
            toolName: definition.function.name, argumentsJSON: #"{"value":1}"#)

        guard case .refused(.unsupportedSchema) = result else {
            return XCTFail("Expected unsupported keyword refusal, got \(result)")
        }

        McpToolCatalogCache.shared.setTools(
            [McpDiscoveredTool(
                name: "numeric_boolean", description: "Invalid schema boolean",
                inputSchemaJSON: #"{"type":"object","properties":{"value":{"type":"object","additionalProperties":1}}}"#,
                serverName: server.name)],
            for: server)
        let numericBoolean = AppToolCatalogMcp.toolDefinitions(servers: [server], permissions: nil)[0]
        let booleanResult = TurnAvailableTools(definitions: [numericBoolean]).validate(
            toolName: numericBoolean.function.name, argumentsJSON: #"{"value":{"other":1}}"#)
        guard case .refused(.unsupportedSchema) = booleanResult else {
            return XCTFail("Expected a non-boolean additionalProperties value to be refused, got \(booleanResult)")
        }
    }

    func testNoProjectCapturesNoOfferedTools() {
        let available = AppToolCatalog.captureTurnAvailableTools(
            for: nil, globalMcpServers: [], contextTokens: nil, webToolsEnabled: true)

        XCTAssertTrue(available.definitions.isEmpty)
        XCTAssertTrue(available.promptDefinitions.isEmpty)
        XCTAssertTrue(available.deferredMcpTools.isEmpty)
    }
}

private extension ToolCallValidationResult {
    var refusal: ToolCallValidationRefusal? {
        guard case .refused(let refusal) = self else { return nil }
        return refusal
    }
}
