import XCTest
@testable import TurboSparkApp

final class AgentSystemTests: XCTestCase {
    func testBuiltInAgentsExistAndHaveCorrectProperties() {
        let agents = AgentManager.shared.builtInAgents
        XCTAssertGreaterThanOrEqual(agents.count, 4)

        let explore = agents.first { $0.name == "explore" }
        XCTAssertNotNil(explore)
        XCTAssertEqual(explore?.displayName, "Codebase Explorer")
        XCTAssertFalse(explore?.isToolAllowed("write_file") ?? true)
        XCTAssertFalse(explore?.isToolAllowed("edit_file") ?? true)
        XCTAssertFalse(explore?.isToolAllowed("apply_patch") ?? true)
        XCTAssertTrue(explore?.isToolAllowed("read_file") ?? false)
        XCTAssertTrue(explore?.isToolAllowed("search_code") ?? false)

        let plan = agents.first { $0.name == "plan" }
        XCTAssertNotNil(plan)
        XCTAssertFalse(plan?.isToolAllowed("write_file") ?? true)

        let gp = agents.first { $0.name == "general-purpose" }
        XCTAssertNotNil(gp)
        XCTAssertTrue(gp?.isToolAllowed("write_file") ?? false)
        XCTAssertTrue(gp?.isToolAllowed("read_file") ?? false)

        let reviewer = agents.first { $0.name == "reviewer" }
        XCTAssertNotNil(reviewer)
        XCTAssertFalse(reviewer?.isToolAllowed("write_file") ?? true)
        XCTAssertTrue(reviewer?.isToolAllowed("read_file") ?? false)
    }

    func testMarkdownAgentParsing() {
        let markdown = """
        ---
        name: test-agent
        display_name: Test Specialist
        description: A specialist for unit testing
        max_turns: 8
        model: gemma4
        disallowed_tools:
          - write_file
          - edit_file
        ---
        You are a testing assistant. Analyze test suites and verify coverage.
        """

        let url = URL(fileURLWithPath: "/tmp/agents/test-agent.md")
        let agent = AgentParser.parseMarkdownContent(markdown, sourceURL: url, scope: .userGlobal, sourceAgent: .claude)

        XCTAssertEqual(agent.name, "test-agent")
        XCTAssertEqual(agent.displayName, "Test Specialist")
        XCTAssertEqual(agent.agentDescription, "A specialist for unit testing")
        XCTAssertEqual(agent.maxTurns, 8)
        XCTAssertEqual(agent.model, "gemma4")
        XCTAssertEqual(agent.sourceAgent, .claude)
        XCTAssertEqual(agent.scope, .userGlobal)
        XCTAssertFalse(agent.isToolAllowed("write_file"))
        XCTAssertFalse(agent.isToolAllowed("edit_file"))
        XCTAssertTrue(agent.isToolAllowed("read_file"))
        XCTAssertTrue(agent.systemPrompt.contains("You are a testing assistant."))
    }

    func testJsonAgentParsing() throws {
        let json = """
        {
            "name": "json-agent",
            "displayName": "JSON Specialist",
            "description": "Agent parsed from JSON",
            "maxTurns": 3,
            "prompt": "Evaluate JSON payloads and schemas.",
            "tools": ["read_file", "search_code"]
        }
        """

        let url = URL(fileURLWithPath: "/tmp/agents/json-agent.json")
        let agent = try AgentParser.parseJSONContent(json, sourceURL: url, scope: .project, sourceAgent: .openCode)

        XCTAssertEqual(agent.name, "json-agent")
        XCTAssertEqual(agent.displayName, "JSON Specialist")
        XCTAssertEqual(agent.maxTurns, 3)
        XCTAssertEqual(agent.scope, .project)
        XCTAssertEqual(agent.sourceAgent, .openCode)
        XCTAssertTrue(agent.isToolAllowed("read_file"))
        XCTAssertTrue(agent.isToolAllowed("search_code"))
        XCTAssertFalse(agent.isToolAllowed("write_file")) // Not in allowed tools whitelist
        XCTAssertTrue(agent.systemPrompt.contains("Evaluate JSON payloads"))
    }

    func testProjectAgentDiscoveryAndPrecedence() throws {
        let tempDir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let projectAgentsDir = tempDir.appendingPathComponent(".turbospark/agents", isDirectory: true)
        try FileManager.default.createDirectory(at: projectAgentsDir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: tempDir) }

        let customAgentMd = """
        ---
        name: explore
        display_name: Custom Project Explorer
        description: Overridden project explorer
        ---
        Custom project exploration instructions.
        """
        try customAgentMd.write(to: projectAgentsDir.appendingPathComponent("explore.md"), atomically: true, encoding: .utf8)

        let discovered = AgentManager.shared.discoverProjectAgents(projectURL: tempDir)
        XCTAssertEqual(discovered.count, 1)
        XCTAssertEqual(discovered[0].displayName, "Custom Project Explorer")

        let effective = AgentManager.shared.resolveEffectiveAgents(projectURL: tempDir)
        let resolvedExplore = effective.first { $0.name == "explore" }
        XCTAssertEqual(resolvedExplore?.displayName, "Custom Project Explorer")
        XCTAssertEqual(resolvedExplore?.scope, .project)
    }

    func testSubagentRunnerSystemPromptConstruction() {
        let agent = AppAgentDefinition(
            name: "explore",
            displayName: "Codebase Explorer",
            agentDescription: "Read only search",
            systemPrompt: "Search only.",
            disallowedTools: ["write_file", "edit_file", "apply_patch"],
            scope: .builtIn
        )

        let project = AppProject(name: "Demo", rootDirectoryPath: "/path/to/demo", customInstructions: "Use Swift 5.9")
        let prompt = SubagentRunner.buildSystemPrompt(for: agent, project: project)

        XCTAssertTrue(prompt.contains("Subagent Role: Codebase Explorer"))
        XCTAssertTrue(prompt.contains("Search only."))
        XCTAssertTrue(prompt.contains("Root codebase directory: `/path/to/demo`"))
        XCTAssertTrue(prompt.contains("Use Swift 5.9"))
        XCTAssertTrue(prompt.contains("FileRead") || prompt.lowercased().contains("fileread"))
        // Disallowed tools must not be advertised in the subagent's tool list
        XCTAssertFalse(prompt.contains("`FileWrite`:"))
        XCTAssertFalse(prompt.contains("`FileEdit`:"))
    }

    func testAppToolRegistryExecutesAgentToolGracefully() async {
        let call = AppToolCall(
            name: "Agent",
            arguments: [
                "prompt": "Find all view controllers",
                "subagent_type": "explore"
            ],
            category: .automation
        )

        // Without a real running session, SubagentRunner reports session unavailable error gracefully without crashing
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("No active model session"))
    }

    func testAgentToggleEnabled() {
        let agentName = "plan"
        let initial = AgentManager.shared.isAgentDisabled(name: agentName)

        AgentManager.shared.setAgentEnabled(false, name: agentName)
        XCTAssertTrue(AgentManager.shared.isAgentDisabled(name: agentName))

        let agent = AgentManager.shared.findAgent(name: agentName)
        XCTAssertEqual(agent?.isEnabled, false)

        // Restore
        AgentManager.shared.setAgentEnabled(!initial, name: agentName)
    }
}
