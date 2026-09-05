import XCTest
@testable import TurboSparkApp

final class McpAndIntegrationToolsTests: XCTestCase {
    private func makeProject() throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(name: "test-mcp-integration", rootDirectoryPath: dir.path)
        return (project, dir)
    }

    // MARK: - call_mcp_tool / callmcptool / mcp_tool

    func testCallMcpToolMissingArguments() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["call_mcp_tool", "callmcptool", "mcp_tool"] {
            let call = AppToolCall(
                name: alias,
                arguments: [:],
                category: .mcp
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertTrue(result.isError, "Missing server and tool arguments should report error")
        }
    }

    // MARK: - list_mcp_resources / read_mcp_resource

    func testListAndReadMcpResourcesValidation() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["list_mcp_resources", "listmcpresources", "list_resources"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["server": "unknown_server"],
                category: .mcp
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            // Unknown server reports error or empty resources without trapping
            XCTAssertNotNil(result.output)
        }

        for alias in ["read_mcp_resource", "readmcpresource", "read_resource"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["server": "unknown_server", "uri": "file:///test.txt"],
                category: .mcp
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertNotNil(result.output)
        }
    }

    // MARK: - web_search / webfetch

    func testWebSearchAndFetchValidation() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        // web_search missing query
        for alias in ["web_search", "websearch", "search_web"] {
            let call = AppToolCall(
                name: alias,
                arguments: [:],
                category: .web
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertTrue(result.isError, "web_search missing query must report error")
        }

        // web_fetch invalid url / missing url
        for alias in ["web_fetch", "webfetch", "fetch_url", "read_url_content"] {
            let call = AppToolCall(
                name: alias,
                arguments: [:],
                category: .web
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertTrue(result.isError, "web_fetch missing url must report error")

            let invalidSchemeCall = AppToolCall(
                name: alias,
                arguments: ["url": "ftp://example.com/file"],
                category: .web
            )
            let invalidSchemeResult = await AppToolRegistry.execute(call: invalidSchemeCall, in: project)
            XCTAssertTrue(invalidSchemeResult.isError, "web_fetch non-http scheme must report error")
        }
    }

    // MARK: - skill

    func testSkillInvocation() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let call = AppToolCall(
            name: "skill",
            arguments: ["name": "unknown_skill_xyz"],
            category: .automation
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        // Resolves or reports unknown skill without process trap
        XCTAssertNotNil(result.output)
    }

    // MARK: - agent / subagent / task

    func testAgentSubagentInvocationWithoutActiveSession() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["agent", "subagent", "task"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["task": "Analyze codebase architecture"],
                category: .automation
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            // Without an active model session attached, reports clean error rather than crash
            XCTAssertNotNil(result.output)
        }
    }
}
