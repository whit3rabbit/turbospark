import XCTest
@testable import TurboSparkApp

final class McpClientEngineTests: XCTestCase {
    func testMcpServerConfigRoundtrip() throws {
        let config = McpServerConfig(
            name: "test-mcp",
            transport: .stdio(command: "python3", args: ["-m", "mcp_server"], env: ["API_KEY": "secret"]),
            isEnabled: true,
            autoApprove: true,
            sourcePath: "/path/to/.mcp.json",
            serverDescription: "Test description",
            discoveredTools: [
                McpDiscoveredTool(name: "query_docs", description: "Search docs", inputSchemaJSON: "{}", serverName: "test-mcp")
            ]
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(config)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(McpServerConfig.self, from: data)

        XCTAssertEqual(decoded.name, "test-mcp")
        XCTAssertEqual(decoded.isEnabled, true)
        XCTAssertEqual(decoded.autoApprove, true)
        XCTAssertEqual(decoded.serverDescription, "Test description")
        XCTAssertEqual(decoded.discoveredTools.count, 1)
        XCTAssertEqual(decoded.discoveredTools.first?.name, "query_docs")

        if case .stdio(let cmd, let args, let env) = decoded.transport {
            XCTAssertEqual(cmd, "python3")
            XCTAssertEqual(args, ["-m", "mcp_server"])
            XCTAssertEqual(env["API_KEY"], "secret")
        } else {
            XCTFail("Expected stdio transport")
        }
    }

    func testMcpSseTransportRoundtrip() throws {
        let url = URL(string: "https://mcp.example.com/events")!
        let config = McpServerConfig(
            name: "remote-mcp",
            transport: .sse(url: url, headers: ["Authorization": "Bearer abc"])
        )

        let encoder = JSONEncoder()
        let data = try encoder.encode(config)

        let decoder = JSONDecoder()
        let decoded = try decoder.decode(McpServerConfig.self, from: data)

        XCTAssertEqual(decoded.name, "remote-mcp")
        if case .sse(let decodedUrl, let headers) = decoded.transport {
            XCTAssertEqual(decodedUrl.absoluteString, "https://mcp.example.com/events")
            XCTAssertEqual(headers["Authorization"], "Bearer abc")
        } else {
            XCTFail("Expected SSE transport")
        }
    }

    func testVariableExpansion() {
        let projectURL = URL(fileURLWithPath: "/Users/test/Code/MyProject")
        let rawArg = "${workspaceFolder}/data/files"
        let expanded = ProjectMcpDetector.expandVariables(rawArg, projectRoot: projectURL)
        XCTAssertEqual(expanded, "/Users/test/Code/MyProject/data/files")

        let rawArg2 = "${projectRoot}/src"
        let expanded2 = ProjectMcpDetector.expandVariables(rawArg2, projectRoot: projectURL)
        XCTAssertEqual(expanded2, "/Users/test/Code/MyProject/src")
    }
}
