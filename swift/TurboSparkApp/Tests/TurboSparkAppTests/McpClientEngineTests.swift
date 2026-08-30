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

    // MARK: - Transport Robustness (T7, T16)

    /// A server that never writes a response must be abandoned at the
    /// stated deadline, not hang forever. The previous implementation read
    /// the pipe directly (`pipe.fileHandleForReading.availableData`) inside
    /// the polling loop, so the loop's own deadline check could only run
    /// BETWEEN blocking reads -- if the child never wrote again, the
    /// deadline was never re-evaluated.
    func testDiscoverToolsTimesOutRatherThanHangingOnASilentServer() async {
        let engine = McpClientEngine()
        let config = McpServerConfig(
            name: "silent-server",
            transport: .stdio(command: "/bin/sh", args: ["-c", "sleep 30"])
        )
        let start = Date()
        do {
            _ = try await engine.discoverTools(for: config, timeoutSeconds: 0.5)
            XCTFail("Expected a timeout error from a server that never responds.")
        } catch {
            // Expected.
        }
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 5.0, "A 0.5s timeout must fire well before the fake server's 30s sleep.")
    }

    /// A server that exits immediately makes every subsequent stdin write
    /// hit a closed pipe (EPIPE). The legacy `FileHandle.write(Data)`
    /// overload raises an uncatchable Objective-C exception on that, which
    /// would crash this very test process rather than being catchable as a
    /// Swift error.
    func testCallingAToolThatDiesImmediatelyThrowsRatherThanCrashing() async {
        let engine = McpClientEngine()
        let config = McpServerConfig(
            name: "dies-immediately",
            transport: .stdio(command: "/bin/sh", args: ["-c", "exit 0"])
        )
        do {
            _ = try await engine.callTool(config: config, toolName: "anything", arguments: [:], timeoutSeconds: 2.0)
            XCTFail("Expected an error calling a tool on a server that has already exited.")
        } catch {
            // Expected: a normal Swift error, not a crash.
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
