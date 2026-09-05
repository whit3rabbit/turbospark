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

        if case .stdio(let cmd, let args, let env, _, _) = decoded.transport {
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

    // MARK: - Working directory and environment (added with `cwd` / `envPassthrough`)

    /// **THE GOTCHA 13 REGRESSION.** A server written before these two fields
    /// existed must still decode, with both at their defaults. Every install on
    /// disk today is one of those.
    func testAConfigWrittenBeforeCwdAndPassthroughStillDecodes() throws {
        let json = """
            {
              "id": "3F2504E0-4F89-11D3-9A0C-0305E82C3301",
              "name": "legacy",
              "transport": {"type": "stdio", "command": "npx", "args": ["-y", "server"]},
              "isEnabled": true,
              "autoApprove": false,
              "discoveredTools": [],
              "createdAt": 0,
              "updatedAt": 0
            }
            """
        let decoded = try JSONDecoder().decode(McpServerConfig.self, from: Data(json.utf8))

        guard case .stdio(let command, let args, _, let cwd, let passthrough) = decoded.transport
        else {
            return XCTFail("Expected stdio transport")
        }
        XCTAssertEqual(command, "npx")
        XCTAssertEqual(args, ["-y", "server"])
        XCTAssertNil(cwd, "An absent cwd must stay absent, not become a path")
        XCTAssertEqual(passthrough, [], "An absent passthrough must be empty, not nil-ish")
    }

    func testCwdAndPassthroughRoundTrip() throws {
        let config = McpServerConfig(
            name: "rooted",
            transport: .stdio(
                command: "python3", args: ["-m", "srv"], env: ["A": "1"],
                cwd: "/tmp/work", envPassthrough: ["GITHUB_TOKEN"]))

        let decoded = try JSONDecoder().decode(
            McpServerConfig.self, from: JSONEncoder().encode(config))

        guard case .stdio(_, _, let env, let cwd, let passthrough) = decoded.transport else {
            return XCTFail("Expected stdio transport")
        }
        XCTAssertEqual(env["A"], "1")
        XCTAssertEqual(cwd, "/tmp/work")
        XCTAssertEqual(passthrough, ["GITHUB_TOKEN"])
    }

    // MARK: childEnvironment

    private var parentFixture: [String: String] {
        [
            "PATH": "/usr/bin",
            "HOME": "/Users/fixture",
            "LANG": "en_US.UTF-8",
            "TMPDIR": "/tmp",
            "GITHUB_TOKEN": "gh-secret",
            "ANTHROPIC_API_KEY": "sk-secret",
        ]
    }

    func testTheBaselineKeysAreAlwaysForwarded() {
        let env = McpClientEngine.childEnvironment(
            parent: parentFixture, passthrough: [], declared: [:])

        XCTAssertEqual(env["PATH"], "/usr/bin")
        XCTAssertEqual(env["HOME"], "/Users/fixture")
        XCTAssertEqual(env["LANG"], "en_US.UTF-8")
        XCTAssertEqual(env["TMPDIR"], "/tmp")
    }

    /// **THE CASE THIS FUNCTION EXISTS FOR** (swift/CLAUDE.md Gotcha 32,
    /// state#32). The child environment is built from scratch, so a variable
    /// nobody named never reaches a third-party server binary. If this ever
    /// goes green while the assertion is inverted, the allowlist has become a
    /// passthrough and every credential in the launching shell is exposed.
    func testAVariableThatWasNotNamedIsNotForwarded() {
        let env = McpClientEngine.childEnvironment(
            parent: parentFixture, passthrough: ["GITHUB_TOKEN"], declared: [:])

        XCTAssertEqual(env["GITHUB_TOKEN"], "gh-secret", "A named key is forwarded")
        XCTAssertNil(env["ANTHROPIC_API_KEY"], "An unnamed key must never be forwarded")
    }

    /// Omitted, not set empty. An empty `HOME` is a different failure from an
    /// unset one, and the second is the truth.
    func testAPassthroughKeyMissingFromTheParentIsOmitted() {
        let env = McpClientEngine.childEnvironment(
            parent: parentFixture, passthrough: ["NOT_SET_ANYWHERE"], declared: [:])

        XCTAssertFalse(env.keys.contains("NOT_SET_ANYWHERE"))
    }

    /// A config naming its own PATH means it.
    func testADeclaredVariableBeatsBothTheBaselineAndThePassthrough() {
        let env = McpClientEngine.childEnvironment(
            parent: parentFixture,
            passthrough: ["GITHUB_TOKEN"],
            declared: ["PATH": "/custom/bin", "GITHUB_TOKEN": "explicit"])

        XCTAssertEqual(env["PATH"], "/custom/bin")
        XCTAssertEqual(env["GITHUB_TOKEN"], "explicit")
    }

    func testABlankPassthroughEntryIsIgnored() {
        let env = McpClientEngine.childEnvironment(
            parent: parentFixture, passthrough: ["", "   ", " GITHUB_TOKEN "], declared: [:])

        XCTAssertEqual(env["GITHUB_TOKEN"], "gh-secret", "Entries are trimmed")
        XCTAssertFalse(env.keys.contains(""))
    }

    // MARK: resolveWorkingDirectory

    /// The caller's value is a DEFAULT, not a choice: `executeMcpCall` passes
    /// the project root for every server alike. A server that states its own
    /// directory has stated a choice, so it wins.
    func testAServerCwdBeatsTheCallersWorkingDirectory() {
        let resolved = McpClientEngine.resolveWorkingDirectory(
            cwd: "/tmp/server-home", fallback: URL(fileURLWithPath: "/tmp/project-root"))

        XCTAssertEqual(resolved?.path, "/tmp/server-home")
    }

    func testWithNoCwdTheCallersWorkingDirectoryIsUsed() {
        let resolved = McpClientEngine.resolveWorkingDirectory(
            cwd: nil, fallback: URL(fileURLWithPath: "/tmp/project-root"))

        XCTAssertEqual(resolved?.path, "/tmp/project-root")
    }

    /// An empty string is what a cleared text field produces. It must fall
    /// through to the caller's value rather than resolving to the filesystem
    /// root, which is where a bare `URL(fileURLWithPath: "")` lands.
    func testAnEmptyCwdFallsThroughRatherThanBecomingTheRoot() {
        let resolved = McpClientEngine.resolveWorkingDirectory(
            cwd: "   ", fallback: URL(fileURLWithPath: "/tmp/project-root"))

        XCTAssertEqual(resolved?.path, "/tmp/project-root")
    }

    func testATildeCwdIsExpanded() {
        let resolved = McpClientEngine.resolveWorkingDirectory(cwd: "~/work", fallback: nil)
        let home = FileManager.default.homeDirectoryForCurrentUser.path

        XCTAssertEqual(resolved?.path, "\(home)/work")
    }

    func testAHomeVariableCwdIsExpanded() {
        let resolved = McpClientEngine.resolveWorkingDirectory(cwd: "${HOME}/work", fallback: nil)
        let home = FileManager.default.homeDirectoryForCurrentUser.path

        XCTAssertEqual(resolved?.path, "\(home)/work")
    }
}
