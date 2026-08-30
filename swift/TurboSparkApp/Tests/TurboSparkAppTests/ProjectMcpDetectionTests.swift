import XCTest
@testable import TurboSparkApp

final class ProjectMcpDetectionTests: XCTestCase {
    private var tempDirURL: URL!

    override func setUp() {
        super.setUp()
        let uniqueName = "ts_mcp_test_\(UUID().uuidString)"
        tempDirURL = FileManager.default.temporaryDirectory.appendingPathComponent(uniqueName, isDirectory: true)
        try? FileManager.default.createDirectory(at: tempDirURL, withIntermediateDirectories: true)
    }

    override func tearDown() {
        if let tempDirURL {
            try? FileManager.default.removeItem(at: tempDirURL)
        }
        super.tearDown()
    }

    func testDetectStandardMcpJson() throws {
        let mcpJsonContent = """
        {
          "mcpServers": {
            "memory-server": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-memory"],
              "env": { "DEBUG": "1" }
            },
            "filesystem-server": {
              "command": "npx",
              "args": ["-y", "@modelcontextprotocol/server-filesystem", "${workspaceFolder}"]
            }
          }
        }
        """
        let fileURL = tempDirURL.appendingPathComponent(".mcp.json")
        try mcpJsonContent.write(to: fileURL, atomically: true, encoding: .utf8)

        let detected = ProjectMcpDetector.detectInProject(rootURL: tempDirURL)
        XCTAssertEqual(detected.count, 1)
        XCTAssertEqual(detected.first?.relativePath, ".mcp.json")

        let servers = detected.first?.servers ?? []
        XCTAssertEqual(servers.count, 2)

        let memory = servers.first { $0.name == "memory-server" }
        XCTAssertNotNil(memory)
        if case .stdio(let cmd, let args, let env) = memory?.transport {
            XCTAssertEqual(cmd, "npx")
            XCTAssertEqual(args, ["-y", "@modelcontextprotocol/server-memory"])
            XCTAssertEqual(env["DEBUG"], "1")
        } else {
            XCTFail("Expected stdio transport")
        }

        let fsServer = servers.first { $0.name == "filesystem-server" }
        XCTAssertNotNil(fsServer)
        if case .stdio(_, let args, _) = fsServer?.transport {
            XCTAssertEqual(args, ["-y", "@modelcontextprotocol/server-filesystem", tempDirURL.path])
        }
    }

    func testDetectOpenCodeJson() throws {
        let opencodeJsonContent = """
        {
          // OpenCode configuration with comments
          "mcp": {
            "opencode-tools": {
              "command": "node",
              "args": ["./tools/server.js", "${projectRoot}"]
            }
          }
        }
        """
        let fileURL = tempDirURL.appendingPathComponent("opencode.json")
        try opencodeJsonContent.write(to: fileURL, atomically: true, encoding: .utf8)

        let detected = ProjectMcpDetector.detectInProject(rootURL: tempDirURL)
        XCTAssertEqual(detected.count, 1)
        XCTAssertEqual(detected.first?.relativePath, "opencode.json")

        let servers = detected.first?.servers ?? []
        XCTAssertEqual(servers.count, 1)
        XCTAssertEqual(servers.first?.name, "opencode-tools")
        if case .stdio(let cmd, let args, _) = servers.first?.transport {
            XCTAssertEqual(cmd, "node")
            XCTAssertEqual(args, ["./tools/server.js", tempDirURL.path])
        } else {
            XCTFail("Expected stdio transport")
        }
    }

    func testDetectOpenCodeArrayCommandFormat() throws {
        let opencodeJsonContent = """
        {
          "mcp": {
            "opencode-array": {
              "type": "local",
              "command": ["npx", "-y", "@modelcontextprotocol/server-postgres", "${workspaceFolder}/db"],
              "enabled": true,
              "autoApprove": true
            }
          }
        }
        """
        let fileURL = tempDirURL.appendingPathComponent("opencode.json")
        try opencodeJsonContent.write(to: fileURL, atomically: true, encoding: .utf8)

        let detected = ProjectMcpDetector.detectInProject(rootURL: tempDirURL)
        XCTAssertEqual(detected.count, 1)

        let server = detected.first?.servers.first
        XCTAssertNotNil(server)
        XCTAssertEqual(server?.name, "opencode-array")
        XCTAssertEqual(server?.isEnabled, true)
        // T14: `autoApprove` is NEVER trusted from a repo-controlled config
        // file, even when the file explicitly declares it, because the
        // permission engine reads this flag to skip confirmation entirely.
        // A cloned `.mcp.json`/`opencode.json` must not be able to grant
        // itself silent tool execution just by being imported.
        XCTAssertEqual(server?.autoApprove, false)

        if case .stdio(let cmd, let args, _) = server?.transport {
            XCTAssertEqual(cmd, "npx")
            XCTAssertEqual(args, ["-y", "@modelcontextprotocol/server-postgres", "\(tempDirURL.path)/db"])
        } else {
            XCTFail("Expected stdio transport")
        }
    }

    func testDetectCursorMcpJson() throws {
        let cursorDir = tempDirURL.appendingPathComponent(".cursor", isDirectory: true)
        try FileManager.default.createDirectory(at: cursorDir, withIntermediateDirectories: true)
        let cursorMcpContent = """
        {
          "mcpServers": {
            "cursor-sse": {
              "url": "http://127.0.0.1:8080/sse",
              "headers": { "Authorization": "Bearer token123" }
            }
          }
        }
        """
        let fileURL = cursorDir.appendingPathComponent("mcp.json")
        try cursorMcpContent.write(to: fileURL, atomically: true, encoding: .utf8)

        let detected = ProjectMcpDetector.detectInProject(rootURL: tempDirURL)
        XCTAssertEqual(detected.count, 1)
        XCTAssertEqual(detected.first?.relativePath, ".cursor/mcp.json")

        let servers = detected.first?.servers ?? []
        XCTAssertEqual(servers.count, 1)
        XCTAssertEqual(servers.first?.name, "cursor-sse")
        if case .sse(let url, let headers) = servers.first?.transport {
            XCTAssertEqual(url.absoluteString, "http://127.0.0.1:8080/sse")
            XCTAssertEqual(headers["Authorization"], "Bearer token123")
        } else {
            XCTFail("Expected sse transport")
        }
    }

    func testDetectMultipleConfigFilesInProject() throws {
        // Create both .mcp.json and opencode.json
        try "{\"mcpServers\": {\"srv1\": {\"command\": \"cmd1\"}}}".write(
            to: tempDirURL.appendingPathComponent(".mcp.json"),
            atomically: true,
            encoding: .utf8
        )
        try "{\"mcp\": {\"srv2\": {\"command\": \"cmd2\"}}}".write(
            to: tempDirURL.appendingPathComponent("opencode.json"),
            atomically: true,
            encoding: .utf8
        )

        let detected = ProjectMcpDetector.detectInProject(rootURL: tempDirURL)
        XCTAssertEqual(detected.count, 2)
        let allServerNames = detected.flatMap { $0.servers.map(\.name) }
        XCTAssertTrue(allServerNames.contains("srv1"))
        XCTAssertTrue(allServerNames.contains("srv2"))
    }

    // MARK: - autoApprove Is Never Trusted From a Repo Config (T14)

    func testAutoApproveIsStrippedRegardlessOfSpelling() throws {
        let content = """
        {
          "mcpServers": {
            "claude-spelling": { "command": "npx", "args": [], "autoApprove": true },
            "alt-spelling-1": { "command": "npx", "args": [], "auto_approve": true },
            "alt-spelling-2": { "command": "npx", "args": [], "alwaysAllow": true }
          }
        }
        """
        let fileURL = tempDirURL.appendingPathComponent(".mcp.json")
        try content.write(to: fileURL, atomically: true, encoding: .utf8)

        let servers = ProjectMcpDetector.detectInProject(rootURL: tempDirURL).first?.servers ?? []
        XCTAssertEqual(servers.count, 3)
        for server in servers {
            XCTAssertFalse(server.autoApprove, "'\(server.name)' must not inherit autoApprove from an untrusted config file, whatever key spelling it used.")
        }
    }

    func testAppProjectBackwardCompatibilityWithMcp() throws {
        // JSON representing an older project without mcpServers field
        let legacyJson = """
        {
          "id": "11111111-2222-3333-4444-555555555555",
          "name": "Legacy Project",
          "rootDirectoryPath": "/tmp/legacy",
          "agentType": "coder",
          "rulePreference": "agents_first",
          "customInstructions": "Instructions",
          "permissions": {
            "mode": "auto",
            "fileRead": "allow",
            "fileWrite": "ask",
            "terminal": "ask",
            "web": "allow",
            "mcp": "ask",
            "automation": "ask"
          },
          "maxAutonomousSteps": 5
        }
        """.data(using: .utf8)!

        let project = try JSONDecoder().decode(AppProject.self, from: legacyJson)
        XCTAssertEqual(project.name, "Legacy Project")
        XCTAssertEqual(project.mcpServers, [])

        // Re-encoding and decoding with MCP servers
        var updatedProject = project
        let mcpServer = McpServerConfig(
            name: "test-server",
            transport: .stdio(command: "echo", args: ["hello"])
        )
        updatedProject.mcpServers.append(mcpServer)

        let encoded = try JSONEncoder().encode(updatedProject)
        let roundtrip = try JSONDecoder().decode(AppProject.self, from: encoded)
        XCTAssertEqual(roundtrip.mcpServers.count, 1)
        XCTAssertEqual(roundtrip.mcpServers.first?.name, "test-server")
    }
}
