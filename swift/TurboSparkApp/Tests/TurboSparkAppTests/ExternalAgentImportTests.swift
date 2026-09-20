import XCTest
@testable import TurboSparkApp

/// The secret-store double `HookSystemTests` also carries, private to each
/// file: hook construction needs one, and nothing here reads secrets.
private final class InMemorySecrets: HookOptionSecretStoring {
    func load(sourceID: String, key: String, storageDirectory: URL) -> String? { nil }
    func save(_ value: String, sourceID: String, key: String, storageDirectory: URL) -> Bool { true }
}

/// The cross-agent opt-in contract: with
/// `autoLoadExternalAgentContent` off (the default), NO surface reads
/// other agent tools' home folders, and the Import wizard's copy path is
/// the way in. Each test builds a fixture home (or plugin roots) so
/// nothing here depends on -- or touches -- the real home directory.
final class ExternalAgentImportTests: XCTestCase {
    var fixtureHome: URL!

    override func setUpWithError() throws {
        fixtureHome = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("ext-agent-home-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: fixtureHome, withIntermediateDirectories: true)
        SkillManager.shared.externalAgentDiscoveryEnabled = false
        AgentManager.shared.externalAgentDiscoveryEnabled = false
    }

    override func tearDownWithError() throws {
        SkillManager.shared.externalAgentDiscoveryEnabled = false
        AgentManager.shared.externalAgentDiscoveryEnabled = false
        try? FileManager.default.removeItem(at: fixtureHome)
    }

    // MARK: - Skills discovery gating

    /// A skill living in another tool's folder is invisible to the default
    /// (flag-off) discovery path and visible when the flag is on. This is
    /// THE behavior the wizard replaces auto-discovery with.
    func testSkillDiscoveryExcludesExternalRootsByDefault() throws {
        // The shared-roots branch is additionally gated on the Default
        // profile (profile isolation), which a fresh test store is.
        try XCTSkipUnless(UserProfileStore.isDefault, "shared roots need the Default profile")
        let skillDir = fixtureHome
            .appendingPathComponent(".claude/skills/fixture-skill", isDirectory: true)
        try FileManager.default.createDirectory(at: skillDir, withIntermediateDirectories: true)
        try """
        ---
        name: fixture-skill
        description: Lives in another tool's folder
        ---
        Body.
        """.write(to: skillDir.appendingPathComponent("SKILL.md"), atomically: true, encoding: .utf8)

        let manager = SkillManager.shared
        XCTAssertFalse(
            manager.discoverUserSkills(home: fixtureHome)
                .contains { $0.name == "fixture-skill" },
            "flag off: the external root must not be read")

        manager.externalAgentDiscoveryEnabled = true
        XCTAssertTrue(
            manager.discoverUserSkills(home: fixtureHome)
                .contains { $0.name == "fixture-skill" },
            "flag on: the external root is read")

        XCTAssertTrue(
            manager.discoverUserSkills(includeExternalAgents: false, home: fixtureHome)
                .contains { $0.name == "fixture-skill" } == false,
            "an explicit false overrides the enabled flag")
    }

    // MARK: - Agents discovery gating

    func testAgentDiscoveryExcludesExternalRootsByDefault() throws {
        try XCTSkipUnless(UserProfileStore.isDefault, "shared roots need the Default profile")
        let agentDir = fixtureHome.appendingPathComponent(".codex/agents", isDirectory: true)
        try FileManager.default.createDirectory(at: agentDir, withIntermediateDirectories: true)
        try """
        ---
        name: codex-explorer
        description: Lives in Codex's folder
        ---
        You explore.
        """.write(to: agentDir.appendingPathComponent("codex-explorer.md"), atomically: true, encoding: .utf8)

        let manager = AgentManager.shared
        XCTAssertFalse(
            manager.discoverUserAgents(home: fixtureHome)
                .contains { $0.name == "codex-explorer" },
            "flag off: the external root must not be read")

        manager.externalAgentDiscoveryEnabled = true
        XCTAssertTrue(
            manager.discoverUserAgents(home: fixtureHome)
                .contains { $0.name == "codex-explorer" },
            "flag on: the external root is read (this also pins the codex root)")
    }

    // MARK: - Agent import (the wizard's copy path)

    func testImportAgentCopiesFileAndDiscoverySeesIt() throws {
        let source = fixtureHome.appendingPathComponent("claude-agent.md")
        try """
        ---
        name: imported-helper
        description: Imported from another tool
        ---
        Help out.
        """.write(to: source, atomically: true, encoding: .utf8)

        let manager = AgentManager.shared
        let imported = try manager.importAgent(from: source, scope: .userGlobal)
        XCTAssertEqual(imported.name, "imported-helper")
        defer { try? FileManager.default.removeItem(at: URL(fileURLWithPath: imported.filePath!)) }

        let target = manager.defaultUserAgentsDirectory
            .appendingPathComponent("claude-agent.md")
        XCTAssertTrue(FileManager.default.fileExists(atPath: target.path), "the file was copied")
        XCTAssertTrue(
            manager.scanDirectory(
                manager.defaultUserAgentsDirectory, scope: .userGlobal, defaultAgent: .turboSpark
            ).contains { $0.name == "imported-helper" },
            "discovery resolves the imported copy")
    }

    func testImportAgentRefusesToOverwriteAnExistingFile() throws {
        let source = fixtureHome.appendingPathComponent("dup-agent.md")
        try """
        ---
        name: dup-agent
        description: First copy
        ---
        One.
        """.write(to: source, atomically: true, encoding: .utf8)

        let manager = AgentManager.shared
        let first = try manager.importAgent(from: source, scope: .userGlobal)
        defer { try? FileManager.default.removeItem(at: URL(fileURLWithPath: first.filePath!)) }

        XCTAssertThrowsError(try manager.importAgent(from: source, scope: .userGlobal)) { error in
            guard case AgentManager.AgentFileError.destinationExists = error else {
                return XCTFail("expected destinationExists, got \(error)")
            }
        }
    }

    // MARK: - Codex TOML parsing

    func testCodexTomlParsesStdioArgsEnvAndRemoteServers() throws {
        let toml = """
        model = "gpt-5"
        # unrelated top-level keys are ignored

        [mcp_servers.memory]
        command = "npx"
        args = ["-y", "@modelcontextprotocol/server-memory"]

        [mcp_servers.fetch]
        command = "uvx"
        args = ["mcp-server-fetch"]

        [mcp_servers.fetch.env]
        HTTP_TIMEOUT = "30"

        [mcp_servers.docs]
        url = "https://mcp.example.com/sse"
        headers = { Authorization = "Bearer abc" }

        [mcp_servers.broken]
        args = ["not-a-server"]
        """
        let servers = ExternalAgentMcpReader.parseCodexToml(toml, sourcePath: "/x/config.toml")

        XCTAssertEqual(servers.count, 3, "broken has no command and is skipped, not guessed at")
        let memory = servers.first { $0.name == "memory" }
        guard case .stdio(let memCmd, let memArgs, _, _, _)? = memory?.transport else {
            return XCTFail("memory must be stdio")
        }
        XCTAssertEqual(memCmd, "npx")
        XCTAssertEqual(memArgs, ["-y", "@modelcontextprotocol/server-memory"])
        XCTAssertTrue(memory?.autoApprove == false, "autoApprove is never trusted from disk")

        let fetch = servers.first { $0.name == "fetch" }
        guard case .stdio(let cmd, _, let env, _, _)? = fetch?.transport else {
            return XCTFail("fetch must be stdio")
        }
        XCTAssertEqual(cmd, "uvx")
        XCTAssertEqual(env["HTTP_TIMEOUT"], "30", "the [mcp_servers.x.env] subtable is honored")

        let docs = servers.first { $0.name == "docs" }
        guard case .sse(let url, let headers)? = docs?.transport else {
            return XCTFail("docs must be sse")
        }
        XCTAssertEqual(url.absoluteString, "https://mcp.example.com/sse")
        XCTAssertEqual(headers["Authorization"], "Bearer abc")
    }

    func testCodexTomlHandlesQuotedNamesAndEscapesWithoutCrossingQuotes() throws {
        let toml = #"""
        [mcp_servers."github.tools"]
        command = "docker"
        args = ["run", "-i", "--rm", "-e", "GITHUB_TOKEN, quote \" inside"]
        """#
        let servers = ExternalAgentMcpReader.parseCodexToml(toml, sourcePath: "/x/config.toml")
        XCTAssertEqual(servers.count, 1)
        XCTAssertEqual(servers.first?.name, "github.tools", "a quoted name keeps its dots")
        guard case .stdio(let cmd, let args, _, _, _)? = servers.first?.transport else {
            return XCTFail("must be stdio")
        }
        XCTAssertEqual(cmd, "docker")
        XCTAssertEqual(args.last, "GITHUB_TOKEN, quote \" inside", "escaped quotes stay intact")
    }

    // MARK: - Global hook gating

    @MainActor
    func testGlobalHookDiscoverySkipsClaudeConfigUnlessEnabled() throws {
        try XCTSkipUnless(UserProfileStore.isDefault, "global candidates differ per profile")
        let turboDir = fixtureHome.appendingPathComponent(".turbospark", isDirectory: true)
        let claudeDir = fixtureHome.appendingPathComponent(".claude", isDirectory: true)
        try FileManager.default.createDirectory(at: turboDir, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: claudeDir, withIntermediateDirectories: true)
        let turboHooks = turboDir.appendingPathComponent("hooks.json")
        try """
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo ours"}]}]}}
        """.write(to: turboHooks, atomically: true, encoding: .utf8)
        let claudeSettings = claudeDir.appendingPathComponent("settings.json")
        try """
        {"hooks": {"Stop": [{"hooks": [{"type": "command", "command": "echo theirs"}]}]}}
        """.write(to: claudeSettings, atomically: true, encoding: .utf8)

        let store = AppHookStore(optionSecretStore: InMemorySecrets())
        var diagnostics: [String] = []

        store.includeClaudeGlobalConfig = false
        let gated = store.discoverGlobalHooks(diagnostics: &diagnostics, home: fixtureHome)
        XCTAssertEqual(
            gated.map(\.command), ["echo ours"],
            "flag off: only TurboSpark's own config is read")

        store.includeClaudeGlobalConfig = true
        let both = store.discoverGlobalHooks(diagnostics: &diagnostics, home: fixtureHome)
        XCTAssertEqual(Set(both.map(\.command)), ["echo ours", "echo theirs"])
    }

    // MARK: - Plugin interop gating

    func testPluginInteropRequiresTheFlag() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tsp-gate-\(UUID().uuidString)", isDirectory: true)
        let claudeRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("claude-gate-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: claudeRoot, withIntermediateDirectories: true)
        defer {
            try? FileManager.default.removeItem(at: root)
            try? FileManager.default.removeItem(at: claudeRoot)
        }

        let pluginDir = claudeRoot.appendingPathComponent("interop", isDirectory: true)
        try FileManager.default.createDirectory(
            at: pluginDir.appendingPathComponent(".claude-plugin"), withIntermediateDirectories: true)
        try Data(#"{"name": "interop"}"#.utf8).write(
            to: pluginDir.appendingPathComponent(".claude-plugin/plugin.json"))

        let manager = PluginManager(turboSparkRoot: root, claudeRoot: claudeRoot)
        XCTAssertTrue(
            manager.resolve(projectURL: nil).plugins.isEmpty,
            "flag off: Claude's root is not read at all")

        manager.includeClaudeInterop = true
        // What `AppModel.applyExternalAgentContentFlag` does on a flag
        // change: resolution is memoized per project, so the flip must be
        // followed by a cache drop or the empty result replays forever.
        manager.invalidateResolutionCache()
        XCTAssertEqual(
            manager.resolve(projectURL: nil).plugins.first?.origin, .claudeInterop)
    }

    // MARK: - Settings persistence

    func testAutoLoadFlagDefaultsOffAndRoundTrips() throws {
        let decodedAbsent = try JSONDecoder().decode(
            MacAppSettings.self, from: Data(#"{"temperature": 0.5}"#.utf8))
        XCTAssertFalse(decodedAbsent.autoLoadExternalAgentContent, "absent key means off")

        let decodedOn = try JSONDecoder().decode(
            MacAppSettings.self,
            from: Data(#"{"autoLoadExternalAgentContent": true}"#.utf8))
        XCTAssertTrue(decodedOn.autoLoadExternalAgentContent)

        let encoded = try JSONEncoder().encode(MacAppSettings(autoLoadExternalAgentContent: true))
        let roundTrip = try JSONDecoder().decode(MacAppSettings.self, from: encoded)
        XCTAssertTrue(roundTrip.autoLoadExternalAgentContent)
    }
}
