import XCTest
@testable import TurboSparkApp

/// Discovery, precedence and the enable cascade, over INJECTABLE temp roots
/// so nothing here touches the real home directory.
final class PluginManagerTests: XCTestCase {
    var root: URL!
    var claudeRoot: URL!
    var manager: PluginManager!
    var userEnable: [String: Bool] = [:]
    var projectEnable: [String: Bool] = [:]
    var claudeEnable: [String: Bool] = [:]

    override func setUpWithError() throws {
        // Resolve AFTER the directories exist: /tmp is /private/tmp on
        // macOS, discovery returns the RESOLVED path, and
        // resolvingSymlinksInPath resolves nothing on a nonexistent path.
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tsp-plugins-\(UUID().uuidString)", isDirectory: true)
        claudeRoot = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("claude-plugins-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: claudeRoot, withIntermediateDirectories: true)
        root = root.resolvingSymlinksInPath()
        claudeRoot = claudeRoot.resolvingSymlinksInPath()
        userEnable = [:]
        projectEnable = [:]
        claudeEnable = [:]
        manager = PluginManager(
            turboSparkRoot: root, claudeRoot: claudeRoot,
            userEnableProvider: { [unowned self] in userEnable },
            projectEnableProvider: { [unowned self] _ in projectEnable },
            claudeEnableProvider: { [unowned self] in claudeEnable })
        // This suite exercises the interop path itself; the app default is
        // opt-in (`includeClaudeInterop` false until
        // `autoLoadExternalAgentContent` is enabled), which
        // `PluginGatingTests` covers.
        manager.includeClaudeInterop = true
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.removeItem(at: claudeRoot)
    }

    @discardableResult
    private func makePlugin(
        _ name: String, in base: URL,
        manifest: String? = nil,
        version: String? = nil,
        marketplace: String? = nil
    ) throws -> URL {
        let dir: URL
        if let marketplace {
            let version = version ?? "1.0.0"
            dir = base
                .appendingPathComponent("cache/\(marketplace)/\(name)/\(version)", isDirectory: true)
        } else {
            dir = base.appendingPathComponent(name, isDirectory: true)
        }
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        if let manifest {
            try FileManager.default.createDirectory(
                at: dir.appendingPathComponent(".claude-plugin"), withIntermediateDirectories: true)
            try Data(manifest.utf8).write(
                to: dir.appendingPathComponent(".claude-plugin/plugin.json"))
        }
        return dir
    }

    // MARK: - Discovery

    func testFlatDirectoryWithManifestIsDiscovered() throws {
        try makePlugin("alpha", in: root, manifest: #"{"name": "alpha", "version": "3.1"}"#)
        let resolution = manager.resolve(projectURL: nil)
        XCTAssertEqual(resolution.plugins.first?.name, "alpha")
        XCTAssertEqual(resolution.plugins.first?.version, "3.1")
        XCTAssertEqual(resolution.plugins.first?.origin, .turboSpark)
        XCTAssertTrue(resolution.plugins.first!.isEnabledAlias(manager, projectURL: nil))
    }

    func testConventionDirectoryWithoutManifestIsSynthesized() throws {
        try makePlugin("bare", in: root)
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("bare/commands"), withIntermediateDirectories: true)
        let plugin = manager.resolve(projectURL: nil).plugins.first { $0.name == "bare" }
        XCTAssertNotNil(plugin)
        // Compared against the DISCOVERED directory, not the fixture root:
        // FileManager returns resolved (/private/var) URLs while a fixture
        // built from NSTemporaryDirectory spells /var, and the two disagree
        // by a prefix that is not the thing under test.
        XCTAssertEqual(
            plugin?.manifest.descriptionText,
            "Plugin from \(plugin!.directoryURL.path)")
    }

    func testPlainDirectoryWithoutSignatureIsIgnored() throws {
        try makePlugin("not-a-plugin", in: root)
        XCTAssertTrue(manager.resolve(projectURL: nil).plugins.isEmpty)
    }

    func testCorruptManifestFailsAloneAndSiblingsLoad() throws {
        try makePlugin("broken", in: root, manifest: "{oops")
        try makePlugin("healthy", in: root, manifest: #"{"name": "healthy"}"#)
        let resolution = manager.resolve(projectURL: nil)
        XCTAssertEqual(resolution.plugins.map(\.name), ["healthy"])
        XCTAssertTrue(resolution.loadErrors.contains { $0.contains("broken") })
    }

    func testMarketplaceCacheDiscoversTheNewestVersion() throws {
        try makePlugin("cached", in: root,
                       manifest: #"{"name": "cached"}"#, version: "1.0.0",
                       marketplace: "my-market")
        try makePlugin("cached", in: root,
                       manifest: #"{"name": "cached"}"#, version: "1.10.0",
                       marketplace: "my-market")
        let resolution = manager.resolve(projectURL: nil)
        XCTAssertEqual(resolution.plugins.count, 1, "one plugin across its version dirs")
        XCTAssertEqual(resolution.plugins.first?.version, "1.10.0")
        XCTAssertEqual(resolution.plugins.first?.marketplaceName, "my-market")
        XCTAssertEqual(resolution.plugins.first?.origin, .marketplace)
        XCTAssertEqual(resolution.plugins.first?.id, "cached@my-market")
    }

    func testTurboSparkRootShadowsClaudeInteropByName() throws {
        try makePlugin("shared", in: root, manifest: #"{"name": "shared"}"#)
        try makePlugin("shared", in: claudeRoot, manifest: #"{"name": "shared"}"#)
        try makePlugin("only-claude", in: claudeRoot, manifest: #"{"name": "only-claude"}"#)
        let resolution = manager.resolve(projectURL: nil)
        XCTAssertEqual(resolution.plugins.map(\.id), ["shared@turbospark", "only-claude@claude"])
        XCTAssertTrue(resolution.loadErrors.contains { $0.contains("shadowed") || $0.contains("skipped") })
        XCTAssertEqual(
            resolution.plugins.first { $0.name == "only-claude" }?.origin, .claudeInterop)
    }

    // MARK: - Enable cascade

    func testAbsentEnableEntryMeansEnabled() throws {
        try makePlugin("a", in: root, manifest: #"{"name": "a"}"#)
        XCTAssertTrue(manager.resolve(projectURL: nil).plugins.allSatisfy {
            manager.isEnabled(pluginID: $0.id, projectURL: nil)
        })
    }

    func testUserDisableIsHonored() throws {
        try makePlugin("a", in: root, manifest: #"{"name": "a"}"#)
        userEnable = ["a@turbospark": false]
        XCTAssertFalse(manager.isEnabled(pluginID: "a@turbospark", projectURL: nil))
        XCTAssertTrue(manager.enabledPlugins(projectURL: nil).isEmpty)
    }

    func testProjectOverrideWinsOverUser() throws {
        try makePlugin("a", in: root, manifest: #"{"name": "a"}"#)
        userEnable = ["a@turbospark": true]
        projectEnable = ["a@turbospark": false]
        let projectURL = URL(fileURLWithPath: "/tmp/some-project")
        XCTAssertFalse(manager.isEnabled(pluginID: "a@turbospark", projectURL: projectURL))
        XCTAssertTrue(manager.isEnabled(pluginID: "a@turbospark", projectURL: nil))
    }

    func testClaudeInteropFallsBackToClaudeCodesOwnSetting() throws {
        try makePlugin("interop", in: claudeRoot, manifest: #"{"name": "interop"}"#)
        claudeEnable = ["interop@claude": false]
        XCTAssertFalse(manager.isEnabled(pluginID: "interop@claude", projectURL: nil))
        // An explicit TurboSpark user setting beats Claude Code's.
        userEnable = ["interop@claude": true]
        XCTAssertTrue(manager.isEnabled(pluginID: "interop@claude", projectURL: nil))
    }

    func testClaudeEnabledPluginsReadsBoolAndArrayForms() throws {
        let settings = root.appendingPathComponent("claude-settings.json")
        try Data(#"""
        {"enabledPlugins": {"a@market": false, "b@market": ["^1.0.0"], "c@market": true}}
        """#.utf8).write(to: settings)
        let map = PluginManager.claudeEnabledPlugins(settingsURL: settings)
        XCTAssertEqual(map["a@market"], false)
        XCTAssertEqual(map["b@market"], true, "the version-constraint form counts as enabled")
        XCTAssertEqual(map["c@market"], true)
    }

    // MARK: - Paths

    func testDataDirectoryIsSanitizedPerPluginAndOrigin() {
        let plugin = LoadedPlugin(
            name: "../evil name", manifest: PluginManifest(name: "../evil name"),
            directoryURL: root, origin: .marketplace, marketplaceName: "mkt")
        let dataDir = manager.dataDirectory(for: plugin)
        XCTAssertTrue(dataDir.path.hasPrefix(root.path + "/data/"))
        let component = dataDir.lastPathComponent
        XCTAssertFalse(component.contains("/"))
        XCTAssertFalse(component.contains("."))
        XCTAssertTrue(component.hasSuffix("-mkt"))
    }
}

/// Enable-state helpers on the test type, kept off `LoadedPlugin` itself.
extension LoadedPlugin {
    func isEnabledAlias(_ manager: PluginManager, projectURL: URL?) -> Bool {
        manager.isEnabled(pluginID: id, projectURL: projectURL)
    }
}

/// The contribution surfaces: skills and commands with the plugin namespace,
/// agents, MCP servers, and hook discovery through the injectable provider.
final class PluginContributionTests: XCTestCase {
    var root: URL!
    var manager: PluginManager!
    var userEnable: [String: Bool] = [:]

    override func setUpWithError() throws {
        root = URL(fileURLWithPath: NSTemporaryDirectory())
            .appendingPathComponent("tsp-contrib-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        root = root.resolvingSymlinksInPath()
        userEnable = [:]
        manager = PluginManager(
            turboSparkRoot: root, claudeRoot: root.appendingPathComponent("unused"),
            userEnableProvider: { [unowned self] in userEnable },
            projectEnableProvider: { _ in [:] },
            claudeEnableProvider: { [:] })
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: root)
    }

    private func write(_ text: String, to url: URL) throws {
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(text.utf8).write(to: url)
    }

    @discardableResult
    private func makePlugin(
        name: String,
        manifest: String,
        extra: (URL) throws -> Void
    ) throws -> LoadedPlugin {
        let dir = root.appendingPathComponent(name, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try write(manifest, to: dir.appendingPathComponent(".claude-plugin/plugin.json"))
        try extra(dir)
        let resolution = manager.resolve(projectURL: nil)
        guard let plugin = resolution.plugins.first else {
            throw NSError(domain: "test", code: 1)
        }
        return plugin
    }

    func testSkillIsNamespacedAndExpandsThePluginRoot() throws {
        _ = try makePlugin(
            name: "reviewer",
            manifest: #"{"name": "reviewer"}"#,
            extra: { dir in
                try self.write(
                    """
                    ---
                    description: Reviews a diff
                    ---
                    Run the linter at ${CLAUDE_PLUGIN_ROOT}/bin/lint.
                    """,
                    to: dir.appendingPathComponent("skills/review/SKILL.md"))
            })

        let skills = manager.pluginSkills(projectURL: nil)
        XCTAssertEqual(skills.count, 1)
        XCTAssertEqual(skills.first?.name, "reviewer:review")
        XCTAssertEqual(skills.first?.scope, SkillScope.plugin(pluginID: "reviewer@turbospark"))
        let pluginDir = manager.resolve(projectURL: nil)
            .plugins.first { $0.name == "reviewer" }!.directoryURL.path
        // The parsed body keeps frontmatter-surrounding whitespace, so the
        // sentence is matched trimmed.
        XCTAssertEqual(
            skills.first!.content.trimmingCharacters(in: .whitespacesAndNewlines),
            "Run the linter at \(pluginDir)/bin/lint.")
    }

    func testNestedCommandDirectoryAddsNamespaceSegments() throws {
        _ = try makePlugin(
            name: "cmds",
            manifest: #"{"name": "cmds"}"#,
            extra: { dir in
                try self.write(
                    "---\ndescription: Lint it\n---\nbody",
                    to: dir.appendingPathComponent("commands/repo/lint.md"))
            })
        let skills = manager.pluginSkills(projectURL: nil)
        XCTAssertEqual(skills.first?.name, "cmds:repo:lint")
    }

    func testManifestCommandObjectMapNamesTheCommandAndSupportsInlineContent() throws {
        _ = try makePlugin(
            name: "mapped",
            manifest: """
            {"name": "mapped", "commands": {"deploy": {"content": "Deploy now", "description": "Ships it"}}}
            """,
            extra: { _ in })
        let skills = manager.pluginSkills(projectURL: nil)
        XCTAssertEqual(skills.first?.name, "mapped:deploy")
        XCTAssertEqual(skills.first?.content, "Deploy now")
        XCTAssertEqual(skills.first?.manifest.description, "Ships it")
    }

    func testDisabledPluginContributesNothing() throws {
        _ = try makePlugin(
            name: "gated",
            manifest: #"{"name": "gated"}"#,
            extra: { dir in
                try self.write("---\ndescription: x\n---\nbody",
                               to: dir.appendingPathComponent("skills/s/SKILL.md"))
            })
        XCTAssertEqual(manager.pluginSkills(projectURL: nil).count, 1)
        userEnable = ["gated@turbospark": false]
        XCTAssertEqual(manager.pluginSkills(projectURL: nil).count, 0)
        XCTAssertEqual(manager.pluginAgents(projectURL: nil).count, 0)
        XCTAssertEqual(manager.pluginMcpServers(projectURL: nil).count, 0)
    }

    func testAgentIsNamespacedAndScoped() throws {
        _ = try makePlugin(
            name: "agentful",
            manifest: #"{"name": "agentful"}"#,
            extra: { dir in
                try self.write(
                    "---\nname: finder\ndescription: Finds things\ntools: [read_file]\n---\nLook around.",
                    to: dir.appendingPathComponent("agents/finder.md"))
            })
        let agents = manager.pluginAgents(projectURL: nil)
        XCTAssertEqual(agents.count, 1)
        XCTAssertEqual(agents.first?.name, "agentful:finder")
        XCTAssertEqual(agents.first?.scope, AppAgentScope.plugin)
        XCTAssertTrue(agents.first!.isToolAllowed("read_file"))
        XCTAssertFalse(agents.first!.isToolAllowed("run_command"))
    }

    func testMcpServersAreRenamedWithThePluginPrefixAndExpanded() throws {
        _ = try makePlugin(
            name: "mcpful",
            manifest: #"{"name": "mcpful", "mcpServers": {"fs": {"command": "${CLAUDE_PLUGIN_ROOT}/bin/serve", "args": ["--root", "${user_config.root}"], "env": {"T": "1"}}}}"#,
            extra: { dir in
                // The conventional root .mcp.json contributes a second server.
                try self.write(
                    #"{"mcpServers": {"dot": {"command": "node", "args": ["server.js"]}}}"#,
                    to: dir.appendingPathComponent(".mcp.json"))
            })

        // An option value for ${user_config.root}, stored the way the hooks
        // options store stores it (sourceID `plugin_<name>`).
        let optionsURL = AppStorageRoot.subdirectory("Hooks")
            .appendingPathComponent("hook_options_values.json")
        try FileManager.default.createDirectory(
            at: optionsURL.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(#"{"plugin_mcpful": {"root": "/chosen/root"}}"#.utf8).write(to: optionsURL)
        defer { try? FileManager.default.removeItem(at: optionsURL) }

        let servers = manager.pluginMcpServers(projectURL: nil)
        XCTAssertEqual(servers.count, 2)
        let names = Set(servers.map(\.name))
        XCTAssertEqual(names, ["plugin:mcpful:fs", "plugin:mcpful:dot"])
        guard let plugin = manager.resolve(projectURL: nil).plugins.first(where: { $0.name == "mcpful" }) else {
            return XCTFail("plugin mcpful was not discovered")
        }
        guard let fs = servers.first(where: { $0.name == "plugin:mcpful:fs" }) else {
            return XCTFail("the namespaced server name plugin:mcpful:fs did not resolve; names were \(names)")
        }
        if case .stdio(let command, let args, let env, _, _) = fs.transport {
            XCTAssertEqual(command, "\(plugin.directoryURL.path)/bin/serve")
            XCTAssertEqual(args, ["--root", "/chosen/root"])
            XCTAssertEqual(env, ["T": "1"])
        } else {
            XCTFail("expected stdio transport")
        }
        XCTAssertEqual(fs.autoApprove, false, "plugin servers never auto-approve")
        XCTAssertTrue(fs.isEnabled)
    }

    func testManifestMcpSymlinkOutsidePluginIsDropped() throws {
        let external = root.appendingPathComponent("external-mcp.json")
        try write(
            #"{"mcpServers": {"escaped": {"command": "/bin/sh"}}}"#,
            to: external)
        let plugin = try makePlugin(
            name: "linked-mcp",
            manifest: #"{"name": "linked-mcp", "mcpServers": "./servers.json"}"#,
            extra: { dir in
                try FileManager.default.createSymbolicLink(
                    at: dir.appendingPathComponent("servers.json"),
                    withDestinationURL: external)
            })

        XCTAssertTrue(manager.mcpServersContributed(by: plugin).isEmpty)
    }
}
