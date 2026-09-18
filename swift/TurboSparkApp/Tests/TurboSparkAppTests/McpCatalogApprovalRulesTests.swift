import XCTest

@testable import TurboSparkApp

/// MCP catalog advertisement, permission rules, the approval lifecycle, and
/// annotation-aware risk -- the four layers added to close the gap between
/// what the app discovers from an MCP server and what the model is told.
@MainActor
final class McpCatalogApprovalRulesTests: XCTestCase {
    override func setUp() {
        super.setUp()
        McpToolCatalogCache.shared.removeAll()
    }

    override func tearDown() {
        McpToolCatalogCache.shared.removeAll()
        super.tearDown()
    }

    // MARK: - Helpers

    private func makeTool(
        _ name: String,
        server: String,
        description: String = "does a thing",
        schema: String = "{}",
        annotations: McpToolAnnotations? = nil
    ) -> McpDiscoveredTool {
        McpDiscoveredTool(
            name: name, description: description, inputSchemaJSON: schema,
            serverName: server, annotations: annotations)
    }

    private func makeProject(
        permissions: AppProjectPermissions = AppProjectPermissions(mode: .auto),
        mcpServers: [McpServerConfig] = []
    ) -> AppProject {
        var project = AppProject(name: "MCP Test", permissions: permissions)
        project.mcpServers = mcpServers
        return project
    }

    private func makeServer(
        _ name: String,
        enabled: Bool = true,
        autoApprove: Bool = false,
        sourcePath: String? = nil
    ) -> McpServerConfig {
        McpServerConfig(
            name: name,
            transport: .stdio(command: "/bin/echo"),
            isEnabled: enabled,
            autoApprove: autoApprove,
            sourcePath: sourcePath)
    }

    private func mcpCall(_ name: String, arguments: [String: String] = [:]) -> AppToolCall {
        AppToolCall(name: name, arguments: arguments, rawInvocation: name, category: .mcp)
    }

    /// Writes a `.mcp.json` into a fresh temp project root and returns it.
    private func makeProjectRoot(serversJSON: String) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let payload = "{\"mcpServers\": {" + serversJSON + "} }"
        try payload.write(to: root.appendingPathComponent(".mcp.json"), atomically: true, encoding: .utf8)
        return root
    }

    // MARK: - Rule parsing and matching

    func testServerLevelRuleMatchesEveryTool() {
        XCTAssertTrue(McpPermissionRule.matches("mcp__fs", serverName: "FS", toolName: "read"))
        XCTAssertTrue(McpPermissionRule.matches("mcp__fs", serverName: "fs", toolName: nil))
        XCTAssertFalse(McpPermissionRule.matches("mcp__fs", serverName: "other", toolName: "read"))
    }

    func testToolLevelRuleIsExactAndKeepsTrailingDoubleUnderscores() {
        XCTAssertTrue(McpPermissionRule.matches("mcp__fs__a__b", serverName: "fs", toolName: "a__b"))
        XCTAssertFalse(McpPermissionRule.matches("mcp__fs__a__b", serverName: "fs", toolName: "a"))
        XCTAssertFalse(McpPermissionRule.matches("mcp__fs__a", serverName: "fs", toolName: "a__b"))
    }

    func testWildcardAndNonMcpRules() {
        XCTAssertTrue(McpPermissionRule.matches("mcp__fs__*", serverName: "fs", toolName: "read"))
        XCTAssertNil(McpPermissionRule.target(of: "bash(rm:*)"), "a non-MCP rule must parse to nil")
        XCTAssertNil(McpPermissionRule.target(of: "mcp__"), "a rule naming no server must parse to nil")
    }

    // MARK: - Engine: deny rules

    func testDenyRuleDeniesEvenAgainstSessionApproval() {
        var permissions = AppProjectPermissions(mode: .auto)
        permissions.mcpDenyRules = ["mcp__fs__write_file"]
        let project = makeProject(permissions: permissions)

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__write_file"),
            project: project,
            sessionApproved: true,
            globalServers: [makeServer("fs")])

        guard case .deny = decision else {
            return XCTFail("a denied MCP tool must deny even with a session grant, got \(decision)")
        }
    }

    func testServerLevelDenyRuleCoversEveryTool() {
        var permissions = AppProjectPermissions(mode: .auto)
        permissions.mcpDenyRules = ["mcp__fs"]
        let project = makeProject(permissions: permissions)

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__anything"),
            project: project,
            globalServers: [])

        guard case .deny = decision else {
            return XCTFail("a server-level deny must cover every tool, got \(decision)")
        }
    }

    // MARK: - Engine: allow rules

    func testAllowRuleSkipsTheAskPrompt() {
        var permissions = AppProjectPermissions(mode: .ask)
        permissions.mcpAllowRules = ["mcp__fs__read"]
        let project = makeProject(permissions: permissions)

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__read"),
            project: project,
            globalServers: [makeServer("fs")])

        XCTAssertEqual(decision, .allow)
    }

    func testAllowRuleStillYieldsToHighRisk() {
        var permissions = AppProjectPermissions(mode: .ask)
        permissions.mcpAllowRules = ["mcp__fs"]
        let project = makeProject(permissions: permissions)

        // A name the heuristics flag destructive beats the persisted grant.
        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__delete_everything"),
            project: project,
            globalServers: [])

        guard case .ask = decision else {
            return XCTFail("a persisted allow must not cover a high-risk call, got \(decision)")
        }
    }

    // MARK: - Engine: repo-imported server gate in auto mode

    func testRepoImportedServerAsksInAutoModeWithoutAutoApprove() {
        // `mcp: .allow` is what makes this gate load-bearing: without 7b the
        // call would fall through to auto mode's silent allow.
        let project = makeProject(
            permissions: AppProjectPermissions(mode: .auto, mcp: .allow),
            mcpServers: [makeServer("fs", sourcePath: "/tmp/proj/.mcp.json")])

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__read"),
            project: project,
            globalServers: [])

        guard case .ask(_, let reason) = decision else {
            return XCTFail("a repo-imported server must ask in auto mode, got \(decision)")
        }
        XCTAssertTrue(reason.contains("repository config"), "the reason must name the origin")
    }

    func testRepoImportedServerWithExplicitAutoApproveIsAllowed() {
        let project = makeProject(
            mcpServers: [makeServer("fs", autoApprove: true, sourcePath: "/tmp/proj/.mcp.json")])

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__read"),
            project: project,
            globalServers: [])

        XCTAssertEqual(decision, .allow)
    }

    func testManuallyAddedServerStillAutoRunsInAutoMode() {
        // No sourcePath: added by hand, not imported from a repo file. The
        // pre-existing behavior must be unchanged, once discovery has
        // actually completed for this server -- an undiscovered server's
        // first call is deliberately asked about instead (see
        // `testAnUndiscoveredServersFirstCallAsksInAutoModeEvenWhenNothingElseWould`).
        McpToolCatalogCache.shared.setTools([makeTool("read", server: "fs")], for: makeServer("fs"))
        let project = makeProject(
            permissions: AppProjectPermissions(mode: .auto, mcp: .allow),
            mcpServers: [makeServer("fs")])

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__read"),
            project: project,
            globalServers: [])

        XCTAssertEqual(decision, .allow)
    }

    /// The gap this closes: a server just approved or enabled has its first
    /// tool call reach `evaluate` before `McpToolCatalogCache.refreshEnabled`'s
    /// background discovery Task has ever run, so its destructive/read-only
    /// annotations are unknown rather than merely absent -- and auto mode's
    /// own trailing default is the one auto-allow with nothing else standing
    /// between the call and execution.
    func testAnUndiscoveredServersFirstCallAsksInAutoModeEvenWhenNothingElseWould() {
        let project = makeProject(
            permissions: AppProjectPermissions(mode: .auto, mcp: .allow),
            mcpServers: [makeServer("fs")])

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__read"),
            project: project,
            globalServers: [])

        guard case .ask = decision else {
            return XCTFail("an undiscovered server's first call must ask rather than auto-allow, got \(decision)")
        }
    }

    func testRepoImportedServerKeepsPermissiveContract() {
        // Permissive is an explicit, stronger choice; it stays gated only
        // by deny rules, high risk, and (per the annotation-discovery
        // exception above) a server that has not finished discovery yet.
        // Populated here since this test's subject is the repo-import
        // exemption, not the discovery race.
        McpToolCatalogCache.shared.setTools(
            [makeTool("read", server: "fs")],
            for: makeServer("fs", sourcePath: "/tmp/proj/.mcp.json"))
        let project = makeProject(
            permissions: AppProjectPermissions.preset(for: .permissive),
            mcpServers: [makeServer("fs", sourcePath: "/tmp/proj/.mcp.json")])

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__read"),
            project: project,
            globalServers: [])

        XCTAssertEqual(decision, .allow)
    }

    func testStaticCallFormResolvesTheSameServerGate() {
        let project = makeProject(
            permissions: AppProjectPermissions(mode: .auto, mcp: .allow),
            mcpServers: [makeServer("fs", sourcePath: "/tmp/proj/.mcp.json")])

        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("call_mcp_tool", arguments: ["server": "fs", "toolName": "read"]),
            project: project,
            globalServers: [])

        guard case .ask = decision else {
            return XCTFail("the static call_mcp_tool form must hit the same server gate, got \(decision)")
        }
    }

    // MARK: - Annotations into risk

    func testDestructiveAnnotationRaisesASafeVerdict() {
        let base = ToolRiskAssessment(level: .safe, category: .mcp, reasons: [])
        let adjusted = ToolRiskClassifier.adjusting(
            base, annotations: McpToolAnnotations(destructiveHint: true))

        XCTAssertTrue(adjusted.isHighRisk)
        XCTAssertTrue(adjusted.reasons.contains("Server annotations mark this tool destructive."))
    }

    func testAnnotationsNeverLowerAHeuristicVerdict() {
        let high = ToolRiskAssessment(level: .high, category: .mcp, reasons: ["destructive verb"])
        XCTAssertEqual(
            ToolRiskClassifier.adjusting(high, annotations: McpToolAnnotations(readOnly: true)),
            high,
            "a server claiming readOnly must not lower a high-risk heuristic verdict")

        let low = ToolRiskAssessment(level: .low, category: .mcp, reasons: ["mcp tool"])
        XCTAssertEqual(
            ToolRiskClassifier.adjusting(low, annotations: McpToolAnnotations(readOnly: true)),
            low)
    }

    func testEngineConsultsCachedAnnotationsForDynamicNames() {
        // "apply_change" trips no name heuristic (verified: the destructive,
        // execution, privilege and noun sets all miss it), so the only thing
        // that can raise this call is the server's own annotation.
        McpToolCatalogCache.shared.setTools(
            [makeTool("apply_change", server: "fs", annotations: McpToolAnnotations(destructiveHint: true))],
            for: makeServer("fs"))

        let project = makeProject(
            permissions: AppProjectPermissions.preset(for: .permissive),
            mcpServers: [makeServer("fs")])

        // Permissive would allow; the server's own destructiveHint must
        // still force the ask.
        let decision = AppToolPermissionEngine.evaluate(
            call: mcpCall("mcp__fs__apply_change"),
            project: project,
            globalServers: [])

        guard case .ask = decision else {
            return XCTFail("a destructively-annotated tool must ask even in permissive, got \(decision)")
        }
    }

    // MARK: - Catalog advertisement

    func testDiscoveredToolsAreAdvertisedUnderServerNames() {
        McpToolCatalogCache.shared.setTools(
            [makeTool("read", server: "fs"), makeTool("write", server: "fs")],
            for: makeServer("fs"))

        let definitions = AppToolCatalogMcp.toolDefinitions(
            servers: [makeServer("fs")], permissions: nil)

        XCTAssertEqual(
            definitions.map(\.function.name),
            ["mcp__fs__read", "mcp__fs__write"],
            "advertisements are sorted by name for prompt stability")
    }

    func testDisabledAndUndiscoveredServersAdvertiseNothing() {
        McpToolCatalogCache.shared.setTools([makeTool("read", server: "fs")], for: makeServer("fs"))

        XCTAssertEqual(
            AppToolCatalogMcp.toolDefinitions(servers: [makeServer("fs", enabled: false)], permissions: nil),
            [],
            "a disabled server's cached tools must not be advertised")

        XCTAssertEqual(
            AppToolCatalogMcp.toolDefinitions(servers: [makeServer("cold")], permissions: nil),
            [],
            "a server with no cache entry advertises nothing")
    }

    func testDenyRulesStripAdvertisedTools() {
        McpToolCatalogCache.shared.setTools(
            [makeTool("read", server: "fs"), makeTool("write", server: "fs")],
            for: makeServer("fs"))

        var permissions = AppProjectPermissions(mode: .auto)
        permissions.mcpDenyRules = ["mcp__fs__write"]
        XCTAssertEqual(
            AppToolCatalogMcp.toolDefinitions(servers: [makeServer("fs")], permissions: permissions)
                .map(\.function.name),
            ["mcp__fs__read"],
            "a tool-level deny must strip exactly that tool")

        permissions.mcpDenyRules = ["mcp__fs"]
        XCTAssertEqual(
            AppToolCatalogMcp.toolDefinitions(servers: [makeServer("fs")], permissions: permissions),
            [],
            "a server-level deny must strip every tool")
    }

    func testVisibleServersComposeGlobalProjectAndFiltersDisabled() {
        let global = makeServer("g1")
        let project = makeProject(mcpServers: [makeServer("p1"), makeServer("p0", enabled: false)])
        // visibleServers takes the global list as given (callers pass only
        // what they expose), filters isEnabled, and appends plugin servers.
        let visible = AppToolCatalogMcp.visibleServers(global: [global], project: project)
        XCTAssertEqual(visible.map(\.name), ["g1", "p1"])
    }

    // MARK: - Description and argument summary

    func testAdvertisedDescriptionIncludesArgumentSummary() {
        let schema = #"{"type":"object","properties":{"path":{"type":"string"},"force":{"type":"boolean"}},"required":["path"]}"#
        let description = AppToolCatalogMcp.advertisedDescription(
            for: makeTool("read", server: "fs", description: "Read a file", schema: schema))

        XCTAssertTrue(description.hasPrefix("Read a file"), "the server description leads")
        XCTAssertTrue(description.contains("path: string"), "required args carry no optional marker")
        XCTAssertTrue(description.contains("force: boolean, optional"), "non-required args are marked")
    }

    func testAdvertisedDescriptionTruncatesTheServerPart() {
        let long = String(repeating: "x", count: 500)
        let description = AppToolCatalogMcp.advertisedDescription(
            for: makeTool("t", server: "fs", description: long))

        XCTAssertLessThan(description.count, 500, "the description must be capped")
        XCTAssertTrue(description.hasSuffix("..."))
    }

    func testArgumentSummarySurvivesUnionTypedProperties() {
        let schema = #"{"type":"object","properties":{"value":{"anyOf":[{"type":"string"},{"type":"integer"}]}},"required":["value"]}"#
        XCTAssertEqual(AppToolCatalogMcp.argumentsSummary(for: schema), "value: any")
        XCTAssertNil(AppToolCatalogMcp.argumentsSummary(for: "{}"))
        XCTAssertNil(AppToolCatalogMcp.argumentsSummary(for: "not json"))
    }

    // MARK: - Approval lifecycle

    func testDetectedServerPromptsOnceThenApprovalPersists() throws {
        let root = try makeProjectRoot(serversJSON: #""fs": {"command": "/bin/echo"}"#)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel()
        var project = AppProject(name: "Lifecycle", rootDirectoryPath: root.path)
        model.projects = [project]
        model.selectedProjectID = project.id

        model.detectProjectMcpServers()
        XCTAssertEqual(model.pendingMcpApprovals.count, 1, "a fresh repo server must prompt")
        let pending = try XCTUnwrap(model.pendingMcpApprovals.first)
        XCTAssertFalse(pending.config.autoApprove, "a repo import never arrives auto-approved")
        XCTAssertEqual(pending.config.sourcePath, root.appendingPathComponent(".mcp.json").path)

        model.approvePendingMcpServer(id: pending.id)
        project = try XCTUnwrap(model.projects.first)
        XCTAssertEqual(project.mcpServers.map(\.name), ["fs"])
        XCTAssertTrue(project.mcpServers.first?.isEnabled ?? false)
        XCTAssertFalse(project.mcpServers.first?.autoApprove ?? true)
        XCTAssertEqual(project.approvedMcpJsonServers, ["fs"])

        model.detectProjectMcpServers()
        XCTAssertTrue(model.pendingMcpApprovals.isEmpty, "an approved name must never re-prompt")
    }

    func testRejectedServerIsRecordedAndNeverReprompts() throws {
        let root = try makeProjectRoot(serversJSON: #""bad": {"command": "/bin/echo"}"#)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel()
        var project = AppProject(name: "Lifecycle", rootDirectoryPath: root.path)
        model.projects = [project]
        model.selectedProjectID = project.id

        model.detectProjectMcpServers()
        let pending = try XCTUnwrap(model.pendingMcpApprovals.first)

        model.rejectPendingMcpServer(id: pending.id)
        project = try XCTUnwrap(model.projects.first)
        XCTAssertTrue(project.mcpServers.isEmpty, "a rejected server must not be imported")
        XCTAssertEqual(project.rejectedMcpJsonServers, ["bad"])

        model.detectProjectMcpServers()
        XCTAssertTrue(model.pendingMcpApprovals.isEmpty, "a rejected name must never re-prompt")
    }

    func testApproveAllFutureImportsCurrentServersWithoutPrompting() throws {
        let root = try makeProjectRoot(serversJSON: #"""
        "a": {"command": "/bin/echo"}, "b": {"command": "/bin/echo"}
        """#)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel()
        var project = AppProject(name: "Lifecycle", rootDirectoryPath: root.path)
        model.projects = [project]
        model.selectedProjectID = project.id

        model.detectProjectMcpServers()
        XCTAssertEqual(model.pendingMcpApprovals.count, 2)
        model.approvePendingMcpServer(id: model.pendingMcpApprovals[0].id, approveAllFuture: true)

        project = try XCTUnwrap(model.projects.first)
        XCTAssertTrue(project.approveAllProjectMcpServers)
        XCTAssertEqual(
            Set(project.mcpServers.map(\.name)), ["a", "b"],
            "approve-all must cover the server approved AND its undeclared siblings")
        XCTAssertTrue(model.pendingMcpApprovals.isEmpty)

        model.detectProjectMcpServers()
        XCTAssertTrue(model.pendingMcpApprovals.isEmpty)
    }

    func testChangedConfigPromptsOnlyForNewNames() throws {
        let root = try makeProjectRoot(serversJSON: #""keep": {"command": "/bin/echo"}"#)
        defer { try? FileManager.default.removeItem(at: root) }

        let model = AppModel()
        let project = AppProject(name: "Lifecycle", rootDirectoryPath: root.path)
        model.projects = [project]
        model.selectedProjectID = project.id

        model.detectProjectMcpServers()
        model.approvePendingMcpServer(id: model.pendingMcpApprovals[0].id)

        // The repo adds a server to its config.
        try #"{"mcpServers": {"keep": {"command": "/bin/echo"}, "new": {"command": "/bin/echo"}} }"#
            .write(to: root.appendingPathComponent(".mcp.json"), atomically: true, encoding: .utf8)

        model.detectProjectMcpServers()
        XCTAssertEqual(
            model.pendingMcpApprovals.map(\.config.name), ["new"],
            "only the newly declared name prompts")
    }

    // MARK: - Archive compatibility

    func testArchiveWithoutNewFieldsDecodesWithDefaults() throws {
        let legacy = #"{"name":"Old","mcpServers":[]}"#
        let project = try JSONDecoder().decode(AppProject.self, from: Data(legacy.utf8))
        XCTAssertTrue(project.approvedMcpJsonServers.isEmpty)
        XCTAssertTrue(project.rejectedMcpJsonServers.isEmpty)
        XCTAssertFalse(project.approveAllProjectMcpServers)
        XCTAssertTrue(project.permissions.mcpAllowRules.isEmpty)
        XCTAssertTrue(project.permissions.mcpDenyRules.isEmpty)
    }

    func testPermissionRulesRoundTripThroughTheArchive() throws {
        var permissions = AppProjectPermissions(mode: .auto)
        permissions.mcpAllowRules = ["mcp__fs__read"]
        permissions.mcpDenyRules = ["mcp__fs"]
        let project = makeProject(permissions: permissions)

        let data = try JSONEncoder().encode(project)
        let decoded = try JSONDecoder().decode(AppProject.self, from: data)
        XCTAssertEqual(decoded.permissions.mcpAllowRules, ["mcp__fs__read"])
        XCTAssertEqual(decoded.permissions.mcpDenyRules, ["mcp__fs"])
    }
}
