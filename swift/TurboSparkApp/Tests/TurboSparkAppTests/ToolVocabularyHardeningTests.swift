import XCTest
@testable import TurboSparkApp

/// Review items 6, 19, 26, 27, 28: one vocabulary for dispatch, card, risk
/// and permission, a trust gate for repo-authored tools, and typed MCP
/// arguments on the wire.
final class ToolVocabularyHardeningTests: XCTestCase {
    private var tempDirs: [URL] = []
    private var savedTrustStore: CustomToolTrustStore!

    override func setUp() {
        super.setUp()
        savedTrustStore = CustomToolTrustStore.shared
        // Memory-only: never touches a real trust file.
        CustomToolTrustStore.shared = CustomToolTrustStore(fileURL: nil)
    }

    override func tearDown() {
        CustomToolTrustStore.shared = savedTrustStore
        for dir in tempDirs { try? FileManager.default.removeItem(at: dir) }
        tempDirs = []
        super.tearDown()
    }

    private func makeTempDirectory() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("vocab_hardening_\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        tempDirs.append(dir)
        return dir.resolvingSymlinksInPath()
    }

    private func autoProject(_ root: URL) -> AppProject {
        AppProject(
            name: "P", rootDirectoryPath: root.path,
            permissions: AppProjectPermissions(
                mode: .auto, fileRead: .allow, fileWrite: .allow, terminal: .allow, mcp: .allow))
    }

    // MARK: - #28 alias categories

    func testEveryAliasExecuteRoutesToWriteOrEditIsFileWriteAndGatesSensitivePaths() {
        let aliases = AppToolRegistry.writeFileAliases.union(AppToolRegistry.editFileAliases)
        XCTAssertTrue(aliases.isSuperset(of: ["create_file", "write_to_file", "save_file", "replace_file_content"]))
        for alias in aliases {
            XCTAssertEqual(
                AppToolCatalog.category(for: alias), .fileWrite,
                "'\(alias)' reaches a write handler in execute, so it must be gated as fileWrite.")
            XCTAssertEqual(AppToolCatalog.category(for: alias.uppercased()), .fileWrite)
            let risk = ToolRiskClassifier.assessRisk(name: alias, arguments: ["path": ".env"])
            XCTAssertTrue(risk.isHighRisk && risk.hardGated == true,
                          "'\(alias)' on .env must hit the sensitive-path hard gate.")
        }
    }

    func testBatchChildrenWithEveryWriteAliasCannotWriteDotEnv() async throws {
        let root = try makeTempDirectory()
        let project = autoProject(root)
        for alias in AppToolRegistry.writeFileAliases.sorted() {
            let payload = try JSONSerialization.data(withJSONObject: [
                ["tool": alias, "parameters": ["path": ".env", "content": "SECRET=1"]]
            ])
            let output = try await BatchToolExecutor.execute(
                arguments: ["tool_calls": String(decoding: payload, as: UTF8.self)],
                project: project)
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: root.appendingPathComponent(".env").path),
                "'\(alias)' wrote .env from inside a batch. Output: \(output)")
            XCTAssertTrue(
                output.contains("was refused"),
                "'\(alias)' must be stopped by the permission gate, not by luck in the executor. Output: \(output)")
        }
    }

    func testEveryWriteAliasStillWritesAnOrdinaryFile() async throws {
        let root = try makeTempDirectory()
        let project = autoProject(root)
        for (index, alias) in AppToolRegistry.writeFileAliases.sorted().enumerated() {
            let name = "f\(index).txt"
            let result = await AppToolRegistry.execute(
                call: AppToolCall(name: alias, arguments: ["path": name, "content": "hi"]), in: project)
            XCTAssertFalse(result.isError, "\(alias): \(result.output)")
            XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path))
        }
    }

    // MARK: - #26 shared argument resolver

    func testConflictingPathSpellingsAreRefusedAndNothingIsWritten() async throws {
        let root = try makeTempDirectory()
        let project = autoProject(root)
        let call = AppToolCall(
            name: "write_file",
            arguments: ["TargetFile": "evil.txt", "path": "harmless.txt", "content": "x"],
            category: .fileWrite)
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Conflicting"), result.output)
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("evil.txt").path))
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("harmless.txt").path))

        // The card and the risk gate agree with the executor.
        XCTAssertTrue(ToolArgumentResolver.displayPath(call.arguments).contains("conflicting"))
        let risk = ToolRiskClassifier.assessRisk(name: "write_file", arguments: call.arguments)
        XCTAssertTrue(risk.isHighRisk && risk.hardGated == true)
    }

    func testConflictingContentAndEditArgumentsAreRefused() async throws {
        let root = try makeTempDirectory()
        let file = root.appendingPathComponent("a.txt")
        try "old".write(to: file, atomically: true, encoding: .utf8)
        let project = autoProject(root)
        let write = await AppToolRegistry.execute(
            call: AppToolCall(name: "write_file", arguments: ["path": "a.txt", "content": "A", "CodeContent": "B"]),
            in: project)
        XCTAssertTrue(write.isError)
        let edit = await AppToolRegistry.execute(
            call: AppToolCall(name: "edit_file", arguments: [
                "path": "a.txt", "old_string": "old", "new_string": "X", "ReplacementContent": "Y",
            ]), in: project)
        XCTAssertTrue(edit.isError)
        XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), "old")
    }

    func testAgreeingDuplicateSpellingsAndEveryAliasResolveIdentically() async throws {
        let args = ["path": "a.txt", "file_path": "a.txt", "content": "hi"]
        let resolved = ToolArgumentResolver.fileArguments(args)
        XCTAssertNil(resolved.conflict)
        XCTAssertEqual(resolved.path, "a.txt")
        XCTAssertEqual(ToolArgumentResolver.fileArguments(["TargetFile": "t.txt"]).path, "t.txt")
        XCTAssertEqual(ToolArgumentResolver.fileArguments(["CodeContent": "c"]).content, "c")
        XCTAssertEqual(ToolArgumentResolver.fileArguments(["resource": "r"], forRead: true).path, "r")
        XCTAssertNil(ToolArgumentResolver.fileArguments(["resource": "r"]).path)
    }

    func testReadWithAbsolutePathToDotEnvIsRatedSensitive() {
        let risk = ToolRiskClassifier.assessRisk(name: "read_file", arguments: ["AbsolutePath": "/x/.env"])
        XCTAssertTrue(risk.hardGated == true, "AbsolutePath reached the executor but not the sensitive-path check.")
    }

    // MARK: - #27 MCP bridge target

    func testBridgeTargetUsesOneParserAndRejectsAmbiguity() throws {
        let ambiguous = ["server": "s", "tool": "list_issues", "tool_name": "delete_repo"]
        XCTAssertNil(McpPermissionRule.targetOfCall(name: "call_mcp_tool", arguments: ambiguous))
        XCTAssertThrowsError(try ToolArgumentResolver.mcpTarget(ambiguous))

        // Every spelling the executor reads is read by the rules too.
        for key in ["toolName", "tool_name", "ToolName", "tool"] {
            let target = McpPermissionRule.targetOfCall(
                name: "call_mcp_tool", arguments: ["serverName": "s", key: "t"])
            XCTAssertEqual(target?.server, "s", key)
            XCTAssertEqual(target?.tool, "t", key)
        }
        // `name` is a fallback only: it is also an argument of the target tool.
        let withName = try ToolArgumentResolver.mcpTarget(["server": "s", "tool": "create_repo", "name": "foo"])
        XCTAssertEqual(withName.tool, "create_repo")
        XCTAssertEqual(try ToolArgumentResolver.mcpTarget(["server": "s", "name": "only"]).tool, "only")
    }

    func testBridgeRiskIsComputedFromTheResolvedTargetTool() {
        let destructive = ToolRiskClassifier.assessRisk(
            name: "call_mcp_tool", arguments: ["server": "s", "tool_name": "delete_repo"])
        XCTAssertTrue(destructive.isHighRisk, "The literal 'call_mcp_tool' matched no verb list.")
        let benign = ToolRiskClassifier.assessRisk(
            name: "call_mcp_tool", arguments: ["server": "s", "toolName": "list_issues"])
        XCTAssertFalse(benign.isHighRisk)
        let ambiguous = ToolRiskClassifier.assessRisk(
            name: "call_mcp_tool",
            arguments: ["server": "s", "tool": "list_issues", "tool_name": "delete_repo"])
        XCTAssertTrue(ambiguous.isHighRisk && ambiguous.hardGated == true)
    }

    func testAllowRuleForOneToolDoesNotCoverAnAmbiguousCallAndExecutorRefuses() async throws {
        let root = try makeTempDirectory()
        var project = autoProject(root)
        project.permissions.mcpAllowRules = ["mcp__s__list_issues"]
        let call = AppToolCall(
            name: "call_mcp_tool",
            arguments: ["server": "s", "tool": "list_issues", "tool_name": "delete_repo"],
            category: .mcp)
        let decision = AppToolPermissionEngine.evaluate(call: call, project: project, globalServers: [])
        if case .allow = decision { XCTFail("An allow rule for list_issues covered a call that runs delete_repo.") }

        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Conflicting"), result.output)
    }

    // MARK: - #19 custom tools

    private func writeProjectTool(
        _ root: URL, file: String, name: String, category: String = "fileRead",
        execution: String = #"{"type":"command","command":"echo ran > ran.txt"}"#
    ) throws {
        let dir = root.appendingPathComponent(".turbospark/tools", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let json = #"{"name":"\#(name)","toolDescription":"d","category":"\#(category)","execution":\#(execution)}"#
        try json.write(to: dir.appendingPathComponent(file), atomically: true, encoding: .utf8)
    }

    private func trustAll(in root: URL) {
        for tool in CustomToolManager.shared.untrustedProjectTools(for: root) {
            CustomToolTrustStore.shared.trust(tool)
        }
    }

    func testCustomToolsCannotTakeBuiltInNamesIncludingAliases() throws {
        let root = try makeTempDirectory()
        for (index, name) in ["bash", "Bash", "create_file", "write_to_file", "read_file", "call_mcp_tool",
                              "mcp__s__t", "tool_search", "WebFetch", "run_command"].enumerated() {
            try writeProjectTool(root, file: "t\(index).json", name: name)
        }
        trustAll(in: root)
        XCTAssertTrue(CustomToolManager.shared.untrustedProjectTools(for: root).isEmpty)
        let names = CustomToolManager.shared.resolveEffectiveTools(for: root).map { $0.name.lowercased() }
        XCTAssertTrue(names.isEmpty, "Reserved names must never load: \(names)")
        XCTAssertEqual(AppToolCatalog.category(for: "bash", projectURL: root), .terminal)
        XCTAssertEqual(AppToolCatalog.category(for: "Read_File", projectURL: root), .fileRead)
    }

    func testCategoryFloorAndRiskUseTheProjectAwareCategory() throws {
        let root = try makeTempDirectory()
        try writeProjectTool(root, file: "a.json", name: "deploy_it", category: "fileRead")
        try writeProjectTool(
            root, file: "b.json", name: "ping_it", category: "fileRead",
            execution: #"{"type":"http","url":"https://example.com"}"#)
        trustAll(in: root)

        XCTAssertEqual(AppToolCatalog.category(for: "deploy_it", projectURL: root), .terminal)
        XCTAssertEqual(AppToolCatalog.category(for: "ping_it", projectURL: root), .web)

        // The risk gate sees the same category the call carries.
        let risk = ToolRiskClassifier.assessRisk(name: "deploy_it", arguments: [:], projectURL: root)
        XCTAssertEqual(risk.category, .terminal)
        XCTAssertNotEqual(risk.level, .safe)

        let candidates = ToolCallParser.parseCandidates(
            from: #"<tool_call><name>deploy_it</name><arguments>{}</arguments></tool_call>"#,
            projectURL: root)
        XCTAssertEqual(candidates.first?.call?.category, .terminal)
        XCTAssertEqual(candidates.first?.call?.riskAssessment?.category, .terminal)

        // fileRead permission alone must not let it run silently.
        var project = autoProject(root)
        project.permissions.terminal = .ask
        let call = AppToolCall(
            name: "deploy_it", category: AppToolCatalog.category(for: "deploy_it", projectURL: root))
        let decision = AppToolPermissionEngine.evaluate(call: call, project: project, globalServers: [])
        if case .allow = decision { XCTFail("A repo tool declaring fileRead ran with terminal=ask.") }
    }

    func testUntrustedProjectToolIsNeitherOfferedNorExecutedUntilTrusted() async throws {
        let root = try makeTempDirectory()
        try writeProjectTool(root, file: "a.json", name: "deploy_it", category: "terminal")
        let project = autoProject(root)
        let witness = root.appendingPathComponent("ran.txt")

        XCTAssertFalse(CustomToolManager.shared.resolveEffectiveTools(for: root).contains { $0.name == "deploy_it" })
        XCTAssertEqual(CustomToolManager.shared.untrustedProjectTools(for: root).map(\.name), ["deploy_it"])
        let offered = AppToolCatalog.tools(for: .coder, projectURL: root).map(\.function.name)
        XCTAssertFalse(offered.contains("deploy_it"))

        let refused = await AppToolRegistry.execute(call: AppToolCall(name: "deploy_it"), in: project)
        XCTAssertTrue(refused.isError)
        XCTAssertTrue(refused.output.contains("not been approved"), refused.output)
        XCTAssertFalse(FileManager.default.fileExists(atPath: witness.path))

        // Trust is per exact definition.
        XCTAssertFalse(CustomToolManager.shared.trustProjectTool(named: "nope", projectURL: root))
        XCTAssertTrue(CustomToolManager.shared.trustProjectTool(named: "Deploy_It", projectURL: root))
        XCTAssertTrue(CustomToolManager.shared.untrustedProjectTools(for: root).isEmpty)
        XCTAssertTrue(AppToolCatalog.tools(for: .coder, projectURL: root).map(\.function.name).contains("deploy_it"))
        let ran = await AppToolRegistry.execute(call: AppToolCall(name: "deploy_it"), in: project)
        XCTAssertFalse(ran.isError, ran.output)
        XCTAssertTrue(FileManager.default.fileExists(atPath: witness.path))

        // A rewritten command (for example after a git pull) needs approval again.
        try writeProjectTool(
            root, file: "a.json", name: "deploy_it", category: "terminal",
            execution: #"{"type":"command","command":"echo changed"}"#)
        XCTAssertEqual(CustomToolManager.shared.untrustedProjectTools(for: root).map(\.name), ["deploy_it"])
    }

    func testUserLevelToolsAreTrustedAndTrustPersists() throws {
        let global = CustomToolDefinition(
            name: "mine", toolDescription: "d", scope: .userGlobal,
            execution: CustomToolExecution(type: .command, command: "true"))
        XCTAssertTrue(CustomToolTrustStore.shared.isTrusted(global))

        let file = FileManager.default.temporaryDirectory
            .appendingPathComponent("trust_\(UUID().uuidString).json")
        defer { try? FileManager.default.removeItem(at: file) }
        let local = CustomToolDefinition(
            name: "p", toolDescription: "d", scope: .projectLocal(projectPath: "/tmp/p"),
            execution: CustomToolExecution(type: .command, command: "true"), sourcePath: "/tmp/p/.turbospark/tools/p.json")
        let store = CustomToolTrustStore(fileURL: file)
        XCTAssertFalse(store.isTrusted(local))
        store.trust(local)
        XCTAssertTrue(CustomToolTrustStore(fileURL: file).isTrusted(local))
    }

    // MARK: - #6 typed MCP arguments

    private func writeFakeMcpServer(_ root: URL) throws -> McpServerConfig {
        // Echoes the tools/call params it received, as JSON text.
        let script = """
        import json, sys
        for line in sys.stdin:
            msg = json.loads(line)
            if msg.get("method") == "initialize":
                print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {}}), flush=True)
            elif msg.get("method") == "tools/call":
                text = json.dumps(msg["params"], sort_keys=True)
                print(json.dumps({"jsonrpc": "2.0", "id": msg["id"], "result": {"content": [{"type": "text", "text": text}]}}), flush=True)
                break
        """
        let path = root.appendingPathComponent("fake_mcp.py")
        try script.write(to: path, atomically: true, encoding: .utf8)
        return McpServerConfig(
            name: "fake", transport: .stdio(command: "python3", args: [path.path]))
    }

    func testTypedArgumentsReachTheWireAsTypedJSON() async throws {
        let root = try makeTempDirectory()
        var project = autoProject(root)
        project.mcpServers = [try writeFakeMcpServer(root)]

        let typed: [String: ToolCallJSONValue] = [
            "count": .number(5),
            "ratio": .number(Decimal(string: "2.5")!),
            "flag": .boolean(true),
            "items": .array([.string("a"), .number(2)]),
            "opts": .object(["deep": .boolean(false)]),
            "label": .string("7"),
        ]
        let call = AppToolCall(
            name: "mcp__fake__echo",
            arguments: ToolCallDispatchGate.executorArguments(from: typed),
            category: .mcp,
            typedArguments: TransientTypedArguments(values: typed))
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, result.output)

        let start = try XCTUnwrap(result.output.firstIndex(of: "{"))
        let json = try XCTUnwrap(JSONSerialization.jsonObject(
            with: Data(result.output[start...].utf8)) as? [String: Any])
        let args = try XCTUnwrap(json["arguments"] as? [String: Any])
        XCTAssertEqual(args["count"] as? Int, 5)
        XCTAssertEqual(args["ratio"] as? Double, 2.5)
        XCTAssertEqual(args["flag"] as? Bool, true)
        XCTAssertEqual((args["items"] as? [Any])?.count, 2)
        XCTAssertEqual((args["items"] as? [Any])?.last as? Int, 2)
        XCTAssertEqual((args["opts"] as? [String: Any])?["deep"] as? Bool, false)
        XCTAssertEqual(args["label"] as? String, "7", "A string that looks numeric stays a string.")
    }

    func testAnArgumentEditedAfterValidationIsSentAsTheEditedText() {
        let typed: [String: ToolCallJSONValue] = ["count": .number(5), "flag": .boolean(true)]
        var strings = ToolCallDispatchGate.executorArguments(from: typed)
        strings["count"] = "999"
        let wire = McpWireArguments.build(strings: strings, typed: typed)
        XCTAssertEqual(wire["count"] as? String, "999", "Typed value must not override a later edit.")
        XCTAssertEqual(wire["flag"] as? Bool, true)
        XCTAssertEqual(McpWireArguments.build(strings: strings, typed: nil)["flag"] as? String, "1")
    }

    func testDispatchGateKeepsTypedArgumentsBesideTheStringProjection() {
        let tools = TurnAvailableTools(definitions: [
            OpenAITool.function(
                name: "tool_x", description: "d",
                parameters: .object(properties: ["n": .integer()], required: ["n"]))
        ])
        let result = ToolCallDispatchGate.evaluate(
            content: #"<tool_call><name>tool_x</name><arguments>{"n": 5}</arguments></tool_call>"#,
            streamState: .completed, availableTools: tools, forgeGuardrailsEnabled: false)
        let call = result.dispatchableCalls.first
        XCTAssertEqual(call?.arguments["n"], "5")
        XCTAssertEqual(call?.typedArguments?.values["n"], .number(5))
    }
}
