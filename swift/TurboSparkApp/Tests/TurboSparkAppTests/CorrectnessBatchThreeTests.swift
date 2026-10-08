import XCTest

@testable import TurboSparkApp

/// Regression tests for the third slice of the "Correctness" medium findings
/// (tools, hooks, MCP, cron, REPL lowering, workflow world).
final class CorrectnessBatchThreeTests: XCTestCase {
    private var tempRoots: [URL] = []

    override func tearDown() {
        for url in tempRoots { try? FileManager.default.removeItem(at: url) }
        tempRoots.removeAll()
        super.tearDown()
    }

    private func makeRoot() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("tsp-corr3-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        tempRoots.append(url)
        // Resolve /var -> /private/var so paths compare with resolved URLs.
        return url.resolvingSymlinksInPath()
    }

    // MARK: - Forge rescue

    private func rescueTools(duplicate: Bool = false) -> TurnAvailableTools {
        let search = OpenAITool.function(
            name: "search",
            description: "Search",
            parameters: .object(
                properties: [
                    "query": .string(),
                    "limit": .integer(),
                    "exact": .boolean(),
                ],
                required: ["query"]))
        return TurnAvailableTools(definitions: duplicate ? [search, search] : [search])
    }

    func testRescuedCallKeepsIntegerAndBooleanArgumentsDispatchable() {
        let content = #"<function=search>{"query":"x","limit":5,"exact":true}</function>"#
        let result = ToolCallDispatchGate.evaluate(
            content: content, streamState: .completed,
            availableTools: rescueTools(), forgeGuardrailsEnabled: true)
        XCTAssertEqual(result.refusals, [], "typed rescued arguments must not be refused")
        XCTAssertEqual(result.dispatchableCalls.map(\.name), ["search"])
        XCTAssertEqual(result.dispatchableCalls.first?.arguments["limit"], "5")
    }

    func testDuplicateToolNamesDoNotTrapTheGuardrailInspection() {
        // A project custom tool shadowing a built-in produced duplicate names;
        // Dictionary(uniqueKeysWithValues:) trapped the whole app on them.
        let content = #"<function=search>{"query":"x"}</function>"#
        let result = ToolCallDispatchGate.evaluate(
            content: content, streamState: .completed,
            availableTools: rescueTools(duplicate: true), forgeGuardrailsEnabled: true)
        XCTAssertEqual(result.dispatchableCalls.map(\.name), ["search"])
    }

    func testRescueFlattensJsonBooleansToTrueFalseNotOneZero() {
        let calls = ForgeGuardrailsEngine.rescueToolCalls(
            from: #"<function=search>{"query":"x","exact":true,"limit":3}</function>"#,
            availableToolNames: ["search"])
        XCTAssertEqual(calls.first?.arguments["exact"], "true")
        XCTAssertEqual(calls.first?.arguments["limit"], "3")
    }

    // MARK: - Hook placeholders

    func testEnvPlaceholderIsSafeInEveryQuotingContext() throws {
        let raw = "${CLAUDE_PLUGIN_ROOT}"
        let expand = { (command: String) in
            AppHookExecutionEngine.expandEnvPlaceholder(
                raw, inCommand: command, envName: "CLAUDE_PLUGIN_ROOT", shell: .bash)
        }
        XCTAssertEqual(expand("bash \(raw)/x"), "bash \"${CLAUDE_PLUGIN_ROOT}\"/x")
        // An already double-quoted placeholder must NOT gain nested quotes.
        XCTAssertEqual(expand("bash \"\(raw)/x\""), "bash \"${CLAUDE_PLUGIN_ROOT}/x\"")
        XCTAssertEqual(expand("echo '\(raw)'"), "echo ''\"${CLAUDE_PLUGIN_ROOT}\"''")

        // Run each against a path with a space and shell metacharacters.
        let dir = try makeRoot().appendingPathComponent("My Plugins; echo pwned", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        for command in ["printf '%s' \(raw)", "printf '%s' \"\(raw)\"", "printf '%s' '\(raw)'"] {
            let process = Process()
            process.executableURL = URL(fileURLWithPath: "/bin/bash")
            process.arguments = ["-c", expand(command)]
            process.environment = ["CLAUDE_PLUGIN_ROOT": dir.path]
            let pipe = Pipe()
            process.standardOutput = pipe
            try process.run()
            let data = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            XCTAssertEqual(String(decoding: data, as: UTF8.self), dir.path, command)
        }
    }

    func testSensitiveUserConfigPlaceholderBecomesAnEnvReference() {
        let expanded = AppHookExecutionEngine.expandEnvPlaceholder(
            "${user_config.api_token}",
            inCommand: "check --token ${user_config.api_token}",
            envName: "CLAUDE_PLUGIN_OPTION_API_TOKEN", shell: .bash)
        XCTAssertEqual(expanded, "check --token \"${CLAUDE_PLUGIN_OPTION_API_TOKEN}\"")
        XCTAssertFalse(expanded.contains("user_config"))
    }

    @MainActor
    func testDisabledPluginHookStaysDisabledAcrossRefreshAndRelaunch() throws {
        let root = try makeRoot()
        let manager = PluginManager(
            turboSparkRoot: root, claudeRoot: root.appendingPathComponent("unused"),
            userEnableProvider: { [:] }, projectEnableProvider: { _ in [:] },
            claudeEnableProvider: { [:] })
        let marker = UUID().uuidString
        let dir = root.appendingPathComponent("p-\(marker)", isDirectory: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent(".claude-plugin"), withIntermediateDirectories: true)
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("hooks"), withIntermediateDirectories: true)
        try Data(#"{"name": "p-\#(marker)"}"#.utf8)
            .write(to: dir.appendingPathComponent(".claude-plugin/plugin.json"))
        try Data(
            """
            {"hooks": {"PreToolUse": [{"matcher": "Bash", "hooks": [{"type": "command", "command": "echo \(marker)"}]}]}}
            """.utf8
        ).write(to: dir.appendingPathComponent("hooks/hooks.json"))

        func makeStore() -> AppHookStore {
            let store = AppHookStore()
            store.pluginProvider = { _ in manager.enabledPlugins(projectURL: nil) }
            store.refresh(projectDirectory: nil)
            return store
        }
        let store = makeStore()
        let hook = try XCTUnwrap(store.hooks.first { $0.command.contains(marker) })
        XCTAssertTrue(hook.isEnabled)
        store.toggleHookEnabled(id: hook.id)
        XCTAssertFalse(try XCTUnwrap(store.hooks.first { $0.command.contains(marker) }).isEnabled)

        store.refresh(projectDirectory: nil)
        XCTAssertFalse(
            try XCTUnwrap(store.hooks.first { $0.command.contains(marker) }).isEnabled,
            "refresh must not re-enable a hook the user switched off")
        let relaunched = makeStore()
        let again = try XCTUnwrap(relaunched.hooks.first { $0.command.contains(marker) })
        XCTAssertFalse(again.isEnabled, "the off switch must survive a relaunch")

        // Leave no disabled entry behind for the shared test storage root.
        relaunched.toggleHookEnabled(id: again.id)
    }

    // MARK: - MCP

    func testRelativeCommandAndCwdResolveAgainstTheProjectNotTheAppCwd() throws {
        let project = try makeRoot()
        let scripts = project.appendingPathComponent("scripts", isDirectory: true)
        try FileManager.default.createDirectory(at: scripts, withIntermediateDirectories: true)
        let script = scripts.appendingPathComponent("server.sh")
        try Data("#!/bin/sh\n".utf8).write(to: script)
        try FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: script.path)

        XCTAssertEqual(
            McpClientEngine.resolveExecutablePath("./scripts/server.sh", relativeTo: project),
            script.path)
        XCTAssertEqual(
            McpClientEngine.resolveExecutablePath("scripts/server.sh", relativeTo: project),
            script.path)
        XCTAssertNil(McpClientEngine.resolveExecutablePath("./scripts/missing.sh", relativeTo: project))

        let cwd = McpClientEngine.resolveWorkingDirectory(cwd: "scripts", fallback: project)
        XCTAssertEqual(cwd?.path, script.deletingLastPathComponent().path)
    }

    func testServerThatExitsImmediatelyFailsFastWithItsStderr() async throws {
        let config = McpServerConfig(
            name: "dies",
            transport: .stdio(
                command: "/bin/sh",
                args: ["-c", "echo missing-api-key >&2; exit 3"]))
        let started = Date()
        do {
            _ = try await McpClientEngine.shared.discoverTools(for: config, timeoutSeconds: 10)
            XCTFail("a dead server must not discover tools")
        } catch {
            let message = error.localizedDescription
            XCTAssertTrue(message.contains("missing-api-key"), message)
            XCTAssertTrue(message.contains("status 3"), message)
            XCTAssertFalse(message.contains("Timed out"), message)
        }
        XCTAssertLessThan(Date().timeIntervalSince(started), 5, "must not wait the full timeout")
    }

    // MARK: - Worktree

    func testExitWorktreeRemoveWithoutPathRefusesInsteadOfClaimingSuccess() async {
        do {
            let output = try await WorktreeExecutor.exit(
                arguments: ["action": "remove"],
                rootURL: FileManager.default.temporaryDirectory)
            XCTFail("a remove with no path removed nothing yet returned: \(output)")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("path"))
        }
    }

    // MARK: - Integer overflow traps

    func testHugeModelSuppliedIntegersDoNotTrap() async throws {
        let root = try makeRoot()
        let file = root.appendingPathComponent("big.txt")
        try Data("one\ntwo\nthree\n".utf8).write(to: file)

        let read = try await AppToolRegistry.readFile(
            relPath: "big.txt", rootURL: root, startLine: 2, limit: Int.max)
        XCTAssertTrue(read.contains("two"))
        let read2 = try await AppToolRegistry.readFile(
            relPath: "big.txt", rootURL: root, startLine: 1, limit: 9_223_372_036_854_774_784)
        XCTAssertTrue(read2.contains("three"))

        _ = try await AppToolRegistry.readFile(relPath: "big.txt", rootURL: root)
        _ = try await AppToolRegistry.editFile(
            relPath: "big.txt", newString: "inserted", rootURL: root,
            command: "insert", insertLine: "-9223372036854775808", position: "before")
        let text = try String(contentsOf: file, encoding: .utf8)
        XCTAssertTrue(text.hasPrefix("inserted"))
    }

    // MARK: - Glob and Grep

    func testGlobMatchesRecursivelyByPatternUnderPath() async throws {
        let root = try makeRoot()
        let sources = root.appendingPathComponent("Sources/Deep", isDirectory: true)
        try FileManager.default.createDirectory(at: sources, withIntermediateDirectories: true)
        try Data("x".utf8).write(to: sources.appendingPathComponent("a.swift"))
        try Data("x".utf8).write(to: sources.appendingPathComponent("b.txt"))
        try Data("x".utf8).write(to: root.appendingPathComponent("top.swift"))

        let project = AppProject(name: "glob", rootDirectoryPath: root.path)
        let recursive = await AppToolRegistry.execute(
            call: AppToolCall(name: "Glob", arguments: ["pattern": "**/*.swift"], category: .fileRead),
            in: project)
        XCTAssertFalse(recursive.isError, recursive.output)
        XCTAssertTrue(recursive.output.contains("Sources/Deep/a.swift"), recursive.output)
        XCTAssertTrue(recursive.output.contains("top.swift"), recursive.output)
        XCTAssertFalse(recursive.output.contains("b.txt"), recursive.output)

        let scoped = await AppToolRegistry.execute(
            call: AppToolCall(
                name: "Glob", arguments: ["pattern": "*.swift", "path": "Sources"], category: .fileRead),
            in: project)
        XCTAssertTrue(scoped.output.contains("Deep/a.swift"), scoped.output)
        XCTAssertFalse(scoped.output.contains("top.swift"), scoped.output)
    }

    func testGrepFallbackHonorsRegexCaseAndFilters() throws {
        let root = try makeRoot()
        try Data("func execute(x) {}\nFUNC loud()\n".utf8)
            .write(to: root.appendingPathComponent("a.swift"))
        try Data("func execute(y) {}\n".utf8).write(to: root.appendingPathComponent("b.txt"))

        let regex = try AppToolRegistry.searchCode(
            pattern: #"func\s+execute\("#, relPath: ".", rootURL: root)
        XCTAssertTrue(regex.contains("a.swift:1"), regex)

        let literal = try AppToolRegistry.searchCode(
            pattern: #"func\s+execute\("#, relPath: ".", rootURL: root, literal: true)
        XCTAssertTrue(literal.contains("No matches"), literal)

        let sensitive = try AppToolRegistry.searchCode(
            pattern: "func", relPath: ".", rootURL: root, caseSensitive: true,
            fileTypes: ["swift"])
        XCTAssertTrue(sensitive.contains("a.swift:1"), sensitive)
        XCTAssertFalse(sensitive.contains("a.swift:2"), sensitive)
        XCTAssertFalse(sensitive.contains("b.txt"), sensitive)

        let globbed = try AppToolRegistry.searchCode(
            pattern: "execute", relPath: ".", rootURL: root, pathFilter: "*.txt")
        XCTAssertTrue(globbed.contains("b.txt"), globbed)
        XCTAssertFalse(globbed.contains("a.swift"), globbed)
    }

    func testSyntextScopeTranslatesPathIntoDirectoryPrefix() throws {
        let root = try makeRoot()
        try FileManager.default.createDirectory(
            at: root.appendingPathComponent("crates/ffi"), withIntermediateDirectories: true)
        XCTAssertEqual(AppToolRegistry.syntextPathScope(relPath: "crates/ffi", rootURL: root), "crates/ffi/")
        XCTAssertNil(AppToolRegistry.syntextPathScope(relPath: ".", rootURL: root))
    }

    // MARK: - Batch

    func testBatchKeepsNestedArrayAndObjectArguments() throws {
        let raw = #"[{"tool":"TodoWrite","parameters":{"todos":[{"content":"a","status":"pending"}],"opts":{"k":1},"n":2}}]"#
        let items = try BatchToolExecutor.parseItems(from: ["tool_calls": raw])
        let todos = try XCTUnwrap(items.first?.parameters["todos"])
        XCTAssertTrue(todos.contains("\"content\":\"a\""), todos)
        XCTAssertEqual(items.first?.parameters["opts"], #"{"k":1}"#)
        XCTAssertEqual(items.first?.parameters["n"], "2")
    }

    // MARK: - Shell benign exit

    func testBenignExitOnlyAppliesToASingleInvocation() {
        XCTAssertEqual(ShellOutputFormatting.benignExitNote(command: "grep -r foo .", exitCode: 1), "No matches found")
        XCTAssertEqual(ShellOutputFormatting.benignExitNote(command: "grep 'a;b$' f 2>&1", exitCode: 1), "No matches found")
        XCTAssertNil(ShellOutputFormatting.benignExitNote(command: "test -f Package.swift && swift build", exitCode: 1))
        XCTAssertNil(ShellOutputFormatting.benignExitNote(command: "[ -d node_modules ] && npm test", exitCode: 1))
        XCTAssertNil(ShellOutputFormatting.benignExitNote(command: "grep x f; make", exitCode: 1))
    }

    // MARK: - Cron

    private var utc: Calendar {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        return calendar
    }

    private func date(_ h: Int, _ m: Int, _ s: Int) -> Date {
        utc.date(from: DateComponents(
            timeZone: TimeZone(identifier: "UTC"), year: 2026, month: 1, day: 5,
            hour: h, minute: m, second: s))!
    }

    func testNextFireDoesNotSkipAMinuteWhenAfterHasSeconds() throws {
        let everyMinute = try XCTUnwrap(CronSchedule("* * * * *"))
        XCTAssertEqual(everyMinute.nextFire(after: date(10, 30, 12), calendar: utc), date(10, 31, 0))
        XCTAssertEqual(everyMinute.nextFire(after: date(10, 30, 0), calendar: utc), date(10, 31, 0))

        let nine = try XCTUnwrap(CronSchedule("0 9 * * *"))
        XCTAssertEqual(nine.nextFire(after: date(8, 59, 30), calendar: utc), date(9, 0, 0))
    }

    func testOneShotWakeupKeepsItsFireTimeAcrossRelaunchAndPauseResume() throws {
        let directory = try makeRoot()
        let fireAt = Date().addingTimeInterval(3_600)
        let scheduler = CronScheduler(directory: directory)
        let job = scheduler.createOneShot(fireAt: fireAt, prompt: "wake", chatID: UUID())

        let relaunched = CronScheduler(directory: directory)
        let loaded = try XCTUnwrap(relaunched.list().first { $0.id == job.id })
        XCTAssertEqual(try XCTUnwrap(loaded.nextFireAt).timeIntervalSince1970, fireAt.timeIntervalSince1970, accuracy: 1)
        XCTAssertTrue(loaded.humanSchedule.hasPrefix("Once at"), loaded.humanSchedule)

        relaunched.setEnabled(id: job.id, enabled: false)
        relaunched.setEnabled(id: job.id, enabled: true)
        let resumed = try XCTUnwrap(relaunched.list().first { $0.id == job.id })
        XCTAssertEqual(try XCTUnwrap(resumed.nextFireAt).timeIntervalSince1970, fireAt.timeIntervalSince1970, accuracy: 1)
    }

    // MARK: - Push notification

    func testPushNotificationDoesNotClaimSystemDeliveryOutsideTheAppBundle() async throws {
        let output = try await PushNotificationExecutor.execute(arguments: ["message": "done"])
        XCTAssertTrue(output.contains("shown in-app only"), output)
        XCTAssertFalse(output.contains("posted successfully"), output)
    }

    // MARK: - Custom tools

    func testCustomToolBaseEnvironmentHasHomebrewPathAndLocale() {
        let env = CustomToolExecutor.baseEnvironment(parent: ["PATH": "/usr/bin:/bin"])
        XCTAssertTrue(env["PATH"]?.contains("/opt/homebrew/bin") == true)
        XCTAssertEqual(env["LANG"], "en_US.UTF-8")
        XCTAssertNil(env["ANTHROPIC_API_KEY"])
    }

    func testCustomToolRunsWithUsablePathAndLang() async throws {
        let tool = CustomToolDefinition(
            name: "envprobe", toolDescription: "probe",
            execution: CustomToolExecution(type: .command, command: "printf '%s|%s' \"$PATH\" \"$LANG\""))
        let output = try await CustomToolExecutor.execute(
            tool: tool, arguments: [:], projectRootURL: try makeRoot())
        XCTAssertTrue(output.contains("/opt/homebrew/bin"), output)
        XCTAssertTrue(output.contains("UTF-8"), output)
    }

    func testCustomToolSchemaWithNumericDefaultKeepsItsProperties() throws {
        let dict: [String: Any] = [
            "name": "counter",
            "command": "echo hi",
            "parameters": [
                "type": "object",
                "properties": ["count": ["type": "integer", "default": 10]],
                "required": ["count"],
            ],
        ]
        let tool = try CustomToolParser.parseDictionary(
            dict, sourceURL: URL(fileURLWithPath: "/tmp/counter.json"), scope: .userGlobal)
        XCTAssertEqual(tool.parameters.required, ["count"])
        XCTAssertEqual(tool.parameters.properties?["count"]?.type, "integer")
    }

    // MARK: - apply_patch

    func testPlainMultiFileUnifiedDiffPatchesEachFile() async throws {
        let root = try makeRoot()
        try Data("a\nb\nc\n".utf8).write(to: root.appendingPathComponent("f1.txt"))
        try Data("x\ny\nz\n".utf8).write(to: root.appendingPathComponent("f2.txt"))
        let patch = """
        --- a/f1.txt
        +++ b/f1.txt
        @@ -1,3 +1,3 @@
         a
        -b
        +B
         c
        --- a/f2.txt
        +++ b/f2.txt
        @@ -1,3 +1,3 @@
         x
        -y
        +Y
         z

        """
        _ = try await ApplyPatchExecutor.apply(patchText: patch, rootURL: root)
        XCTAssertEqual(try String(contentsOf: root.appendingPathComponent("f1.txt"), encoding: .utf8), "a\nB\nc\n")
        XCTAssertEqual(try String(contentsOf: root.appendingPathComponent("f2.txt"), encoding: .utf8), "x\nY\nz\n")
    }

    func testRemovedLineThatLooksLikeAFileHeaderIsHunkBody() async throws {
        let root = try makeRoot()
        try Data("select 1\n-- old comment\nselect 2\n".utf8).write(to: root.appendingPathComponent("q.sql"))
        let patch = """
        --- a/q.sql
        +++ b/q.sql
        @@ -1,3 +1,3 @@
         select 1
        --- old comment
        +-- new comment
         select 2

        """
        _ = try await ApplyPatchExecutor.apply(patchText: patch, rootURL: root)
        XCTAssertEqual(
            try String(contentsOf: root.appendingPathComponent("q.sql"), encoding: .utf8),
            "select 1\n-- new comment\nselect 2\n")
    }

    // MARK: - Snapshot BOM

    func testBomFileCanBeEditedAfterReading() async throws {
        let root = try makeRoot()
        let file = root.appendingPathComponent("bom.csproj")
        try Data([0xEF, 0xBB, 0xBF] + Array("<a>old</a>\n".utf8)).write(to: file)

        _ = try await AppToolRegistry.readFile(relPath: "bom.csproj", rootURL: root)
        let url = try AppToolRegistry.resolveSecurePath(relPath: "bom.csproj", rootURL: root)
        let stale = await FileSnapshotStore.shared.isStale(url: url)
        XCTAssertFalse(stale, "a BOM file read moments ago must not be stale")
        _ = try await AppToolRegistry.editFile(
            relPath: "bom.csproj", oldString: "old", newString: "new", rootURL: root)
        XCTAssertTrue(try String(contentsOf: file, encoding: .utf8).contains("new"))
    }

    // MARK: - NotebookEdit

    private func writeNotebook(_ root: URL, withIDs: Bool) throws {
        func cell(_ id: String, _ source: String) -> String {
            let idField = withIDs ? "\"id\": \"\(id)\"," : ""
            return #"{\#(idField) "cell_type": "code", "metadata": {}, "source": ["\#(source)"], "outputs": [], "execution_count": null}"#
        }
        let json = #"{"cells": [\#(cell("c1", "one")), \#(cell("c2", "two")), \#(cell("c3", "three"))], "metadata": {}, "nbformat": 4, "nbformat_minor": \#(withIDs ? 5 : 4)}"#
        try Data(json.utf8).write(to: root.appendingPathComponent("n.ipynb"))
    }

    private func notebookSources(_ root: URL) throws -> [String] {
        let data = try Data(contentsOf: root.appendingPathComponent("n.ipynb"))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        return (object["cells"] as? [[String: Any]] ?? []).map {
            ($0["source"] as? [String] ?? []).joined()
        }
    }

    func testNotebookEditRefusesAmbiguousOrUnknownTargets() async throws {
        let root = try makeRoot()
        try writeNotebook(root, withIDs: true)

        for arguments in [
            ["notebook_path": "n.ipynb", "edit_mode": "delete"],
            ["notebook_path": "n.ipynb", "edit_mode": "replace", "new_source": "x"],
            ["notebook_path": "n.ipynb", "edit_mode": "replace", "cell_id": "nope", "new_source": "x"],
        ] {
            do {
                _ = try await NotebookEditExecutor.execute(arguments: arguments, rootURL: root)
                XCTFail("must refuse: \(arguments)")
            } catch {}
        }
        XCTAssertEqual(try notebookSources(root), ["one", "two", "three"], "nothing may change on a refusal")
    }

    func testNotebookEditInsertAnchorsAfterTheCellAndMintsAFreshID() async throws {
        let root = try makeRoot()
        try writeNotebook(root, withIDs: true)
        _ = try await NotebookEditExecutor.execute(
            arguments: ["notebook_path": "n.ipynb", "edit_mode": "insert", "cell_id": "c1", "new_source": "new"],
            rootURL: root)
        XCTAssertEqual(try notebookSources(root), ["one", "new", "two", "three"])
        let data = try Data(contentsOf: root.appendingPathComponent("n.ipynb"))
        let object = try XCTUnwrap(JSONSerialization.jsonObject(with: data) as? [String: Any])
        let ids = (object["cells"] as? [[String: Any]] ?? []).compactMap { $0["id"] as? String }
        XCTAssertEqual(Set(ids).count, ids.count, "cell ids must stay unique")
    }

    func testNotebookWithoutIDsAcceptsCellNSpelling() async throws {
        let root = try makeRoot()
        try writeNotebook(root, withIDs: false)
        _ = try await NotebookEditExecutor.execute(
            arguments: ["notebook_path": "n.ipynb", "edit_mode": "replace", "cell_id": "cell-1", "new_source": "TWO"],
            rootURL: root)
        XCTAssertEqual(try notebookSources(root), ["one", "TWO", "three"])
    }

    // MARK: - REPL lowering

    private func lower(_ source: String) throws -> (program: REPLBoundedProgram, wrapper: String) {
        guard case let .program(program) = REPLBoundedScriptParser.parse(source: source) else {
            XCTFail("expected a program for: \(source)")
            throw CancellationError()
        }
        let wrapper = REPLLoweredWrapperBuilder.build(
            program: program, source: source, settleGlobal: "__settle", renderGlobal: "__render")
        return (program, wrapper)
    }

    func testTrailingObjectLiteralAssignmentDoesNotEmitEmptyReturn() throws {
        let (_, wrapper) = try lower("const data = await Promise.resolve(1);\nconst config = { a: data };")
        XCTAssertFalse(wrapper.contains("return (;)"), wrapper)
        XCTAssertTrue(wrapper.contains("globalThis.config = ({ a: data });"), wrapper)

        let (_, bare) = try lower("await Promise.resolve(1);\nconst f = () => { return 1; };")
        XCTAssertFalse(bare.contains("(;)"), bare)
    }

    func testObjectLiteralExpressionsAreNotCutAtTheClosingBrace() throws {
        let (program, wrapper) = try lower("const x = true ? {a: 1} : {a: 2}\nconst y = await Promise.resolve(x.a)")
        XCTAssertEqual(program.statements.count, 2)
        XCTAssertTrue(wrapper.contains("globalThis.x = (true ? {a: 1} : {a: 2});"), wrapper)

        let (indexed, indexedWrapper) = try lower("const v = {a: 1}['a']\nawait Promise.resolve(v)")
        XCTAssertEqual(indexed.statements.count, 2)
        XCTAssertTrue(indexedWrapper.contains("globalThis.v = ({a: 1}['a']);"), indexedWrapper)
    }

    func testAsyncAndGeneratorFunctionDeclarationsPersist() throws {
        let (_, wrapper) = try lower("async function load() { return 1; }\nfunction* gen() { yield 1; }\nconst x = await load();")
        XCTAssertTrue(wrapper.contains("globalThis.load = async function load()"), wrapper)
        XCTAssertTrue(wrapper.contains("globalThis.gen = function* gen()"), wrapper)
    }

    // MARK: - Workflow world

    func testGlobIgnoresUnrelatedEscapingSymlinks() async throws {
        let root = try makeRoot()
        try Data("x".utf8).write(to: root.appendingPathComponent("notes.md"))
        try FileManager.default.createSymbolicLink(
            atPath: root.appendingPathComponent("python3").path, withDestinationPath: "/bin/ls")
        let world = WorkflowWorld(workspaceRoot: root, executor: WorkflowWorldTestExecutor())
        let observation = try await world.read(
            .glob(pattern: "*.md"),
            identity: WorkflowRequestIdentity(
                site: WorkflowSiteKey(lane: "main", siteIndex: 0, ordinal: 0), inputHash: "t"))
        XCTAssertEqual(observation.outputText, "notes.md")
    }

    func testRootGrepSkipsVcsMetadataAndBinaryFilesWithoutSpendingTheBudget() async throws {
        let root = try makeRoot()
        let objects = root.appendingPathComponent(".git/objects", isDirectory: true)
        try FileManager.default.createDirectory(at: objects, withIntermediateDirectories: true)
        // Larger than the default scan budget, and binary (NUL bytes).
        try Data(count: 9 * 1_024 * 1_024).write(to: objects.appendingPathComponent("pack"))
        try Data("let a = 1 // TODO fix\n".utf8).write(to: root.appendingPathComponent("a.swift"))
        let world = WorkflowWorld(workspaceRoot: root, executor: WorkflowWorldTestExecutor())
        let observation = try await world.read(
            .grep(pattern: "TODO", pathHint: nil),
            identity: WorkflowRequestIdentity(
                site: WorkflowSiteKey(lane: "main", siteIndex: 0, ordinal: 0), inputHash: "t"))
        XCTAssertTrue(observation.outputText.contains("a.swift:1:"), observation.outputText)
    }
}
