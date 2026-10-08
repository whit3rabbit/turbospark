import SwiftUI
import XCTest

@testable import TurboSparkApp

/// Regression tests for the low-severity tool and theme fixes (cron, hex
/// colors, TodoWrite, entities, Exa errors, custom tools, worktrees, MCP
/// discovery, hook input, snapshot backups, read gates).
final class LowRoundToolFixesTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("lowround-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private func utcDate(_ y: Int, _ m: Int, _ d: Int, _ h: Int, _ min: Int) -> (Date, Calendar) {
        var calendar = Calendar(identifier: .gregorian)
        calendar.timeZone = TimeZone(identifier: "UTC")!
        let date = calendar.date(from: DateComponents(
            year: y, month: m, day: d, hour: h, minute: min))!
        return (date, calendar)
    }

    // MARK: - Cron

    func testStepDayOfMonthIsNotTreatedAsRestricted() throws {
        let schedule = try XCTUnwrap(CronSchedule("0 9 */2 * 1"))
        // 2026-10-05 is an odd-day Monday, 2026-10-12 an even-day Monday.
        let (odd, calendar) = utcDate(2026, 10, 5, 9, 0)
        let (even, _) = utcDate(2026, 10, 12, 9, 0)
        XCTAssertTrue(schedule.matches(odd, calendar: calendar))
        XCTAssertFalse(
            schedule.matches(even, calendar: calendar),
            "'*/2' day-of-month is unrestricted for the OR rule, so it ANDs with Monday")
    }

    func testDescribeDoesNotMislabelRestrictedSchedules() throws {
        XCTAssertEqual(try XCTUnwrap(CronSchedule("*/5 * * * *")).describe(), "Every 5 minutes")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("*/5 * * 6 *")).describe(), "Custom schedule")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("*/5 * 1 * *")).describe(), "Custom schedule")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("30 * 15 * *")).describe(), "Custom schedule")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("0 9 * * 1")).describe(), "Weekly on Monday at 09:00")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("0 9 * * 1-5")).describe(), "Weekdays at 09:00")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("0 9 * * *")).describe(), "Daily at 09:00")
        XCTAssertEqual(try XCTUnwrap(CronSchedule("0 9 */2 * 1")).describe(), "Custom schedule")
    }

    func testImpossibleScheduleHasNoNextFireAndIsRefusedAtCreate() throws {
        let schedule = try XCTUnwrap(CronSchedule("0 9 31 2 *"))
        XCTAssertNil(schedule.nextFire(after: Date()))

        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let scheduler = CronScheduler(directory: dir)
        XCTAssertThrowsError(
            try scheduler.create(cron: "0 9 31 2 *", prompt: "x", recurring: true, chatID: UUID()))
        XCTAssertTrue(scheduler.list().isEmpty)
    }

    func testSkippingSearchStillFindsOrdinaryFires() throws {
        let (start, calendar) = utcDate(2026, 10, 7, 10, 30)
        // Next first-of-month 09:00 is 2026-11-01.
        let monthly = try XCTUnwrap(CronSchedule("0 9 1 * *"))
        let fire = try XCTUnwrap(monthly.nextFire(after: start, calendar: calendar))
        XCTAssertEqual(fire, utcDate(2026, 11, 1, 9, 0).0)
        // Every minute is still the next minute.
        let every = try XCTUnwrap(CronSchedule("* * * * *"))
        XCTAssertEqual(
            try XCTUnwrap(every.nextFire(after: start, calendar: calendar)),
            utcDate(2026, 10, 7, 10, 31).0)
        // A February 29 schedule has no fire in the one-year window from a
        // non-leap start.
        let leap = try XCTUnwrap(CronSchedule("0 0 29 2 *"))
        XCTAssertNil(leap.nextFire(after: start, calendar: calendar))
    }

    func testConcurrentCreatesAllPersist() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let scheduler = CronScheduler(directory: dir)
        let chat = UUID()
        DispatchQueue.concurrentPerform(iterations: 24) { index in
            _ = try? scheduler.create(
                cron: "*/5 * * * *", prompt: "job \(index)", recurring: true, chatID: chat)
        }
        XCTAssertEqual(scheduler.list().count, 24)
        // The last write on disk must be the newest snapshot, not a stale one.
        let reloaded = CronScheduler(directory: dir)
        XCTAssertEqual(reloaded.list().count, 24)
    }

    func testScheduleWakeupStopAcceptsFlattenedBoolean() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let scheduler = CronScheduler(directory: dir)
        let chat = UUID()
        for flag in ["1", "true", "TRUE"] {
            let text = try CronScheduler.executeScheduleWakeup(
                arguments: ["stop": flag, "delaySeconds": "600", "prompt": "x"],
                chatID: chat, scheduler: scheduler)
            XCTAssertEqual(text, "No active wakeup loop to stop.")
        }
        XCTAssertTrue(scheduler.list().isEmpty, "stop must not schedule a wakeup")
    }

    // MARK: - Theme

    func testHexColorRejectsPartiallyValidStrings() {
        XCTAssertNotNil(Color(hex: "#1D6FE0"))
        XCTAssertNotNil(Color(hex: "1D6FE0FF"))
        XCTAssertNil(Color(hex: "#1D6FEG"), "a valid hex prefix followed by junk is not a color")
        XCTAssertNil(Color(hex: "0x1D6FE0"), "only a leading '#' is stripped")
        XCTAssertNil(Color(hex: "#12345"))
        XCTAssertNil(Color(hex: ""))
    }

    // MARK: - TodoWrite

    func testTodoWriteRefusesUnparseablePayloadInsteadOfClearing() {
        var updates = 0
        TodoWriteExecutor.onTodosUpdated = { _, _ in updates += 1 }
        defer { TodoWriteExecutor.onTodosUpdated = nil }

        XCTAssertThrowsError(try TodoWriteExecutor.execute(arguments: [:]))
        XCTAssertThrowsError(try TodoWriteExecutor.execute(arguments: ["todos": "{\"todos\": [{\"title\": \"x\"}"]))
        XCTAssertThrowsError(try TodoWriteExecutor.execute(arguments: ["todos": "[{\"title\": \"x\"}]"]))
        XCTAssertEqual(updates, 0, "a failed parse must not reach the persisted checklist")

        XCTAssertNoThrow(try TodoWriteExecutor.execute(arguments: ["todos": "[]"]))
        XCTAssertEqual(updates, 1, "an explicit empty list still clears")
    }

    func testTodoIdsAreStableAndUnique() throws {
        let json = """
            [{"id":"1","content":"a"},{"id":"1","content":"b"},{"content":"c"},{"content":"d"}]
            """
        let first = try TodoWriteExecutor.parseTodos(from: ["todos": json])
        let second = try TodoWriteExecutor.parseTodos(from: ["todos": json])
        XCTAssertEqual(Set(first.map(\.id)).count, 4)
        XCTAssertEqual(first.map(\.id), second.map(\.id), "ids must not change between updates")
    }

    // MARK: - Web

    func testEntityDecodingIsSinglePass() {
        XCTAssertEqual(WebFetchExecutor.decodeHTMLEntities("&amp;lt;div&amp;gt;"), "&lt;div&gt;")
        XCTAssertEqual(WebFetchExecutor.decodeHTMLEntities("&lt;div&gt; &amp; more"), "<div> & more")
        XCTAssertEqual(WebFetchExecutor.decodeHTMLEntities("&amp;#39;"), "&#39;")
        XCTAssertEqual(WebFetchExecutor.decodeHTMLEntities("it&#39;s"), "it's")
    }

    func testExaToolErrorsAreDetected() {
        let rpcError = #"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"rate limit"}}"#
        XCTAssertEqual(WebSearchExecutor.exaErrorMessage(in: rpcError), "rate limit")
        let toolError = #"{"result":{"isError":true,"content":[{"type":"text","text":"quota exceeded"}]}}"#
        XCTAssertEqual(WebSearchExecutor.exaErrorMessage(in: toolError), "quota exceeded")
        let sse = "event: message\ndata: " + toolError
        XCTAssertEqual(WebSearchExecutor.exaErrorMessage(in: sse), "quota exceeded")
        let ok = #"{"result":{"content":[{"type":"text","text":"Title: A\nURL: https://a.test"}]}}"#
        XCTAssertNil(WebSearchExecutor.exaErrorMessage(in: ok))
    }

    // MARK: - Custom tools

    private func declaredParams(_ names: [String]) -> JSONSchema {
        JSONSchema.object(
            properties: Dictionary(uniqueKeysWithValues: names.map { ($0, JSONSchemaProperty(type: "string")) }))
    }

    func testUndeclaredArgumentCannotHijackEnvironmentReference() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let tool = CustomToolDefinition(
            name: "show_home", toolDescription: "d",
            execution: CustomToolExecution(type: .command, command: "echo $HOME"))
        let output = try await CustomToolExecutor.execute(
            tool: tool, arguments: ["home": "/attacker/chosen"], projectRootURL: dir)
        XCTAssertFalse(output.contains("/attacker/chosen"), output)
    }

    func testDeclaredArgumentStillSubstitutes() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let tool = CustomToolDefinition(
            name: "say", toolDescription: "d", parameters: declaredParams(["msg"]),
            execution: CustomToolExecution(type: .command, command: "echo {{msg}}"))
        let output = try await CustomToolExecutor.execute(
            tool: tool, arguments: ["msg": "hello"], projectRootURL: dir)
        XCTAssertTrue(output.contains("hello"), output)
    }

    func testCustomToolNonZeroExitIsAnError() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let tool = CustomToolDefinition(
            name: "deploy", toolDescription: "d",
            execution: CustomToolExecution(
                type: .command, command: "echo 'error: auth failed' >&2; exit 3"))
        do {
            _ = try await CustomToolExecutor.execute(tool: tool, arguments: [:], projectRootURL: dir)
            XCTFail("a non-zero exit must throw")
        } catch {
            let text = error.localizedDescription
            XCTAssertTrue(text.contains("status 3"), text)
            XCTAssertTrue(text.contains("auth failed"), text)
        }
    }

    func testCustomToolTimeoutIsAnError() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let tool = CustomToolDefinition(
            name: "slow", toolDescription: "d",
            execution: CustomToolExecution(
                type: .command, command: "echo partial; sleep 20", timeoutSeconds: 1))
        do {
            _ = try await CustomToolExecutor.execute(tool: tool, arguments: [:], projectRootURL: dir)
            XCTFail("a timeout must throw")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("timed out"), error.localizedDescription)
        }
    }

    // MARK: - Planning / worktree

    func testAskUserQuestionMultiSelectAcceptsFlattenedBoolean() throws {
        for flag in ["1", "true"] {
            let items = try AskUserQuestionExecutor.parseQuestions(
                from: ["question": "Pick", "multiSelect": flag])
            XCTAssertEqual(items.first?.multiSelect, true, flag)
        }
        let off = try AskUserQuestionExecutor.parseQuestions(
            from: ["question": "Pick", "multiSelect": "0"])
        XCTAssertEqual(off.first?.multiSelect, false)
    }

    private func runGit(_ args: [String], in dir: URL) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = args
        process.currentDirectoryURL = dir
        process.standardOutput = FileHandle.nullDevice
        process.standardError = FileHandle.nullDevice
        try process.run()
        process.waitUntilExit()
        XCTAssertEqual(process.terminationStatus, 0, "git \(args)")
    }

    func testEnterWorktreeOnExistingDirectoryFails() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        try runGit(["init", "-q"], in: dir)
        try runGit(
            ["-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "--allow-empty", "-m", "i"],
            in: dir)
        // Git accepts an EMPTY existing directory, so make it a real folder.
        try FileManager.default.createDirectory(
            at: dir.appendingPathComponent("src"), withIntermediateDirectories: true)
        try "x".write(
            to: dir.appendingPathComponent("src/keep.txt"), atomically: true, encoding: .utf8)
        do {
            let text = try await WorktreeExecutor.enter(
                arguments: ["name": "feat-x", "path": "src"], rootURL: dir)
            XCTFail("an existing non-worktree directory must not report success: \(text)")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Failed to create git worktree"))
        }
    }

    // MARK: - Hooks / MCP

    func testHookUpdatedInputContainersAreJSON() throws {
        let stdout = """
            {"hookSpecificOutput":{"hookEventName":"PreToolUse","updatedInput":{
              "edits":[{"file_path":"a","old_string":"x","new_string":"y"}],
              "path":"a.txt","flag":true}}}
            """
        let outcome = AppHookResponseParser.parseHookOutput(
            stdout: stdout, stderr: "", exitCode: 0, event: .preToolUse)
        guard case .structured(let response) = outcome,
            let input = response.hookSpecificOutput?.updatedInput
        else { return XCTFail("expected structured outcome") }
        XCTAssertEqual(input["path"], "a.txt")
        let edits = try XCTUnwrap(input["edits"])
        let parsed = try JSONSerialization.jsonObject(with: Data(edits.utf8)) as? [[String: String]]
        XCTAssertEqual(parsed?.first?["old_string"], "x")
    }

    func testMcpRpcErrorsAreSurfaced() {
        XCTAssertThrowsError(
            try McpClientEngine.throwIfRpcError(
                ["error": ["message": "GITHUB_TOKEN not set"]], serverName: "gh", step: "initialize")
        ) { error in
            XCTAssertTrue(error.localizedDescription.contains("GITHUB_TOKEN not set"))
            XCTAssertTrue(error.localizedDescription.contains("gh"))
        }
        XCTAssertNoThrow(
            try McpClientEngine.throwIfRpcError(["result": [:]], serverName: "gh", step: "tools/list"))
    }

    @MainActor
    func testAgentAutoAsksForMcpServerThatHasNotAdvertisedYet() {
        McpToolCatalogCache.shared.removeAll()
        defer { McpToolCatalogCache.shared.removeAll() }
        var project = AppProject(
            name: "p", permissions: AppProjectPermissions(mode: .agentAuto, mcp: .allow))
        project.mcpServers = [
            McpServerConfig(name: "srv", transport: .stdio(command: "/bin/echo"), isEnabled: true)
        ]
        let decision = AppToolPermissionEngine.evaluate(
            call: AppToolCall(
                name: "mcp__srv__apply", arguments: [:], rawInvocation: "mcp__srv__apply", category: .mcp),
            project: project, globalServers: [])
        guard case .ask = decision else {
            return XCTFail("agentAuto must ask while annotations are unknown, got \(decision)")
        }
    }

    // MARK: - Files

    func testSnapshotBackupsAreBoundedAndClearedByReset() async {
        let store = FileSnapshotStore()
        for index in 0..<(FileSnapshotStore.maximumTrackedFiles + 20) {
            await store.recordBackup(url: URL(fileURLWithPath: "/tmp/f\(index).txt"), content: "c")
        }
        let held = await store.backupCount
        XCTAssertEqual(held, FileSnapshotStore.maximumTrackedFiles)
        await store.reset()
        let afterReset = await store.backupCount
        XCTAssertEqual(afterReset, 0)
    }

    func testSnipRefusesOversizeFile() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let big = dir.appendingPathComponent("big.log")
        FileManager.default.createFile(atPath: big.path, contents: nil)
        let handle = try FileHandle(forWritingTo: big)
        try handle.truncate(atOffset: UInt64(AppFileReadLimits.maximumBytes + 1))
        try handle.close()
        XCTAssertThrowsError(try SnipExecutor.execute(arguments: ["path": "big.log"], rootURL: dir)) {
            XCTAssertTrue($0.localizedDescription.contains("limit"))
        }
    }

    func testInsertAndPatternReplaceRequireAPriorRead() async throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let file = dir.appendingPathComponent("config.yaml")
        try "a: 1\nb: 2\n".write(to: file, atomically: true, encoding: .utf8)
        await FileSnapshotStore.shared.removeSnapshot(url: file)

        do {
            _ = try await AppToolRegistry.editFile(
                relPath: "config.yaml", newString: "", rootURL: dir,
                command: "pattern_replace", regexPattern: "[\\s\\S]*")
            XCTFail("pattern_replace on an unread file must be refused")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("not been read"))
        }
        XCTAssertEqual(try String(contentsOf: file, encoding: .utf8), "a: 1\nb: 2\n")

        do {
            _ = try await AppToolRegistry.editFile(
                relPath: "config.yaml", newString: "z", rootURL: dir,
                command: "insert", insertLine: "1", position: "after")
            XCTFail("insert on an unread file must be refused")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("not been read"))
        }
    }

    func testOversizeReadMessageDoesNotPointAtLineBounds() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let big = dir.appendingPathComponent("big.log")
        FileManager.default.createFile(atPath: big.path, contents: nil)
        let handle = try FileHandle(forWritingTo: big)
        try handle.truncate(atOffset: UInt64(AppFileReadLimits.maximumBytes + 1))
        try handle.close()
        XCTAssertThrowsError(try AppFileReadLimits.readTextFile(at: big, describing: "big.log")) {
            let text = $0.localizedDescription
            XCTAssertFalse(text.contains("Use start_line"), text)
            XCTAssertTrue(text.contains("sed -n"), text)
        }
    }

    // MARK: - Terminal

    func testShellTimeoutHasOneSecondFloor() {
        XCTAssertEqual(ShellCommandRunner.clampedTimeoutSeconds(timeoutMs: 120), 1)
        XCTAssertEqual(ShellCommandRunner.clampedTimeoutSeconds(timeoutMs: 5_000), 5)
    }
}
