import XCTest
@testable import TurboSparkApp

/// The user-facing kill surfaces: the process-tree sweep under every
/// `terminateAndReap` caller (a killed shell must not leave its `sleep 30 &`
/// children running), the strip's manager accessors, and the AppModel kill
/// paths (`killBackgroundShell`, `stopBackgroundWork(forDeletedChat:)`,
/// `stopAll`, and the shutdown sweep). Real `/bin/zsh` children throughout;
/// every command is short or something the teardown kills.
@MainActor
final class KillSurfacesTests: XCTestCase {
    var appModel: AppModel!

    override func setUp() {
        super.setUp()
        BackgroundShellManager.shared.resetForTests()
        ShellCwdTracker.shared.resetForTests()
        appModel = AppModel()
    }

    override func tearDown() {
        BackgroundShellManager.shared.resetForTests()
        ShellCwdTracker.shared.resetForTests()
        appModel = nil
        super.tearDown()
    }

    // MARK: - helpers

    private func temporaryProject() -> AppProject {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return AppProject(name: "Test", rootDirectoryPath: root.path)
    }

    private func launchBackground(_ command: String, chatID: UUID?) async -> AppToolResult {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": command, "run_in_background": "true"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: temporaryProject(), chatID: chatID)
        XCTAssertFalse(result.isError, "background launch of \(command) failed: \(result.output)")
        return result
    }

    /// Waits until the grandchild's pid file exists (the shell writes it a
    /// few milliseconds after spawn -- killing before that leaves nothing
    /// for the sweep's assertion to name).
    private func waitForPidFile(_ pidFile: String) async throws {
        for _ in 0..<100 {
            if FileManager.default.fileExists(atPath: pidFile) { return }
            try await Task.sleep(nanoseconds: 50_000_000)
        }
        XCTFail("the grandchild never wrote its pid to \(pidFile)")
    }

    /// A shell command that spawns a grandchild, writes the grandchild's pid
    /// to `pidFile`, and stays alive waiting on it. Everything the kill
    /// tests below assert hangs on this pid being DEAD afterwards -- a
    /// plain `terminate()` of the shell cannot reach it (zsh does not
    /// forward signals), so a survivor means the sweep is gone.
    private func grandchildCommand(writingTo pidFile: String) -> String {
        let escape = "import os; os.setsid(); os.execv('/bin/sleep', ['sleep', '30'])"
        return "python3 -c \"\(escape)\" & echo $! > '\(pidFile)'; wait"
    }

    private func readGrandchildPid(_ pidFile: String) throws -> String {
        try String(contentsOfFile: pidFile, encoding: .utf8)
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// `kill -0` exits 0 only while the pid lives. A short settle first:
    /// the sweep runs synchronously inside the kill, but the probe must not
    /// race the process table's own teardown.
    private func assertGrandchild(_ pidFile: String, alive expected: Bool) async throws {
        let pid = try readGrandchildPid(pidFile)
        try await Task.sleep(nanoseconds: 400_000_000)
        let check = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", "kill -0 \(pid) 2>/dev/null && echo ALIVE || echo DEAD"],
            timeoutSeconds: 5
        )
        let verdict = check.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertEqual(
            verdict, expected ? "ALIVE" : "DEAD",
            "grandchild \(pid) (from \(pidFile)) is \(verdict) where \(expected ? "ALIVE" : "DEAD") was required")
    }

    // MARK: - the tree sweep

    func testTimeoutKillReachesTheGrandchild() async throws {
        let pidFile = NSTemporaryDirectory() + "treekill-timeout-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: pidFile) }
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/zsh"),
            arguments: ["-c", grandchildCommand(writingTo: pidFile)],
            timeoutSeconds: 0.8
        )
        XCTAssertTrue(result.timedOut)
        try await assertGrandchild(pidFile, alive: false)
    }

    func testKillingABackgroundShellReachesItsGrandchild() async throws {
        let pidFile = NSTemporaryDirectory() + "treekill-bg-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: pidFile) }
        let chatID = UUID()
        _ = await launchBackground(grandchildCommand(writingTo: pidFile), chatID: chatID)
        try await waitForPidFile(pidFile)
        let record = try XCTUnwrap(
            BackgroundShellManager.shared.runningRecords(chatID: chatID).first)
        XCTAssertEqual(BackgroundShellManager.shared.kill(record), true)
        try await assertGrandchild(pidFile, alive: false)
    }

    func testShutdownSweepReachesTheGrandchild() async throws {
        let pidFile = NSTemporaryDirectory() + "treekill-quit-\(UUID().uuidString)"
        defer { try? FileManager.default.removeItem(atPath: pidFile) }
        _ = await launchBackground(grandchildCommand(writingTo: pidFile), chatID: nil)
        try await waitForPidFile(pidFile)
        BackgroundShellManager.shared.killAllForShutdown()
        try await assertGrandchild(pidFile, alive: false)
    }

    // MARK: - the strip's manager accessors

    func testRunningRecordsFilterByChatAndKillAllSweepsEverything() async throws {
        let chatA = UUID()
        let chatB = UUID()
        _ = await launchBackground("sleep 20", chatID: chatA)
        _ = await launchBackground("sleep 21", chatID: chatA)
        _ = await launchBackground("sleep 22", chatID: chatB)

        XCTAssertEqual(BackgroundShellManager.shared.runningRecords().count, 3)
        XCTAssertEqual(BackgroundShellManager.shared.runningRecords(chatID: chatA).count, 2)
        XCTAssertEqual(BackgroundShellManager.shared.runningRecords(chatID: chatB).count, 1)
        XCTAssertEqual(
            BackgroundShellManager.shared.runningRecords(chatID: UUID()).count, 0,
            "an unrelated chat's shells must not resolve")

        XCTAssertEqual(BackgroundShellManager.shared.killAll(), 3)
        XCTAssertTrue(
            BackgroundShellManager.shared.runningRecords().isEmpty,
            "killAll must empty the running set")
        // The finished records stay resolvable PER CHAT (a nil-chat lookup
        // is scoped, not global) and read .killed, not .failed with a
        // signal exit -- BashOutput's killed summary depends on that.
        let idsA = BackgroundShellManager.shared.knownShellIDs(chatID: chatA)
        let idsB = BackgroundShellManager.shared.knownShellIDs(chatID: chatB)
        XCTAssertEqual(idsA.count + idsB.count, 3)
        for id in idsA {
            XCTAssertEqual(
                BackgroundShellManager.shared.record(id: id, chatID: chatA)?.state,
                .killed, "\(id) must read killed")
        }
        for id in idsB {
            XCTAssertEqual(
                BackgroundShellManager.shared.record(id: id, chatID: chatB)?.state,
                .killed, "\(id) must read killed")
        }
    }

    func testALaunchedShellIsPushedToThePublishedSummaries() async throws {
        _ = await launchBackground("sleep 20", chatID: nil)
        // The change hook is delivered via DispatchQueue.main.async; yield
        // the main actor so the block runs.
        try await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertEqual(appModel.backgroundShellSummaries.count, 1)
        let summary = try XCTUnwrap(appModel.backgroundShellSummaries.first)
        XCTAssertEqual(summary.commandHead, "sleep 20", "the head is the command's first line")

        let record = try XCTUnwrap(BackgroundShellManager.shared.runningRecords().first)
        BackgroundShellManager.shared.kill(record)
        try await Task.sleep(nanoseconds: 200_000_000)
        XCTAssertTrue(
            appModel.backgroundShellSummaries.isEmpty,
            "a killed shell must leave the strip")
    }

    // MARK: - AppModel.killBackgroundShell

    func testKillBackgroundShellAppliesTheStripVisibilityRule() async throws {
        let selected = appModel.selectedChatID
        let other = UUID()
        _ = await launchBackground("sleep 20", chatID: selected)
        _ = await launchBackground("sleep 21", chatID: other)

        XCTAssertTrue(
            appModel.killBackgroundShell(
                id: BackgroundShellManager.shared.runningRecords(chatID: selected).first!.id),
            "the selected chat's shell is visible, so it is killable")
        XCTAssertFalse(
            appModel.killBackgroundShell(
                id: BackgroundShellManager.shared.runningRecords(chatID: other).first!.id),
            "another chat's shell must not be killable from here")
        XCTAssertFalse(
            appModel.killBackgroundShell(id: "bg_99999"),
            "an unknown id is false, not a throw")
    }

    // MARK: - deleteChat and shutdown cleanup

    func testStopBackgroundWorkForDeletedChatEndsItsAgentAndShells() async throws {
        let chatID = UUID()
        let agentID = "bga_delete"
        appModel.backgroundAgentRuns[agentID] = SubagentRunState(
            id: agentID, mode: .background, chatID: chatID,
            agentName: "explore", displayName: "Explore")
        let agentTask = Task<SubagentRunResult, Never> {
            SubagentRunResult(
                agentName: "explore", status: "completed", finalResponse: "",
                totalTurns: 0, totalToolCalls: 0, durationSeconds: 0, runID: "run-x")
        }
        appModel.backgroundAgentTasks[agentID] = agentTask
        _ = await launchBackground("sleep 20", chatID: chatID)
        _ = await launchBackground("sleep 21", chatID: UUID())

        appModel.stopBackgroundWork(forDeletedChat: chatID)

        XCTAssertTrue(agentTask.isCancelled, "the chat's agent task must be cancelled")
        XCTAssertEqual(
            appModel.killedBackgroundAgentIDs.contains(agentID), true,
            "the run must read as killed, not cancelled-by-chance")
        XCTAssertTrue(
            BackgroundShellManager.shared.runningRecords(chatID: chatID).isEmpty,
            "the chat's shells must be gone")
        XCTAssertEqual(
            BackgroundShellManager.shared.runningRecords().count, 1,
            "another chat's shell must be untouched")
    }

    func testStopAllSweepsAgentsAndShells() async throws {
        let agentID = "bga_stopall"
        appModel.backgroundAgentRuns[agentID] = SubagentRunState(
            id: agentID, mode: .background, chatID: nil,
            agentName: "explore", displayName: "Explore")
        let agentTask = Task<SubagentRunResult, Never> {
            SubagentRunResult(
                agentName: "explore", status: "completed", finalResponse: "",
                totalTurns: 0, totalToolCalls: 0, durationSeconds: 0, runID: "run-y")
        }
        appModel.backgroundAgentTasks[agentID] = agentTask
        _ = await launchBackground("sleep 20", chatID: nil)

        appModel.stopAll()

        XCTAssertTrue(agentTask.isCancelled)
        XCTAssertTrue(
            BackgroundShellManager.shared.runningRecords().isEmpty,
            "Stop All must kill every running shell, any chat")
    }

    func testShutdownSweepCancelsAgentTasksAndKillsShells() {
        let agentID = "bga_shutdown"
        appModel.backgroundAgentRuns[agentID] = SubagentRunState(
            id: agentID, mode: .background, chatID: nil,
            agentName: "explore", displayName: "Explore")
        let agentTask = Task<SubagentRunResult, Never> {
            SubagentRunResult(
                agentName: "explore", status: "completed", finalResponse: "",
                totalTurns: 0, totalToolCalls: 0, durationSeconds: 0, runID: "run-z")
        }
        appModel.backgroundAgentTasks[agentID] = agentTask

        appModel.stopAllBackgroundWorkForShutdown()

        XCTAssertTrue(agentTask.isCancelled)
        XCTAssertTrue(
            appModel.backgroundAgentTasks.isEmpty,
            "the shutdown sweep owns the task table's teardown")
        XCTAssertTrue(
            BackgroundShellManager.shared.runningRecords().isEmpty,
            "no shell may outlive the process")
    }

    // MARK: - command head

    func testCommandHeadTakesTheFirstLineAndCaps() {
        XCTAssertEqual(AppModel.commandHead(of: "echo hi\necho more"), "echo hi")
        XCTAssertEqual(AppModel.commandHead(of: "   echo padded   "), "echo padded")
        let long = String(repeating: "x", count: 200)
        let head = AppModel.commandHead(of: long)
        XCTAssertEqual(head.count, 99, "96 chars plus an ellipsis")
        XCTAssertTrue(head.hasSuffix("..."))
    }
}
