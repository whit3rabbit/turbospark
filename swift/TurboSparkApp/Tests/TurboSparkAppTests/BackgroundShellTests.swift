import XCTest
@testable import TurboSparkApp

/// The background shell lifecycle, end to end through `execute`: Bash with
/// `run_in_background` returns an id immediately, BashOutput retrieves the
/// output (optionally waiting for completion), KillShell stops a running
/// shell, and ids are scoped to the conversation that started them.
///
/// These run the REAL `/bin/zsh` children, so every command here is
/// deliberately short or a `sleep` that the teardown kills.
final class BackgroundShellTests: XCTestCase {
    override func setUp() {
        super.setUp()
        BackgroundShellManager.shared.resetForTests()
        ShellCwdTracker.shared.resetForTests()
    }

    override func tearDown() {
        BackgroundShellManager.shared.resetForTests()
        ShellCwdTracker.shared.resetForTests()
        super.tearDown()
    }

    /// A project rooted at a real temporary directory (same shape as
    /// `RunCommandToolTests.temporaryProject`).
    private func temporaryProject() -> AppProject {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return AppProject(name: "Test", rootDirectoryPath: root.path)
    }

    private func launchBackground(_ command: String, project: AppProject, chatID: UUID? = nil) async -> AppToolResult {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": command, "run_in_background": "true"],
            category: .terminal
        )
        return await AppToolRegistry.execute(call: call, in: project, chatID: chatID)
    }

    func testBackgroundLaunchReturnsImmediatelyWithAnID() async throws {
        let project = temporaryProject()
        let start = Date()
        let result = await launchBackground("echo started", project: project)
        let elapsed = Date().timeIntervalSince(start)

        XCTAssertFalse(result.isError, result.output)
        XCTAssertLessThan(elapsed, 10, "backgrounding must not wait for the command to finish")
        XCTAssertTrue(result.output.contains("running in background with ID"), result.output)
        XCTAssertTrue(result.output.contains("BashOutput"), "the model needs the retrieval tool named")
        XCTAssertTrue(result.output.contains("KillShell"), "the model needs the kill tool named")
    }

    func testBashOutputWaitsForCompletionAndReturnsTheOutput() async throws {
        let project = temporaryProject()
        _ = await launchBackground("echo bg-marker-9137", project: project)
        let id = try XCTUnwrap(BackgroundShellManager.shared.knownShellIDs(chatID: nil).first)

        let retrieve = AppToolCall(
            name: "bash_output",
            arguments: ["task_id": id, "wait_seconds": "30"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: retrieve, in: project, chatID: nil)
        XCTAssertFalse(result.isError, result.output)
        XCTAssertTrue(result.output.contains("completed (exit 0)"), result.output)
        XCTAssertTrue(result.output.contains("bg-marker-9137"), result.output)
    }

    func testKillShellStopsARunningBackgroundCommand() async throws {
        let project = temporaryProject()
        _ = await launchBackground("sleep 30", project: project)
        let id = try XCTUnwrap(BackgroundShellManager.shared.knownShellIDs(chatID: nil).first)

        let poll = AppToolCall(
            name: "bash_output",
            arguments: ["task_id": id, "wait_seconds": "0"],
            category: .terminal
        )
        let pollResult = await AppToolRegistry.execute(call: poll, in: project, chatID: nil)
        XCTAssertTrue(pollResult.output.contains("still running"), pollResult.output)

        let kill = AppToolCall(
            name: "kill_shell",
            arguments: ["task_id": id],
            category: .terminal
        )
        let killResult = await AppToolRegistry.execute(call: kill, in: project, chatID: nil)
        XCTAssertFalse(killResult.isError, killResult.output)
        XCTAssertTrue(killResult.output.contains("terminated"), killResult.output)

        let after = AppToolCall(
            name: "bash_output",
            arguments: ["task_id": id, "wait_seconds": "5"],
            category: .terminal
        )
        let afterResult = await AppToolRegistry.execute(call: after, in: project, chatID: nil)
        XCTAssertTrue(afterResult.output.contains("killed"), afterResult.output)
    }

    func testKillOnAnAlreadyFinishedShellReportsItDidNothing() async throws {
        let project = temporaryProject()
        _ = await launchBackground("true", project: project)
        let id = try XCTUnwrap(BackgroundShellManager.shared.knownShellIDs(chatID: nil).first)
        // Let the shell finish before killing it.
        try await Task.sleep(nanoseconds: 500_000_000)

        let kill = AppToolCall(
            name: "kill_shell",
            arguments: ["task_id": id],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: kill, in: project, chatID: nil)
        XCTAssertFalse(result.isError, result.output)
        XCTAssertTrue(result.output.contains("already finished"), result.output)
    }

    func testUnknownShellIDIsAnErrorNamingTheKnownIds() async throws {
        let project = temporaryProject()
        _ = await launchBackground("echo present", project: project)
        let knownID = try XCTUnwrap(BackgroundShellManager.shared.knownShellIDs(chatID: nil).first)

        let retrieve = AppToolCall(
            name: "bash_output",
            arguments: ["task_id": "bg_404"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: retrieve, in: project, chatID: nil)
        XCTAssertTrue(result.isError, result.output)
        XCTAssertTrue(result.output.contains("Unknown background shell id"), result.output)
        XCTAssertTrue(
            result.output.contains(knownID),
            "the error should name an id that DOES resolve, so the model can self-correct")
    }

    func testShellIDsAreScopedToTheirConversation() async throws {
        let project = temporaryProject()
        let chatA = UUID()
        _ = await launchBackground("echo scoped", project: project, chatID: chatA)
        let id = try XCTUnwrap(BackgroundShellManager.shared.knownShellIDs(chatID: chatA).first)

        let retrieve = AppToolCall(
            name: "bash_output",
            arguments: ["task_id": id],
            category: .terminal
        )
        let fromB = await AppToolRegistry.execute(
            call: retrieve, in: project, chatID: UUID())
        XCTAssertTrue(fromB.isError, "another conversation's shell id must not resolve: \(fromB.output)")

        let fromA = await AppToolRegistry.execute(call: retrieve, in: project, chatID: chatA)
        XCTAssertFalse(fromA.isError, fromA.output)
    }

    func testTooManyRunningShellsIsRefused() throws {
        let project = temporaryProject()
        let rootURL = try XCTUnwrap(project.rootDirectoryURL)
        for _ in 0..<BackgroundShellManager.maxRunningShells {
            _ = try BackgroundShellManager.shared.launch(
                command: "sleep 30",
                startDirectory: rootURL,
                environment: ShellOutputFormatting.shellEnvironment(),
                chatID: nil,
                description: nil)
        }
        XCTAssertThrowsError(
            try BackgroundShellManager.shared.launch(
                command: "sleep 30",
                startDirectory: rootURL,
                environment: ShellOutputFormatting.shellEnvironment(),
                chatID: nil,
                description: nil),
            "the ceiling must bound how many processes this app hands out"
        ) { error in
            XCTAssertTrue("\(error)".contains("Too many background commands"))
        }
    }
}
