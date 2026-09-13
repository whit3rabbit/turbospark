import XCTest
@testable import TurboSparkApp

final class AutomationAndEnvironmentToolsTests: XCTestCase {
    private func makeProject() throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(name: "test-auto-env", rootDirectoryPath: dir.path)
        return (project, dir)
    }

    // MARK: - sleep / delay

    func testSleepSuccessAndAliases() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["sleep", "delay"] {
            let start = Date()
            let call = AppToolCall(
                name: alias,
                arguments: ["seconds": "0.05"],
                category: .automation
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            let elapsed = Date().timeIntervalSince(start)

            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertGreaterThanOrEqual(elapsed, 0.04)
        }
    }

    func testSleepClampNegativeAndExcessive() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let negativeCall = AppToolCall(
            name: "sleep",
            arguments: ["seconds": "-5"],
            category: .automation
        )
        let result = await AppToolRegistry.execute(call: negativeCall, in: project)
        XCTAssertFalse(result.isError, "Negative sleep should clamp to 0 without error")
    }

    // MARK: - push_notification / notify

    func testPushNotification() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        final class DispatchedBox: @unchecked Sendable {
            var title: String?
            var message: String?
        }
        let box = DispatchedBox()

        PushNotificationExecutor.onNotificationPushed = { title, msg in
            box.title = title
            box.message = msg
        }
        defer { PushNotificationExecutor.onNotificationPushed = nil }

        for alias in ["push_notification", "pushnotification", "notify"] {
            let call = AppToolCall(
                name: alias,
                arguments: [
                    "title": "Build Finished",
                    "message": "All assets packaged cleanly"
                ],
                category: .automation
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertEqual(box.title, "Build Finished")
            XCTAssertEqual(box.message, "All assets packaged cleanly")
        }
    }

    // MARK: - config / config_tool

    func testConfigToolInspection() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["config", "config_tool"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["action": "get"],
                category: .automation
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertTrue(result.output.contains("interaction.mode") || result.output.contains("TurboSpark Configuration"))
        }
    }

    // MARK: - ctx_inspect / ctxinspect

    func testCtxInspectTelemetry() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["ctx_inspect", "ctxinspect"] {
            let call = AppToolCall(
                name: alias,
                arguments: [:],
                category: .automation
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertTrue(result.output.contains("Active Project"))
            XCTAssertTrue(result.output.contains("Registered Tools") || result.output.contains("Context Inspection"))
        }
    }

    // MARK: - enter_worktree / exit_worktree

    func testWorktreeToolNonGitDirHandledGracefully() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["enter_worktree", "enterworktree"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["branch": "feature-test"],
                category: .terminal
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            // In a non-git dir, git command exits non-zero and reports error gracefully
            XCTAssertTrue(result.isError, "Worktree in non-git directory should report error cleanly")
        }

        for alias in ["exit_worktree", "exitworktree"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["action": "remove", "path": "feature-test"],
                category: .terminal
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertTrue(result.isError, "Exit worktree in non-git directory should report error cleanly")
        }
    }

    func testWorktreeArgumentsCannotInjectShellCommands() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let marker = "WORKTREE_INJECTION_PROOF"
        let call = AppToolCall(
            name: "enter_worktree",
            arguments: ["branch": "feature-test", "path": "worktree\"; touch \(marker); #"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: project)

        XCTAssertTrue(result.isError, "The non-git fixture should reject worktree creation")
        XCTAssertFalse(FileManager.default.fileExists(atPath: dir.appendingPathComponent(marker).path))
    }

    func testWorktreeEnterRejectsInvalidBranchAndEscapingPath() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let invalidBranch = AppToolCall(
            name: "enter_worktree",
            arguments: ["branch": "bad..branch"],
            category: .terminal
        )
        let invalidResult = await AppToolRegistry.execute(call: invalidBranch, in: project)
        XCTAssertTrue(invalidResult.isError)
        XCTAssertTrue(invalidResult.output.contains("Invalid git branch name"))

        let escapingPath = AppToolCall(
            name: "enter_worktree",
            arguments: ["branch": "feature-test", "path": "../outside"],
            category: .terminal
        )
        let escapingResult = await AppToolRegistry.execute(call: escapingPath, in: project)
        XCTAssertTrue(escapingResult.isError)
        XCTAssertTrue(escapingResult.output.contains("Path escapes the project root"))
    }
}
