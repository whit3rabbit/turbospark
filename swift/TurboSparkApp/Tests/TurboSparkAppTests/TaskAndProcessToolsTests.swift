import XCTest
@testable import TurboSparkApp

final class TaskAndProcessToolsTests: XCTestCase {
    private func makeProject() throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(name: "test-tasks", rootDirectoryPath: dir.path)
        return (project, dir)
    }

    // MARK: - Task Manager Lifecycle

    func testTaskManagerFullFlow() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        // 1. task_create
        let createCall = AppToolCall(
            name: "task_create",
            arguments: [
                "title": "Compile Assets",
                "command": "echo 'compiling...'",
                "description": "Compiles static web assets"
            ],
            category: .automation
        )
        let createResult = await AppToolRegistry.execute(call: createCall, in: project)
        XCTAssertFalse(createResult.isError, "task_create should succeed: \(createResult.output)")
        XCTAssertTrue(createResult.output.contains("Compile Assets"))

        // Extract task id from output
        let taskId: String
        if let start = createResult.output.range(of: "ID '"),
           let end = createResult.output[start.upperBound...].range(of: "'") {
            taskId = String(createResult.output[start.upperBound..<end.lowerBound])
        } else {
            XCTFail("Failed to parse task id from create output: \(createResult.output)")
            return
        }

        // 2. task_list
        for alias in ["task_list", "tasklist"] {
            let listCall = AppToolCall(name: alias, arguments: [:], category: .automation)
            let listResult = await AppToolRegistry.execute(call: listCall, in: project)
            XCTAssertFalse(listResult.isError, "task_list should succeed: \(listResult.output)")
            XCTAssertTrue(listResult.output.contains(taskId))
        }

        // 3. task_get
        for alias in ["task_get", "taskget"] {
            let getCall = AppToolCall(
                name: alias,
                arguments: ["id": taskId],
                category: .automation
            )
            let getResult = await AppToolRegistry.execute(call: getCall, in: project)
            XCTAssertFalse(getResult.isError, "task_get should succeed: \(getResult.output)")
            XCTAssertTrue(getResult.output.contains(taskId))
            XCTAssertTrue(getResult.output.contains("Compile Assets"))
        }

        // 4. task_update
        for alias in ["task_update", "taskupdate"] {
            let updateCall = AppToolCall(
                name: alias,
                arguments: [
                    "id": taskId,
                    "status": "running",
                    "progress": "50%"
                ],
                category: .automation
            )
            let updateResult = await AppToolRegistry.execute(call: updateCall, in: project)
            XCTAssertFalse(updateResult.isError, "task_update should succeed: \(updateResult.output)")
        }

        // 5. task_output
        for alias in ["task_output", "taskoutput"] {
            let outCall = AppToolCall(
                name: alias,
                arguments: ["id": taskId],
                category: .automation
            )
            let outResult = await AppToolRegistry.execute(call: outCall, in: project)
            XCTAssertFalse(outResult.isError, "task_output should succeed: \(outResult.output)")
        }

        // 6. task_stop
        for alias in ["task_stop", "taskstop", "task_cancel"] {
            let stopCall = AppToolCall(
                name: alias,
                arguments: ["id": taskId],
                category: .automation
            )
            let stopResult = await AppToolRegistry.execute(call: stopCall, in: project)
            XCTAssertFalse(stopResult.isError, "task_stop should succeed: \(stopResult.output)")
        }
    }

    func testTaskManagerInvalidIDs() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let getCall = AppToolCall(
            name: "task_get",
            arguments: ["id": "nonexistent-id"],
            category: .automation
        )
        let getResult = await AppToolRegistry.execute(call: getCall, in: project)
        XCTAssertTrue(getResult.isError, "Getting non-existent task must fail")
    }

    // MARK: - run_command / bash / shell / exec / terminal

    func testRunCommandSuccessAndAliases() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["run_command", "bash", "shell", "exec", "terminal"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["command": "echo 'hello from \(alias)'"],
                category: .terminal
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertTrue(result.output.contains("hello from \(alias)"))
        }
    }

    func testRunCommandNonZeroExitCode() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "exit 1"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError, "Non-zero exit code should be reported as error")
    }
}
