import XCTest
@testable import TurboSparkApp

final class TaskOutputBlockDeletedTests: XCTestCase {
    func testDeletedStatusRemovesTask() throws {
        let chatID = UUID()
        let task = TaskManager.shared.createTask(subject: "doomed", description: "", chatID: chatID)
        let keep = TaskManager.shared.createTask(subject: "keeper", description: "", chatID: chatID)
        defer { _ = TaskManager.shared.updateTask(id: keep.id, status: "deleted", chatID: chatID) }

        let out = try TaskManager.executeUpdate(
            arguments: ["taskId": task.id, "status": "deleted"], chatID: chatID)
        XCTAssertTrue(out.contains(task.id))
        XCTAssertNil(TaskManager.shared.getTask(id: task.id))
        XCTAssertEqual(TaskManager.shared.listTasks(chatID: chatID).map(\.id), [keep.id])
        XCTAssertThrowsError(try TaskManager.executeGet(arguments: ["taskId": task.id]))
    }

    func testTaskOutputNonBlockingReturnsImmediately() async throws {
        let task = TaskManager.shared.createTask(subject: "nb", description: "")
        defer { _ = TaskManager.shared.updateTask(id: task.id, status: "deleted") }
        let start = Date()
        let out = try await TaskManager.executeOutput(
            arguments: ["task_id": task.id, "block": "false", "timeout": "5000"])
        XCTAssertLessThan(Date().timeIntervalSince(start), 1.0)
        XCTAssertTrue(out.contains("no recorded output"))
    }

    func testTaskOutputBlocksUntilCompleted() async throws {
        let task = TaskManager.shared.createTask(subject: "wait", description: "")
        defer { _ = TaskManager.shared.updateTask(id: task.id, status: "deleted") }
        let id = task.id
        Task {
            try? await Task.sleep(nanoseconds: 300_000_000)
            _ = TaskManager.shared.updateTask(id: id, status: "completed", output: "all done")
        }
        let out = try await TaskManager.executeOutput(
            arguments: ["task_id": id, "block": "true", "timeout": "5000"])
        XCTAssertEqual(out, "all done")
    }

    func testTaskOutputBlockTimesOut() async throws {
        let task = TaskManager.shared.createTask(subject: "slow", description: "")
        defer { _ = TaskManager.shared.updateTask(id: task.id, status: "deleted") }
        let start = Date()
        let out = try await TaskManager.executeOutput(
            arguments: ["task_id": task.id, "block": "true", "timeout": "300"])
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertGreaterThanOrEqual(elapsed, 0.25)
        XCTAssertLessThan(elapsed, 3.0)
        XCTAssertTrue(out.contains("Timed out"), out)
    }

    func testTaskOutputSchemaOnlyRequiresTaskID() {
        let required = TaskItemToolDefinitions.taskOutput.function.parameters.required
        XCTAssertEqual(required, ["task_id"])
    }
}
