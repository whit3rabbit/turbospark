import XCTest
@testable import TurboSparkApp

final class BatchToolTests: XCTestCase {
    func testBatchExecutionExecutesCallsConcurrently() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "alpha content".write(to: dir.appendingPathComponent("a.txt"), atomically: true, encoding: .utf8)
        try "beta content".write(to: dir.appendingPathComponent("b.txt"), atomically: true, encoding: .utf8)
        let project = AppProject(name: "batch_test", rootDirectoryPath: dir.path)

        let batchArgs = [
            "tool_calls": """
            [
                {"tool": "read_file", "parameters": {"path": "a.txt"}},
                {"tool": "read_file", "parameters": {"path": "b.txt"}}
            ]
            """
        ]

        let call = AppToolCall(name: "batch", arguments: batchArgs, category: .automation)
        let result = await AppToolRegistry.execute(call: call, in: project)

        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("alpha content"))
        XCTAssertTrue(result.output.contains("beta content"))
        XCTAssertTrue(result.output.contains("2 succeeded, 0 failed"))
    }

    func testBatchHandlesPartialFailureWithoutAbortingSiblings() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        try "good content".write(to: dir.appendingPathComponent("valid.txt"), atomically: true, encoding: .utf8)
        let project = AppProject(name: "batch_test", rootDirectoryPath: dir.path)

        let batchArgs = [
            "tool_calls": """
            [
                {"tool": "read_file", "parameters": {"path": "valid.txt"}},
                {"tool": "read_file", "parameters": {"path": "nonexistent.txt"}}
            ]
            """
        ]

        let call = AppToolCall(name: "batch", arguments: batchArgs, category: .automation)
        let result = await AppToolRegistry.execute(call: call, in: project)

        XCTAssertFalse(result.isError, "Batch wrapper itself succeeds while reporting individual failure")
        XCTAssertTrue(result.output.contains("1 succeeded, 1 failed"))
        XCTAssertTrue(result.output.contains("good content"))
        XCTAssertTrue(result.output.contains("FAILED"))
    }

    func testBatchRejectsRecursiveBatching() async throws {
        let batchArgs = [
            "tool_calls": """
            [
                {"tool": "batch", "parameters": {}}
            ]
            """
        ]

        let call = AppToolCall(name: "batch", arguments: batchArgs, category: .automation)
        let result = await AppToolRegistry.execute(call: call, in: nil)

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Recursive batch execution is not allowed"))
    }

    func testBatchEnforcesLimitOf25Calls() async throws {
        var items: [String] = []
        for i in 1...26 {
            items.append("{\"tool\": \"read_file\", \"parameters\": {\"path\": \"\(i).txt\"}}")
        }
        let batchArgs = ["tool_calls": "[\(items.joined(separator: ","))]"]

        let call = AppToolCall(name: "batch", arguments: batchArgs, category: .automation)
        let result = await AppToolRegistry.execute(call: call, in: nil)

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Batch size exceeds maximum limit of 25"))
    }

    func testBatchCannotBypassDeniedTerminalPermission() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let marker = dir.appendingPathComponent("should-not-exist")
        let project = AppProject(
            name: "batch_test", rootDirectoryPath: dir.path,
            permissions: AppProjectPermissions(
                mode: .auto, terminal: .deny, automation: .allow))
        let batchArgs = [
            "tool_calls": """
            [{"tool":"run_command","parameters":{"command":"touch should-not-exist"}}]
            """
        ]

        let result = await AppToolRegistry.execute(
            call: AppToolCall(name: "batch", arguments: batchArgs, category: .automation),
            in: project)

        XCTAssertTrue(result.output.contains("1 failed"))
        XCTAssertTrue(result.output.contains("Terminal & Shell Execution category is set to Deny"))
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path))
    }

    func testBatchHighRiskChildNeedsFreshApproval() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(
            UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let project = AppProject(
            name: "batch_test", rootDirectoryPath: dir.path,
            permissions: AppProjectPermissions(
                mode: .auto, terminal: .allow, automation: .allow))
        let batchArgs = [
            "tool_calls": """
            [{"tool":"run_command","parameters":{"command":"rm -rf should-not-exist"}}]
            """
        ]

        let risk = ToolRiskClassifier.assessRisk(name: "batch", arguments: batchArgs)
        let result = await AppToolRegistry.execute(
            call: AppToolCall(name: "batch", arguments: batchArgs, category: .automation),
            in: project)

        XCTAssertEqual(risk.level, .high)
        XCTAssertEqual(risk.category, .terminal)
        XCTAssertTrue(result.output.contains("Nested call requires separate approval"))
        XCTAssertTrue(result.output.contains("1 failed"))
    }
}
