import XCTest
@testable import TurboSparkApp

final class MultiEditTests: XCTestCase {
    func testAtomicMultiEditSuccessAcrossMultipleFiles() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let fileA = dir.appendingPathComponent("fileA.txt")
        let fileB = dir.appendingPathComponent("fileB.txt")
        try "func oldA() {}".write(to: fileA, atomically: true, encoding: .utf8)
        try "func oldB() {}".write(to: fileB, atomically: true, encoding: .utf8)
        let project = AppProject(name: "multiedit_test", rootDirectoryPath: dir.path)

        let editsPayload = """
        [
            {"file_path": "fileA.txt", "old_string": "oldA", "new_string": "newA"},
            {"file_path": "fileB.txt", "old_string": "oldB", "new_string": "newB"}
        ]
        """

        let call = AppToolCall(name: "multiedit", arguments: ["edits": editsPayload], category: .fileWrite)
        let result = await AppToolRegistry.execute(call: call, in: project)

        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("Atomic multiedit completed successfully"))

        let textA = try String(contentsOf: fileA, encoding: .utf8)
        let textB = try String(contentsOf: fileB, encoding: .utf8)
        XCTAssertEqual(textA, "func newA() {}")
        XCTAssertEqual(textB, "func newB() {}")
    }

    func testAtomicMultiEditAbortsAndRollsBackWhenTargetNotFound() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let fileA = dir.appendingPathComponent("fileA.txt")
        let fileB = dir.appendingPathComponent("fileB.txt")
        try "originalA".write(to: fileA, atomically: true, encoding: .utf8)
        try "originalB".write(to: fileB, atomically: true, encoding: .utf8)
        let project = AppProject(name: "multiedit_test", rootDirectoryPath: dir.path)

        let editsPayload = """
        [
            {"file_path": "fileA.txt", "old_string": "originalA", "new_string": "mutatedA"},
            {"file_path": "fileB.txt", "old_string": "nonexistentTarget", "new_string": "mutatedB"}
        ]
        """

        let call = AppToolCall(name: "multiedit", arguments: ["edits": editsPayload], category: .fileWrite)
        let result = await AppToolRegistry.execute(call: call, in: project)

        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Pre-flight check failed"))

        // File A should remain untouched because pre-flight aborted before any write
        let textA = try String(contentsOf: fileA, encoding: .utf8)
        let textB = try String(contentsOf: fileB, encoding: .utf8)
        XCTAssertEqual(textA, "originalA")
        XCTAssertEqual(textB, "originalB")
    }
}
