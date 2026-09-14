import XCTest
@testable import TurboSparkApp

final class FuzzyEditWhitespaceTests: XCTestCase {
    func testEditFileWithNormalizedWhitespaceAndIndentationMismatch() async throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }

        let fileURL = dir.appendingPathComponent("code.swift")
        let diskContent = """
        struct Sample {
            func execute() {
                let x = 10
                let y = 20
                print(x + y)
            }
        }
        """
        try diskContent.write(to: fileURL, atomically: true, encoding: .utf8)
        let project = AppProject(name: "fuzzy_edit_test", rootDirectoryPath: dir.path)

        // Model generated target with different indentation (2 spaces instead of 4)
        let targetWithMismatchedIndentation = """
          let x = 10
          let y = 20
          print(x + y)
        """

        let replacement = """
                let x = 100
                let y = 200
                print(x + y)
        """

        let call = AppToolCall(
            name: "edit_file",
            arguments: [
                "path": "code.swift",
                "old_string": targetWithMismatchedIndentation,
                "new_string": replacement
            ],
            category: .fileWrite
        )
        let result = await AppToolRegistry.execute(call: call, in: project)

        XCTAssertFalse(result.isError, "Normalized whitespace matching should succeed: \(result.output)")

        let updated = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertTrue(updated.contains("let x = 100"))
        XCTAssertTrue(updated.contains("let y = 200"))
        XCTAssertFalse(updated.contains("let x = 10\n"))
    }
}
