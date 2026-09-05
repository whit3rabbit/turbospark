import XCTest
@testable import TurboSparkApp

final class CoreFileToolsTests: XCTestCase {
    private func makeProject() throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(name: "test-core-files", rootDirectoryPath: dir.path)
        return (project, dir)
    }

    // MARK: - list_directory / ls / glob

    func testListDirectorySuccessAndAliases() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        try "hello".write(to: dir.appendingPathComponent("file1.txt"), atomically: true, encoding: .utf8)
        try "world".write(to: dir.appendingPathComponent("file2.swift"), atomically: true, encoding: .utf8)
        let subDir = dir.appendingPathComponent("subdir", isDirectory: true)
        try FileManager.default.createDirectory(at: subDir, withIntermediateDirectories: true)
        try "nested".write(to: subDir.appendingPathComponent("nested.txt"), atomically: true, encoding: .utf8)

        for alias in ["list_directory", "list_dir", "ls", "glob"] {
            let call = AppToolCall(name: alias, arguments: ["path": "."], category: .fileRead)
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertTrue(result.output.contains("file1.txt"), "Should contain file1.txt")
            XCTAssertTrue(result.output.contains("file2.swift"), "Should contain file2.swift")
            XCTAssertTrue(result.output.contains("subdir"), "Should contain subdir")
        }
    }

    func testListDirectoryNonExistentPath() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let call = AppToolCall(name: "list_directory", arguments: ["path": "nonexistent_dir"], category: .fileRead)
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError, "Listing non-existent directory should report error")
    }

    // MARK: - read_file / view_file / cat / read

    func testReadFileSuccessAndAliases() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let content = "Line 1\nLine 2\nLine 3\nLine 4\nLine 5"
        try content.write(to: dir.appendingPathComponent("test.txt"), atomically: true, encoding: .utf8)

        for alias in ["read_file", "view_file", "cat", "fileread", "read"] {
            let call = AppToolCall(name: alias, arguments: ["path": "test.txt"], category: .fileRead)
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed")
            XCTAssertTrue(result.output.contains("Line 1"))
            XCTAssertTrue(result.output.contains("Line 5"))
        }
    }

    func testReadFileWithBoundsAndMissingFile() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let content = (1...20).map { "Item \($0)" }.joined(separator: "\n")
        try content.write(to: dir.appendingPathComponent("items.txt"), atomically: true, encoding: .utf8)

        let rangeCall = AppToolCall(
            name: "read_file",
            arguments: ["path": "items.txt", "start_line": "5", "end_line": "8"],
            category: .fileRead
        )
        let rangeResult = await AppToolRegistry.execute(call: rangeCall, in: project)
        XCTAssertFalse(rangeResult.isError)
        XCTAssertTrue(rangeResult.output.contains("Item 5"))
        XCTAssertTrue(rangeResult.output.contains("Item 8"))
        XCTAssertFalse(rangeResult.output.contains("Item 4"))
        XCTAssertFalse(rangeResult.output.contains("Item 9"))

        let missingCall = AppToolCall(name: "read_file", arguments: ["path": "missing.txt"], category: .fileRead)
        let missingResult = await AppToolRegistry.execute(call: missingCall, in: project)
        XCTAssertTrue(missingResult.isError, "Reading missing file should fail")
    }

    // MARK: - write_file / save_file / write

    func testWriteFileSuccessAndOverwrites() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for (idx, alias) in ["write_file", "save_file", "filewrite", "write"].enumerated() {
            let filePath = "nested/dir/test_\(idx).txt"
            let fileContent = "Content for \(alias)"
            let call = AppToolCall(
                name: alias,
                arguments: ["path": filePath, "content": fileContent],
                category: .fileWrite
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed writing file")

            let written = try String(contentsOf: dir.appendingPathComponent(filePath), encoding: .utf8)
            XCTAssertEqual(written, fileContent)
        }
    }

    func testWriteFileSandboxEscapeRefusal() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let call = AppToolCall(
            name: "write_file",
            arguments: ["path": "../../../../../tmp/evil.txt", "content": "bad"],
            category: .fileWrite
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError, "Path escaping project root must be refused")
    }

    // MARK: - edit_file / fileedit / edit

    func testEditFileSuccessAndAliases() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        for alias in ["edit_file", "fileedit", "edit"] {
            let fileName = "\(alias)_test.txt"
            let initial = "let x = 1\nlet y = 2\nlet z = 3\n"
            try initial.write(to: dir.appendingPathComponent(fileName), atomically: true, encoding: .utf8)

            let call = AppToolCall(
                name: alias,
                arguments: [
                    "path": fileName,
                    "target": "let y = 2",
                    "replacement": "let y = 42"
                ],
                category: .fileWrite
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")

            let modified = try String(contentsOf: dir.appendingPathComponent(fileName), encoding: .utf8)
            XCTAssertTrue(modified.contains("let y = 42"))
            XCTAssertFalse(modified.contains("let y = 2"))
        }
    }

    func testEditFileTargetNotFound() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let fileName = "not_found.txt"
        try "alpha beta gamma".write(to: dir.appendingPathComponent(fileName), atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "edit_file",
            arguments: [
                "path": fileName,
                "target": "delta epsilon",
                "replacement": "omega"
            ],
            category: .fileWrite
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertTrue(result.isError, "Target not found should report error")
    }

    // MARK: - apply_patch / applypatch

    func testApplyPatchSuccess() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let fileName = "code.rs"
        let initial = "fn main() {\n    println!(\"old\");\n}\n"
        try initial.write(to: dir.appendingPathComponent(fileName), atomically: true, encoding: .utf8)

        let patch = """
        --- a/code.rs
        +++ b/code.rs
        @@ -1,3 +1,3 @@
         fn main() {
        -    println!("old");
        +    println!("new");
         }
        """

        let call = AppToolCall(
            name: "apply_patch",
            arguments: ["path": fileName, "patch": patch],
            category: .fileWrite
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "apply_patch should succeed: \(result.output)")

        let modified = try String(contentsOf: dir.appendingPathComponent(fileName), encoding: .utf8)
        XCTAssertTrue(modified.contains("println!(\"new\");"))
    }

    // MARK: - search_code / grep / search

    func testSearchCodeSuccessAndAliases() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        try "func calculateMetrics() -> Int { return 42 }".write(
            to: dir.appendingPathComponent("Math.swift"),
            atomically: true,
            encoding: .utf8
        )
        try "struct MetricsPayload { let count: Int }".write(
            to: dir.appendingPathComponent("Models.swift"),
            atomically: true,
            encoding: .utf8
        )

        for alias in ["search_code", "grep", "search"] {
            let call = AppToolCall(
                name: alias,
                arguments: ["pattern": "Metrics", "path": "."],
                category: .fileRead
            )
            let result = await AppToolRegistry.execute(call: call, in: project)
            XCTAssertFalse(result.isError, "Alias \(alias) should succeed: \(result.output)")
            XCTAssertTrue(result.output.contains("Math.swift") || result.output.contains("Models.swift"))
        }
    }
}
