import XCTest
@testable import TurboSparkApp

final class EnhancedToolsAndHttpTests: XCTestCase {
    private func makeProject() throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let project = AppProject(name: "test-enhanced-tools", rootDirectoryPath: dir.path)
        return (project, dir)
    }

    // MARK: - read_file Modes

    func testReadFileStatsMode() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let content = "Alpha Bravo Charlie\nDelta Echo Foxtrot\n"
        let fileURL = dir.appendingPathComponent("sample.txt")
        try content.write(to: fileURL, atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "sample.txt", "mode": "stats"],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "Stats mode should succeed: \(result.output)")
        XCTAssertTrue(result.output.contains("Characters:"), "Should report character count")
        XCTAssertTrue(result.output.contains("Lines: 2"), "Should report line count")
        XCTAssertTrue(result.output.contains("Words: 6"), "Should report word count")
        XCTAssertTrue(result.output.contains("SHA256:"), "Should report SHA256 checksum")
    }

    func testReadFilePreviewMode() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        var lines: [String] = []
        for i in 1...30 {
            lines.append("Line \(i) content")
        }
        let fileURL = dir.appendingPathComponent("long.txt")
        try lines.joined(separator: "\n").write(to: fileURL, atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "long.txt", "mode": "preview", "limit": "10"],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "Preview mode should succeed: \(result.output)")
        XCTAssertTrue(result.output.contains("Line 1 content"))
        XCTAssertTrue(result.output.contains("Line 10 content"))
        XCTAssertTrue(result.output.contains("lines omitted"))
    }

    func testReadFileSearchMode() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let content = """
        start
        needle_target_alpha
        middle
        needle_target_beta
        end
        """
        let fileURL = dir.appendingPathComponent("searchable.txt")
        try content.write(to: fileURL, atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "read_file",
            arguments: [
                "path": "searchable.txt",
                "mode": "search",
                "search_pattern": "needle_target",
                "context_lines": "1"
            ],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "Search mode should succeed: \(result.output)")
        XCTAssertTrue(result.output.contains("needle_target_alpha"))
        XCTAssertTrue(result.output.contains("needle_target_beta"))
        XCTAssertTrue(result.output.contains("Found 2 matching line(s)"))
    }

    func testReadFileDiffModeBetweenFiles() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let original = "A\nB\nC\n"
        let modified = "A\nB modified\nC\n"
        let origURL = dir.appendingPathComponent("file_a.txt")
        let modURL = dir.appendingPathComponent("file_b.txt")
        try original.write(to: origURL, atomically: true, encoding: .utf8)
        try modified.write(to: modURL, atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "read_file",
            arguments: [
                "path": "file_a.txt",
                "mode": "diff",
                "comparison_path": "file_b.txt"
            ],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "Diff mode should succeed: \(result.output)")
        XCTAssertTrue(result.output.contains("B modified"))
    }

    func testReadFileTimeMachineMode() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let fileURL = dir.appendingPathComponent("history.txt")
        try "V1".write(to: fileURL, atomically: true, encoding: .utf8)

        let call = AppToolCall(
            name: "read_file",
            arguments: ["path": "history.txt", "mode": "time_machine"],
            category: .fileRead
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "Time machine mode should succeed: \(result.output)")
        XCTAssertTrue(result.output.contains("history"))
    }

    // MARK: - edit_file / editor Commands

    func testEditFileInsertCommand() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let initial = "Line 1\nLine 3\n"
        let fileURL = dir.appendingPathComponent("insert_test.txt")
        try initial.write(to: fileURL, atomically: true, encoding: .utf8)

        // Read first to track snapshot
        let readCall = AppToolCall(name: "read_file", arguments: ["path": "insert_test.txt"], category: .fileRead)
        _ = await AppToolRegistry.execute(call: readCall, in: project)

        let call = AppToolCall(
            name: "edit_file",
            arguments: [
                "path": "insert_test.txt",
                "command": "insert",
                "new_string": "Line 2",
                "insert_line": "1",
                "position": "after"
            ],
            category: .fileWrite
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "Insert command should succeed: \(result.output)")

        let updated = try String(contentsOf: fileURL, encoding: .utf8)
        let lines = updated.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        XCTAssertEqual(lines[0], "Line 1")
        XCTAssertEqual(lines[1], "Line 2")
        XCTAssertEqual(lines[2], "Line 3")
    }

    func testEditFilePatternReplaceCommand() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let initial = "Item #001: active\nItem #002: pending\nItem #003: active"
        let fileURL = dir.appendingPathComponent("regex_test.txt")
        try initial.write(to: fileURL, atomically: true, encoding: .utf8)

        // Read first to track snapshot
        let readCall = AppToolCall(name: "read_file", arguments: ["path": "regex_test.txt"], category: .fileRead)
        _ = await AppToolRegistry.execute(call: readCall, in: project)

        let call = AppToolCall(
            name: "editor",
            arguments: [
                "path": "regex_test.txt",
                "command": "pattern_replace",
                "regex_pattern": "Item #([0-9]+): active",
                "new_string": "Record #$1: done",
                "replace_all": "true"
            ],
            category: .fileWrite
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "Pattern replace should succeed: \(result.output)")

        let updated = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertTrue(updated.contains("Record #001: done"))
        XCTAssertTrue(updated.contains("Record #003: done"))
        XCTAssertTrue(updated.contains("Item #002: pending"))
    }

    func testEditFileUndoEditCommand() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }

        let initial = "Original Pristine Content\n"
        let fileURL = dir.appendingPathComponent("undo_test.txt")
        try initial.write(to: fileURL, atomically: true, encoding: .utf8)

        // Read first to track snapshot
        let readCall = AppToolCall(name: "read_file", arguments: ["path": "undo_test.txt"], category: .fileRead)
        let readResult = await AppToolRegistry.execute(call: readCall, in: project)
        XCTAssertFalse(readResult.isError)

        // Make an edit first
        let editCall = AppToolCall(
            name: "edit_file",
            arguments: [
                "path": "undo_test.txt",
                "old_string": "Original Pristine Content",
                "new_string": "Overwritten Content"
            ],
            category: .fileWrite
        )
        let editResult = await AppToolRegistry.execute(call: editCall, in: project)
        XCTAssertFalse(editResult.isError)
        let editedContent = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertTrue(editedContent.contains("Overwritten Content"))

        // Now trigger undo_edit
        let undoCall = AppToolCall(
            name: "editor",
            arguments: [
                "path": "undo_test.txt",
                "command": "undo_edit"
            ],
            category: .fileWrite
        )
        let undoResult = await AppToolRegistry.execute(call: undoCall, in: project)
        XCTAssertFalse(undoResult.isError, "Undo edit should succeed: \(undoResult.output)")
        XCTAssertTrue(undoResult.output.contains("rolled back"))

        let restoredContent = try String(contentsOf: fileURL, encoding: .utf8)
        XCTAssertEqual(restoredContent, initial)
    }

    // MARK: - HttpRequest Tool Integration

    func testHttpRequestRegistrationAndCategory() {
        XCTAssertTrue(AppToolRegistry.isImplemented("HttpRequest"))
        XCTAssertTrue(AppToolRegistry.isImplemented("http_request"))
        XCTAssertTrue(AppToolRegistry.isImplemented("httprequest"))
        XCTAssertEqual(AppToolCatalog.category(for: "http_request"), .web)
        XCTAssertEqual(AppToolCatalog.category(for: "HttpRequest"), .web)
        XCTAssertFalse(AppToolRegistry.workspaceRootedToolNames.contains("http_request"))
    }

    func testHttpRequestMissingUrlFails() async {
        let call = AppToolCall(name: "HttpRequest", arguments: [:], category: .web)
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Missing 'url'"))
    }

    func testHttpRequestRejectsPrivateAndLoopbackHosts() async {
        let privateUrls = [
            "http://localhost:8080/api",
            "http://127.0.0.1:3000/status",
            "http://169.254.169.254/metadata",
            "http://[::1]:9090/"
        ]

        for url in privateUrls {
            let call = AppToolCall(
                name: "http_request",
                arguments: ["url": url],
                category: .web
            )
            let result = await AppToolRegistry.execute(call: call, in: nil)
            XCTAssertTrue(result.isError, "URL \(url) should be rejected for SSRF safety")
            XCTAssertTrue(result.output.contains("refused") || result.output.contains("private") || result.output.contains("SSRF"))
        }
    }
}
