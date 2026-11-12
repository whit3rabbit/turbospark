import XCTest
@testable import TurboSparkApp

/// Regression tests for the B6 tool edge-case fixes (one MARK per fix).
final class ToolEdgeCaseB6Tests: XCTestCase {
    private func makeProject() throws -> (AppProject, URL) {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return (AppProject(name: "b6", rootDirectoryPath: dir.path), dir)
    }

    // MARK: - NotebookEdit

    private let markdownNotebook = """
    {"cells":[{"cell_type":"markdown","metadata":{},"source":["# T\\n"]}],"metadata":{},"nbformat":4,"nbformat_minor":2}
    """

    func testNotebookReplaceWithoutCellTypeKeepsMarkdown() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("n.ipynb")
        try markdownNotebook.write(to: url, atomically: true, encoding: .utf8)
        let result = await AppToolRegistry.execute(
            call: AppToolCall(name: "notebook_edit", arguments: ["path": "n.ipynb", "cell_index": "0", "source": "# New\n"], category: .fileWrite),
            in: project)
        XCTAssertFalse(result.isError, result.output)
        let json = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        let cell = (json?["cells"] as? [[String: Any]])?.first
        XCTAssertEqual(cell?["cell_type"] as? String, "markdown")
        XCTAssertNil(cell?["outputs"])
        XCTAssertNil(cell?["execution_count"])
    }

    func testNotebookRefusesStaleFile() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("s.ipynb")
        try markdownNotebook.write(to: url, atomically: true, encoding: .utf8)
        await FileSnapshotStore.shared.recordSnapshot(url: url, content: markdownNotebook)
        try (markdownNotebook + "\n").write(to: url, atomically: true, encoding: .utf8)
        let result = await AppToolRegistry.execute(
            call: AppToolCall(name: "notebook_edit", arguments: ["path": "s.ipynb", "cell_index": "0", "source": "x"], category: .fileWrite),
            in: project)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("modified on disk"), result.output)
    }

    func testNotebookRefusesOversizedFile() async throws {
        let (project, dir) = try makeProject()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("big.ipynb")
        FileManager.default.createFile(atPath: url.path, contents: nil)
        let handle = try FileHandle(forWritingTo: url)
        try handle.truncate(atOffset: UInt64(AppFileReadLimits.maximumBytes + 1))
        try handle.close()
        let result = await AppToolRegistry.execute(
            call: AppToolCall(name: "notebook_edit", arguments: ["path": "big.ipynb", "cell_index": "0", "source": "x"], category: .fileWrite),
            in: project)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("limit"), result.output)
    }

    // MARK: - ApplyPatchInput

    func testApplyPatchInputDecodesAlternateKeyBesideNonStringValue() throws {
        let json = #"{"patchText":"*** Begin Patch\n*** End Patch","validate_timeout_ms":5000}"#
        let input = try JSONDecoder().decode(ApplyPatchInput.self, from: Data(json.utf8))
        XCTAssertEqual(input.patchText, "*** Begin Patch\n*** End Patch")
        XCTAssertEqual(input.validateTimeoutMs, 5000)
        let plain = try JSONDecoder().decode(ApplyPatchInput.self, from: Data(#"{"patch":"p"}"#.utf8))
        XCTAssertEqual(plain.patchText, "p")
    }

    // MARK: - MultiEdit

    func testMultiEditWriteFailureRollsBackAndDiscardsBackups() async throws {
        let (project, dir) = try makeProject()
        let sub = dir.appendingPathComponent("sub", isDirectory: true)
        try FileManager.default.createDirectory(at: sub, withIntermediateDirectories: true)
        let a = dir.appendingPathComponent("a.txt")
        let b = sub.appendingPathComponent("b.txt")
        try "oldA".write(to: a, atomically: true, encoding: .utf8)
        try "oldB".write(to: b, atomically: true, encoding: .utf8)
        // An unwritable directory makes the atomic write of b.txt (temp file
        // creation) fail after a.txt was already rewritten.
        try FileManager.default.setAttributes([.posixPermissions: 0o555], ofItemAtPath: sub.path)
        defer {
            try? FileManager.default.setAttributes([.posixPermissions: 0o755], ofItemAtPath: sub.path)
            try? FileManager.default.removeItem(at: dir)
        }
        let edits = #"[{"file_path":"a.txt","old_string":"oldA","new_string":"newA"},{"file_path":"sub/b.txt","old_string":"oldB","new_string":"newB"}]"#
        let result = await AppToolRegistry.execute(
            call: AppToolCall(name: "multiedit", arguments: ["edits": edits], category: .fileWrite), in: project)
        XCTAssertTrue(result.isError, result.output)
        XCTAssertEqual(try String(contentsOf: a, encoding: .utf8), "oldA")
        XCTAssertEqual(try String(contentsOf: b, encoding: .utf8), "oldB")
        let backupA = await FileSnapshotStore.shared.restoreBackup(url: a)
        let backupB = await FileSnapshotStore.shared.restoreBackup(url: b)
        XCTAssertNil(backupA)
        XCTAssertNil(backupB)
    }
}
