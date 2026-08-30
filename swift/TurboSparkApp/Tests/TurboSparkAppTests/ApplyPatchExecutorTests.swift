import XCTest
@testable import TurboSparkApp

/// Regression tests for `ApplyPatchExecutor` (T3, T4): a delete hunk left
/// `isDeleteFile` set without ever setting `currentFile`, so `flushCurrentFile`
/// returned early WITHOUT resetting either flag, and the next file in the
/// same patch inherited `isDeleteFile == true` and got `removeItem`'d instead
/// of edited. Separately, `applyHunkLines` ignored the `@@` hunk header's
/// starting line number and walked the whole file from index 0, so any hunk
/// not starting at line 1 applied its changes at the wrong offset.
final class ApplyPatchExecutorTests: XCTestCase {
    private func makeWorkspace() throws -> URL {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    // MARK: - T3: stale isDeleteFile must not survive into the next file

    func testDeletingOneFileDoesNotDeleteTheNextFileInTheSamePatch() throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }

        let toDelete = root.appendingPathComponent("old.txt")
        try "gone soon".write(to: toDelete, atomically: true, encoding: .utf8)

        let survivor = root.appendingPathComponent("Package.swift")
        try "// swift-tools-version: 5.9\nlet x = 1\n".write(to: survivor, atomically: true, encoding: .utf8)

        let patch = """
        diff --git a/old.txt b/old.txt
        deleted file mode 100644
        index 0000000..0000000
        --- a/old.txt
        +++ /dev/null
        @@ -1,1 +0,0 @@
        -gone soon
        diff --git a/Package.swift b/Package.swift
        --- a/Package.swift
        +++ b/Package.swift
        @@ -2,1 +2,1 @@
        -let x = 1
        +let x = 2
        """

        let output = try ApplyPatchExecutor.apply(patchText: patch, rootURL: root)

        XCTAssertFalse(FileManager.default.fileExists(atPath: toDelete.path), "old.txt should have been deleted.")
        // The actual regression: Package.swift must still EXIST and must
        // have been EDITED, not removed by a stale isDeleteFile flag.
        XCTAssertTrue(FileManager.default.fileExists(atPath: survivor.path), "Package.swift must survive a preceding delete in the same patch.")
        let updated = try String(contentsOf: survivor, encoding: .utf8)
        XCTAssertTrue(updated.contains("let x = 2"))

        XCTAssertEqual(output.applied.filter { $0.type == "delete" }.count, 1)
        XCTAssertEqual(output.applied.filter { $0.type == "update" }.count, 1)
    }

    // MARK: - T4: hunks must apply at their stated offset, not at index 0

    func testEditAtANonZeroOffsetAppliesAtTheCorrectLine() throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }

        let file = root.appendingPathComponent("numbers.txt")
        let original = (1...10).map { "line\($0)" }.joined(separator: "\n") + "\n"
        try original.write(to: file, atomically: true, encoding: .utf8)

        // Replace line 7 only. A hunk-offset-blind applier (walking from
        // index 0) would instead touch line 1, since the removal/addition
        // pair is the FIRST thing it sees in the hunk body.
        let patch = """
        diff --git a/numbers.txt b/numbers.txt
        --- a/numbers.txt
        +++ b/numbers.txt
        @@ -7,1 +7,1 @@
        -line7
        +REPLACED
        """
        _ = try ApplyPatchExecutor.apply(patchText: patch, rootURL: root)

        let result = try String(contentsOf: file, encoding: .utf8)
        let resultLines = result.components(separatedBy: "\n")
        XCTAssertEqual(resultLines[0], "line1", "Line 1 must be untouched by a hunk targeting line 7.")
        XCTAssertEqual(resultLines[6], "REPLACED")
        XCTAssertEqual(resultLines[7], "line8")
    }

    func testStaleContextIsRejectedRatherThanAppliedBlindly() throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }

        let file = root.appendingPathComponent("numbers.txt")
        // The file on disk does NOT match what the hunk expects at line 7.
        let original = (1...10).map { "line\($0)" }.joined(separator: "\n") + "\n"
        try original.write(to: file, atomically: true, encoding: .utf8)

        let patch = """
        diff --git a/numbers.txt b/numbers.txt
        --- a/numbers.txt
        +++ b/numbers.txt
        @@ -7,1 +7,1 @@
        -this text does not exist in the file
        +REPLACED
        """

        XCTAssertThrowsError(try ApplyPatchExecutor.apply(patchText: patch, rootURL: root)) { error in
            let msg = (error as NSError).localizedDescription
            XCTAssertTrue(msg.contains("mismatch") || msg.contains("changed"), "Expected a context-mismatch error, got: \(msg)")
        }

        // The file must be left untouched by a rejected hunk.
        let unchanged = try String(contentsOf: file, encoding: .utf8)
        XCTAssertEqual(unchanged, original)
    }

    func testMultipleHunksInOneFileEachApplyAtTheirOwnOffset() throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }

        let file = root.appendingPathComponent("numbers.txt")
        let original = (1...10).map { "line\($0)" }.joined(separator: "\n") + "\n"
        try original.write(to: file, atomically: true, encoding: .utf8)

        let patch = """
        diff --git a/numbers.txt b/numbers.txt
        --- a/numbers.txt
        +++ b/numbers.txt
        @@ -2,1 +2,1 @@
        -line2
        +FIRST
        @@ -9,1 +9,1 @@
        -line9
        +SECOND
        """
        _ = try ApplyPatchExecutor.apply(patchText: patch, rootURL: root)

        let resultLines = try String(contentsOf: file, encoding: .utf8).components(separatedBy: "\n")
        XCTAssertEqual(resultLines[1], "FIRST")
        XCTAssertEqual(resultLines[8], "SECOND")
        XCTAssertEqual(resultLines[0], "line1")
        XCTAssertEqual(resultLines[9], "line10")
    }

    func testNewFileCreationStillWorks() throws {
        let root = try makeWorkspace()
        defer { try? FileManager.default.removeItem(at: root) }

        let patch = """
        diff --git a/new.txt b/new.txt
        new file mode 100644
        --- /dev/null
        +++ b/new.txt
        @@ -0,0 +1,2 @@
        +hello
        +world
        """
        _ = try ApplyPatchExecutor.apply(patchText: patch, rootURL: root)

        let created = root.appendingPathComponent("new.txt")
        XCTAssertTrue(FileManager.default.fileExists(atPath: created.path))
        let contents = try String(contentsOf: created, encoding: .utf8)
        XCTAssertEqual(contents, "hello\nworld")
    }
}
