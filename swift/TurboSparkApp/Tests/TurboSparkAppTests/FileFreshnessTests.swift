import XCTest

@testable import TurboSparkApp

/// The freshness contract across the three write paths: `edit_file` had it
/// from the start (hash-based, `FileSnapshotStore`); `write_file` and
/// `apply_patch`'s delete branch gained it in this pass. The rule under
/// test: a DESTRUCTIVE write over an existing file requires that this
/// session has read the file's current bytes -- creating a NEW file is
/// never gated, and a patch's update branch stays exempt because its hunk
/// context lines are a stronger check than any hash.
@MainActor
final class FileFreshnessTests: XCTestCase {
    var root: URL!

    override func setUp() {
        super.setUp()
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("freshness-\(UUID().uuidString)", isDirectory: true)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        await FileSnapshotStore.shared.reset()
        if let root {
            try? FileManager.default.removeItem(at: root)
        }
        root = nil
        try? await super.tearDown()
    }

    private func writeFixture(_ relPath: String, _ content: String) throws -> URL {
        let url = root.appendingPathComponent(relPath)
        try content.write(to: url, atomically: true, encoding: .utf8)
        return url
    }

    private func readError(of error: Error) -> String {
        (error as NSError).localizedDescription
    }

    // MARK: - write_file

    func testWriteFileCreatesANewFileWithoutAnyGate() async throws {
        let output = try await AppToolRegistry.writeFile(
            relPath: "new.txt", content: "fresh", rootURL: root)
        XCTAssertTrue(output.contains("Successfully wrote"))
        XCTAssertEqual(try String(contentsOf: root.appendingPathComponent("new.txt"), encoding: .utf8), "fresh")
    }

    func testWriteFileRefusesAnExistingFileThatWasNeverRead() async throws {
        _ = try writeFixture("existing.txt", "user content")
        do {
            _ = try await AppToolRegistry.writeFile(
                relPath: "existing.txt", content: "clobber", rootURL: root)
            XCTFail("write_file over an unread existing file must be refused")
        } catch {
            XCTAssertTrue(
                readError(of: error).contains("has not been read this session"),
                "Unexpected error: \(readError(of: error))")
        }
        XCTAssertEqual(
            try String(contentsOf: root.appendingPathComponent("existing.txt"), encoding: .utf8),
            "user content", "The unread file must be untouched.")
    }

    func testWriteFileAllowsAnExistingFileAfterReadingIt() async throws {
        _ = try writeFixture("tracked.txt", "read me")
        _ = try await AppToolRegistry.readFile(
            relPath: "tracked.txt", rootURL: root, startLine: nil, endLine: nil)
        let output = try await AppToolRegistry.writeFile(
            relPath: "tracked.txt", content: "new bytes", rootURL: root)
        XCTAssertTrue(output.contains("Successfully wrote"))
        XCTAssertEqual(
            try String(contentsOf: root.appendingPathComponent("tracked.txt"), encoding: .utf8),
            "new bytes")
    }

    func testWriteFileRefusesAFileChangedOnDiskSinceTheRead() async throws {
        let url = try writeFixture("changed.txt", "v1")
        _ = try await AppToolRegistry.readFile(
            relPath: "changed.txt", rootURL: root, startLine: nil, endLine: nil)
        // The user (or another process) edits behind the model's back.
        try "v2 from the user".write(to: url, atomically: true, encoding: .utf8)
        do {
            _ = try await AppToolRegistry.writeFile(
                relPath: "changed.txt", content: "model overwrite", rootURL: root)
            XCTFail("write_file over a file changed since its read must be refused")
        } catch {
            XCTAssertTrue(
                readError(of: error).contains("changed since it was last read"),
                "Unexpected error: \(readError(of: error))")
        }
        XCTAssertEqual(
            try String(contentsOf: url, encoding: .utf8), "v2 from the user",
            "The user's edit must win.")
    }

    // MARK: - apply_patch

    private func deletePatch(for relPath: String, content: String) -> String {
        """
        diff --git a/\(relPath) b/\(relPath)
        --- a/\(relPath)
        +++ /dev/null
        @@ -1,1 +0,0 @@
        -\(content)
        """
    }

    func testApplyPatchDeleteOfAnUntrackedFileStillWorks() async throws {
        _ = try writeFixture("untracked.txt", "never read")
        let output = try await ApplyPatchExecutor.apply(
            patchText: deletePatch(for: "untracked.txt", content: "never read"),
            rootURL: root)
        XCTAssertTrue(output.applied.contains { $0.type == "delete" })
        XCTAssertFalse(FileManager.default.fileExists(atPath: root.appendingPathComponent("untracked.txt").path))
    }

    func testApplyPatchDeleteRefusesAFileChangedSinceTheRead() async throws {
        let url = try writeFixture("doomed.txt", "v1")
        _ = try await AppToolRegistry.readFile(
            relPath: "doomed.txt", rootURL: root, startLine: nil, endLine: nil)
        try "v2 from the user".write(to: url, atomically: true, encoding: .utf8)
        do {
            // The patch's only body line names V1, but a delete has NO
            // context verification -- without the snapshot gate this would
            // remove the user's v2 without a word.
            _ = try await ApplyPatchExecutor.apply(
                patchText: deletePatch(for: "doomed.txt", content: "v1"),
                rootURL: root)
            XCTFail("delete of a file changed since its read must be refused")
        } catch {
            XCTAssertTrue(
                readError(of: error).contains("modified on disk"),
                "Unexpected error: \(readError(of: error))")
        }
        XCTAssertTrue(FileManager.default.fileExists(atPath: url.path), "The file must survive.")
    }

    func testApplyPatchRecordsSnapshotsSoEditFileReadsFreshAfterwards() async throws {
        let url = try writeFixture("patched.txt", "line one\nline two\n")
        _ = try await AppToolRegistry.readFile(
            relPath: "patched.txt", rootURL: root, startLine: nil, endLine: nil)
        let patch = """
        diff --git a/patched.txt b/patched.txt
        --- a/patched.txt
        +++ b/patched.txt
        @@ -2,1 +2,1 @@
        -line two
        +REPLACED
        """
        _ = try await ApplyPatchExecutor.apply(patchText: patch, rootURL: root)
        // The store must now hold the POST-patch hash: without this,
        // `edit_file` (and `write_file`) refuse every subsequent call on
        // the file until the model re-reads content it just wrote.
        let stale = await FileSnapshotStore.shared.isStale(url: url)
        XCTAssertFalse(stale, "A file a patch just wrote must read fresh.")
        _ = try await AppToolRegistry.editFile(
            relPath: "patched.txt", oldString: "REPLACED", newString: "DONE",
            replaceAll: false, rootURL: root)
        XCTAssertEqual(
            try String(contentsOf: url, encoding: .utf8), "line one\nDONE\n")
    }
}
