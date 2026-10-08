import XCTest
@testable import TurboSparkApp

/// Review items #17, #18, #20, #23: symlinked-directory containment for new
/// files, apply_patch new-file overwrite, search_code symlink reads, and the
/// read_file time_machine pipe deadlock.
final class FileToolPathSafetyTests: XCTestCase {
    private var base: URL!
    private var root: URL!
    private var outside: URL!

    override func setUpWithError() throws {
        let fm = FileManager.default
        base = fm.temporaryDirectory.appendingPathComponent("fts-\(UUID().uuidString)", isDirectory: true)
        root = base.appendingPathComponent("proj", isDirectory: true)
        outside = base.appendingPathComponent("outside", isDirectory: true)
        try fm.createDirectory(at: root, withIntermediateDirectories: true)
        try fm.createDirectory(at: outside, withIntermediateDirectories: true)
        try fm.createSymbolicLink(at: root.appendingPathComponent("link"), withDestinationURL: outside)
        // Dangling: destination does not exist yet.
        try fm.createSymbolicLink(
            at: root.appendingPathComponent("dangling"),
            withDestinationURL: outside.appendingPathComponent("not-yet"))
        try fm.createDirectory(at: root.appendingPathComponent("real"), withIntermediateDirectories: true)
        try fm.createSymbolicLink(
            at: root.appendingPathComponent("inlink"),
            withDestinationURL: root.appendingPathComponent("real"))
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: base)
    }

    private func outsideEntries() -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: outside.path)) ?? []).sorted()
    }

    // MARK: - #20

    func testResolveSecurePathRefusesNewFileAndNestedFileUnderEscapingSymlink() {
        for rel in ["link/new.txt", "link/sub/deeper/new.txt", "dangling"] {
            XCTAssertThrowsError(try AppToolRegistry.resolveSecurePath(relPath: rel, rootURL: root), rel)
        }
    }

    func testResolveSecurePathAllowsNewFileUnderInRootSymlink() throws {
        let url = try AppToolRegistry.resolveSecurePath(relPath: "inlink/sub/new.txt", rootURL: root)
        XCTAssertTrue(url.path.hasSuffix("/proj/real/sub/new.txt"), url.path)
    }

    func testValidateWritePathRefusesNewFileUnderEscapingSymlink() {
        let candidate = root.appendingPathComponent("link/sub/new.txt")
        XCTAssertThrowsError(try AppToolSandbox.validateWritePath(candidate, rootURL: root))
    }

    func testWriteFileRefusesNewAndNestedFilesUnderEscapingSymlink() async {
        for rel in ["link/new.txt", "link/a/b/new.txt"] {
            do {
                _ = try await AppToolRegistry.writeFile(relPath: rel, content: "x", rootURL: root)
                XCTFail("write_file should refuse \(rel)")
            } catch {}
        }
        XCTAssertEqual(outsideEntries(), [], "nothing may be created outside the root")
    }

    func testWriteFileStillWorksUnderInRootSymlink() async throws {
        _ = try await AppToolRegistry.writeFile(relPath: "inlink/sub/ok.txt", content: "ok", rootURL: root)
        let written = root.appendingPathComponent("real/sub/ok.txt")
        XCTAssertEqual(try String(contentsOf: written, encoding: .utf8), "ok")
    }

    func testApplyPatchRefusesNewFileUnderEscapingSymlink() async {
        let patch = """
        diff --git a/link/sub/n.txt b/link/sub/n.txt
        --- /dev/null
        +++ b/link/sub/n.txt
        @@ -0,0 +1,1 @@
        +pwned
        """
        do {
            _ = try await ApplyPatchExecutor.apply(patchText: patch, rootURL: root)
            XCTFail("apply_patch should refuse a new file under an escaping symlink")
        } catch {}
        XCTAssertEqual(outsideEntries(), [])
    }

    func testNotebookEditAndMultiEditRefuseExistingTargetsThroughEscapingSymlink() async throws {
        try "{\"cells\":[]}".write(to: outside.appendingPathComponent("n.ipynb"), atomically: true, encoding: .utf8)
        try "hello".write(to: outside.appendingPathComponent("f.txt"), atomically: true, encoding: .utf8)
        do {
            _ = try await NotebookEditExecutor.execute(
                arguments: ["notebook_path": "link/n.ipynb", "edit_mode": "insert", "new_source": "x"],
                rootURL: root)
            XCTFail("notebook_edit should refuse")
        } catch {}
        do {
            _ = try await MultiEditExecutor.execute(
                arguments: ["edits": "[{\"file_path\":\"link/f.txt\",\"old_string\":\"hello\",\"new_string\":\"bye\"}]"],
                rootURL: root)
            XCTFail("multi_edit should refuse")
        } catch {}
        XCTAssertEqual(try String(contentsOf: outside.appendingPathComponent("f.txt"), encoding: .utf8), "hello")
    }

    func testReplBrokerCanonicalTargetRefusesNewFileUnderEscapingSymlink() {
        XCTAssertNil(REPLFileAccessBroker.canonicalTarget(for: "link/new.txt", under: [root.path]))
        XCTAssertNil(REPLFileAccessBroker.canonicalTarget(for: "link/a/b.txt", under: [root.path]))
        XCTAssertNotNil(REPLFileAccessBroker.canonicalTarget(for: "inlink/new.txt", under: [root.path]))
    }

    // MARK: - #17

    func testApplyPatchNewFileHeaderRefusesToOverwriteExistingFile() async throws {
        let existing = root.appendingPathComponent("keep.txt")
        try "user edits".write(to: existing, atomically: true, encoding: .utf8)
        let patch = """
        diff --git a/keep.txt b/keep.txt
        --- /dev/null
        +++ b/keep.txt
        @@ -0,0 +1,1 @@
        +clobber
        """
        do {
            _ = try await ApplyPatchExecutor.apply(patchText: patch, rootURL: root)
            XCTFail("expected refusal")
        } catch {}
        XCTAssertEqual(try String(contentsOf: existing, encoding: .utf8), "user edits")
    }

    func testApplyPatchNewFileStillCreatesMissingFile() async throws {
        let patch = """
        diff --git a/sub/fresh.txt b/sub/fresh.txt
        --- /dev/null
        +++ b/sub/fresh.txt
        @@ -0,0 +1,2 @@
        +one
        +two
        """
        _ = try await ApplyPatchExecutor.apply(patchText: patch, rootURL: root)
        XCTAssertEqual(
            try String(contentsOf: root.appendingPathComponent("sub/fresh.txt"), encoding: .utf8),
            "one\ntwo")
    }

    // MARK: - #23

    func testSearchCodeSkipsSymlinkedFilesButFindsRegularOnes() throws {
        try "TOPSECRET outside".write(to: outside.appendingPathComponent("secret.txt"), atomically: true, encoding: .utf8)
        try FileManager.default.createSymbolicLink(
            at: root.appendingPathComponent("notes.txt"),
            withDestinationURL: outside.appendingPathComponent("secret.txt"))
        try "TOPSECRET inside".write(to: root.appendingPathComponent("plain.txt"), atomically: true, encoding: .utf8)
        let result = try AppToolRegistry.searchCode(pattern: "TOPSECRET", relPath: ".", rootURL: root)
        XCTAssertTrue(result.contains("plain.txt"), result)
        XCTAssertFalse(result.contains("notes.txt"), result)
        XCTAssertFalse(result.contains("outside"), result)
    }

    // MARK: - #18

    func testTimeMachineReturnsWhenGitOutputExceedsPipeBuffer() async throws {
        func git(_ args: [String]) throws {
            let p = Process()
            p.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            p.arguments = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"] + args
            p.currentDirectoryURL = root
            p.standardOutput = FileHandle.nullDevice
            p.standardError = FileHandle.nullDevice
            try p.run()
            p.waitUntilExit()
            XCTAssertEqual(p.terminationStatus, 0, args.joined(separator: " "))
        }
        try git(["init", "-q"])
        let file = root.appendingPathComponent("big.txt")
        for rev in 0..<3 {
            let body = (0..<4_000).map { "rev\(rev) line \($0) padding padding" }.joined(separator: "\n")
            try body.write(to: file, atomically: true, encoding: .utf8)
            try git(["add", "big.txt"])
            try git(["commit", "-q", "-m", "r\(rev)"])
        }
        let rootCopy = root!
        let task = Task {
            try await AppToolRegistry.readFile(
                relPath: "big.txt", rootURL: rootCopy, mode: "time_machine", numRevisions: 3)
        }
        let watchdog = Task {
            try await Task.sleep(nanoseconds: 30_000_000_000)
            task.cancel()
        }
        let out = try await task.value
        watchdog.cancel()
        XCTAssertTrue(out.contains("commit"), "expected git log -p output, got: \(out.prefix(200))")
    }
}
