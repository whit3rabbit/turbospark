import XCTest

@testable import TurboSparkApp

/// The end-of-turn card of files a turn touched (component audit,
/// file-tree). Aggregation is pure: only calls that ran without error
/// count, a path appears once, and the tree folds single-child folders.
final class TurnFileTreeTests: XCTestCase {

    private let root = "/Users/me/proj"

    private func turn(_ steps: [(name: String, args: [String: String], output: String, isError: Bool)]) -> [AppChatMessage] {
        steps.map { step in
            let call = AppToolCall(name: step.name, arguments: step.args, status: .completed)
            let result = AppToolResult(callID: call.id, output: step.output, isError: step.isError)
            return AppChatMessage(role: .assistant, content: "", toolCalls: [call], toolResults: [result])
        }
    }

    private func ok(_ name: String, _ args: [String: String], output: String = "ok") -> (String, [String: String], String, Bool) {
        (name, args, output, false)
    }

    private func summary(_ tree: TurnFileTree?) -> [String] {
        (tree?.rows ?? []).map { row in
            switch row {
            case .directory(let name, let depth):
                return String(repeating: " ", count: depth) + name + "/"
            case .file(let entry, let name, let depth):
                var line = String(repeating: " ", count: depth) + name + " \(entry.change)"
                if let a = entry.additions, let d = entry.deletions { line += " +\(a) -\(d)" }
                return line
            }
        }
    }

    // MARK: - Threshold

    func testOneFileGivesNoCard() {
        let messages = turn([ok("edit_file", ["path": "a.swift", "old_string": "x", "new_string": "y"])])
        XCTAssertNil(TurnFileTree.make(messages: messages, rootPath: root))
    }

    func testReadsAloneGiveNoCard() {
        let messages = turn([
            ok("read_file", ["path": "a.swift"]),
            ok("read_file", ["path": "b.swift"]),
        ])
        XCTAssertNil(TurnFileTree.make(messages: messages, rootPath: root))
    }

    func testOneChangeAndOneReadGiveACard() throws {
        let messages = turn([
            ok("read_file", ["path": "b.swift"]),
            ok("edit_file", ["path": "a.swift", "old_string": "x", "new_string": "y\nz"]),
        ])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(tree.changedCount, 1)
        XCTAssertEqual(tree.readCount, 1)
        XCTAssertEqual(summary(tree), ["a.swift modified +2 -1", "b.swift read"])
    }

    // MARK: - Only calls that ran

    func testFailedAndUnansweredCallsTouchNothing() {
        var messages = turn([
            ok("edit_file", ["path": "a.swift", "old_string": "x", "new_string": "y"]),
            ("write_file", ["path": "b.swift", "content": "z"], "denied", true),
        ])
        // A call with no result yet (pending approval) did not run either.
        messages.append(AppChatMessage(
            role: .assistant, content: "",
            toolCalls: [AppToolCall(name: "write_file", arguments: ["path": "c.swift", "content": "q"])]))
        XCTAssertNil(TurnFileTree.make(messages: messages, rootPath: root))
    }

    func testConflictingPathArgumentsAreSkipped() {
        let messages = turn([
            ok("edit_file", ["path": "a.swift", "old_string": "x", "new_string": "y"]),
            ok("write_file", ["path": "b.swift", "file_path": "c.swift", "content": "z"]),
        ])
        XCTAssertNil(TurnFileTree.make(messages: messages, rootPath: root))
    }

    // MARK: - Dedupe and merge

    func testRepeatedEditsMergeCountsUnderOnePath() throws {
        let messages = turn([
            ok("edit_file", ["path": "src/a.swift", "old_string": "x", "new_string": "y\nz"]),
            ok("edit_file", ["path": "./src/a.swift", "old_string": "p\nq", "new_string": "r"]),
            ok("edit_file", ["path": "\(root)/src/b.swift", "old_string": "x", "new_string": "y"]),
        ])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(tree.entries.map(\.path), ["src/a.swift", "src/b.swift"])
        XCTAssertEqual(summary(tree), ["src/", " a.swift modified +3 -3", " b.swift modified +1 -1"])
    }

    func testReadThenWriteIsModifiedAndAWriteAloneIsWritten() throws {
        let messages = turn([
            ok("read_file", ["path": "a.txt"]),
            ok("write_file", ["path": "a.txt", "content": "new"]),
            ok("write_file", ["path": "b.txt", "content": "new"]),
        ])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(tree.entries.map(\.change), [.modified, .written])
        // A whole-file write is not a diff, so it has no counts.
        XCTAssertNil(tree.entries[0].additions)
        XCTAssertNil(tree.entries[1].additions)
    }

    func testAnUnknownCountMakesTheMergedCountUnknown() throws {
        let messages = turn([
            ok("edit_file", ["path": "a.txt", "old_string": "x", "new_string": "y"]),
            ok("write_file", ["path": "a.txt", "content": "z"]),
            ok("edit_file", ["path": "b.txt", "old_string": "x", "new_string": "y"]),
        ])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(tree.entries[0].change, .modified)
        XCTAssertNil(tree.entries[0].additions)
        XCTAssertNil(tree.entries[0].deletions)
    }

    func testAReadAfterAChangeKeepsTheChangeAndItsCall() throws {
        let messages = turn([
            ok("edit_file", ["path": "a.txt", "old_string": "x", "new_string": "y"]),
            ok("read_file", ["path": "a.txt"]),
            ok("read_file", ["path": "b.txt"]),
        ])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(tree.entries[0].change, .modified)
        XCTAssertEqual(tree.entries[0].origin.messageID, messages[0].id)
        XCTAssertEqual(tree.entries[0].origin.callID, messages[0].toolCalls[0].id)
    }

    func testCreatedThenEditedStaysAdded() throws {
        let messages = turn([
            ok("document_write", ["path": "docs/r.docx", "spec": "{}"]),
            ok("edit_file", ["path": "notes.md", "old_string": "x", "new_string": "y"]),
        ])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(summary(tree), ["docs/", " r.docx added", "notes.md modified +1 -1"])
    }

    // MARK: - multiedit and apply_patch

    func testMultieditCountsPerFile() throws {
        let edits = #"[{"file_path":"a.swift","old_string":"x","new_string":"y\nz"},"#
            + #"{"file_path":"b.swift","old_string":"p\nq","new_string":"r"},"#
            + #"{"file_path":"a.swift","old_string":"m","new_string":"n"}]"#
        let messages = turn([ok("multiedit", ["edits": edits])])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(summary(tree), ["a.swift modified +3 -2", "b.swift modified +1 -2"])
    }

    func testApplyPatchReadsTheOperationsThatRan() throws {
        let output = "Applied patch to 3 file(s):\n  ADD new.txt\n  UPDATE src/old.txt\n  DELETE gone.txt"
        let messages = turn([ok("apply_patch", ["patch_text": "irrelevant"], output: output)])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        // Several files in one patch: per-file counts are not known.
        XCTAssertEqual(summary(tree), ["src/", " old.txt modified", "gone.txt deleted", "new.txt added"])
        XCTAssertEqual(tree.changedCount, 3)
    }

    func testSingleFilePatchUsesThePatchCounts() throws {
        let patch = "--- a/x.txt\n+++ b/x.txt\n@@ -1,2 +1,2 @@\n-a\n+b\n+c\n keep"
        let output = "Applied patch to 1 file(s):\n  UPDATE x.txt"
        let messages = turn([
            ok("apply_patch", ["patch_text": patch], output: output),
            ok("read_file", ["path": "y.txt"]),
        ])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(summary(tree), ["x.txt modified +2 -1", "y.txt read"])
    }

    // MARK: - Tree and paths

    func testSingleChildFoldersAreCompressedAndFoldersComeFirst() throws {
        let messages = turn([
            ok("edit_file", ["path": "Sources/App/Views/a.swift", "old_string": "x", "new_string": "y"]),
            ok("edit_file", ["path": "Sources/App/Views/b.swift", "old_string": "x", "new_string": "y"]),
            ok("edit_file", ["path": "README.md", "old_string": "x", "new_string": "y"]),
        ])
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(summary(tree), [
            "Sources/App/Views/",
            " a.swift modified +1 -1",
            " b.swift modified +1 -1",
            "README.md modified +1 -1",
        ])
    }

    func testPathsOutsideTheRootStayAbsolute() {
        XCTAssertEqual(TurnFileTree.displayPath("/etc/hosts", rootPath: root), "/etc/hosts")
        XCTAssertEqual(TurnFileTree.displayPath("\(root)/a//b/./c.txt", rootPath: root), "a/b/c.txt")
        XCTAssertEqual(TurnFileTree.displayPath("/Users/me/project2/a.txt", rootPath: root), "/Users/me/project2/a.txt")
        XCTAssertEqual(TurnFileTree.displayPath("./a.txt", rootPath: nil), "a.txt")
        XCTAssertNil(TurnFileTree.displayPath("  ", rootPath: root))
        XCTAssertNil(TurnFileTree.displayPath(root, rootPath: root))
    }

    func testUserMessagesAndOtherToolsAreIgnored() throws {
        var messages = turn([
            ok("run_command", ["command": "rm a.txt"]),
            ok("edit_file", ["path": "a.txt", "old_string": "x", "new_string": "y"]),
            ok("edit_file", ["path": "b.txt", "old_string": "x", "new_string": "y"]),
        ])
        messages.insert(AppChatMessage(role: .user, content: "go"), at: 0)
        let tree = try XCTUnwrap(TurnFileTree.make(messages: messages, rootPath: root))
        XCTAssertEqual(tree.entries.map(\.path), ["a.txt", "b.txt"])
    }
}
