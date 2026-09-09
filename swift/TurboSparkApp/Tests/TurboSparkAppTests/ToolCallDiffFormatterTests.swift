import Foundation
import XCTest
@testable import TurboSpark
@testable import TurboSparkApp

final class ToolCallDiffFormatterTests: XCTestCase {
    func testReplaceFileContentSummary() {
        let args = [
            "TargetFile": "/path/to/messages.rs",
            "TargetContent": "line 1\nline 2\nline 3",
            "ReplacementContent": "line 1\nline 2\nline 3\nline 4\nline 5\nline 6\nline 7"
        ]

        let summary = ToolCallDiffFormatter.summarize(callName: "replace_file_content", arguments: args)
        XCTAssertEqual(summary.action, "Edited")
        XCTAssertEqual(summary.target, "messages.rs")
        XCTAssertEqual(summary.deletions, 3)
        XCTAssertEqual(summary.additions, 7)
        XCTAssertNil(summary.lineRange)
    }

    func testWriteToFileSummary() {
        let args = [
            "TargetFile": "src/lib.rs",
            "CodeContent": "use std::io;\n\npub fn hello() {}\n"
        ]

        let summary = ToolCallDiffFormatter.summarize(callName: "write_to_file", arguments: args)
        XCTAssertEqual(summary.action, "Wrote")
        XCTAssertEqual(summary.target, "lib.rs")
        // Three rendered lines; the trailing newline does not start a
        // fourth, phantom one.
        XCTAssertEqual(summary.additions, 3)
        XCTAssertNil(summary.deletions)
    }

    func testCountLinesTrailingNewlineAndEmptyBodies() {
        // Direct pins on the two shapes the old `max(1, components)` count
        // got wrong: an empty body read 1, and a trailing newline added a
        // line no editor would render.
        XCTAssertEqual(ToolCallDiffFormatter.countLines(""), 0)
        XCTAssertEqual(ToolCallDiffFormatter.countLines("a"), 1)
        XCTAssertEqual(ToolCallDiffFormatter.countLines("a\n"), 1)
        XCTAssertEqual(ToolCallDiffFormatter.countLines("a\nb"), 2)
        XCTAssertEqual(ToolCallDiffFormatter.countLines("a\nb\n"), 2)
    }

    func testPureInsertionEditHidesItsDeletionBadge() {
        let summary = ToolCallDiffFormatter.summarize(
            callName: "edit_file",
            arguments: [
                "file_path": "Sources/App.swift",
                "old_string": "",
                "new_string": "let x = 1\n"
            ]
        )
        XCTAssertNil(summary.deletions, "an empty old_string removes nothing; no -0 badge")
        XCTAssertEqual(summary.additions, 1)
    }

    func testViewFileSummaryWithLineRange() {
        let args = [
            "AbsolutePath": "/Users/test/responses.rs",
            "StartLine": "563",
            "EndLine": "722"
        ]

        let summary = ToolCallDiffFormatter.summarize(callName: "view_file", arguments: args)
        XCTAssertEqual(summary.action, "Read")
        XCTAssertEqual(summary.target, "responses.rs")
        XCTAssertEqual(summary.lineRange, "(563-722)")
    }

    func testRunCommandSummary() {
        let args = [
            "CommandLine": "cargo test -p turbospark-gpu"
        ]

        let summary = ToolCallDiffFormatter.summarize(callName: "run_command", arguments: args)
        XCTAssertEqual(summary.action, "Ran")
        XCTAssertEqual(summary.target, "cargo test -p turbospark-gpu")
    }

    func testGrepSearchSummary() {
        let args = [
            "Query": "ToolCallCardView"
        ]

        let summary = ToolCallDiffFormatter.summarize(callName: "grep_search", arguments: args)
        XCTAssertEqual(summary.action, "Searched")
        XCTAssertEqual(summary.target, "\"ToolCallCardView\"")
    }

    func testApplyPatchSummary() {
        let patch = """
        --- a/src/main.rs
        +++ b/src/main.rs
        @@ -1,3 +1,4 @@
        -old line
        +new line 1
        +new line 2
         common line
        """
        let summary = ToolCallDiffFormatter.summarize(
            callName: "apply_patch",
            arguments: ["patch": patch]
        )
        XCTAssertEqual(summary.action, "Patched")
        XCTAssertEqual(summary.target, "main.rs")
        XCTAssertEqual(summary.additions, 2)
        XCTAssertEqual(summary.deletions, 1)
    }

    func testWebFetchSummary() {
        let summary = ToolCallDiffFormatter.summarize(
            callName: "web_fetch",
            arguments: ["url": "https://api.github.com/repos/turbospark/releases"]
        )
        XCTAssertEqual(summary.action, "Fetched")
        XCTAssertTrue(summary.target.contains("api.github.com"))
    }

    func testListDirectorySummary() {
        let rootSummary = ToolCallDiffFormatter.summarize(
            callName: "list_directory",
            arguments: ["path": "."]
        )
        XCTAssertEqual(rootSummary.action, "Listed")
        XCTAssertEqual(rootSummary.target, "workspace")

        let subSummary = ToolCallDiffFormatter.summarize(
            callName: "ls",
            arguments: ["path": "swift/TurboSparkApp"]
        )
        XCTAssertEqual(subSummary.action, "Listed")
        XCTAssertEqual(subSummary.target, "TurboSparkApp")
    }

    func testSkillSummary() {
        let summary = ToolCallDiffFormatter.summarize(
            callName: "skill",
            arguments: ["name": "refactor-clean"]
        )
        XCTAssertEqual(summary.action, "Skill")
        XCTAssertEqual(summary.target, "refactor-clean")
    }

    func testMcpToolSummary() {
        let summary = ToolCallDiffFormatter.summarize(
            callName: "call_mcp_tool",
            arguments: ["server": "context7", "toolName": "resolve-library-id"]
        )
        XCTAssertEqual(summary.action, "MCP: context7")
        XCTAssertEqual(summary.target, "resolve-library-id")

        let prefixedSummary = ToolCallDiffFormatter.summarize(
            callName: "mcp__github__get_issue",
            arguments: [:]
        )
        XCTAssertEqual(prefixedSummary.action, "MCP: github")
        XCTAssertEqual(prefixedSummary.target, "get_issue")
    }

    func testEditFileWithStandardKeys() {
        let summary = ToolCallDiffFormatter.summarize(
            callName: "edit_file",
            arguments: [
                "file_path": "Sources/App.swift",
                "old_string": "let x = 1\n",
                "new_string": "let x = 2\nlet y = 3\n"
            ]
        )
        XCTAssertEqual(summary.action, "Edited")
        XCTAssertEqual(summary.target, "App.swift")
        XCTAssertEqual(summary.deletions, 1)
        XCTAssertEqual(summary.additions, 2)
    }
}
