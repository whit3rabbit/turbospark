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
        XCTAssertEqual(summary.additions, 4)
        XCTAssertNil(summary.deletions)
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
}
