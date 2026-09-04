import XCTest

@testable import TurboSparkApp

/// The parser had NO tests of its own while it existed twice, which is part
/// of why the two copies could drift: a case written against one of them
/// proved nothing about the other, so nobody wrote one against either.
/// Extracting it makes the seam reachable, and this is what it does.
final class ToolCallParserTests: XCTestCase {
    func testEveryXMLBlockIsReturnedInTheOrderItAppears() {
        let reply = """
        first
        <tool_call><name>read_file</name><arguments>{"path": "a.txt"}</arguments></tool_call>
        then
        <tool_call><name>read_file</name><arguments>{"path": "b.txt"}</arguments></tool_call>
        """
        let calls = ToolCallParser.parse(from: reply)
        XCTAssertEqual(calls.count, 2, "The loop runs one and RECORDS the rest (state#37).")
        XCTAssertEqual(calls.map { $0.arguments["path"] }, ["a.txt", "b.txt"])
    }

    func testTheMarkdownFormIsAFallbackAndNotASecondPass() {
        let both = """
        <tool_call><name>read_file</name><arguments>{"path": "a.txt"}</arguments></tool_call>
        ```tool_call
        {"name": "read_file", "arguments": {"path": "a.txt"}}
        ```
        """
        XCTAssertEqual(
            ToolCallParser.parse(from: both).count, 1,
            "A reply carrying both is one that fenced its own XML; reading it twice would run "
                + "the same call twice.")

        let markdownOnly = """
        ```json_tool_call
        {"tool": "run_command", "arguments": {"command": "ls"}}
        ```
        """
        let fallback = ToolCallParser.parse(from: markdownOnly)
        XCTAssertEqual(fallback.count, 1, "And the fallback still fires when it is the only form.")
        XCTAssertEqual(fallback.first?.name, "run_command", "`tool` is accepted beside `name`.")
    }

    func testABlockWithNoNameIsNotACall() {
        let reply = "<tool_call><arguments>{\"path\": \"a.txt\"}</arguments></tool_call>"
        XCTAssertTrue(
            ToolCallParser.parse(from: reply).isEmpty,
            "A call with no tool to invoke is not a call.")
    }

    func testAParsedCallStartsAtPendingApproval() {
        let reply = "<tool_call><name>run_command</name><arguments>{\"command\": \"ls\"}</arguments></tool_call>"
        let call = ToolCallParser.parse(from: reply).first
        XCTAssertEqual(
            call?.status, .pendingApproval,
            "A call that has not been through the permission engine is not an approved one.")
        XCTAssertEqual(call?.category, .terminal, "And it is classified as it is parsed.")
        XCTAssertNotNil(call?.riskAssessment)
    }

    func testNonStringArgumentValuesSurviveAsStrings() {
        let reply = """
        <tool_call><name>read_file</name>
        <arguments>{"path": "a.txt", "limit": 40, "replace_all": true}</arguments></tool_call>
        """
        let arguments = ToolCallParser.parse(from: reply).first?.arguments ?? [:]
        // What every downstream reader is written against: `Int(...)` on the
        // one and `== "true"` on the other.
        XCTAssertEqual(arguments["limit"], "40")
        XCTAssertEqual(arguments["replace_all"], "1")
        XCTAssertEqual(arguments["path"], "a.txt")
    }
}
