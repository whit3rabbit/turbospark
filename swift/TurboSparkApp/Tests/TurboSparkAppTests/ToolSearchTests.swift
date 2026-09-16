import XCTest

@testable import TurboSparkApp

final class ToolSearchTests: XCTestCase {
    private let descriptors = [
        DeferredToolDescriptor(
            name: "mcp__github__create_issue",
            serverName: "github",
            toolName: "create_issue",
            description: "Create an issue in a repository.",
            inputSchemaJSON: #"{"type":"object","properties":{"repo":{"type":"string"},"title":{"type":"string"}},"required":["repo","title"]}"#),
        DeferredToolDescriptor(
            name: "mcp__slack__send_message",
            serverName: "slack",
            toolName: "send_message",
            description: "Send a message to a Slack channel.",
            inputSchemaJSON: #"{"type":"object","properties":{"channel":{"type":"string"},"text":{"type":"string"}},"required":["channel","text"]}"#),
    ]

    func testBridgeDefinitionsAreRegisteredAndImplemented() {
        XCTAssertEqual(
            ToolSearchToolDefinitions.all.map { $0.function.name },
            ["tool_search", "tool_describe", "tool_call"])
        for name in ["tool_search", "tool_describe", "tool_call"] {
            XCTAssertTrue(AppToolRegistry.isImplemented(name))
        }

        let autonomousNames = AppToolCatalog.tools(for: .autonomous).map { $0.function.name }
        for name in ["tool_search", "tool_describe", "tool_call"] {
            XCTAssertEqual(autonomousNames.filter { $0 == name }.count, 1)
        }
        XCTAssertEqual(AppToolCatalog.category(for: "tool_search"), .fileRead)
        XCTAssertEqual(AppToolCatalog.category(for: "tool_describe"), .fileRead)
        XCTAssertEqual(AppToolCatalog.category(for: "tool_call"), .mcp)
    }

    func testDeferredMcpPreHookContinueFalseStopsAnOtherwiseAllowedCall() throws {
        let decision = AppHookPreToolUseDecision(
            behavior: .allow,
            preventContinuation: true,
            continuationStopReason: "production blocked")

        let stop = try XCTUnwrap(AppToolRegistry.deferredMcpPreHookStop(decision))
        XCTAssertTrue(stop.isError)
        XCTAssertEqual(stop.reason, "production blocked")
        XCTAssertEqual(stop.output, "Deferred MCP call blocked by hook: production blocked")
    }

    func testSearchRunsEachQueryIndependentlyAndStemsTerms() throws {
        let output = ToolSearchCatalog.search(
            queries: ["issues", "send message"], descriptors: descriptors)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any])
        let results = try XCTUnwrap(json["results"] as? [[String: Any]])
        XCTAssertEqual(results.count, 2)
        XCTAssertEqual(results[0]["matches"] as? [String], ["mcp__github__create_issue"])
        XCTAssertEqual(results[1]["matches"] as? [String], ["mcp__slack__send_message"])
        let tools = try XCTUnwrap(json["tools"] as? [String: Any])
        XCTAssertTrue(tools["mcp__github__create_issue"] != nil)
        XCTAssertTrue(tools["mcp__slack__send_message"] != nil)
    }

    func testSearchFiltersLongQueriesThatOnlyMatchAnIncidentalTool() throws {
        let output = ToolSearchCatalog.search(
            queries: ["create issue Slack channel text"],
            descriptors: descriptors)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any])
        let results = try XCTUnwrap(json["results"] as? [[String: Any]])
        XCTAssertEqual(results[0]["matches"] as? [String], ["mcp__slack__send_message"])
        XCTAssertFalse((results[0]["matches"] as? [String] ?? []).contains("mcp__github__create_issue"))
    }

    func testDescribeReturnsFullSchemaAndReportsUnknownNames() throws {
        let output = ToolSearchCatalog.describe(
            names: ["mcp__github__create_issue", "mcp__missing__tool"],
            descriptors: descriptors)
        let json = try XCTUnwrap(JSONSerialization.jsonObject(with: Data(output.utf8)) as? [String: Any])
        let tools = try XCTUnwrap(json["tools"] as? [String: Any])
        let issue = try XCTUnwrap(tools["mcp__github__create_issue"] as? [String: Any])
        let parameters = try XCTUnwrap(issue["parameters"] as? [String: Any])
        XCTAssertEqual(parameters["type"] as? String, "object")
        XCTAssertEqual(issue["required"] as? [String], ["repo", "title"])
        XCTAssertEqual(json["not_found"] as? [String], ["mcp__missing__tool"])
    }

    func testPromptListingIsBoundedAndContainsTheDiscoveryInstructions() {
        let listing = ToolSearchCatalog.promptListing(
            descriptors: descriptors, contextTokens: 1000)
        XCTAssertTrue(listing.contains("tool_search"))
        XCTAssertTrue(listing.contains("mcp__github__create_issue"))
        XCTAssertLessThanOrEqual(listing.count, 8000)
    }
}
