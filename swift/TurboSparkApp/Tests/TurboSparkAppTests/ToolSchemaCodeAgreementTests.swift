import XCTest

@testable import TurboSparkApp

/// Pins the model-facing schemas of a few tools to what their payload types
/// and executors actually read, so an advertised field cannot drift from the
/// code again (issue C2).
final class ToolSchemaCodeAgreementTests: XCTestCase {
    private func props(_ tool: OpenAITool) -> Set<String> {
        Set(tool.function.parameters.properties?.keys.map { $0 } ?? [])
    }

    /// Encodes the decoded payload back to JSON keys. A field the type does
    /// not model is dropped by the decoder, so it never reappears here.
    private func roundTripKeys<T: Codable>(_ type: T.Type, _ json: String) throws -> Set<String> {
        let value = try JSONDecoder().decode(T.self, from: Data(json.utf8))
        let data = try JSONEncoder().encode(value)
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        return Set(object.keys)
    }

    func testExitWorktreeSchemaPathIsADecodableField() throws {
        XCTAssertTrue(props(ArtifactWorktreeDefinitions.exitWorktree).contains("path"))
        let keys = try roundTripKeys(
            ExitWorktreeInput.self, #"{"action":"remove","path":"wt","discard_changes":true}"#)
        XCTAssertEqual(keys, props(ArtifactWorktreeDefinitions.exitWorktree))
    }

    func testEnterWorktreeSchemaDescribesCreationNotAnExistingWorktree() {
        let desc = ArtifactWorktreeDefinitions.enterWorktree.function.parameters.properties?["path"]?.description ?? ""
        XCTAssertFalse(desc.lowercased().contains("existing git worktree"), desc)
        XCTAssertTrue(desc.lowercased().contains("create"), desc)
    }

    func testPushNotificationSchemaAdvertisesTitle() throws {
        let schema = MonitoringNotificationDefinitions.pushNotification
        XCTAssertTrue(props(schema).contains("title"))
        XCTAssertFalse(schema.function.parameters.required?.contains("title") ?? false)
        let keys = try roundTripKeys(
            PushNotificationInput.self, #"{"message":"m","title":"t","status":"proactive"}"#)
        XCTAssertEqual(keys, props(schema))
    }

    func testReportFindingsItemSchemaAdvertisesFieldsTheExecutorRenders() throws {
        let findings = PlanningInteractiveToolDefinitions.reportFindings
            .function.parameters.properties?["findings"]
        let itemProps = Set(findings?.items?.value.properties?.keys.map { $0 } ?? [])
        XCTAssertTrue(itemProps.isSuperset(of: ["category", "verdict"]), "\(itemProps)")
        // Every advertised item field must survive decoding.
        let json = #"{"file":"f","line":1,"summary":"s","failure_scenario":"x","category":"c","verdict":"v"}"#
        let keys = try roundTripKeys(CodeFindingItem.self, json)
        XCTAssertTrue(keys.isSuperset(of: itemProps), "\(keys) vs \(itemProps)")
        let out = try ReportFindingsExecutor.execute(arguments: [
            "findings": "[\(json)]"
        ])
        XCTAssertTrue(out.contains("[c]") && out.contains("Verdict: v"), out)
    }

    func testGrepSchemaOmitsDecodedButIgnoredOffsetAndMultiline() throws {
        let names = props(FileSearchToolDefinitions.grep)
        XCTAssertTrue(names.isDisjoint(with: ["offset", "multiline"]), "\(names)")
        // Still decoded, so a model that sends them does not fail.
        let input = try JSONDecoder().decode(
            GrepInput.self, from: Data(#"{"pattern":"x","offset":3,"multiline":true}"#.utf8))
        XCTAssertEqual(input.offset, 3)
    }

    func testAgentIsolationIsToleratedButNotAdvertised() throws {
        XCTAssertFalse(props(AgentToolDefinitions.agent).contains("isolation"))
        let input = try JSONDecoder().decode(
            AgentInput.self,
            from: Data(#"{"description":"d","prompt":"p","isolation":"worktree"}"#.utf8))
        XCTAssertEqual(input.isolation, "worktree")
    }

    func testWebFetchPromptIsToleratedButNotAdvertised() throws {
        XCTAssertFalse(props(WebToolDefinitions.webFetch).contains("prompt"))
        let input = try JSONDecoder().decode(
            WebFetchInput.self, from: Data(#"{"url":"https://example.com","prompt":"x"}"#.utf8))
        XCTAssertEqual(input.url, "https://example.com")
    }

    func testBatchDoesNotClaimEverythingRunsConcurrently() {
        let fn = BatchToolDefinitions.batch.function
        XCTAssertFalse(fn.description.contains("Executes all calls concurrently"))
        XCTAssertFalse(fn.description.contains("in parallel (up to"))
        XCTAssertTrue(fn.description.contains("one at a time"))
        let calls = fn.parameters.properties?["tool_calls"]?.description ?? ""
        XCTAssertFalse(calls.contains("execute concurrently"), calls)
        XCTAssertTrue(calls.contains("serialized"), calls)
    }
}
