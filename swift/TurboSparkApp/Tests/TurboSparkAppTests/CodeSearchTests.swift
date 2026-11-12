import XCTest
@testable import TurboSparkApp

final class CodeSearchTests: XCTestCase {
    private var previousProvider: (@Sendable @MainActor () -> Bool)?

    // Every `AppModel` installs itself as the static web-tools provider, so
    // whichever one an earlier test left alive decides whether a web tool is
    // "disabled" before it ever validates its arguments. These tests are about
    // the executor, not that switch.
    override func setUp() async throws {
        try await super.setUp()
        previousProvider = AppToolRegistry.webToolsEnabledProvider
        AppToolRegistry.webToolsEnabledProvider = nil
    }

    override func tearDown() async throws {
        AppToolRegistry.webToolsEnabledProvider = previousProvider
        try await super.tearDown()
    }

    func testCodeSearchRejectsEmptyQuery() async {
        let call = AppToolCall(
            name: "codesearch",
            arguments: ["query": "   "],
            category: .web
        )
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertTrue(result.isError)
        XCTAssertTrue(result.output.contains("Missing required 'query'"))
    }

    func testCodeSearchAdvertisedAndCategorizedCorrectly() {
        let category = AppToolCatalog.category(for: "codesearch")
        XCTAssertEqual(category, .web)

        let aliasCategory = AppToolCatalog.category(for: "code_search")
        XCTAssertEqual(aliasCategory, .web)

        let risk = ToolRiskClassifier.assessRisk(name: "codesearch", arguments: ["query": "Swift async"])
        XCTAssertEqual(risk.level, .safe)
    }
}
