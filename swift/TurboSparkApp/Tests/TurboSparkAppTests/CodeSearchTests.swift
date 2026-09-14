import XCTest
@testable import TurboSparkApp

final class CodeSearchTests: XCTestCase {
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
