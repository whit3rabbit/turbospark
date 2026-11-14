import Foundation
import XCTest

@testable import TurboSparkApp

/// ZP-06 Gen UI: the closed schema is the security boundary, so each refusal
/// is pinned. There is no web view on this path, so there is nothing to load,
/// and these tests prove that markup stays inert data.
final class GenUiTests: XCTestCase {
    private func parse(_ object: [String: Any]) -> Result<GenUiSpec, GenUiError> {
        let data = try! JSONSerialization.data(withJSONObject: object)
        return GenUiParser.parse(String(decoding: data, as: UTF8.self))
    }

    private let slider: [String: Any] = ["type": "slider", "id": "rate", "label": "Rate", "min": 0, "max": 10, "step": 0.5, "value": 5, "unit": "%"]

    private func spec(_ components: [[String: Any]]) -> [String: Any] {
        ["title": "T", "components": components]
    }

    private func failure(_ result: Result<GenUiSpec, GenUiError>) -> String? {
        if case .failure(let error) = result { return error.message }
        return nil
    }

    func testAValidCardParsesEveryComponentKind() throws {
        let result = parse(spec([
            ["type": "text", "text": "Hello", "style": "heading"],
            slider,
            ["type": "stepper", "id": "n", "label": "N", "min": 1, "max": 9, "step": 1, "value": 2],
            ["type": "toggle", "id": "flag", "label": "Flag", "value": true],
            ["type": "picker", "id": "mode", "label": "Mode", "options": ["a", "b"], "value": 1],
            ["type": "readout", "label": "Rate", "bind": "rate", "unit": "%"],
            ["type": "table", "header": ["a", "b"], "rows": [["x", 1], ["y", 2.5]]],
            ["type": "bars", "title": "B", "items": [["label": "a", "value": 3], ["label": "b", "value": 1]]],
            ["type": "follow_up", "label": "Ask", "template": "rate {rate}, mode {mode}, flag {flag}, n {n}"],
        ]))
        let parsed = try result.get()
        XCTAssertEqual(parsed.components.count, 9)
        XCTAssertEqual(Set(parsed.controlIDs), ["rate", "n", "flag", "mode"])
    }

    func testUnknownTypeAndUnknownFieldsRejectTheWholeSpec() {
        XCTAssertNotNil(failure(parse(spec([["type": "html", "html": "<b>x</b>"]]))))
        XCTAssertNotNil(failure(parse(spec([["type": "webview", "url": "https://example.com"]]))))
        var withExtra = slider
        withExtra["onchange"] = "alert(1)"
        XCTAssertTrue(failure(parse(spec([withExtra])))?.contains("unknown field 'onchange'") == true)
        XCTAssertNotNil(failure(parse(["title": "T", "components": [slider], "script": "x"])))
    }

    func testMarkupAndScriptStayInertText() throws {
        let hostile = "<script>alert(1)</script><img src=x onerror=alert(2)> [link](https://evil.example) \u{1b}[31m"
        let parsed = try parse(spec([["type": "text", "text": hostile]])).get()
        guard case .text(let text, _) = parsed.components[0] else { return XCTFail("not text") }
        XCTAssertFalse(text.contains("\u{1b}"), "control characters are dropped")
        // The string survives as plain data; the renderer draws it with a
        // verbatim Text, which has no markup or link handling at all.
        XCTAssertTrue(text.contains("<script>"))
    }

    func testBoundsAreEnforced() {
        XCTAssertEqual(GenUiParser.parse(String(repeating: " ", count: GenUiLimits.maxSpecBytes + 1)), .failure(.tooLarge))
        let many = (0..<(GenUiLimits.maxComponents + 1)).map { _ in ["type": "text", "text": "x"] as [String: Any] }
        XCTAssertNotNil(failure(parse(spec(many))))
        var steps = slider
        steps["step"] = 0.0001
        XCTAssertTrue(failure(parse(spec([steps])))?.contains("steps") == true)
        var outside = slider
        outside["value"] = 11
        XCTAssertNotNil(failure(parse(spec([outside]))))
        var inverted = slider
        inverted["min"] = 10
        inverted["max"] = 0
        XCTAssertNotNil(failure(parse(spec([inverted]))))
        let longText = String(repeating: "a", count: GenUiLimits.maxTextLength + 1)
        XCTAssertNotNil(failure(parse(spec([["type": "text", "text": longText]]))))
        let rows = (0..<(GenUiLimits.maxTableRows + 1)).map { _ in ["a"] }
        XCTAssertNotNil(failure(parse(spec([["type": "table", "header": ["h"], "rows": rows]]))))
    }

    func testIdsAreUniqueWellFormedAndBound() {
        XCTAssertNotNil(failure(parse(spec([slider, slider]))))
        var bad = slider
        bad["id"] = "Bad Id"
        XCTAssertNotNil(failure(parse(spec([bad]))))
        XCTAssertNotNil(failure(parse(spec([["type": "readout", "label": "x", "bind": "nothing"]]))))
        XCTAssertNotNil(failure(parse(["title": "T", "components": []])))
        XCTAssertEqual(GenUiParser.parse("[1]"), .failure(.notJSON))
        XCTAssertEqual(GenUiParser.parse("not json"), .failure(.notJSON))
    }

    func testFollowUpTemplateIsPlainSubstitutionAndClipped() throws {
        let parsed = try parse(spec([slider, ["type": "toggle", "id": "flag", "label": "F", "value": true],
                                     ["type": "follow_up", "label": "Ask", "template": "r={rate} f={flag} u={unknown} {rate}{rate}"]])).get()
        var values = GenUiValues(parsed)
        values.numbers["rate"] = 7.5
        XCTAssertEqual(values.expand("r={rate} f={flag} u={unknown}", in: parsed), "r=7.5 f=on u={unknown}")
        let long = String(repeating: "{rate}", count: 1_000)
        XCTAssertLessThanOrEqual(values.expand(long, in: parsed).count, GenUiLimits.maxFollowUpLength)
    }

    func testExecutorReportsAcceptanceAndNamesTheFailingPath() throws {
        let good = String(decoding: try JSONSerialization.data(withJSONObject: spec([slider])), as: UTF8.self)
        XCTAssertTrue(try GenUiRenderExecutor.execute(arguments: ["spec": good]).contains("1 components"))
        let bad = String(decoding: try JSONSerialization.data(withJSONObject: spec([["type": "html"]])), as: UTF8.self)
        XCTAssertThrowsError(try GenUiRenderExecutor.execute(arguments: ["spec": bad])) {
            XCTAssertTrue($0.localizedDescription.contains("components[0]"))
        }
        XCTAssertThrowsError(try GenUiRenderExecutor.execute(arguments: [:]))
    }

    func testToolSitsAtTheSafeAutomationRungWithNoToolAuthority() {
        XCTAssertEqual(AppToolCatalog.builtInCategory(for: "gen_ui_render"), .automation)
        let assessment = ToolRiskClassifier.assessRisk(name: "gen_ui_render", arguments: ["spec": "{}"])
        XCTAssertEqual(assessment.level, .safe)
        XCTAssertTrue(AppToolRegistry.supportedToolNames.contains("gen_ui_render"))
        XCTAssertFalse(AppToolRegistry.workspaceRootedToolNames.contains("gen_ui_render"), "works without a project")
        XCTAssertEqual(ToolPresentation.resolve("gen_ui_render").label, "Show Card")
    }
}
