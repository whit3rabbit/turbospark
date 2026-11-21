import Foundation
import XCTest
@testable import TurboSparkApp

/// The fence info string parser behind the chat code block's file title and
/// marked lines. Anything it does not fully understand must come back as the
/// raw info string with no title and no marks, so the block renders exactly
/// as it did before the parser existed.
final class CodeFenceInfoTests: XCTestCase {
    private func parse(_ info: String?) -> CodeFenceInfo { CodeFenceInfo.parse(info) }

    func testNilAndPlainLanguagePassThrough() {
        XCTAssertEqual(parse(nil), CodeFenceInfo(language: nil, title: nil, markedLines: nil))
        XCTAssertEqual(parse(""), CodeFenceInfo(language: "", title: nil, markedLines: nil))
        XCTAssertEqual(parse("swift"), CodeFenceInfo(language: "swift", title: nil, markedLines: nil))
        XCTAssertEqual(parse("chart"), CodeFenceInfo(language: "chart", title: nil, markedLines: nil))
    }

    func testQuotedTitleAttribute() {
        XCTAssertEqual(
            parse(#"swift title="Sources/Foo Bar.swift""#),
            CodeFenceInfo(language: "swift", title: "Sources/Foo Bar.swift", markedLines: nil))
        XCTAssertEqual(
            parse("python filename='main.py'"),
            CodeFenceInfo(language: "python", title: "main.py", markedLines: nil))
        XCTAssertEqual(
            parse("js title=app.js"),
            CodeFenceInfo(language: "js", title: "app.js", markedLines: nil))
    }

    func testColonTitleForm() {
        XCTAssertEqual(
            parse("swift:Foo.swift"),
            CodeFenceInfo(language: "swift", title: "Foo.swift", markedLines: nil))
    }

    func testMarkedLines() {
        XCTAssertEqual(
            parse("swift {3,5-7}"),
            CodeFenceInfo(language: "swift", title: nil, markedLines: IndexSet([3, 5, 6, 7])))
        XCTAssertEqual(
            parse("ts{1, 4}"),
            CodeFenceInfo(language: "ts", title: nil, markedLines: IndexSet([1, 4])))
        XCTAssertEqual(
            parse(#"swift title="a.swift" {2}"#),
            CodeFenceInfo(language: "swift", title: "a.swift", markedLines: IndexSet([2])))
    }

    func testChartFenceWithTitleKeepsChartLanguage() {
        XCTAssertEqual(parse(#"chart title="Sales""#).language, "chart")
    }

    func testUnknownOrMalformedFallsBackToRaw() {
        let cases = [
            "swift foo=bar",
            "swift linenums",
            #"swift title="unterminated"#,
            "swift {3",
            "swift {7-5}",
            "swift {0}",
            "swift {a}",
            "swift {}",
            "swift {1} {2}",
            #"swift title="a" title="b""#,
            "swift title=",
            "swift:",
            ":Foo.swift",
            "swift {99999999}",
        ]
        for info in cases {
            XCTAssertEqual(
                parse(info), CodeFenceInfo(language: info, title: nil, markedLines: nil),
                "expected raw fallback for \(info)")
        }
    }

    func testLinesSplitKeepsAttributesAndDropsTrailingNewline() {
        var text = AttributedString("let a = 1\r\nlet b = 2\n\nend\n")
        text.inlinePresentationIntent = .code
        let lines = CodeFenceInfo.lines(of: text)
        XCTAssertEqual(lines.map { String($0.characters) }, ["let a = 1", "let b = 2", "", "end"])
        XCTAssertEqual(lines.first?.inlinePresentationIntent, .code)
    }
}
