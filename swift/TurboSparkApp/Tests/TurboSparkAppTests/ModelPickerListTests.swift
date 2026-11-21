import XCTest
import TurboSpark
@testable import TurboSparkApp

/// The chat model picker's filter, grouping, ranking and keyboard highlight.
final class ModelPickerListTests: XCTestCase {
    private let providerID = UUID(uuidString: "00000000-0000-0000-0000-000000000001")!

    private func local(_ alias: String, family: String, bytes: UInt64 = 0, variant: String? = nil) -> InstalledModel {
        InstalledModel(alias: alias, repo: "org/\(alias)", path: "/m/\(alias)", family: family,
                       installBytes: bytes, variant: variant)
    }

    private var installed: [InstalledModel] {
        [
            local("qwen3-8b", family: "qwen3", bytes: 5_000_000_000, variant: "Q4_K_M"),
            local("gptoss-20b", family: "gpt_oss", bytes: 12_000_000_000),
            local("llama3-8b-qwen-distill", family: "llama"),
        ]
    }

    private var remote: [ModelPickerRow] {
        [ModelPickerRow.remote(providerID: providerID, providerName: "Acme", modelID: "acme-large",
                               displayName: "Acme Large", selected: nil)]
    }

    private func rows(_ query: String, selectedAlias: String? = nil) -> [ModelPickerSection] {
        let locals = installed.map { ModelPickerRow.local($0, selectedAlias: selectedAlias) }
        return ModelPickerList.sections(local: locals, remote: remote, query: query)
    }

    // MARK: - Rows from sources

    func testLocalRowCarriesSizeAndQuantAndTreatsZeroBytesAsUnknown() {
        let sized = ModelPickerRow.local(installed[0], selectedAlias: "qwen3-8b")
        XCTAssertEqual(sized.installBytes, 5_000_000_000)
        XCTAssertEqual(sized.quant, "Q4_K_M")
        XCTAssertTrue(sized.isSelected)
        XCTAssertEqual(sized.fitAlias, "qwen3-8b")

        // An install whose size was never recorded reports 0; that is not a size.
        let unsized = ModelPickerRow.local(installed[2], selectedAlias: "qwen3-8b")
        XCTAssertNil(unsized.installBytes)
        XCTAssertFalse(unsized.isSelected)
    }

    func testRemoteRowHasNoDiskOrFitFacts() {
        let row = ModelPickerRow.remote(providerID: providerID, providerName: "Acme", modelID: "acme-large",
                                        displayName: "Acme Large",
                                        selected: RemoteModelSelection(providerID: providerID, modelID: "acme-large"))
        XCTAssertEqual(row.title, "Acme: Acme Large")
        XCTAssertNil(row.installBytes)
        XCTAssertNil(row.fitAlias)
        XCTAssertTrue(row.isSelected)
        XCTAssertNotEqual(row.id, ModelPickerRow.local(installed[0], selectedAlias: nil).id)
    }

    // MARK: - Grouping

    func testEmptyQueryKeepsLocalThenRemoteInSourceOrder() {
        let sections = rows("")
        XCTAssertEqual(sections.map(\.group), [.local, .remote])
        XCTAssertEqual(sections[0].rows.map(\.title), ["qwen3-8b", "gptoss-20b", "llama3-8b-qwen-distill"])
        XCTAssertEqual(sections[1].rows.map(\.title), ["Acme: Acme Large"])
    }

    func testEmptyGroupsAreDropped() {
        let locals = installed.map { ModelPickerRow.local($0, selectedAlias: nil) }
        XCTAssertEqual(ModelPickerList.sections(local: locals, remote: [], query: "").map(\.group), [.local])
        XCTAssertEqual(rows("acme").map(\.group), [.remote])
        XCTAssertTrue(rows("no-such-model").isEmpty)
    }

    // MARK: - Filtering and ranking

    func testFilterIsCaseInsensitiveAndMatchesFamilyQuantAndProvider() {
        XCTAssertEqual(rows("GPT").flatMap(\.rows).map(\.title), ["gptoss-20b"])
        XCTAssertEqual(rows("gpt_oss").flatMap(\.rows).map(\.title), ["gptoss-20b"])
        XCTAssertEqual(rows("q4_k").flatMap(\.rows).map(\.title), ["qwen3-8b"])
        XCTAssertEqual(rows("acme").flatMap(\.rows).map(\.title), ["Acme: Acme Large"])
    }

    func testEveryWordMustMatch() {
        XCTAssertEqual(rows("8b llama").flatMap(\.rows).map(\.title), ["llama3-8b-qwen-distill"])
        XCTAssertTrue(rows("8b acme").isEmpty)
    }

    func testAliasPrefixRanksBeforeSubstringBeforeOtherFields() {
        // "qwen" prefixes qwen3-8b, sits inside llama3-8b-qwen-distill.
        XCTAssertEqual(rows("qwen").flatMap(\.rows).map(\.title), ["qwen3-8b", "llama3-8b-qwen-distill"])
        // "llama" prefixes the alias of one row and matches no other.
        let locals = [
            ModelPickerRow.local(local("tiny", family: "llama"), selectedAlias: nil),
            ModelPickerRow.local(local("my-llama", family: "other"), selectedAlias: nil),
            ModelPickerRow.local(local("llama-1b", family: "other"), selectedAlias: nil),
        ]
        let ranked = ModelPickerList.sections(local: locals, remote: [], query: "llama").flatMap(\.rows).map(\.title)
        XCTAssertEqual(ranked, ["llama-1b", "my-llama", "tiny"])
    }

    func testWhitespaceOnlyQueryIsNoFilter() {
        XCTAssertEqual(rows("   ").flatMap(\.rows).count, 4)
    }

    // MARK: - Keyboard highlight

    func testInitialHighlightIsTheSelectedRowElseFirst() {
        let flat = ModelPickerList.flattened(rows("", selectedAlias: "gptoss-20b"))
        XCTAssertEqual(ModelPickerList.initialHighlight(flat), 1)
        XCTAssertEqual(ModelPickerList.initialHighlight(ModelPickerList.flattened(rows(""))), 0)
        XCTAssertNil(ModelPickerList.initialHighlight([]))
    }

    func testMoveHighlightClampsAndHandlesEmpty() {
        XCTAssertEqual(ModelPickerList.moveHighlight(from: 0, by: 1, count: 3), 1)
        XCTAssertEqual(ModelPickerList.moveHighlight(from: 2, by: 1, count: 3), 2)
        XCTAssertEqual(ModelPickerList.moveHighlight(from: 0, by: -1, count: 3), 0)
        XCTAssertEqual(ModelPickerList.moveHighlight(from: nil, by: 1, count: 3), 0)
        // A stale index past a shrunken list is clamped to the last row, then moved.
        XCTAssertEqual(ModelPickerList.moveHighlight(from: 7, by: -1, count: 3), 1)
        XCTAssertEqual(ModelPickerList.moveHighlight(from: 7, by: 1, count: 3), 2)
        XCTAssertNil(ModelPickerList.moveHighlight(from: 1, by: 1, count: 0))
    }
}
