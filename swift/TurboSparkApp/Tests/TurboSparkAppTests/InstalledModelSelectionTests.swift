import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// The rules of multi-select delete: which rows can be ticked, what the bar
/// says it frees, what the confirmation and the toast say.
final class InstalledModelSelectionTests: XCTestCase {
    private func row(_ alias: String, bytes: UInt64 = 0, scanned: Bool = false) -> InstalledModel {
        if scanned {
            // What the app's scanners build: no registry identity.
            return InstalledModel(alias: alias, repo: "lm studio/\(alias)", path: "/scan/\(alias)", family: "qwen",
                                  installBytes: bytes)
        }
        return InstalledModel(
            alias: alias, repo: "o/\(alias)", path: "/lib/\(alias)", family: "qwen", installBytes: bytes,
            library: "internal", libraryName: "This Mac", installId: "internal:text/\(alias)")
    }

    func testAvailableEngineRowIsSelectable() {
        XCTAssertNil(InstalledModelSelection.block(for: row("a"), status: nil, isLoaded: false))
    }

    func testOfflineMissingScannedAndLoadedRowsAreBlockedWithTheirOwnReason() {
        XCTAssertEqual(InstalledModelSelection.block(for: row("a"), status: .notConnected, isLoaded: false), .notConnected)
        XCTAssertEqual(InstalledModelSelection.block(for: row("a"), status: .missing, isLoaded: false), .missing)
        XCTAssertEqual(InstalledModelSelection.block(for: row("s", scanned: true), status: nil, isLoaded: false), .scanned)
        XCTAssertEqual(InstalledModelSelection.block(for: row("a"), status: nil, isLoaded: true), .loaded)
        // An away drive wins over loaded: the more fundamental reason.
        XCTAssertEqual(InstalledModelSelection.block(for: row("a"), status: .notConnected, isLoaded: true), .notConnected)
        let reasons = [Optional(InstalledModelSelection.Block.notConnected), .missing, .scanned, .loaded]
            .compactMap { $0?.reason(language: .english) }
        XCTAssertEqual(Set(reasons).count, 4, "each block says its own reason")
        XCTAssertTrue(reasons.allSatisfy { !$0.isEmpty })
    }

    func testBytesFreedCountsOnlyTheSelection() {
        let models = [row("a", bytes: 100), row("b", bytes: 250), row("c", bytes: 4)]
        XCTAssertEqual(InstalledModelSelection.bytesFreed(of: ["/lib/a", "/lib/c"], in: models), 104)
        XCTAssertEqual(InstalledModelSelection.bytesFreed(of: [], in: models), 0)
        XCTAssertEqual(InstalledModelSelection.bytesFreed(of: ["/lib/gone"], in: models), 0)
    }

    func testEffectiveSelectionDropsRowsThatAreNoLongerSelectable() {
        XCTAssertEqual(
            InstalledModelSelection.effective(selection: ["a", "b"], selectable: ["b", "c"]), ["b"])
    }

    func testSelectAllThenDeselectAll() {
        let selectable: Set<String> = ["a", "b"]
        let all = InstalledModelSelection.toggledAll(selection: ["a"], selectable: selectable)
        XCTAssertEqual(all, selectable)
        XCTAssertEqual(InstalledModelSelection.toggledAll(selection: all, selectable: selectable), [])
        XCTAssertEqual(InstalledModelSelection.toggledAll(selection: [], selectable: []), [])
    }

    func testDetailSelectionMovesToARemainingModel() {
        let remaining = [row("a"), row("b")]
        XCTAssertEqual(
            InstalledModelSelection.detailSelection(current: "/lib/b", deleted: [], remaining: remaining), "/lib/b")
        XCTAssertEqual(
            InstalledModelSelection.detailSelection(current: "/lib/x", deleted: ["/lib/x"], remaining: remaining), "/lib/a")
        XCTAssertNil(InstalledModelSelection.detailSelection(current: "/lib/x", deleted: ["/lib/x"], remaining: []))
    }

    func testDialogNamesTheCountAndTheSpace() {
        let one = InstalledModelBulkDeleteDialog(count: 1, bytes: 1_000_000_000, language: .english)
        let many = InstalledModelBulkDeleteDialog(count: 3, bytes: 3_000_000_000, language: .english)
        XCTAssertTrue(one.title.contains("1 model"), one.title)
        XCTAssertTrue(many.title.contains("3 models"), many.title)
        XCTAssertEqual(many.confirmTitle, "Delete 3 Models")
        XCTAssertEqual(one.confirmTitle, "Delete 1 Model")
        XCTAssertTrue(many.message.contains(MetricFormat.storage(3_000_000_000)), many.message)
        XCTAssertTrue(many.message.contains("cannot be undone"))
    }

    func testDialogAndSummaryAreTranslatedInEveryLanguage() {
        for language in AppLanguage.allCases where language != .english && language != .system {
            let dialog = InstalledModelBulkDeleteDialog(count: 3, bytes: 3_000_000_000, language: language)
            let english = InstalledModelBulkDeleteDialog(count: 3, bytes: 3_000_000_000, language: .english)
            XCTAssertNotEqual(dialog.title, english.title, "\(language) title")
            XCTAssertNotEqual(dialog.message, english.message, "\(language) message")
            XCTAssertNotEqual(dialog.confirmTitle, english.confirmTitle, "\(language) button")
            XCTAssertFalse(dialog.title.contains("%"), dialog.title)
        }
    }

    func testOutcomeSeparatesDeletedFromRemaining() {
        let requested = [row("a"), row("b"), row("c")]
        let outcome = InstalledModelBulkDeleteOutcome.make(requested: requested, remainingPaths: ["/lib/b"])
        XCTAssertEqual(outcome.deleted, ["a", "c"])
        XCTAssertEqual(outcome.failed, ["b"])
        XCTAssertTrue(outcome.hasFailures)
        XCTAssertEqual(outcome.summary(language: .english), "Deleted 2, not deleted: b")
    }

    func testSummaryWhenEverythingWentOrNothingDid() {
        let requested = [row("a"), row("b")]
        let ok = InstalledModelBulkDeleteOutcome.make(requested: requested, remainingPaths: [])
        XCTAssertFalse(ok.hasFailures)
        XCTAssertEqual(ok.summary(language: .english), "Deleted 2 models")
        let none = InstalledModelBulkDeleteOutcome.make(requested: requested, remainingPaths: ["/lib/a", "/lib/b"])
        XCTAssertEqual(none.summary(language: .english), "Not deleted: a, b")
    }

    func testSummaryCutsALongFailureList() {
        let requested = (1...5).map { row("m\($0)") }
        let outcome = InstalledModelBulkDeleteOutcome.make(
            requested: requested, remainingPaths: Set(requested.map(\.path)))
        XCTAssertEqual(outcome.summary(language: .english), "Not deleted: m1, m2, m3, ...")
    }
}
