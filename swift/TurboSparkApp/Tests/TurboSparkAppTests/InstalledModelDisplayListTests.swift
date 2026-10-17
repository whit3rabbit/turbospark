import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.4: the Installed Models list shows offline and missing library rows
/// once each, next to the rows `AppModel.installed` already holds.
///
/// `installed` is the available-only list every picker and load gate reads, and
/// it also holds rows the app scanned from other apps' folders (a placeholder
/// id, no library). `LibraryState.allInstalled` has every engine row. The list
/// the manager shows is a merge of the two, and it is pure so each rule is a
/// table row here.
@MainActor
final class InstalledModelDisplayListTests: LibraryStateTestCase {
    // MARK: Fixtures

    /// An engine row of library `library` (nil is This Mac).
    private func engineRow(
        _ alias: String, library: ModelLibrary? = nil, availability: LibraryAvailability = .available,
        path: String? = nil
    ) -> InstalledModel {
        var row = LibraryFixtures.row(alias, library: library ?? LibraryFixtures.internalLibrary, availability: availability)
        if let path {
            row = InstalledModel(
                alias: row.alias, repo: row.repo, path: path, family: row.family, library: row.library,
                libraryName: row.libraryName, installId: row.installId, availability: row.availability,
                locationKind: row.locationKind, connectName: row.connectName)
        }
        return row
    }

    /// What the app's scanners build for another app's folder.
    private func scanned(_ alias: String, path: String) -> InstalledModel {
        InstalledModel(alias: alias, repo: "lm studio/\(alias)", path: path, family: "qwen")
    }

    private func merge(_ installed: [InstalledModel], _ all: [InstalledModel]) -> [InstalledModel] {
        InstalledModelDisplayList.merge(installed: installed, libraryRows: all)
    }

    // MARK: The rules

    func testOfflineAndMissingLibraryRowsAreAddedOnce() {
        let here = engineRow("here")
        let away = engineRow("away", library: t7, availability: .notConnected)
        let gone = engineRow("gone", library: t7, availability: .missing)
        let live = engineRow("live", library: t7)

        let shown = merge([here, live], [here, live, away, gone])

        XCTAssertEqual(shown.map(\.alias), ["here", "live", "away", "gone"])
        XCTAssertEqual(shown.map(\.availability), [.available, .available, .notConnected, .missing])
        XCTAssertEqual(Set(shown.map(\.installId)).count, shown.count, "no row twice")
    }

    func testAnAvailableRowInBothListsIsShownOnceWithTheInstalledRow() {
        let live = engineRow("live", library: t7)

        let shown = merge([live], [live])

        XCTAssertEqual(shown, [live])
    }

    func testAScannedPlaceholderRowIsShownOnceAndKeepsItsPlace() {
        let one = scanned("lm-one", path: "/Users/test/.lmstudio/models/one")
        let two = scanned("lm-two", path: "/Users/test/.lmstudio/models/two")
        let here = engineRow("here")

        let shown = merge([here, one, two], [here, engineRow("away", library: t7, availability: .notConnected)])

        XCTAssertEqual(shown.map(\.alias), ["here", "lm-one", "lm-two", "away"])
    }

    func testAPlaceholderIsNeverMatchedByItsDefaultInstallId() {
        // The default id of a scanned row is `internal:text/<alias>`: the same
        // spelling a This Mac row whose folder is gone has. They are two models.
        let placeholder = scanned("shared", path: "/Users/test/.lmstudio/models/shared")
        let missing = engineRow("shared", availability: .missing, path: "/Users/test/.turbospark/models/text/shared.gturbo")
        XCTAssertEqual(placeholder.installId, missing.installId, "the fixture must collide on id")

        let shown = merge([placeholder], [missing])

        XCTAssertEqual(shown.map(\.path), [placeholder.path, missing.path])
    }

    func testARowInBothListsIsMatchedByPathWhenTheIdsDiffer() {
        let path = "/Volumes/T7/TurboSpark/text/qwen.gturbo"
        let installedCopy = engineRow("qwen", library: t7, path: path)
        let listed = InstalledModel(
            alias: "qwen", repo: "owner/qwen", path: path, family: "qwen", library: t7.id,
            libraryName: t7.name, installId: "other-id:text/qwen", availability: .missing,
            locationKind: .library)

        let shown = merge([installedCopy], [listed])

        XCTAssertEqual(shown.count, 1, "one model at one path is one row")
    }

    func testAStaleAvailableRowTakesTheFresherOfflineAnswer() {
        // The drive was unplugged and the library list already knows; the
        // picker list has not been read again yet.
        let stale = engineRow("qwen", library: t7)
        let fresh = engineRow("qwen", library: t7, availability: .notConnected)

        let shown = merge([engineRow("first"), stale], [fresh])

        XCTAssertEqual(shown.map(\.alias), ["first", "qwen"], "it keeps its place")
        XCTAssertEqual(shown.last?.availability, .notConnected)
    }

    func testAnAvailableLibraryRowOnlyTheLibraryListKnowsIsNotAdded() {
        // The drive just came back and `installed` is not read again yet. The
        // pickers do not offer it either, and the next refresh brings it.
        let live = engineRow("live", library: t7)

        XCTAssertEqual(merge([], [live]), [])
    }

    func testRowsWithNoPathAreNotCollapsed() {
        let a = scanned("a", path: "")
        let b = scanned("b", path: "")

        XCTAssertEqual(merge([a, b], []).map(\.alias), ["a", "b"])
    }

    func testTheOrderIsStableAcrossCalls() {
        let installed = [engineRow("c"), engineRow("a"), engineRow("b", library: t7)]
        let all = installed + [
            engineRow("z", library: t7, availability: .missing),
            engineRow("y", library: t7, availability: .notConnected),
        ]

        let first = merge(installed, all)
        let second = merge(installed, all)

        XCTAssertEqual(first.map(\.alias), ["c", "a", "b", "z", "y"])
        XCTAssertEqual(first, second)
    }

    func testTheInputListsAreNotChanged() {
        let installed = [engineRow("here")]
        let all = [engineRow("away", library: t7, availability: .notConnected)]
        _ = merge(installed, all)
        XCTAssertEqual(installed.map(\.alias), ["here"])
        XCTAssertEqual(all.map(\.alias), ["away"])
    }

    // MARK: On the model

    func testTheModelListsOfflineRowsWithoutChangingInstalled() async {
        let live = engineRow("live", library: t7)
        let away = engineRow("away", library: t7, availability: .notConnected)
        service.with { $0.installed = [live, away] }
        let state = makeState()
        await state.refresh()
        let model = await makeModel(using: state)
        model.installed = [live]

        XCTAssertEqual(model.installed.map(\.alias), ["live"], "every other consumer still sees available rows only")
        XCTAssertEqual(model.installedForDisplay.map(\.alias), ["live", "away"])
        XCTAssertEqual(state.loadableInstalled.map(\.alias), ["live"])
    }

    func testTheDisplayListFollowsLibraryStateWhenItPublishes() async {
        let live = engineRow("live", library: t7)
        service.with { $0.installed = [live] }
        let state = makeState()
        await state.refresh()
        let model = await makeModel(using: state)
        model.installed = [live]
        XCTAssertEqual(model.installedForDisplay.map(\.alias), ["live"])

        // The drive is unplugged: the library list learns first.
        service.with { $0.installed = [engineRow("live", library: t7, availability: .notConnected)] }
        await state.refresh()

        XCTAssertEqual(model.installedForDisplay.map(\.availability), [.notConnected])
        XCTAssertEqual(model.installed.map(\.availability), [.available], "the picker list is untouched")
    }

    func testTheListViewObservesTheLibraryState() throws {
        // A view that read `model.installed` alone would not redraw when a
        // drive is unplugged: `AppModel` does not publish `libraryState`.
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let view = try String(
            contentsOf: packageRoot.appendingPathComponent(
                "Sources/TurboSparkApp/Installation/ModelManagerView.swift"), encoding: .utf8)
        XCTAssertTrue(view.contains("@ObservedObject private var libraryState: LibraryState"))
        XCTAssertTrue(view.contains("model.libraryState"))
        XCTAssertTrue(view.contains("installedForDisplay"))
        let pane = try String(
            contentsOf: packageRoot.appendingPathComponent(
                "Sources/TurboSparkApp/Installation/InstalledModelDetailPaneView.swift"), encoding: .utf8)
        XCTAssertTrue(pane.contains("@ObservedObject private var libraryState: LibraryState"))
    }
}
