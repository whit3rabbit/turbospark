import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.9: the Model Hub offered "Download and Install" for a model installed on
/// a drive that is away. It reads `AppModel.installed`, which holds available rows
/// only, so the model looked absent, and the download then ended in
/// `alias_elsewhere`. The hub now says where the model is instead.
@MainActor
final class HubOfflineInstallTests: LibraryStateTestCase {
    private func resolve(
        alias: String, isInstalled: Bool = false, rows: [InstalledModel]
    ) -> HubInstallState {
        HubInstallState.resolve(alias: alias, isInstalled: isInstalled, rows: rows)
    }

    func testAnAliasInstalledOnAnAwayDriveShowsWhereItIs() async {
        service.with {
            $0.installed = [LibraryFixtures.row("qwen", library: t7, availability: .notConnected)]
        }
        let state = makeState()
        await state.refresh()

        XCTAssertEqual(
            resolve(alias: "qwen", rows: state.allInstalled), .installedOffline(libraryName: "T7"),
            "not Download: the install would be refused as alias_elsewhere")
    }

    func testAnAliasNotInstalledAnywhereIsOfferedForDownload() async {
        let state = makeState()
        await state.refresh()
        XCTAssertEqual(resolve(alias: "brand-new", rows: state.allInstalled), .download)
    }

    func testAnAliasInstalledAndAvailableIsInstalled() async {
        let state = makeState()
        await state.refresh()
        XCTAssertEqual(resolve(alias: "local", isInstalled: true, rows: state.allInstalled), .installed)
    }

    func testAnInstalledAliasStaysInstalledEvenIfAnOfflineRowSharesItsName() {
        // The engine keeps alias plus modality unique, so this is a stale list
        // during a reconnect; the usable install wins.
        let rows = [LibraryFixtures.row("qwen", library: t7, availability: .notConnected)]
        XCTAssertEqual(resolve(alias: "qwen", isInstalled: true, rows: rows), .installed)
    }

    func testOnlyAnAwayRowOfTheSameModalityCounts() {
        let image = InstalledModel(
            alias: "qwen", repo: "o/qwen", path: "/Volumes/T7/TurboSpark/image/qwen", family: "qwen",
            modality: "image", library: t7.id, libraryName: t7.name, installId: "t7-id:image/qwen",
            availability: .notConnected, locationKind: .library, connectName: "T7")
        XCTAssertEqual(resolve(alias: "qwen", rows: [image]), .download, "an image with that name is not this text model")
    }

    func testAMissingRowIsNotAnAwayDrive() {
        // The drive is connected and the folder is gone: there is nothing to
        // plug in, and reinstalling into the same library is allowed.
        let missing = LibraryFixtures.row("qwen", library: t7, availability: .missing)
        XCTAssertEqual(resolve(alias: "qwen", rows: [missing]), .download)
    }

    func testAnotherAliasOnTheAwayDriveDoesNotMatter() {
        let other = LibraryFixtures.row("llama", library: t7, availability: .notConnected)
        XCTAssertEqual(resolve(alias: "qwen", rows: [other]), .download)
    }

    func testTheStatementNamesTheDriveAndCarriesAGlyph() {
        let state = HubInstallState.installedOffline(libraryName: "T7\u{202E}x")
        let statement = state.statement(language: .english)
        XCTAssertNotNil(statement)
        XCTAssertTrue(statement?.contains("T7") ?? false)
        XCTAssertFalse(statement?.contains("\u{202E}") ?? true, "a drive's name is escaped")
        XCTAssertEqual(state.glyph, LibraryRowStatus.notConnected.glyph, "the same minus as every Not connected row")
        XCTAssertNil(HubInstallState.download.statement(language: .english))
        XCTAssertNil(HubInstallState.installed.statement(language: .english))
    }

    func testOnlyTheDownloadStateOffersDownload() {
        XCTAssertTrue(HubInstallState.download.offersDownload)
        XCTAssertFalse(HubInstallState.installed.offersDownload)
        XCTAssertFalse(HubInstallState.installedOffline(libraryName: "T7").offersDownload)
    }

    func testTheHubDetailPaneAsksTheStateBeforeItOffersADownload() throws {
        let sources = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/TurboSparkApp")
        let code = MoveStoreRetirementTests.codeWithoutComments(
            try String(
                contentsOf: sources.appendingPathComponent("Installation/ModelDetailPaneView.swift"),
                encoding: .utf8))
        XCTAssertTrue(code.contains("HubInstallState.resolve"), "the pane must ask whether the model is on an away drive")
        XCTAssertTrue(code.contains(".installedOffline"), "and draw the statement for it")
        // The Download & Install button lives only in the branch for a model
        // that is installed nowhere.
        let download = try XCTUnwrap(code.range(of: "Download & Install"))
        let offline = try XCTUnwrap(code.range(of: ".installedOffline"))
        XCTAssertLessThan(offline.lowerBound, download.lowerBound, "the offline branch comes before the download")
    }
}
