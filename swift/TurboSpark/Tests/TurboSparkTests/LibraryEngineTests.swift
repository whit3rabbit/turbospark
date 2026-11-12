import XCTest

@testable import TurboSpark

/// The real engine, linked as the RELEASE archive, against a throwaway home
/// that has only This Mac. Release builds ignore the `TURBOSPARK_TEST_*`
/// location overrides, so no fake drive can be mounted here: what this proves
/// is that the hand-written header, the wrappers and the error codes agree
/// for the paths a single-volume machine can reach. Drives are covered by the
/// Rust FFI tests.
final class LibraryEngineTests: XCTestCase {

    private var home: URL!

    override func setUpWithError() throws {
        try super.setUpWithError()
        // Never the developer's real ~/.turbospark: installed.json and the
        // library list live in the store root.
        home = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-library-test-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: home, withIntermediateDirectories: true)
        try TurboSparkCatalog.setStoreRoot(home.path)
    }

    override func tearDown() {
        try? TurboSparkCatalog.setStoreRoot(nil)
        if let home { try? FileManager.default.removeItem(at: home) }
        super.tearDown()
    }

    private func fault(
        _ body: () throws -> Void, file: StaticString = #filePath, line: UInt = #line
    ) -> LibraryFault? {
        do {
            try body()
            XCTFail("expected the call to throw", file: file, line: line)
            return nil
        } catch let error as TurboSparkError {
            let parsed = error.libraryFault
            XCTAssertNotNil(parsed, "no code in: \(error.message)", file: file, line: line)
            return parsed
        } catch {
            XCTFail("expected a TurboSparkError, got \(error)", file: file, line: line)
            return nil
        }
    }

    // MARK: - The library list

    func testAFreshHomeListsOnlyThisMacAsTheDefault() throws {
        let libraries = try TurboSparkLibraries.list()
        XCTAssertEqual(libraries.count, 1)
        let thisMac = try XCTUnwrap(libraries.first)
        XCTAssertEqual(thisMac.id, "internal")
        XCTAssertTrue(thisMac.isInternal)
        XCTAssertTrue(thisMac.isDefault)
        XCTAssertTrue(thisMac.available)
        XCTAssertFalse(thisMac.duplicate)
        XCTAssertEqual(thisMac.modelCount, 0)
        XCTAssertEqual(thisMac.name, "This Mac", "the English literal, localized by the app")
        XCTAssertEqual(try XCTUnwrap(thisMac.root).hasSuffix("/models"), true)
    }

    func testSettingTheDefaultToThisMacKeepsTheList() throws {
        try TurboSparkLibraries.setDefault(id: "internal")
        XCTAssertEqual(try TurboSparkLibraries.list().map(\.id), ["internal"])
    }

    // MARK: - Library errors, from the engine

    func testAddingAPathThatIsNotThereIsRefusedAsNotADirectory() throws {
        let missing = home.appendingPathComponent("no-such-folder").path
        let parsed = try XCTUnwrap(fault { _ = try TurboSparkLibraries.add(path: missing) })
        XCTAssertEqual(parsed.code, .refusal(.notADirectory))
        XCTAssertEqual(parsed.status, .invalidArgument)
        XCTAssertFalse(parsed.sentence.isEmpty)
    }

    func testRemovingThisMacIsRefusedAsInternalLibrary() throws {
        let parsed = try XCTUnwrap(fault { try TurboSparkLibraries.remove(id: "internal") })
        XCTAssertEqual(parsed.code, .internalLibrary)
        XCTAssertEqual(try TurboSparkLibraries.list().count, 1, "This Mac is still listed")
    }

    func testAnIdTheListDoesNotHaveIsRefusedAsUnknownLibrary() throws {
        XCTAssertEqual(
            fault { try TurboSparkLibraries.remove(id: "no-such-id") }?.code, .unknownLibrary)
        XCTAssertEqual(
            fault { try TurboSparkLibraries.setDefault(id: "no-such-id") }?.code, .unknownLibrary)
    }

    func testDiscoveringWhereThereIsNoLibraryAnswersNil() throws {
        XCTAssertNil(try TurboSparkLibraries.discover(volumeRoot: home.path))
        XCTAssertNil(
            try TurboSparkLibraries.discover(volumeRoot: home.appendingPathComponent("gone").path))
    }

    func testForgettingAnInstallThatIsNotListedIsNotAnError() throws {
        try TurboSparkLibraries.forget(installId: "internal:text/never-installed")
        try TurboSparkLibraries.forget(installId: "no-such-id:text/never-installed")
    }

    func testImportingARootThatIsNotThereIsRefusedAsNotConnected() throws {
        let parsed = try XCTUnwrap(
            fault {
                _ = try TurboSparkLibraries.importMovedRoot(
                    self.home.appendingPathComponent("old-store").path)
            })
        XCTAssertEqual(parsed.code, .notConnected)
    }

    // MARK: - Moves and deletes with ids nothing holds

    func testAMoveFromALibraryThatIsNotListedIsRefusedAsUnknownInstall() async throws {
        do {
            _ = try await TurboSparkLibraries.move(
                installId: "no-such-id:text/x", to: "internal", freeBytes: 1 << 40)
            XCTFail("expected the move to throw")
        } catch let error as TurboSparkError {
            XCTAssertEqual(error.libraryFault?.code, .unknownInstall, error.message)
        }
    }

    func testAMoveToALibraryThatIsNotListedIsRefusedAsUnknownLibrary() async throws {
        do {
            _ = try await TurboSparkLibraries.move(
                installId: "internal:text/x", to: "no-such-id", freeBytes: 1 << 40)
            XCTFail("expected the move to throw")
        } catch let error as TurboSparkError {
            XCTAssertEqual(error.libraryFault?.code, .unknownLibrary, error.message)
        }
    }

    func testAMoveWithoutAMeasuredFreeSpaceIsRefusedBeforeAnythingElse() async throws {
        // Nothing here is listed either, yet the free space is what is
        // refused first: the app passes nil only when it could not measure.
        do {
            _ = try await TurboSparkLibraries.move(
                installId: "no-such-id:text/x", to: "no-such-id", freeBytes: nil)
            XCTFail("expected the move to throw")
        } catch let error as TurboSparkError {
            XCTAssertEqual(error.libraryFault?.code, .freeSpaceUnknown, error.message)
        }
    }

    func testCancellingWhenNoMoveRunsSignalsNothing() {
        XCTAssertFalse(TurboSparkLibraries.cancelMove())
    }

    func testDeletingAnInstallNobodyHoldsIsRefusedAsUnknownInstall() throws {
        XCTAssertEqual(
            fault { try TurboSparkCatalog.deleteInstall(installId: "internal:text/none") }?.code,
            .unknownInstall)
        // The alias call keeps working for This Mac and says the same.
        XCTAssertEqual(
            fault { try TurboSparkCatalog.delete("none") }?.code, .unknownInstall)
    }

    func testDeletingFromALibraryThatIsNotListedIsRefusedAsUnknownLibrary() throws {
        XCTAssertEqual(
            fault { try TurboSparkCatalog.deleteInstall(installId: "no-such-id:text/x") }?.code,
            .unknownLibrary)
        XCTAssertEqual(
            fault { try TurboSparkCatalog.deleteImageInstall(installId: "no-such-id:image/x") }?.code,
            .unknownLibrary)
    }

    // MARK: - Install destinations reach the engine

    func testAnInstallIntoALibraryThatIsNotListedIsRefusedBeforeAnyRequest() async throws {
        // The destination is settled before the repo probe, so this is
        // refused without a request. A repository that does not exist keeps
        // the test harmless if the destination were ever dropped on the way.
        let stream = TurboSparkCatalog.install(
            repo: "turbospark-test-nonexistent/none", alias: "library-smoke", library: "no-such-id")
        do {
            for try await _ in stream {}
            XCTFail("expected the install to throw")
        } catch let error as TurboSparkError {
            XCTAssertEqual(error.libraryFault?.code, .unknownLibrary, error.message)
        }
    }

    func testAnAudioInstallIntoALibraryThatIsNotListedIsRefusedBeforeAnyRequest() throws {
        // A stale identity, so the call could never download even if the
        // destination were dropped on the way: it would fail on the profile
        // instead, with no code.
        let stale = AudioProfileIdentity(
            task: .textToSpeech, alias: "no-such-profile", repository: "x/y",
            revision: String(repeating: "0", count: 40), assetFingerprint: "none")
        let parsed = try XCTUnwrap(
            fault { _ = try AudioCatalog.install(stale, library: "no-such-id") })
        XCTAssertEqual(parsed.code, .unknownLibrary)
    }

    // MARK: - Rows and reports

    private func writeInstalled(_ rows: [[String: Any]]) throws {
        let data = try JSONSerialization.data(withJSONObject: rows)
        try data.write(to: home.appendingPathComponent("installed.json"))
    }

    private func row(_ alias: String, path: String) -> [String: Any] {
        [
            "alias": alias, "repo": "owner/\(alias)", "revision": "main", "path": path,
            "family": "llama", "install_bytes": 4096, "installed_on": "2026-10-14",
            "status": "verified",
        ]
    }

    /// Three rows the engine can tell apart on one machine with one volume: a
    /// This Mac folder that is there, one that is gone, and a custom folder
    /// under a volume that is not mounted.
    private func writeThreeRows() throws -> (here: String, gone: String, away: String, drive: String) {
        let here = home.appendingPathComponent("models/text/here.gturbo")
        try FileManager.default.createDirectory(at: here, withIntermediateDirectories: true)
        let gone = home.appendingPathComponent("models/text/gone.gturbo").path
        let drive = "TurboSparkNoSuchDrive-\(UUID().uuidString)"
        let away = "/Volumes/\(drive)/models/away.gturbo"
        try writeInstalled([
            row("here", path: here.path), row("gone", path: gone), row("away", path: away),
        ])
        return (here.path, gone, away, drive)
    }

    func testTheAllRowsListingKeepsEveryRowAndSaysWhatEachIs() throws {
        let fixture = try writeThreeRows()

        let rows = try TurboSparkCatalog.installedWithAvailability()
        XCTAssertEqual(rows.map(\.alias), ["away", "gone", "here"], "sorted by alias")

        let here = try XCTUnwrap(rows.first { $0.alias == "here" })
        XCTAssertEqual(here.availability, .available)
        XCTAssertEqual(here.locationKind, .thisMac)
        XCTAssertEqual(here.library, "internal")
        XCTAssertEqual(here.libraryName, "This Mac")
        XCTAssertEqual(here.installId, "internal:text/here")
        XCTAssertNil(here.connectName)
        XCTAssertEqual(here.path, fixture.here)

        let gone = try XCTUnwrap(rows.first { $0.alias == "gone" })
        XCTAssertEqual(gone.availability, .missing)
        XCTAssertEqual(gone.locationKind, .thisMac)

        let away = try XCTUnwrap(rows.first { $0.alias == "away" })
        XCTAssertEqual(away.availability, .notConnected)
        XCTAssertEqual(away.locationKind, .custom)
        XCTAssertNil(away.library, "a custom location has no library")
        XCTAssertEqual(away.connectName, fixture.drive)
    }

    func testInstalledKeepsItsOldContractAndListsOnlyRowsThatCanBeOpened() throws {
        _ = try writeThreeRows()
        // The app feeds this list to its pickers and gating unfiltered, so a
        // row that cannot be opened must never appear in it.
        let usable = try TurboSparkCatalog.installed()
        XCTAssertEqual(usable.map(\.alias), ["here"])
        XCTAssertTrue(usable.allSatisfy { $0.availability == .available })
    }

    func testInstalledOnAFreshHomeIsEmptyAndTheAudioReportDecodes() throws {
        XCTAssertTrue(try TurboSparkCatalog.installed().isEmpty)
        XCTAssertTrue(try TurboSparkCatalog.installedWithAvailability().isEmpty)
        let report = try AudioCatalog.installed()
        XCTAssertTrue(report.installed.isEmpty)
        XCTAssertTrue(report.notConnected.isEmpty)
        XCTAssertTrue(try TurboSparkCatalog.imageInstalled().isEmpty)
    }

    func testAThisMacImageIdAnswersWithTheAliasDeletesMessageAndNoInventedCode() throws {
        // `internal:image/<alias>` is ts_image_delete of the alias, whose
        // refusal carries no code: the wrapper must not invent one.
        do {
            try TurboSparkCatalog.deleteImageInstall(installId: "internal:image/none")
            XCTFail("expected the delete to throw")
        } catch let error as TurboSparkError {
            XCTAssertNil(error.libraryFault, error.message)
            XCTAssertTrue(error.message.contains("not installed"), error.message)
        }
    }
}
