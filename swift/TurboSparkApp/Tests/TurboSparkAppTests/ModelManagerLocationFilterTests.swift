import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.5: the Installed Models list can be grouped under library names and
/// narrowed to what can be opened now (requirements 3.5 and 3.6).
@MainActor
final class ModelManagerLocationFilterTests: XCTestCase {
    private let internalLibrary = LibraryFixtures.internalLibrary
    private let t7 = LibraryFixtures.external("t7-id", name: "T7")
    private let archive = LibraryFixtures.external("arch-id", name: "Archive", available: false)
    private let spare = LibraryFixtures.external("spare-id", name: "Spare")

    private func engine(
        _ alias: String, _ library: ModelLibrary, _ availability: LibraryAvailability = .available
    ) -> InstalledModel {
        LibraryFixtures.row(alias, library: library, availability: availability)
    }

    private func scanned(_ alias: String) -> InstalledModel {
        InstalledModel(alias: alias, repo: "lm studio/\(alias)", path: "/x/\(alias)", family: "qwen")
    }

    private func custom(_ alias: String) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "x/y", path: "/data/\(alias)", family: "qwen", library: nil,
            libraryName: "custom location", installId: "internal:text/\(alias)", locationKind: .custom)
    }

    private func filter(
        _ rows: [InstalledModel], availableOnly: Bool = false, search: String = ""
    ) throws -> [InstalledModel] {
        let store = ModelOrganizationStore(
            metadataFile: FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).json"),
            exclusionsFile: FileManager.default.temporaryDirectory.appendingPathComponent("\(UUID().uuidString).json"))
        return ModelManagerView.filterModels(
            installed: rows, searchText: search, showFavoritesOnly: false, availableOnly: availableOnly,
            architectureFilter: .all, drafterFilter: .all, sourceFilter: .all, selectedTag: nil,
            sortOption: .name, orgStore: store)
    }

    // MARK: Available now

    func testAvailableNowHidesNotConnectedAndMissingRows() throws {
        let rows = [
            engine("here", internalLibrary), engine("live", t7), engine("away", archive, .notConnected),
            engine("gone", t7, .missing), scanned("lm"),
        ]

        XCTAssertEqual(try filter(rows, availableOnly: true).map(\.alias), ["here", "live", "lm"])
        XCTAssertEqual(try filter(rows, availableOnly: false).count, 5, "off shows everything")
    }

    func testAvailableNowCombinesWithTheOtherFilters() throws {
        // The fixtures' family is "qwen", which the search also reads, so the
        // search term is one only the aliases carry.
        let rows = [engine("alpha-live", t7), engine("alpha-away", archive, .notConnected), engine("beta", t7)]

        XCTAssertEqual(try filter(rows, availableOnly: true, search: "alpha").map(\.alias), ["alpha-live"])
    }

    func testAvailableNowDecidesFromTheRowsAvailabilityAndNothingElse() throws {
        // A row of an available library whose folder is gone is hidden, and a
        // row of a library the list does not know is kept while it is available.
        let gone = engine("gone", t7, .missing)
        let unknownLibrary = InstalledModel(
            alias: "orphan", repo: "x/y", path: "/p", family: "qwen", library: "nobody", libraryName: "Nobody",
            installId: "nobody:text/orphan", availability: .available, locationKind: .library)

        XCTAssertEqual(try filter([gone, unknownLibrary], availableOnly: true).map(\.alias), ["orphan"])
    }

    func testTheSelectedRowStaysSelectedWhenTheFilterHidesIt() {
        let away = engine("away", archive, .notConnected)
        let rows = [engine("live", t7), away]

        // The detail pane looks the selection up in the list the filters read
        // from, so hiding a row does not rebind the pane to another model.
        XCTAssertEqual(ModelManagerView.selectedRow(path: away.path, in: rows)?.alias, "away")
        XCTAssertNil(ModelManagerView.selectedRow(path: "/nowhere", in: rows))
        XCTAssertNil(ModelManagerView.selectedRow(path: nil, in: rows))
    }

    // MARK: Group by location

    func testGroupsAreNamedByLibraryAndOrderedThisMacThenConnectedThenAwayThenCustom() {
        let rows = [
            engine("a-away", archive, .notConnected),
            custom("c-custom"),
            engine("b-live", t7),
            scanned("d-scanned"),
            engine("e-here", internalLibrary),
            engine("f-spare", spare),
            engine("g-away-too", archive, .notConnected),
        ]
        let groups = ModelManagerView.groupModels(
            filteredModels: rows, groupOption: .location,
            libraries: [internalLibrary, archive, t7, spare], language: .english)

        XCTAssertEqual(groups.map(\.groupTitle), ["This Mac", "T7", "Spare", "Archive", "Custom Folder"])
        XCTAssertEqual(groups[0].models.map(\.alias), ["e-here"])
        XCTAssertEqual(groups[3].models.map(\.alias), ["a-away", "g-away-too"])
        XCTAssertEqual(groups[4].models.map(\.alias), ["c-custom", "d-scanned"], "custom folders and scanned folders")
        XCTAssertTrue(groups.allSatisfy(\.titleIsVerbatim), "a library name is never looked up as a key")
    }

    func testAGroupIsOmittedWhenItHasNoRows() {
        let groups = ModelManagerView.groupModels(
            filteredModels: [engine("b-live", t7)], groupOption: .location,
            libraries: [internalLibrary, t7, archive], language: .english)

        XCTAssertEqual(groups.map(\.groupTitle), ["T7"])
    }

    func testTwoLibrariesWithTheSameNameStaySeparateGroups() {
        let first = LibraryFixtures.external("one-id", name: "Backup", volume: "Backup")
        let second = LibraryFixtures.external("two-id", name: "Backup", volume: "Backup 1")
        let groups = ModelManagerView.groupModels(
            filteredModels: [engine("a", first), engine("b", second)], groupOption: .location,
            libraries: [internalLibrary, first, second], language: .english)

        XCTAssertEqual(groups.map(\.groupTitle), ["Backup", "Backup"])
        XCTAssertEqual(Set(groups.map(\.id)).count, 2, "the id is the library's, not its name")
    }

    func testTheGroupTitleIsTheEscapedLibraryNameAndThisMacIsLocalized() {
        let hostile = LibraryFixtures.external("evil-id", name: "Evil\u{202E}fdp.exe")
        let groups = ModelManagerView.groupModels(
            filteredModels: [engine("a", hostile), engine("b", internalLibrary)], groupOption: .location,
            libraries: [internalLibrary, hostile], language: .german)

        XCTAssertNotEqual(groups[0].groupTitle, "This Mac")
        XCTAssertEqual(groups[1].groupTitle, UntrustedText.display("Evil\u{202E}fdp.exe"))
        XCTAssertFalse(groups[1].groupTitle.contains("\u{202E}"))
    }

    func testARowOfALibraryTheListDoesNotKnowIsGroupedByItsOwnNameAfterTheConnectedOnes() {
        let orphan = InstalledModel(
            alias: "orphan", repo: "x/y", path: "/p", family: "qwen", library: "nobody", libraryName: "Old Drive",
            installId: "nobody:text/orphan", availability: .missing, locationKind: .library)
        let groups = ModelManagerView.groupModels(
            filteredModels: [orphan, engine("live", t7)], groupOption: .location,
            libraries: [internalLibrary, t7], language: .english)

        XCTAssertEqual(groups.map(\.groupTitle), ["T7", "Old Drive"])
    }

    func testTheOtherGroupModesAreUnchanged() {
        let rows = [engine("a", t7), scanned("b")]
        XCTAssertEqual(
            ModelManagerView.groupModels(filteredModels: rows, groupOption: .none).map(\.groupTitle),
            ["All Models"])
        let none = ModelManagerView.groupModels(filteredModels: rows, groupOption: .none)
        XCTAssertFalse(none[0].titleIsVerbatim, "the old titles are still catalog keys")
        XCTAssertEqual(ModelManagerView.GroupOption.location.rawValue, "By Location")
        XCTAssertEqual(
            ModelManagerView.GroupOption.allCases.map(\.rawValue),
            ["None", "By Source", "By Architecture", "By Family", "By Location"])
    }

    // MARK: Persistence

    private func freshDefaults() -> UserDefaults {
        UserDefaults(suiteName: "model-manager-prefs-\(UUID().uuidString)")!
    }

    func testTheToggleAndTheGroupModeSurviveARelaunch() {
        let defaults = freshDefaults()
        let first = ModelManagerFilterPreferences(defaults: defaults)
        XCTAssertEqual(first.groupOption, .none)
        XCTAssertFalse(first.availableOnly, "off until the user turns it on")

        first.groupOption = .location
        first.availableOnly = true

        let relaunched = ModelManagerFilterPreferences(defaults: defaults)
        XCTAssertEqual(relaunched.groupOption, .location)
        XCTAssertTrue(relaunched.availableOnly)

        relaunched.availableOnly = false
        relaunched.groupOption = .family
        let again = ModelManagerFilterPreferences(defaults: defaults)
        XCTAssertFalse(again.availableOnly)
        XCTAssertEqual(again.groupOption, .family)
    }

    func testAnUnknownStoredGroupModeFallsBackToNone() {
        let defaults = freshDefaults()
        defaults.set("By Planet", forKey: ModelManagerFilterPreferences.groupOptionKey)

        XCTAssertEqual(ModelManagerFilterPreferences(defaults: defaults).groupOption, .none)
    }

    func testThePreferencesUseTheirOwnKeysAndTheViewReadsThem() throws {
        XCTAssertEqual(ModelManagerFilterPreferences.groupOptionKey, "TurboSpark.modelManager.groupOption")
        XCTAssertEqual(ModelManagerFilterPreferences.availableOnlyKey, "TurboSpark.modelManager.availableOnly")
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let view = try String(
            contentsOf: packageRoot.appendingPathComponent(
                "Sources/TurboSparkApp/Installation/ModelManagerView.swift"), encoding: .utf8)
        XCTAssertTrue(view.contains("ModelManagerFilterPreferences"), "the view must use the persisted object")
        XCTAssertFalse(
            view.contains("@State private var groupOption"),
            "a plain @State group mode is forgotten at relaunch")
    }
}
