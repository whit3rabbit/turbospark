import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.9 (requirements 2.1 to 2.3 and 2.5): the pure model behind the install
/// destination picker.
///
/// **THE MODEL IS PURE.** It takes the library list, the free space the state
/// read, the default destination and This Mac's free space, so every rule here is
/// a table test with no view and no engine. The picker view draws what it says.
@MainActor
final class InstallDestinationModelTests: XCTestCase {
    private let thisMac = LibraryFixtures.internalLibrary
    private let t7 = LibraryFixtures.external("t7-id", name: "T7")
    private let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)

    private let gb: Int64 = 1_000_000_000

    private func internalDefault() -> DefaultInstallDestination {
        DefaultInstallDestination(
            libraryId: "internal", library: thisMac, fallbackBecauseNotConnected: false,
            notConnectedLibraryName: nil)
    }

    private func model(
        _ libraries: [ModelLibrary], free: [String: Int64] = [:], thisMacFree: UInt64? = 100_000_000_000,
        defaultDestination: DefaultInstallDestination? = nil, language: AppLanguage = .english
    ) -> InstallDestinationModel {
        InstallDestinationModel(
            libraries: libraries, freeBytes: free,
            destinationDefault: defaultDestination ?? internalDefault(), thisMacFreeBytes: thisMacFree,
            language: language)
    }

    // MARK: Shown only when there is a choice (requirement 2.1)

    func testThePickerIsHiddenWhenThisMacIsTheOnlyDestination() {
        let destinations = model([thisMac])
        XCTAssertFalse(destinations.isShown)
        XCTAssertEqual(destinations.choices.map(\.id), ["internal"])
        XCTAssertNil(destinations.libraryForInstall(selection: nil), "no picker: the install goes to This Mac")
    }

    func testThePickerIsHiddenBeforeTheFirstListing() {
        // No library has been listed yet. This Mac is still a destination, so an
        // install started this early has somewhere to go.
        let destinations = model([])
        XCTAssertFalse(destinations.isShown)
        XCTAssertEqual(destinations.choices.map(\.id), ["internal"])
        XCTAssertEqual(destinations.resolve(selection: nil).choice?.id, "internal")
    }

    func testThePickerIsShownWithTwoDestinations() {
        let destinations = model([thisMac, t7])
        XCTAssertTrue(destinations.isShown)
        XCTAssertEqual(destinations.choices.map(\.id), ["internal", "t7-id"])
        XCTAssertEqual(destinations.choices.map(\.isThisMac), [true, false])
    }

    func testALibraryThatIsNotConnectedIsNotADestination() {
        let destinations = model([thisMac, spare])
        XCTAssertFalse(destinations.isShown, "one connected destination is no choice")
        XCTAssertEqual(destinations.choices.map(\.id), ["internal"])
    }

    func testThisMacComesFirstWhateverTheListOrder() {
        let destinations = model([t7, thisMac])
        XCTAssertEqual(destinations.choices.map(\.id), ["internal", "t7-id"])
    }

    // MARK: Default and fallback (requirement 2.2)

    func testTheDefaultLibraryIsPreselected() {
        let destinations = model(
            [thisMac, t7],
            defaultDestination: DefaultInstallDestination(
                libraryId: t7.id, library: t7, fallbackBecauseNotConnected: false,
                notConnectedLibraryName: nil))
        XCTAssertEqual(destinations.defaultId, "t7-id")
        XCTAssertEqual(destinations.resolve(selection: nil).choice?.id, "t7-id")
        XCTAssertEqual(destinations.libraryForInstall(selection: nil), "t7-id")
        XCTAssertNil(destinations.fallbackNote)
    }

    func testThisMacAsTheDefaultIsPreselectedAndPassesNoLibrary() {
        let destinations = model([thisMac, t7])
        XCTAssertEqual(destinations.resolve(selection: nil).choice?.id, "internal")
        XCTAssertNil(destinations.libraryForInstall(selection: nil))
        XCTAssertNil(destinations.fallbackNote)
    }

    func testAnOfflineDefaultFallsBackToThisMacAndSaysSo() throws {
        let offlineDefault = DefaultInstallDestination(
            libraryId: "internal", library: thisMac, fallbackBecauseNotConnected: true,
            notConnectedLibraryName: "Spare")
        let destinations = model([thisMac, t7, spare], defaultDestination: offlineDefault)

        XCTAssertEqual(destinations.defaultId, "internal", "the offline default is NOT preselected")
        XCTAssertEqual(destinations.resolve(selection: nil).choice?.id, "internal")
        XCTAssertNil(destinations.libraryForInstall(selection: nil))
        let note = try XCTUnwrap(destinations.fallbackNote)
        XCTAssertTrue(note.contains("Spare"), note)
        XCTAssertTrue(note.contains("not connected"), note)
        XCTAssertTrue(note.contains("This Mac"), note)
    }

    func testTheFallbackNoteShowsEvenWhenNoPickerDoes() throws {
        // The default library is away and nothing else is connected: there is no
        // menu to draw, but the install still goes to This Mac and the user who
        // chose a drive for new downloads is told why it did not.
        let offlineDefault = DefaultInstallDestination(
            libraryId: "internal", library: thisMac, fallbackBecauseNotConnected: true,
            notConnectedLibraryName: "Spare")
        let destinations = model([thisMac, spare], defaultDestination: offlineDefault)
        XCTAssertFalse(destinations.isShown)
        XCTAssertNotNil(destinations.fallbackNote)
    }

    func testTheFallbackNameIsEscapedAndCut() throws {
        let name = "Spare\u{202E}\n" + String(repeating: "x", count: 200)
        let offlineDefault = DefaultInstallDestination(
            libraryId: "internal", library: thisMac, fallbackBecauseNotConnected: true,
            notConnectedLibraryName: name)
        let note = try XCTUnwrap(model([thisMac, t7], defaultDestination: offlineDefault).fallbackNote)
        XCTAssertFalse(note.contains("\u{202E}"), "a direction override must not reach the screen")
        XCTAssertFalse(note.contains("\n"))
        XCTAssertTrue(note.contains("\\u{202e}"), note)
        XCTAssertLessThan(note.count, 200)
    }

    // MARK: Free space per destination (requirement 2.1)

    func testEachChoiceCarriesItsOwnFreeSpace() throws {
        let destinations = model([thisMac, t7], free: ["t7-id": 900 * gb], thisMacFree: 100_000_000_000)
        let mac = try XCTUnwrap(destinations.choices.first)
        let drive = try XCTUnwrap(destinations.choices.last)

        XCTAssertEqual(mac.freeBytes, 100_000_000_000)
        XCTAssertEqual(drive.freeBytes, 900_000_000_000)
        XCTAssertEqual(drive.freeText, ModelLibraryRowModel.freeSpaceText(900 * gb, language: .english))
        XCTAssertEqual(drive.menuTitle, "T7, \(drive.freeText ?? "")")
        XCTAssertTrue(drive.menuTitle.contains("free"), drive.menuTitle)
    }

    func testFreeSpaceIsAbsentWhenTheSystemCannotRead() throws {
        let destinations = model([thisMac, t7], free: [:], thisMacFree: nil)
        for choice in destinations.choices {
            XCTAssertNil(choice.freeBytes, choice.name)
            XCTAssertNil(choice.freeText, "an unreadable capacity is absent, never zero")
        }
        XCTAssertEqual(destinations.choices.last?.menuTitle, "T7")
        XCTAssertFalse(destinations.choices.last?.menuTitle.contains("0") ?? true)
    }

    func testALibraryNeverBorrowsThisMacsFreeSpace() throws {
        // The drive's capacity could not be read. It must not show the boot
        // disk's number in its place.
        let destinations = model([thisMac, t7], free: [:], thisMacFree: 100_000_000_000)
        XCTAssertNil(destinations.choices.last?.freeBytes)
    }

    // MARK: Names come from a drive

    func testThisMacIsLabelledByItsFlagAndNeverByItsName() {
        let renamed = ModelLibrary(
            id: "internal", name: "Totally Not This Mac", root: "/Users/test/.turbospark/models",
            available: true, isDefault: true, isInternal: true)
        let destinations = model([renamed, t7], language: .german)
        XCTAssertEqual(destinations.choices.first?.name, AppStrings.localized("This Mac", language: .german))
        XCTAssertEqual(destinations.choices.first?.isThisMac, true)
    }

    func testALibraryNamedThisMacIsStillALibrary() {
        let impostor = LibraryFixtures.external("impostor-id", name: "This Mac")
        let destinations = model([thisMac, impostor])
        let shown = destinations.choices.last
        XCTAssertEqual(shown?.isThisMac, false, "the flag decides, not the name")
        XCTAssertEqual(shown?.id, "impostor-id")
        XCTAssertEqual(
            destinations.libraryForInstall(selection: "impostor-id"), "impostor-id",
            "an install into it names the library, not This Mac")
    }

    func testNamesAreEscapedAndCut() throws {
        let hostile = LibraryFixtures.external(
            "evil-id", name: "T7\u{202E}gnp.exe\u{0007}\n" + String(repeating: "y", count: 300))
        let destinations = model([thisMac, hostile])
        let choice = try XCTUnwrap(destinations.choices.last)

        XCTAssertFalse(choice.name.contains("\u{202E}"))
        XCTAssertFalse(choice.name.contains("\u{0007}"))
        XCTAssertFalse(choice.name.contains("\n"))
        XCTAssertTrue(choice.name.contains("\\u{202e}"), choice.name)
        XCTAssertLessThanOrEqual(choice.name.count, InstallDestinationModel.nameLimit)
        XCTAssertFalse(choice.menuTitle.contains("\u{202E}"))
    }

    func testANameWithNothingInItReadsAsAnUnknownLibrary() {
        let blank = LibraryFixtures.external("blank-id", name: "  ")
        let destinations = model([thisMac, blank])
        XCTAssertEqual(destinations.choices.last?.name, AppStrings.localized("Unknown library", language: .english))
    }

    func testThisMacIsNamedInEveryLanguage() {
        for language in AppLanguage.allCases where language != .system {
            let destinations = model([thisMac, t7], language: language)
            XCTAssertEqual(
                destinations.choices.first?.name, AppStrings.localized("This Mac", language: language),
                language.rawValue)
        }
    }

    // MARK: A choice that went stale (requirement 2.4)

    func testALibraryThatLeftAfterTheChoiceIsNeverQuietlyReplacedByThisMac() throws {
        let destinations = model([thisMac, spare, t7])

        guard case .notConnected(let name) = destinations.resolve(selection: "spare-id") else {
            return XCTFail("a listed library that is away resolves as not connected")
        }
        XCTAssertEqual(name, "Spare")
        XCTAssertEqual(
            destinations.libraryForInstall(selection: "spare-id"), "spare-id",
            "the install must name the drive the user chose, so it is refused, not rerouted")
        let stale = try XCTUnwrap(
            destinations.menuChoices(for: "spare-id").first { $0.id == "spare-id" },
            "the menu keeps showing what is selected")
        XCTAssertFalse(stale.isConnected)
        XCTAssertTrue(stale.menuTitle.contains("Spare"), stale.menuTitle)
        XCTAssertTrue(stale.menuTitle.contains("Not connected"), stale.menuTitle)
        XCTAssertEqual(destinations.menuChoices(for: "t7-id").map(\.id), ["internal", "t7-id"])
    }

    func testALibraryThatIsNoLongerListedResolvesAsUnknown() {
        let destinations = model([thisMac, t7])
        XCTAssertEqual(destinations.resolve(selection: "gone-id"), .unknown)
        XCTAssertEqual(destinations.libraryForInstall(selection: "gone-id"), "gone-id")
    }

    // MARK: The gate's view of a destination (requirement 2.3)

    func testTheGateReadsTheChosenLibrarysFreeSpaceNotThisMacs() {
        let destinations = model([thisMac, t7], free: ["t7-id": 900 * gb], thisMacFree: 5_000_000_000)

        let drive = destinations.gateDestination(library: "t7-id")
        XCTAssertEqual(drive.freeBytes, 900_000_000_000)
        XCTAssertEqual(drive.libraryName, "T7")
        XCTAssertNil(drive.notConnectedName)

        let mac = destinations.gateDestination(library: nil)
        XCTAssertEqual(mac.freeBytes, 5_000_000_000)
        XCTAssertNil(mac.libraryName, "This Mac keeps the sentence it always had")
        XCTAssertNil(mac.notConnectedName)

        XCTAssertEqual(destinations.gateDestination(library: "internal"), mac, "internal is This Mac")
    }

    func testTheGateNamesADriveThatIsAway() {
        let destinations = model([thisMac, spare])
        let away = destinations.gateDestination(library: "spare-id")
        XCTAssertEqual(away.notConnectedName, "Spare")
        XCTAssertNil(away.freeBytes, "an away drive has no free space to report")
    }

    func testTheGateNamesAnUnknownLibrary() {
        let unknown = model([thisMac, t7]).gateDestination(library: "gone-id")
        XCTAssertEqual(unknown.notConnectedName, AppStrings.localized("Unknown library", language: .english))
    }

    func testTheSelectionNilFollowsTheDefaultButTheLibraryNilIsThisMac() {
        let driveDefault = DefaultInstallDestination(
            libraryId: t7.id, library: t7, fallbackBecauseNotConnected: false, notConnectedLibraryName: nil)
        let destinations = model([thisMac, t7], defaultDestination: driveDefault)
        XCTAssertEqual(destinations.resolve(selection: nil).choice?.id, "t7-id")
        XCTAssertEqual(destinations.resolve(library: nil).choice?.id, "internal", "the API's nil is This Mac")
    }

    // MARK: Built from the live state

    func testItIsBuiltFromTheLibraryStateTheAppHolds() async throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("install-destination-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let service = FakeModelLibraryService(libraries: [thisMac, t7, spare])
        let state = LibraryState(
            service: service, probe: FakeDriveProbe(),
            dismissalsURL: directory.appendingPathComponent("d.json"),
            sleep: { _ in }, freeSpace: { url in url.path == "/Volumes/T7/TurboSpark" ? 321_000_000_000 : nil })
        await state.refresh()

        let destinations = InstallDestinationModel(state: state, thisMacFreeBytes: 77_000_000_000)

        XCTAssertEqual(destinations.choices.map(\.id), ["internal", "t7-id"], "the offline library is out")
        XCTAssertEqual(destinations.choices.last?.freeBytes, 321_000_000_000)
        XCTAssertEqual(destinations.choices.first?.freeBytes, 77_000_000_000)
    }
}
