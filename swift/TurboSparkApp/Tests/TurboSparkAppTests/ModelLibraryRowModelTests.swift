import AppKit
import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.1 (requirements 1.3, 3.8, 5.1, 5.2, 5.7): what one library row says,
/// as a pure value, so every field and every wording is pinned without a view.
final class ModelLibraryRowModelTests: XCTestCase {
    private let usb10 = DriveConnection(
        kind: .usb, bitsPerSecond: 10_000_000_000, volumeUUID: "U10", isLocal: true)
    private let usb2 = DriveConnection(
        kind: .usb, bitsPerSecond: 480_000_000, volumeUUID: "U2", isLocal: true)

    private var thisMac: ModelLibrary { LibraryFixtures.internalLibrary }

    private var t7: ModelLibrary {
        ModelLibrary(
            id: "t7-id", name: "T7", root: "/Volumes/T7/TurboSpark", available: true,
            modelCount: 12, isDefault: false, lastSeen: "2026-10-01")
    }

    private var offlineT7: ModelLibrary { LibraryFixtures.external("t7-id", name: "T7", available: false) }

    private func row(
        _ library: ModelLibrary, connection: DriveConnection? = nil, free: Int64? = nil,
        measured: Double? = nil, pending: Bool = false, pendingRoot: String? = nil,
        language: AppLanguage = .english
    ) -> ModelLibraryRowModel {
        ModelLibraryRowModel(
            library: library, isPendingConversion: pending, pendingRoot: pendingRoot,
            connection: connection, freeBytes: free, measuredMBps: measured, language: language)
    }

    // MARK: - This Mac

    func testThisMacShowsItsFieldsAndNothingAboutALink() {
        let internalDrive = DriveConnection(kind: .internalDrive, bitsPerSecond: nil)
        let row = row(thisMac, connection: internalDrive, free: 100_000_000_000)
        XCTAssertEqual(row.kind, .thisMac)
        XCTAssertEqual(row.title, "This Mac")
        XCTAssertEqual(row.location, "/Users/test/.turbospark/models")
        XCTAssertEqual(row.countText, "2 models")
        XCTAssertEqual(row.freeSpaceText, "100 GB free")
        XCTAssertNil(row.connectionText, "requirement 5.7: nothing about a link for This Mac")
        XCTAssertNil(row.slowLinkWarning)
        XCTAssertNil(row.measuredSpeedText)
        XCTAssertEqual(row.status, .connected)
        XCTAssertEqual(row.statusText, "Connected")
        XCTAssertTrue(row.isDefault)
    }

    func testThisMacIsLocalizedFromTheFlagNotTheEnglishName() {
        XCTAssertEqual(row(thisMac, language: .german).title, "Dieser Mac")
        XCTAssertEqual(row(thisMac, language: .arabic).title, "\u{647}\u{630}\u{627} \u{627}\u{644}\u{640} Mac")
        // A marker may name its library "This Mac". That is a name, not This
        // Mac: it keeps what the marker says and it can be removed.
        let lookalike = ModelLibrary(
            id: "lookalike", name: "This Mac", root: "/Volumes/X/TurboSpark", available: true)
        let lookalikeRow = row(lookalike, language: .german)
        XCTAssertEqual(lookalikeRow.kind, .library)
        XCTAssertEqual(lookalikeRow.title, "This Mac")
        XCTAssertTrue(lookalikeRow.offeredActions.contains(.remove))
    }

    func testThisMacCannotBeRemovedOrSpeedTested() {
        let row = row(thisMac)
        XCTAssertFalse(row.offeredActions.contains(.remove))
        XCTAssertFalse(row.offeredActions.contains(.testSpeed))
        XCTAssertNil(row.availability(of: .remove, changesBlocked: false))
        XCTAssertNil(row.availability(of: .testSpeed, changesBlocked: false))
        XCTAssertEqual(row.offeredActions, [.reveal], "it is already the default, so only Reveal remains")
    }

    func testThisMacOffersSetAsDefaultWhenAnotherLibraryIsDefault() {
        let notDefault = ModelLibrary(
            id: "internal", name: "This Mac", root: "/Users/test/.turbospark/models",
            available: true, modelCount: 2, isDefault: false, isInternal: true)
        XCTAssertEqual(row(notDefault).offeredActions, [.setDefault, .reveal])
    }

    // MARK: - A connected external library

    func testAConnectedLibraryShowsEveryField() {
        let row = row(t7, connection: usb10, free: 900_000_000_000, measured: 412.3)
        XCTAssertEqual(row.kind, .library)
        XCTAssertEqual(row.title, "T7")
        XCTAssertEqual(row.location, "/Volumes/T7/TurboSpark")
        XCTAssertEqual(row.countText, "12 models")
        XCTAssertEqual(row.freeSpaceText, "900 GB free")
        XCTAssertEqual(row.connectionText, "Connection: USB 10 Gb/s")
        XCTAssertEqual(row.measuredSpeedText, "Measured read speed: 412 MB/s")
        XCTAssertNil(row.slowLinkWarning)
        XCTAssertEqual(row.status, .connected)
        XCTAssertEqual(row.statusText, "Connected", "the last-seen date belongs to a library that is away")
        XCTAssertFalse(row.isDefault)
        XCTAssertEqual(row.offeredActions, [.setDefault, .testSpeed, .reveal, .remove])
    }

    func testTheDefaultLibraryIsMarkedAndNotOfferedAgain() {
        let isDefault = ModelLibrary(
            id: "t7-id", name: "T7", root: "/Volumes/T7/TurboSpark", available: true, isDefault: true)
        let row = row(isDefault, connection: usb10)
        XCTAssertTrue(row.isDefault)
        XCTAssertEqual(row.offeredActions, [.testSpeed, .reveal, .remove])
        XCTAssertTrue(row.accessibilityLabel.contains("Default library for new downloads"))
    }

    func testTheFreeSpaceIsAbsentWhenTheDriveCannotBeRead() {
        XCTAssertNil(row(t7, connection: usb10, free: nil).freeSpaceText)
    }

    func testTheCountIsPluralInEnglish() {
        func count(_ n: Int) -> String? {
            row(ModelLibrary(id: "x", name: "X", root: "/Volumes/X/TurboSpark", available: true, modelCount: n)).countText
        }
        XCTAssertEqual(count(0), "0 models")
        XCTAssertEqual(count(1), "1 model")
        XCTAssertEqual(count(2), "2 models")
    }

    func testTheCountFollowsEachLanguagesPluralRules() {
        func count(_ n: Int, _ language: AppLanguage) -> String? {
            row(
                ModelLibrary(id: "x", name: "X", root: "/Volumes/X/TurboSpark", available: true, modelCount: n),
                language: language
            ).countText
        }
        XCTAssertEqual(count(1, .german), "1 Modell")
        XCTAssertEqual(count(2, .german), "2 Modelle")
        XCTAssertEqual(count(1, .russian), "1 \u{43c}\u{43e}\u{434}\u{435}\u{43b}\u{44c}")
        XCTAssertEqual(count(2, .russian), "2 \u{43c}\u{43e}\u{434}\u{435}\u{43b}\u{438}")
        XCTAssertEqual(count(5, .russian), "5 \u{43c}\u{43e}\u{434}\u{435}\u{43b}\u{435}\u{439}")
        XCTAssertEqual(count(21, .russian), "21 \u{43c}\u{43e}\u{434}\u{435}\u{43b}\u{44c}")
        XCTAssertEqual(count(0, .arabic), "\u{644}\u{627} \u{62a}\u{648}\u{62c}\u{62f} \u{646}\u{645}\u{627}\u{630}\u{62c}")
        XCTAssertEqual(count(2, .hebrew), "\u{5e9}\u{5e0}\u{5d9} \u{5de}\u{5d5}\u{5d3}\u{5dc}\u{5d9}\u{5dd}")
        XCTAssertEqual(count(7, .japanese), "7\u{500b}\u{306e}\u{30e2}\u{30c7}\u{30eb}")
    }

    // MARK: - Connection text (requirement 5.1)

    func testTheConnectionReadsTypeAndSpeedOrUnknown() {
        func text(_ connection: DriveConnection?) -> String? {
            row(t7, connection: connection).connectionText
        }
        XCTAssertEqual(text(usb10), "Connection: USB 10 Gb/s")
        XCTAssertEqual(text(usb2), "Connection: USB 480 Mb/s")
        XCTAssertEqual(
            text(DriveConnection(kind: .usb, bitsPerSecond: 5_000_000_000)), "Connection: USB 5 Gb/s")
        XCTAssertEqual(text(DriveConnection(kind: .usb, bitsPerSecond: nil)), "Connection: USB")
        XCTAssertEqual(text(DriveConnection(kind: .thunderbolt, bitsPerSecond: nil)), "Connection: Thunderbolt")
        XCTAssertEqual(
            text(DriveConnection(kind: .internalDrive, bitsPerSecond: nil)), "Connection: Internal drive")
        XCTAssertEqual(text(DriveConnection(kind: .other("SATA"), bitsPerSecond: nil)), "Connection: SATA")
        // The system gave nothing: "Unknown", and a missing probe says the same.
        XCTAssertEqual(text(DriveConnection(kind: .unknown, bitsPerSecond: nil)), "Connection: Unknown")
        XCTAssertEqual(text(nil), "Connection: Unknown")
        // A rate without a type is still a rate.
        XCTAssertEqual(
            text(DriveConnection(kind: .unknown, bitsPerSecond: 10_000_000_000)), "Connection: 10 Gb/s")
    }

    func testAProtocolNameFromTheSystemIsEscaped() {
        let text = row(t7, connection: DriveConnection(kind: .other("SA\nTA\u{202E}"), bitsPerSecond: nil))
            .connectionText
        XCTAssertEqual(text, "Connection: SA\\nTA\\u{202e}")
    }

    // MARK: - The slow-link warning (requirements 5.2 and 5.7)

    func testAUsb2LinkWarnsAndNamesTheCable() {
        let row = row(t7, connection: usb2)
        XCTAssertEqual(
            row.slowLinkWarning, "Slow connection. Use a USB-C cable rated for 10 Gb/s or faster.")
    }

    func testAMeasurementUnder150MBpsWarnsEvenOnAFastLink() {
        XCTAssertNotNil(row(t7, connection: usb10, measured: 120).slowLinkWarning)
        XCTAssertNil(row(t7, connection: usb10, measured: 160).slowLinkWarning)
    }

    func testNoWarningWithoutAConnectionAndNeverForThisMacOrAnInternalDrive() {
        XCTAssertNil(row(t7, connection: nil).slowLinkWarning)
        XCTAssertNil(row(thisMac, connection: usb2, measured: 10).slowLinkWarning)
        XCTAssertNil(
            row(t7, connection: DriveConnection(kind: .internalDrive, bitsPerSecond: nil), measured: 10)
                .slowLinkWarning)
    }

    func testThisMacShowsNoConnectionEvenIfOneIsPassed() {
        XCTAssertNil(row(thisMac, connection: usb2).connectionText)
    }

    // MARK: - A library that is away (requirement 3.8)

    func testAnOfflineLibraryReadsNotConnectedWithItsLastSeenDate() {
        let row = row(offlineT7, connection: usb10, free: 5, measured: 300)
        XCTAssertEqual(row.status, .notConnected)
        XCTAssertEqual(row.statusText, "Not connected, last seen Oct 1, 2026")
        XCTAssertNil(row.location, "an unplugged library has no root")
        XCTAssertNil(row.freeSpaceText)
        XCTAssertNil(row.connectionText)
        XCTAssertNil(row.slowLinkWarning)
        XCTAssertNil(row.measuredSpeedText, "a speed belongs to a drive that is there")
        XCTAssertEqual(row.countText, "3 models", "the marker's count is still known")
    }

    func testStatusIsNeverColorAlone() {
        let connected = row(t7, connection: usb10)
        let away = row(offlineT7)
        XCTAssertFalse(connected.statusGlyph.isEmpty)
        XCTAssertFalse(away.statusGlyph.isEmpty)
        XCTAssertNotEqual(connected.statusGlyph, away.statusGlyph, "two states, two shapes")
        XCTAssertFalse(connected.statusText.isEmpty)
        XCTAssertFalse(away.statusText.isEmpty)
        XCTAssertNotEqual(connected.statusText, away.statusText)
    }

    func testAnOfflineLibraryNeverSeenSaysOnlyNotConnected() {
        let never = ModelLibrary(id: "x", name: "X", root: nil, available: false, lastSeen: nil)
        XCTAssertEqual(row(never).statusText, "Not connected")
    }

    func testTheLastSeenDateIsInTheLanguagesOwnFormat() {
        let german = row(offlineT7, language: .german).statusText
        XCTAssertTrue(german.hasPrefix("Nicht verbunden"), german)
        XCTAssertTrue(german.contains("2026"), german)
        XCTAssertFalse(german.contains("Oct"), german)
    }

    func testAMalformedLastSeenIsShownEscapedNotTrusted() {
        let odd = ModelLibrary(
            id: "x", name: "X", root: nil, available: false, lastSeen: "when\n\u{202E}ever")
        let text = row(odd).statusText
        XCTAssertFalse(text.contains("\n"))
        XCTAssertFalse(text.unicodeScalars.contains("\u{202E}"))
        XCTAssertTrue(text.contains("\\n"))
    }

    func testOfflineActionsAreDisabledWithAReasonAndTheRestStayAvailable() {
        let row = row(offlineT7)
        XCTAssertEqual(row.offeredActions, [.setDefault, .testSpeed, .reveal, .remove])
        let speed = row.availability(of: .testSpeed, changesBlocked: false)
        XCTAssertEqual(speed?.isEnabled, false)
        XCTAssertEqual(speed?.disabledReason, "Connect the drive to use this.")
        let reveal = row.availability(of: .reveal, changesBlocked: false)
        XCTAssertEqual(reveal?.isEnabled, false)
        XCTAssertEqual(reveal?.disabledReason, "Connect the drive to use this.")
        XCTAssertEqual(row.availability(of: .remove, changesBlocked: false)?.isEnabled, true)
        XCTAssertEqual(row.availability(of: .setDefault, changesBlocked: false)?.isEnabled, true)
        XCTAssertNil(row.revealPath)
    }

    func testRevealIsEnabledOnlyForAConnectedLibraryAndNamesItsRoot() {
        let row = row(t7, connection: usb10)
        XCTAssertEqual(row.availability(of: .reveal, changesBlocked: false)?.isEnabled, true)
        XCTAssertEqual(row.revealPath, "/Volumes/T7/TurboSpark")
    }

    // MARK: - Busy

    func testChangesAreDisabledNotHiddenWhileAnInstallOrMoveRuns() {
        let row = row(t7, connection: usb10)
        XCTAssertEqual(row.offeredActions, [.setDefault, .testSpeed, .reveal, .remove], "none are hidden")
        for action in [LibraryRowAction.setDefault, .remove] {
            let state = row.availability(of: action, changesBlocked: true)
            XCTAssertEqual(state?.isEnabled, false, "\(action)")
            XCTAssertEqual(
                state?.disabledReason, "Not available while a model is installing or moving.")
        }
        XCTAssertEqual(row.availability(of: .reveal, changesBlocked: true)?.isEnabled, true)
        XCTAssertEqual(row.availability(of: .testSpeed, changesBlocked: true)?.isEnabled, true)
    }

    // MARK: - The moved-store placeholder (requirement 9.3)

    func testThePendingPlaceholderIsNotConnectedAndOffersOnlyToStopWaiting() {
        let placeholder = ModelLibrary(id: "pending-moved-store", name: "Old Models", root: nil, available: false)
        let row = row(placeholder, pending: true, pendingRoot: "/Volumes/Old/Old Models")
        XCTAssertEqual(row.kind, .pendingConversion)
        XCTAssertEqual(row.title, "Old Models")
        XCTAssertEqual(row.location, "/Volumes/Old/Old Models")
        XCTAssertEqual(row.status, .notConnected)
        XCTAssertEqual(row.statusText, "Not connected")
        XCTAssertEqual(
            row.explanation, "Your earlier model store converts to a library when its drive is connected.")
        XCTAssertNil(row.countText, "nothing is known about its models yet")
        XCTAssertFalse(row.isDefault)
        XCTAssertEqual(row.offeredActions, [.abandonConversion])
        XCTAssertNil(row.availability(of: .remove, changesBlocked: false), "no normal Remove for the placeholder")
        XCTAssertNil(row.availability(of: .setDefault, changesBlocked: false))
        // Stopping the wait changes no library, so an install does not block it.
        XCTAssertEqual(row.availability(of: .abandonConversion, changesBlocked: true)?.isEnabled, true)
    }

    // MARK: - Names and paths from a drive

    func testNamesAndPathsFromADriveAreEscapedAndCut() {
        let hostile = ModelLibrary(
            id: "h", name: "evil\n\u{202E}gnp.exe", root: "/Volumes/a\nb/TurboSpark", available: true)
        let hostileRow = row(hostile)
        XCTAssertEqual(hostileRow.title, "evil\\n\\u{202e}gnp.exe")
        XCTAssertEqual(hostileRow.location, "/Volumes/a\\nb/TurboSpark")
        XCTAssertFalse(hostileRow.accessibilityLabel.contains("\n"))

        let long = ModelLibrary(
            id: "l", name: String(repeating: "n", count: 500),
            root: "/Volumes/" + String(repeating: "p", count: 500), available: true)
        XCTAssertLessThanOrEqual(row(long).title.count, UntrustedText.defaultLimit)
        XCTAssertTrue(row(long).title.hasSuffix("..."))
        XCTAssertEqual(row(long).location?.count, 509, "a long path is kept whole for the tooltip")
        let huge = ModelLibrary(
            id: "g", name: "G", root: "/Volumes/" + String(repeating: "q", count: 5000), available: true)
        XCTAssertLessThanOrEqual(row(huge).location?.count ?? 0, 1024)
    }

    // MARK: - VoiceOver

    func testTheAccessibilityLabelReadsInTheDocumentedOrder() {
        let isDefault = ModelLibrary(
            id: "t7-id", name: "T7", root: "/Volumes/T7/TurboSpark", available: true,
            modelCount: 12, isDefault: true)
        let label = row(isDefault, connection: usb2, free: 900_000_000_000, measured: 100).accessibilityLabel
        let order = [
            "T7", "Connected", "/Volumes/T7/TurboSpark", "12 models", "900 GB free",
            "Connection: USB 480 Mb/s", "Slow connection.", "Measured read speed: 100 MB/s",
            "Default library for new downloads",
        ]
        var searchFrom = label.startIndex
        for part in order {
            guard let range = label.range(of: part, range: searchFrom..<label.endIndex) else {
                return XCTFail("\"\(part)\" is missing or out of order in: \(label)")
            }
            searchFrom = range.upperBound
        }
    }

    func testTheOfflineAccessibilityLabelCarriesTheStatusAndDate() {
        let label = row(offlineT7).accessibilityLabel
        XCTAssertTrue(label.contains("Not connected, last seen Oct 1, 2026"), label)
    }

    // MARK: - Symbols

    func testEveryGlyphIsARealSymbol() {
        var names: [String] = []
        for status in LibraryRowStatus.allCases { names.append(status.glyph) }
        for action in LibraryRowAction.allCases { names.append(action.systemImage) }
        let rows = [
            row(thisMac), row(t7, connection: usb2), row(offlineT7),
            row(
                ModelLibrary(id: "p", name: "P", root: nil, available: false), pending: true,
                pendingRoot: "/Volumes/P"),
        ]
        for row in rows { names.append(row.iconName) }
        names.append(contentsOf: [LibraryRowGlyph.warning, LibraryRowGlyph.defaultBadge, LibraryRowGlyph.info])
        for name in names {
            XCTAssertNotNil(
                NSImage(systemSymbolName: name, accessibilityDescription: nil), "\(name) is not an SF Symbol")
        }
    }

    // MARK: - Every language

    func testEveryRowStringIsTranslatedIntoEveryLanguage() {
        let library = ModelLibrary(
            id: "t7-id", name: "T7", root: "/Volumes/T7/TurboSpark", available: true, modelCount: 12,
            isDefault: true, lastSeen: "2026-10-01")
        let english = row(library, connection: usb2, free: 900_000_000_000, measured: 100)
        let away = row(offlineT7)
        let placeholder = row(
            ModelLibrary(id: "p", name: "P", root: nil, available: false), pending: true, pendingRoot: "/V/P")
        let languages = AppLanguage.allCases.filter { $0 != .system }
        XCTAssertEqual(languages.count, 21)
        var untranslated: [String] = []
        for language in languages where language != .english {
            let connected = row(library, connection: usb2, free: 900_000_000_000, measured: 100, language: language)
            let offline = row(offlineT7, language: language)
            let pending = row(
                ModelLibrary(id: "p", name: "P", root: nil, available: false), pending: true, pendingRoot: "/V/P",
                language: language)
            let pairs: [(String, String?, String?)] = [
                ("count", connected.countText, english.countText),
                ("free", connected.freeSpaceText, english.freeSpaceText),
                ("connection", connected.connectionText, english.connectionText),
                ("status", connected.statusText, english.statusText),
                ("warning", connected.slowLinkWarning, english.slowLinkWarning),
                ("measured", connected.measuredSpeedText, english.measuredSpeedText),
                ("away", offline.statusText, away.statusText),
                ("explanation", pending.explanation, placeholder.explanation),
            ]
            for (name, translated, source) in pairs {
                guard let translated, let source else {
                    XCTFail("\(language.rawValue): \(name) is missing")
                    continue
                }
                if translated == source { untranslated.append("\(language.rawValue): \(name)") }
                XCTAssertFalse(translated.contains("%@") || translated.contains("%lld"), "\(language.rawValue): \(name): \(translated)")
            }
            XCTAssertNotEqual(connected.accessibilityLabel, english.accessibilityLabel)
        }
        XCTAssertEqual(untranslated, [], "these fell back to English")
    }

    // MARK: - Status and action titles are catalog keys

    func testStatusAndActionTitlesAreStable() {
        XCTAssertEqual(LibraryRowStatus.connected.title, "Connected")
        XCTAssertEqual(LibraryRowStatus.notConnected.title, "Not connected")
        XCTAssertEqual(LibraryRowAction.setDefault.title, "Set as default")
        XCTAssertEqual(LibraryRowAction.testSpeed.title, "Test speed")
        XCTAssertEqual(LibraryRowAction.reveal.title, "Reveal in Finder")
        XCTAssertEqual(LibraryRowAction.remove.title, "Remove")
        XCTAssertEqual(LibraryRowAction.abandonConversion.title, "Stop waiting for this drive")
        XCTAssertFalse(LibraryRowAction.remove.isDestructive)
        XCTAssertFalse(LibraryRowAction.abandonConversion.isDestructive)
        XCTAssertFalse(LibraryRowAction.setDefault.isDestructive)
    }
}

/// The list the section shows, from the state, in the order the design gives:
/// This Mac first, then the others as listed, then the placeholder.
@MainActor
final class ModelLibraryRowsOrderTests: LibraryStateTestCase {
    func testRowsPutThisMacFirstAndThePlaceholderLast() async {
        service.with {
            $0.libraries = [self.t7, LibraryFixtures.internalLibrary]
        }
        let state = makeState()
        await state.refresh()
        state.setPendingMovedStore(PendingMovedStore(root: "/Volumes/Old/Store", name: "Store"))
        let rows = ModelLibraryRowModel.rows(from: state, language: .english)
        XCTAssertEqual(rows.map(\.title), ["This Mac", "T7", "Store"])
        XCTAssertEqual(rows.map(\.kind), [.thisMac, .library, .pendingConversion])
        XCTAssertEqual(rows.last?.location, "/Volumes/Old/Store")
    }

    func testRowsCarryTheStatesFreeSpaceConnectionAndMeasuredSpeed() async {
        let state = makeState()
        await state.refresh()
        state.recordMeasuredSpeed(480, for: t7.id)
        let rows = ModelLibraryRowModel.rows(from: state, language: .english)
        let external = rows.first { $0.id == t7.id }
        XCTAssertEqual(external?.freeSpaceText, "900 GB free")
        XCTAssertEqual(external?.connectionText, "Connection: USB 10 Gb/s")
        XCTAssertEqual(external?.measuredSpeedText, "Measured read speed: 480 MB/s")
    }

    func testAnEmptyStateYieldsNoRows() {
        let state = makeState()
        XCTAssertTrue(ModelLibraryRowModel.rows(from: state, language: .english).isEmpty)
    }
}
