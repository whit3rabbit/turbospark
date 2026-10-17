import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.4: what one Installed Models row says about where a model is and
/// whether it can be opened (requirements 3.1, 3.7, 5.3 and 5.7).
///
/// The model is pure, so each wording, glyph, rounding and order is a table row.
/// The view draws these strings and nothing else.
final class InstalledModelRowModelTests: XCTestCase {
    private let usb10 = DriveConnection(kind: .usb, bitsPerSecond: 10_000_000_000, volumeUUID: "u", isLocal: true)

    private func library(
        _ id: String = "t7-id", name: String = "T7", available: Bool = true
    ) -> ModelLibrary {
        LibraryFixtures.external(id, name: name, available: available)
    }

    private func row(
        _ alias: String = "qwen", library lib: ModelLibrary? = nil,
        availability: LibraryAvailability = .available, bytes: UInt64 = 7_000_000_000,
        kind: LibraryLocationKind? = nil, libraryName: String? = nil
    ) -> InstalledModel {
        let isThisMac = lib == nil
        let resolvedKind = kind ?? (isThisMac ? .thisMac : .library)
        return InstalledModel(
            alias: alias, repo: "owner/\(alias)", path: "/tmp/\(alias).gturbo", family: "qwen",
            installBytes: bytes, library: isThisMac ? "internal" : lib?.id,
            libraryName: libraryName ?? (isThisMac ? "This Mac" : lib?.name),
            installId: "\(lib?.id ?? "internal"):text/\(alias)", availability: availability,
            locationKind: resolvedKind,
            connectName: availability == .notConnected ? lib?.name : nil)
    }

    private func model(
        _ row: InstalledModel, library: ModelLibrary? = nil, connection: DriveConnection? = nil,
        measured: Double? = nil, language: AppLanguage = .english
    ) -> InstalledModelRowModel {
        InstalledModelRowModel(
            model: row, library: library, connection: connection, measuredMBps: measured,
            isMoE: false, physicalMemoryBytes: 64 << 30, language: language)
    }

    // MARK: Status

    func testAnOfflineRowReadsNotConnectedWithAGlyphAndIsDimmed() {
        let lib = library(available: false)
        let m = model(row(library: lib, availability: .notConnected), library: lib)

        XCTAssertEqual(m.status, .notConnected)
        XCTAssertEqual(m.statusText, "Not connected")
        XCTAssertEqual(m.statusGlyph, "minus.circle.fill")
        XCTAssertTrue(m.isDimmed)
    }

    func testAMissingRowReadsMissingWithItsOwnGlyphAndIsDimmed() {
        let lib = library()
        let m = model(row(library: lib, availability: .missing), library: lib, connection: usb10)

        XCTAssertEqual(m.status, .missing)
        XCTAssertEqual(m.statusText, "Missing")
        XCTAssertEqual(m.statusGlyph, "exclamationmark.triangle.fill")
        XCTAssertTrue(m.isDimmed)
    }

    func testAnAvailableRowHasNoStatusAndIsNotDimmed() {
        let m = model(row())

        XCTAssertNil(m.status)
        XCTAssertNil(m.statusText)
        XCTAssertNil(m.statusGlyph)
        XCTAssertFalse(m.isDimmed)
    }

    func testStatusIsAGlyphAndAWordNeverColorAlone() {
        for status in InstalledRowStatus.allCases {
            XCTAssertFalse(status.glyph.isEmpty, "\(status)")
            XCTAssertFalse(status.title.isEmpty, "\(status)")
        }
        XCTAssertEqual(
            Set(InstalledRowStatus.allCases.map(\.glyph)).count, InstalledRowStatus.allCases.count,
            "two states that share a shape are told apart by color alone")
        XCTAssertEqual(
            Set(InstalledRowStatus.allCases.map(\.title)).count, InstalledRowStatus.allCases.count)
    }

    func testTheStatusWordFollowsTheLanguage() {
        let lib = library(available: false)
        let german = model(row(library: lib, availability: .notConnected), library: lib, language: .german)
        XCTAssertNotEqual(german.statusText, "Not connected")
        XCTAssertFalse(german.statusText?.isEmpty ?? true)
    }

    // MARK: Location badge

    func testThisMacIsLocalizedFromTheLocationKindNotFromTheName() {
        let english = model(row())
        XCTAssertEqual(english.locationText, "This Mac")
        XCTAssertEqual(english.locationGlyph, "laptopcomputer")
        XCTAssertEqual(english.locationKind, .thisMac)

        let german = model(row(), language: .german)
        XCTAssertNotEqual(german.locationText, "This Mac")
        XCTAssertFalse(german.locationText?.isEmpty ?? true)
    }

    func testALibraryShowsItsNameThroughTheUntrustedTextFilter() {
        let hostile = "Backup\u{202E}gpj.exe\u{0007}"
        let lib = library(name: hostile)
        let m = model(row(library: lib), library: lib)

        XCTAssertEqual(m.locationText, UntrustedText.display(hostile))
        XCTAssertFalse(m.locationText?.contains("\u{202E}") ?? true, "a direction override is shown escaped")
        XCTAssertEqual(m.locationGlyph, "externaldrive")
        XCTAssertEqual(m.locationKind, .library)
    }

    func testALibraryNamedThisMacIsStillADriveBecauseTheKindDecides() {
        let lib = library(name: "This Mac")
        let m = model(row(library: lib), library: lib)

        XCTAssertEqual(m.locationKind, .library)
        XCTAssertEqual(m.locationGlyph, "externaldrive")
    }

    func testTheNameFallsBackToTheRowsNameWhenTheLibraryIsNotListed() {
        let lib = library()
        let m = model(row(library: lib, libraryName: "Old Drive"), library: nil)

        XCTAssertEqual(m.locationText, "Old Drive")
        XCTAssertEqual(m.locationKind, .library)
    }

    func testALibraryWithNoNameAtAllIsUnknown() {
        let lib = library()
        let nameless = InstalledModel(
            alias: "qwen", repo: "x/y", path: "/tmp/q", family: "qwen", library: lib.id, libraryName: nil,
            installId: "t7-id:text/qwen", availability: .available, locationKind: .library)

        XCTAssertEqual(model(nameless).locationText, "Unknown library")
    }

    func testACustomFolderAndAnUnknownLibraryHaveTheirOwnLabels() {
        let custom = InstalledModel(
            alias: "c", repo: "x/y", path: "/data/c", family: "qwen", library: nil, libraryName: "custom location",
            installId: "internal:text/c", availability: .available, locationKind: .custom)
        let unknown = InstalledModel(
            alias: "u", repo: "x/y", path: "/data/u", family: "qwen", library: "gone-id",
            libraryName: "unknown library", installId: "gone-id:text/u", availability: .missing,
            locationKind: .unknown)

        XCTAssertEqual(model(custom).locationText, "Custom Folder")
        XCTAssertEqual(model(custom).locationGlyph, "folder")
        XCTAssertEqual(model(unknown).locationText, "Unknown library")
        XCTAssertEqual(model(unknown).locationGlyph, "questionmark.circle")
    }

    func testARowTheAppScannedHasNoLocationBadge() {
        let scanned = InstalledModel(alias: "lm", repo: "lm studio/lm", path: "/x/lm", family: "qwen")

        let m = model(scanned)

        XCTAssertNil(m.locationText, "the source badge already says LM Studio or Custom Folder")
        XCTAssertEqual(m.locationKind, .scanned)
    }

    // MARK: The load-time estimate

    func testAModelOnAnExternalLibraryHasAnEstimateFromTheLinkRate() {
        let lib = library()
        // 7 GB at the 10 Gb/s tier's 700 MB/s is 10 s.
        let m = model(row(library: lib, bytes: 7_000_000_000), library: lib, connection: usb10)

        XCTAssertEqual(m.estimatedLoadSeconds, 10)
        XCTAssertEqual(m.estimateText, "About 10 sec to load")
    }

    func testAMeasuredSpeedWinsOverTheLinkRate() {
        let lib = library()
        // 7 GB at 100 MB/s is 70 s, shown as 2 min (rounded up to a minute).
        let m = model(row(library: lib, bytes: 7_000_000_000), library: lib, connection: usb10, measured: 100)

        XCTAssertEqual(m.estimatedLoadSeconds, 70)
        XCTAssertEqual(m.estimateText, "About 2 min to load")
    }

    func testThisMacHasNoEstimateAndNoSpeedAdvice() {
        let m = model(
            row(bytes: 30_000_000_000),
            connection: DriveConnection(kind: .internalDrive, bitsPerSecond: nil, volumeUUID: "x", isLocal: true))

        XCTAssertNil(m.estimateText)
        XCTAssertNil(m.estimatedLoadSeconds)
    }

    func testAnExternalModelWithNothingKnownAboutTheLinkHasNoEstimate() {
        let lib = library()
        let none = model(row(library: lib), library: lib, connection: nil)
        let unreadable = model(
            row(library: lib), library: lib,
            connection: DriveConnection(kind: .unknown, bitsPerSecond: nil, volumeUUID: nil, isLocal: true))

        XCTAssertNil(none.estimateText)
        XCTAssertNil(unreadable.estimateText, "an unknown link is never guessed")
    }

    func testAnOfflineOrMissingRowHasNoEstimate() {
        let away = library(available: false)
        let offline = model(row(library: away, availability: .notConnected), library: away, connection: usb10)
        let lib = library()
        let missing = model(row(library: lib, availability: .missing), library: lib, connection: usb10)

        XCTAssertNil(offline.estimateText)
        XCTAssertNil(missing.estimateText, "there is nothing to load")
    }

    func testTheEstimateIsRoundedUpAndNeverUnderstated() {
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(3), 3)
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(9), 9)
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(10), 10)
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(11), 15)
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(47), 50)
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(59), 60)
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(61), 120)
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(3_599), 3_600)
        XCTAssertEqual(InstalledModelRowModel.displaySeconds(3_601), 3_900)
        for seconds in [1, 7, 12, 59, 61, 600, 3_599, 4_000, 90_000] {
            XCTAssertGreaterThanOrEqual(InstalledModelRowModel.displaySeconds(seconds), seconds)
        }
    }

    func testTheDurationIsFormattedForTheLanguage() {
        XCTAssertEqual(InstalledModelRowModel.durationText(seconds: 120, language: .english), "2 min")
        XCTAssertEqual(InstalledModelRowModel.durationText(seconds: 3_900, language: .english), "1 hr, 5 min")
        XCTAssertEqual(InstalledModelRowModel.durationText(seconds: 20, language: .english), "20 sec")
        XCTAssertNotEqual(
            InstalledModelRowModel.durationText(seconds: 120, language: .german), "2 min",
            "the unit follows the language")
    }

    // MARK: Reasons Load is not offered

    func testLoadIsBlockedWithTheConnectStatementForAnOfflineRow() {
        let lib = library(name: "T7", available: false)
        let m = model(row(library: lib, availability: .notConnected), library: lib)

        XCTAssertEqual(m.loadBlockedReason, "Connect T7 to load")
    }

    func testTheConnectStatementNamesTheVolumeForACustomFolder() {
        let custom = InstalledModel(
            alias: "c", repo: "x/y", path: "/Volumes/Backup/c", family: "qwen", library: nil,
            libraryName: "custom location", installId: "internal:text/c", availability: .notConnected,
            locationKind: .custom, connectName: "Backup")

        XCTAssertEqual(model(custom).loadBlockedReason, "Connect Backup to load")
    }

    func testLoadIsBlockedForAMissingRowAndOpenForAnAvailableOne() {
        let lib = library()
        XCTAssertEqual(
            model(row(library: lib, availability: .missing), library: lib).loadBlockedReason,
            "The files for this model are missing.")
        XCTAssertNil(model(row()).loadBlockedReason)
    }

    // MARK: What leaves the list

    func testOnlyARowWithReachableFilesOffersDelete() {
        let lib = library()
        let away = library(available: false)
        let available = model(row(library: lib), library: lib)
        let offline = model(row(library: away, availability: .notConnected), library: away)
        let missing = model(row(library: lib, availability: .missing), library: lib)
        let scanned = model(InstalledModel(alias: "lm", repo: "lm studio/lm", path: "/x/lm", family: "qwen"))

        XCTAssertEqual([available, scanned].map(\.offersDelete), [true, true])
        XCTAssertEqual([available, scanned].map(\.offersRemoveFromList), [false, false])
        XCTAssertEqual([offline, missing].map(\.offersDelete), [false, false], "never Delete for a drive that is away")
        // The engine forgets a row only with its library connected, so a row on
        // a drive that is away offers no removal at all (it would always fail).
        XCTAssertEqual([offline, missing].map(\.offersRemoveFromList), [false, true])
    }

    // MARK: The view draws what the model says

    func testTheRowViewDrawsTheStatusAsAGlyphAndAWordAndTheActionBarFollowsTheModel() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let installation = packageRoot.appendingPathComponent("Sources/TurboSparkApp/Installation")
        let row = try String(
            contentsOf: installation.appendingPathComponent("InstalledModelRowView.swift"), encoding: .utf8)
        // Color alone would not survive color blindness: the glyph and the word
        // are both drawn, from the model, inside the one status block.
        let rowStatus = try XCTUnwrap(
            Self.block(in: row, from: "if let status = rowModel.status", to: "Feature & Source Badges"))
        XCTAssertTrue(rowStatus.contains("Image(systemName: status.glyph)"), "the glyph")
        XCTAssertTrue(rowStatus.contains("Text(verbatim: text)"), "the word")
        XCTAssertTrue(rowStatus.contains("rowModel.statusText"))
        XCTAssertTrue(row.contains("rowModel.estimateText"))
        XCTAssertTrue(row.contains("rowModel.locationText"))

        let bar = try String(
            contentsOf: installation.appendingPathComponent("InstalledModelActionBarView.swift"), encoding: .utf8)
        XCTAssertTrue(bar.contains("if rowModel.offersDelete"))
        XCTAssertTrue(bar.contains("if rowModel.offersRemoveFromList"))
        XCTAssertTrue(bar.contains("rowModel.loadBlockedReason"))

        let hero = try String(
            contentsOf: installation.appendingPathComponent("InstalledModelHeroHeaderView.swift"), encoding: .utf8)
        let heroStatus = try XCTUnwrap(
            Self.block(in: hero, from: "if let status = rowModel.status", to: "} else if isCurrentlyLoaded"))
        XCTAssertTrue(heroStatus.contains("Image(systemName: status.glyph)"), "the glyph")
        XCTAssertTrue(heroStatus.contains("Text(verbatim: text)"), "the word")
    }

    /// The text from the first `start` to the next `end` after it.
    private static func block(in source: String, from start: String, to end: String) -> String? {
        guard let from = source.range(of: start),
            let to = source.range(of: end, range: from.upperBound..<source.endIndex)
        else { return nil }
        return String(source[from.lowerBound..<to.lowerBound])
    }

    // MARK: What VoiceOver reads

    func testTheRowIsReadInTheOrderNameLibraryStatusSizeEstimate() {
        let lib = library(available: true)
        let available = model(row("qwen", library: lib, bytes: 7_000_000_000), library: lib, connection: usb10)
        let size = MetricFormat.storage(7_000_000_000)

        XCTAssertEqual(
            available.accessibilityParts,
            ["qwen", "T7", size, "About 10 sec to load"])

        let away = library(available: false)
        let offline = model(row("qwen", library: away, availability: .notConnected, bytes: 7_000_000_000), library: away)
        XCTAssertEqual(offline.accessibilityParts, ["qwen", "T7", "Not connected", size])
    }

    func testAnAccessibilityLabelNeverDropsTheStatusOfADimmedRow() {
        let away = library(available: false)
        let offline = model(row(library: away, availability: .notConnected), library: away)
        XCTAssertTrue(offline.accessibilityLabel.contains("Not connected"))
        let lib = library()
        let missing = model(row(library: lib, availability: .missing), library: lib)
        XCTAssertTrue(missing.accessibilityLabel.contains("Missing"))
    }
}
