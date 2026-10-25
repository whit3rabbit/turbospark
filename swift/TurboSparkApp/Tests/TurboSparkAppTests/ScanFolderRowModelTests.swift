import AppKit
import Foundation
import SwiftUI
import XCTest

@testable import TurboSparkApp

/// Task 6.12 (requirement 3.9): what one additional-scan-folder row says.
///
/// The model is pure, so each state's words, glyph, dimming, Reveal and spoken
/// label are pinned without a view. The row draws these fields and nothing else.
final class ScanFolderRowModelTests: XCTestCase {
    private let path = "/Volumes/T7/TurboSpark/models"

    private func row(
        _ reading: ScanFolderReading?, path: String? = nil, language: AppLanguage = .english
    ) -> ScanFolderRowModel {
        ScanFolderRowModel(path: path ?? self.path, reading: reading, language: language)
    }

    private var away: ScanFolderReading {
        ScanFolderReading(status: .notConnected(volumeName: "T7"), modelCount: 0)
    }

    private var absent: ScanFolderReading {
        ScanFolderReading(status: .notFound, modelCount: 0)
    }

    private func connected(_ count: Int) -> ScanFolderReading {
        ScanFolderReading(status: .connected, modelCount: count)
    }

    // MARK: - Not connected

    func testAFolderOnAMissingVolumeSaysNotConnectedWithAGlyphAndTheWord() {
        let model = row(away)
        XCTAssertEqual(model.status, .notConnected(volumeName: "T7"))
        XCTAssertEqual(model.statusText, "Not connected")
        // The glyph the library rows and the installed rows use for the same
        // condition, so the three surfaces read alike.
        XCTAssertEqual(model.statusGlyph, LibraryRowStatus.notConnected.glyph)
        XCTAssertEqual(model.statusGlyph, InstalledRowStatus.notConnected.glyph)
        XCTAssertTrue(model.isOffline)
        XCTAssertTrue(model.isDimmed)
    }

    func testANotConnectedFolderNeverShowsAZeroCount() {
        let model = row(away)
        XCTAssertNil(model.countText, "an unplugged drive is not a drive with zero models")
        let spoken = [model.statusText, model.hintText, model.accessibilityLabel, model.displayPath]
            .compactMap { $0 }.joined(separator: " | ")
        XCTAssertFalse(spoken.contains("Models"), spoken)
        XCTAssertFalse(spoken.contains("0"), spoken)
    }

    func testANotConnectedFolderNamesTheDriveToConnect() {
        let model = row(away)
        XCTAssertEqual(model.hintText, "Connect T7 to scan this folder.")
    }

    func testRevealIsDisabledWithAReasonWhileTheDriveIsAway() {
        let reveal = row(away).reveal
        XCTAssertFalse(reveal.isEnabled)
        XCTAssertEqual(reveal.disabledReason, "Connect the drive to use this.")
    }

    func testRemoveIsAlwaysOffered() {
        // Removing a folder from the list changes no file, so a dead path is
        // exactly when the user wants it.
        for reading in [away, absent, connected(2), nil] {
            let remove = row(reading).remove
            XCTAssertTrue(remove.isEnabled)
            XCTAssertNil(remove.disabledReason)
        }
    }

    func testTheSpokenLabelIsOneSentenceInTheOrderPathStatusDrive() {
        let model = row(away)
        XCTAssertEqual(
            model.accessibilityLabel,
            "/Volumes/T7/TurboSpark/models. Not connected. Connect T7 to scan this folder")
    }

    // MARK: - Folder not found

    func testAMissingFolderOnAMountedVolumeSaysFolderNotFoundWithADifferentGlyph() {
        let model = row(absent)
        XCTAssertEqual(model.statusText, "Folder not found")
        XCTAssertNil(model.countText)
        XCTAssertNil(model.hintText, "there is no drive to connect")
        XCTAssertTrue(model.isOffline)
        XCTAssertTrue(model.isDimmed)
        // A triangle, not the minus of a drive that is away: two causes, two shapes.
        XCTAssertEqual(model.statusGlyph, InstalledRowStatus.missing.glyph)
        XCTAssertNotEqual(model.statusGlyph, row(away).statusGlyph)
        XCTAssertEqual(
            model.accessibilityLabel, "/Volumes/T7/TurboSpark/models. Folder not found")
    }

    func testRevealIsDisabledForAFolderThatIsNotThere() {
        // Finder would open nothing: a button that silently does nothing is a
        // defect, so it says why instead.
        let reveal = row(absent).reveal
        XCTAssertFalse(reveal.isEnabled)
        XCTAssertEqual(reveal.disabledReason, "Folder not found")
    }

    // MARK: - Connected

    func testAConnectedFolderShowsItsCountAndNoStatus() {
        let model = row(connected(3))
        XCTAssertEqual(model.countText, "Models: 3")
        XCTAssertNil(model.statusText)
        XCTAssertNil(model.statusGlyph)
        XCTAssertNil(model.hintText)
        XCTAssertFalse(model.isOffline)
        XCTAssertFalse(model.isDimmed)
        XCTAssertTrue(model.reveal.isEnabled)
        XCTAssertNil(model.reveal.disabledReason)
    }

    func testAConnectedFolderWithNoModelsStillSaysZero() {
        // A real zero is information; only an unreachable folder has none.
        XCTAssertEqual(row(connected(0)).countText, "Models: 0")
    }

    func testAFolderNotReadYetShowsNeitherACountNorAStatus() {
        // Before the first reading lands the row must not claim zero models for
        // a drive it has not looked at.
        let model = row(nil)
        XCTAssertNil(model.countText)
        XCTAssertNil(model.statusText)
        XCTAssertNil(model.statusGlyph)
        XCTAssertFalse(model.isOffline)
        XCTAssertFalse(model.isDimmed)
        XCTAssertTrue(model.reveal.isEnabled)
    }

    // MARK: - Names from a drive

    func testTheVolumeNameIsEscapedAndCut() {
        let hostile = row(ScanFolderReading(status: .notConnected(volumeName: "T7\u{202E}gpj\u{0007}"), modelCount: 0))
        let text = [hostile.hintText, hostile.accessibilityLabel].compactMap { $0 }.joined()
        XCTAssertFalse(text.contains("\u{202E}"), "a direction override must not reach the screen")
        XCTAssertFalse(text.contains("\u{0007}"))
        XCTAssertTrue(hostile.hintText?.contains("T7") == true)

        let long = row(
            ScanFolderReading(status: .notConnected(volumeName: String(repeating: "A", count: 400)), modelCount: 0))
        XCTAssertLessThan(long.hintText?.count ?? .max, 120)
        XCTAssertTrue(long.hintText?.contains("...") == true)
    }

    func testThePathIsEscapedForDisplayButKeptWhole() {
        let hostile = "/Volumes/T7\u{202E}gpj/models"
        let model = row(away, path: hostile)
        XCTAssertEqual(model.path, hostile, "Reveal and Remove act on the real path")
        XCTAssertFalse(model.displayPath.contains("\u{202E}"))
        XCTAssertFalse(model.accessibilityLabel.contains("\u{202E}"))
    }

    func testAnOrdinaryPathIsShownAsItIs() {
        XCTAssertEqual(row(connected(1)).displayPath, path)
        XCTAssertEqual(row(connected(1), path: "~/Models/\u{00E9}t\u{00E9}").displayPath, "~/Models/\u{00E9}t\u{00E9}")
    }

    // MARK: - Languages

    func testTheStatusWordsFollowTheLanguage() {
        for language in AppLanguage.allCases where language != .system && language != .english {
            XCTAssertNotEqual(row(away, language: language).statusText, "Not connected", language.rawValue)
            XCTAssertNotEqual(row(absent, language: language).statusText, "Folder not found", language.rawValue)
            let hint = row(away, language: language).hintText ?? ""
            XCTAssertTrue(hint.contains("T7"), "\(language.rawValue): \(hint)")
            XCTAssertFalse(hint.contains("%@"), "\(language.rawValue): \(hint)")
            XCTAssertNotEqual(hint, "Connect T7 to scan this folder.", language.rawValue)
        }
    }
}

/// Task 6.12: the row and the section draw every state without clipping, and an
/// offline folder in the list does not stop the section from building.
@MainActor
final class ScanFolderRowViewTests: LibraryStateTestCase {
    private func rows(_ language: AppLanguage = .english) -> [ScanFolderRowModel] {
        [
            ScanFolderRowModel(
                path: "/Users/me/Models", reading: ScanFolderReading(status: .connected, modelCount: 3),
                language: language),
            ScanFolderRowModel(
                path: "/Volumes/Samsung Portable SSD T7 Shield 2 TB/Backups/Large Language Models/text",
                reading: ScanFolderReading(
                    status: .notConnected(volumeName: "Samsung Portable SSD T7 Shield 2 TB"), modelCount: 0),
                language: language),
            ScanFolderRowModel(
                path: "/Volumes/T7/vanished", reading: ScanFolderReading(status: .notFound, modelCount: 0),
                language: language),
            ScanFolderRowModel(path: "/Users/me/new", reading: nil, language: language),
        ]
    }

    private func hosted(_ content: some View, width: CGFloat) -> NSHostingView<AnyView> {
        let host = NSHostingView(rootView: AnyView(content.frame(width: width).appThemed()))
        host.frame = NSRect(x: 0, y: 0, width: width, height: 10)
        let window = NSWindow(contentRect: host.frame, styleMask: [.titled], backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        return host
    }

    private func drewSomething(_ host: NSHostingView<AnyView>) -> Bool {
        host.frame = NSRect(origin: .zero, size: host.fittingSize)
        host.layoutSubtreeIfNeeded()
        guard let bitmap = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { return false }
        host.cacheDisplay(in: host.bounds, to: bitmap)
        guard let data = bitmap.bitmapData else { return false }
        let first = data[0]
        return (0..<min(bitmap.bytesPerRow * bitmap.pixelsHigh, 400_000)).contains { data[$0] != first }
    }

    func testEveryRowStateRendersAtNarrowAndWideWidthsWithoutWidening() {
        for language in [AppLanguage.english, .arabic, .german] {
            for width in [420.0, 760.0] {
                for row in rows(language) {
                    let host = hosted(ScanFolderRowView(row: row, reveal: {}, remove: {}), width: width)
                    XCTAssertEqual(host.fittingSize.width, width, accuracy: 1, "\(language.rawValue) \(width) \(row.path)")
                    XCTAssertGreaterThan(host.fittingSize.height, 20, row.path)
                    XCTAssertTrue(drewSomething(host), "\(language.rawValue) \(width): \(row.path) drew nothing")
                }
            }
        }
    }

    func testAnOfflineRowIsTallerThanAConnectedOneBecauseItNamesTheDrive() {
        let all = rows()
        let connected = hosted(ScanFolderRowView(row: all[0], reveal: {}, remove: {}), width: 420)
        let offline = hosted(ScanFolderRowView(row: all[1], reveal: {}, remove: {}), width: 420)
        XCTAssertGreaterThan(offline.fittingSize.height, connected.fittingSize.height)
    }

    func testTheSectionBuildsAndRendersWithAnOfflineFolderConfigured() async {
        let model = await makeModel(using: makeState())
        model.customModelDirectories = ["/Volumes/TurboSpark-offline-\(UUID().uuidString)/models"]
        for width in [420.0, 760.0] {
            let host = hosted(CustomModelFoldersSectionView(model: model), width: width)
            XCTAssertEqual(host.fittingSize.width, width, accuracy: 1)
            XCTAssertTrue(drewSomething(host), "the section drew nothing at \(width)")
        }
    }
}
