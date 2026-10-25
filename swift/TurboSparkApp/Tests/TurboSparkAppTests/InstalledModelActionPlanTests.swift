import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.7 (requirements 3.2, 3.7, 5.4, 5.5): what the action bar of an
/// installed model offers, per state.
///
/// The plan is pure; the bar draws it and nothing else. A statement or a warning
/// is a glyph AND a sentence, and VoiceOver reads the sentence before the buttons.
final class InstalledModelActionPlanTests: XCTestCase {
    private let gib: UInt64 = 1 << 30
    private let usb2 = DriveConnection(kind: .usb, bitsPerSecond: 480_000_000, volumeUUID: "u", isLocal: true)
    private let usb5 = DriveConnection(kind: .usb, bitsPerSecond: 5_000_000_000, volumeUUID: "u", isLocal: true)
    private let usb10 = DriveConnection(kind: .usb, bitsPerSecond: 10_000_000_000, volumeUUID: "u", isLocal: true)

    private let t7 = LibraryFixtures.external("t7-id", name: "T7")
    private var spare: ModelLibrary { LibraryFixtures.external("spare-id", name: "Spare", available: false) }

    private func row(
        _ alias: String = "qwen", in lib: ModelLibrary? = nil, bytes: UInt64 = 7_000_000_000,
        family: String = "qwen", availability: LibraryAvailability = .available
    ) -> InstalledModel {
        InstalledModel(
            alias: alias, repo: "owner/\(alias)", path: "/tmp/\(alias).gturbo", family: family,
            installBytes: bytes, library: lib?.id ?? "internal", libraryName: lib?.name ?? "This Mac",
            installId: "\(lib?.id ?? "internal"):text/\(alias)", availability: availability,
            locationKind: lib == nil ? .thisMac : .library,
            connectName: availability == .notConnected ? lib?.name : nil)
    }

    private func rowModel(
        _ model: InstalledModel, library: ModelLibrary?, connection: DriveConnection?,
        measured: Double? = nil, isMoE: Bool = false
    ) -> InstalledModelRowModel {
        InstalledModelRowModel(
            model: model, library: library, connection: connection, measuredMBps: measured,
            isMoE: isMoE, physicalMemoryBytes: 64 << 30, language: .english)
    }

    private func plan(_ rowModel: InstalledModelRowModel, loaded: Bool = false) -> InstalledModelActionPlan {
        InstalledModelActionPlan(rowModel: rowModel, isCurrentlyLoaded: loaded)
    }

    // MARK: - The row model carries the gate

    func testTheRowModelCarriesTheGateAndItsSentences() {
        let dense = rowModel(row(in: t7), library: t7, connection: usb2)
        XCTAssertEqual(dense.gate, .warn(.slowDrive(estimateSeconds: 175)))
        XCTAssertEqual(dense.warningText, LoadGateText.text(for: .slowDrive(estimateSeconds: 175), language: .english))
        XCTAssertNil(dense.loadBlockedReason, "a warning never blocks a load")

        let offline = rowModel(row(in: spare, availability: .notConnected), library: spare, connection: nil)
        XCTAssertEqual(offline.gate, .notConnected(libraryName: "Spare"))
        XCTAssertEqual(offline.loadBlockedReason, "Connect Spare to load")
        XCTAssertNil(offline.warningText)

        let open = rowModel(row(), library: LibraryFixtures.internalLibrary, connection: nil)
        XCTAssertEqual(open.gate, .open)
        XCTAssertNil(open.warningText)
        XCTAssertNil(open.loadBlockedReason)
    }

    // MARK: - Per state

    func testAnOpenModelOffersLoadAndStartChatAndNoNotice() {
        let p = plan(rowModel(row(in: t7), library: t7, connection: usb10))

        XCTAssertEqual(p.loadControl, .load)
        XCTAssertNil(p.notice)
        XCTAssertEqual(p.startChat, .enabled)
        XCTAssertTrue(p.revealEnabled)
    }

    func testAThisMacModelOffersLoadWhateverTheLinkSays() {
        let p = plan(rowModel(row(bytes: 500 * gib, family: "gemma4"), library: LibraryFixtures.internalLibrary,
            connection: usb2, isMoE: true))
        XCTAssertEqual(p.loadControl, .load)
        XCTAssertNil(p.notice)
    }

    func testANotConnectedModelReplacesLoadWithAStatementAndOffersNoStartChat() {
        let p = plan(rowModel(row(in: spare, availability: .notConnected), library: spare, connection: nil))

        XCTAssertEqual(p.loadControl, .none, "no Load button at all")
        XCTAssertEqual(p.startChat, .hidden)
        XCTAssertEqual(
            p.notice,
            InstalledModelActionPlan.Notice(
                kind: .statement, text: "Connect Spare to load", glyph: InstalledRowStatus.notConnected.glyph))
        XCTAssertFalse(p.revealEnabled)
    }

    func testAMissingModelStatesThatItsFilesAreMissing() {
        let p = plan(rowModel(row(in: t7, availability: .missing), library: t7, connection: usb10))

        XCTAssertEqual(p.loadControl, .none)
        XCTAssertEqual(p.startChat, .hidden)
        XCTAssertEqual(
            p.notice,
            InstalledModelActionPlan.Notice(
                kind: .statement, text: "The files for this model are missing.",
                glyph: InstalledRowStatus.missing.glyph))
        XCTAssertFalse(p.revealEnabled)
    }

    func testADenseModelOnASlowLinkOffersLoadAnywayWithTheSlowLoadWarning() throws {
        let m = rowModel(row(in: t7), library: t7, connection: usb2)
        let p = plan(m)

        XCTAssertEqual(p.loadControl, .loadAnyway)
        let notice = try XCTUnwrap(p.notice)
        XCTAssertEqual(notice.kind, .warning)
        XCTAssertEqual(notice.glyph, LoadGateText.warningGlyph)
        XCTAssertEqual(notice.text, m.warningText)
        XCTAssertTrue(notice.text.contains("loading will take"), notice.text)
        XCTAssertTrue(p.revealEnabled, "the files are there")
    }

    func testAStreamingModelOffersLoadAnywayWithTheGenerationWarning() {
        let m = rowModel(row(in: t7, bytes: 100 * gib, family: "gemma4"), library: t7, connection: usb5, isMoE: true)
        let p = plan(m)

        XCTAssertEqual(p.loadControl, .loadAnyway)
        XCTAssertEqual(p.notice?.kind, .warning)
        XCTAssertEqual(p.notice?.text, m.warningText)
        XCTAssertTrue(p.notice?.text.contains("generation speed will be limited by the drive") == true)
    }

    func testStartChatIsDisabledForAnUnconfirmedWarnedModelWithTheWarningAsItsReason() {
        let m = rowModel(row(in: t7), library: t7, connection: usb2)
        let p = plan(m)

        // Start Chat would load the model and skip the warning.
        XCTAssertEqual(p.startChat, .disabled(reason: m.warningText!))
        XCTAssertNotEqual(p.startChat, .enabled)
    }

    func testALoadedModelOffersUnloadAndNoWarning() {
        let warned = rowModel(row(in: t7), library: t7, connection: usb2)
        XCTAssertNotNil(warned.warningText, "the fixture must be a warned model")

        let p = plan(warned, loaded: true)

        XCTAssertEqual(p.loadControl, .unload)
        XCTAssertNil(p.notice, "no warning on a model that is already loaded")
        XCTAssertEqual(p.startChat, .enabled, "it is loaded: Start Chat does not load anything")
    }

    func testALoadedOpenModelOffersUnload() {
        let p = plan(rowModel(row(), library: LibraryFixtures.internalLibrary, connection: nil), loaded: true)
        XCTAssertEqual(p.loadControl, .unload)
        XCTAssertNil(p.notice)
        XCTAssertEqual(p.startChat, .enabled)
    }

    // MARK: - Glyph plus text

    func testEveryNoticeIsAGlyphAndASentence() {
        let notices = [
            plan(rowModel(row(in: spare, availability: .notConnected), library: spare, connection: nil)).notice,
            plan(rowModel(row(in: t7, availability: .missing), library: t7, connection: usb10)).notice,
            plan(rowModel(row(in: t7), library: t7, connection: usb2)).notice,
            plan(rowModel(row(in: t7, bytes: 100 * gib), library: t7, connection: usb5, isMoE: true)).notice,
        ]
        XCTAssertEqual(notices.compactMap { $0 }.count, 4)
        for notice in notices.compactMap({ $0 }) {
            XCTAssertFalse(notice.glyph.isEmpty)
            XCTAssertFalse(notice.text.isEmpty)
            XCTAssertTrue(notice.text.hasSuffix(".") || notice.text.hasPrefix("Connect"), notice.text)
        }
        // Three different shapes for three different meanings.
        XCTAssertEqual(Set(notices.compactMap { $0?.glyph }).count, 3)
    }

    func testTheStatementsAreRenderedInEveryLanguage() {
        for language in AppLanguage.allCases where language != .system {
            let offline = InstalledModelRowModel(
                model: row(in: spare, availability: .notConnected), library: spare, connection: nil,
                measuredMBps: nil, isMoE: false, physicalMemoryBytes: 64 << 30, language: language)
            let planned = InstalledModelActionPlan(rowModel: offline, isCurrentlyLoaded: false)
            XCTAssertEqual(planned.notice?.text.contains("Spare"), true, language.rawValue)
            XCTAssertEqual(planned.notice?.text.contains("%@"), false, language.rawValue)
        }
    }
}
