import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.7 (requirements 3.2, 3.7, 5.4, 5.5, 5.7): whether a model may be loaded
/// now, and on what terms.
///
/// The gate is pure, so every rule of the speed advice is a table row here and the
/// thresholds are not re-derived: the cases use the advisor's own numbers (USB 2.0
/// is 40 MB/s, 5 Gb/s is 400 MB/s, a measurement under 500 MB/s limits a streaming
/// model, a model streams above memory minus 4 GiB).
final class LoadGateTests: XCTestCase {
    private let gib: UInt64 = 1 << 30
    private let memory: UInt64 = 64 << 30

    private let usb2 = DriveConnection(kind: .usb, bitsPerSecond: 480_000_000, volumeUUID: "u", isLocal: true)
    private let usb5 = DriveConnection(kind: .usb, bitsPerSecond: 5_000_000_000, volumeUUID: "u", isLocal: true)
    private let usb10 = DriveConnection(kind: .usb, bitsPerSecond: 10_000_000_000, volumeUUID: "u", isLocal: true)
    private let thunderbolt = DriveConnection(kind: .thunderbolt, bitsPerSecond: nil, volumeUUID: "u", isLocal: true)
    private let internalDrive = DriveConnection(kind: .internalDrive, bitsPerSecond: nil, volumeUUID: "u", isLocal: true)
    private let unknownLink = DriveConnection(kind: .unknown, bitsPerSecond: nil, volumeUUID: nil, isLocal: true)

    private func library(
        _ id: String = "t7-id", name: String = "T7", available: Bool = true
    ) -> ModelLibrary {
        LibraryFixtures.external(id, name: name, available: available)
    }

    private func row(
        _ alias: String = "qwen", in lib: ModelLibrary? = nil, bytes: UInt64 = 7_000_000_000,
        family: String = "qwen", availability: LibraryAvailability = .available
    ) -> InstalledModel {
        let isThisMac = lib == nil
        return InstalledModel(
            alias: alias, repo: "owner/\(alias)", path: "/tmp/\(alias).gturbo", family: family,
            installBytes: bytes, library: isThisMac ? "internal" : lib?.id,
            libraryName: isThisMac ? "This Mac" : lib?.name,
            installId: "\(lib?.id ?? "internal"):text/\(alias)", availability: availability,
            locationKind: isThisMac ? .thisMac : .library,
            connectName: availability == .notConnected ? lib?.name : nil)
    }

    private func gate(
        _ model: InstalledModel, library: ModelLibrary?, connection: DriveConnection?,
        measured: Double? = nil, isMoE: Bool = false
    ) -> LoadGate {
        LoadGate.evaluate(
            model: model, library: library, connection: connection, measuredMBps: measured,
            isMoE: isMoE, physicalMemoryBytes: memory)
    }

    // MARK: - Requirement 5.7: nothing for This Mac

    func testThisMacNeverWarnsEvenForAHugeMixtureOfExperts() {
        let huge = row(bytes: 500 * gib, family: "gemma4")
        let thisMac = LibraryFixtures.internalLibrary

        // The honest link, a hostile one (a caller that handed over another
        // volume's connection) and none at all.
        for connection in [internalDrive, usb2, usb5, nil] as [DriveConnection?] {
            XCTAssertEqual(
                gate(huge, library: thisMac, connection: connection, measured: 10, isMoE: true), .open,
                "This Mac with \(String(describing: connection))")
        }
    }

    func testAnInternalDriveLibraryNeverWarns() {
        let secondDisk = library("disk2-id", name: "Models")
        XCTAssertEqual(
            gate(row(in: secondDisk, bytes: 500 * gib), library: secondDisk, connection: internalDrive, isMoE: true),
            .open)
    }

    func testAFolderAnotherAppOwnsNeverWarns() {
        // A scan's placeholder row, and a custom folder: no library, no advice.
        let scanned = InstalledModel(alias: "lm", repo: "lm studio/lm", path: "/lm/lm.gguf", family: "gemma4",
            installBytes: 500 * gib)
        let custom = InstalledModel(
            alias: "c", repo: "x/c", path: "/c.gturbo", family: "gemma4", installBytes: 500 * gib,
            library: nil, libraryName: "custom location", installId: ":text/c", locationKind: .custom)
        XCTAssertEqual(gate(scanned, library: nil, connection: usb2, isMoE: true), .open)
        XCTAssertEqual(gate(custom, library: nil, connection: usb2, isMoE: true), .open)
        // Even with a library that happens to be at hand.
        XCTAssertEqual(gate(scanned, library: library(), connection: usb2, isMoE: true), .open)
    }

    // MARK: - Dense models on a slow link (the slow-load warning)

    func testAFastExternalLinkOpens() {
        let lib = library()
        XCTAssertEqual(gate(row(in: lib), library: lib, connection: usb10), .open)
        XCTAssertEqual(gate(row(in: lib), library: lib, connection: thunderbolt), .open)
    }

    func testADenseModelOnUsb2WarnsThatLoadingIsSlowWithTheEstimate() {
        let lib = library()
        // 7 GB at USB 2.0's 40 MB/s: 175 seconds.
        XCTAssertEqual(
            gate(row(in: lib, bytes: 7_000_000_000), library: lib, connection: usb2),
            .warn(.slowDrive(estimateSeconds: 175)))
    }

    func testASlowMeasurementOnAFastLinkIsTheSameDenseWarning() {
        let lib = library()
        // 100 MB/s measured is under the 150 MB/s floor: 7 GB takes 70 seconds.
        XCTAssertEqual(
            gate(row(in: lib, bytes: 7_000_000_000), library: lib, connection: usb10, measured: 100),
            .warn(.slowDrive(estimateSeconds: 70)))
    }

    func testAModelOfUnknownSizeStillWarnsButHasNoEstimate() {
        let lib = library()
        XCTAssertEqual(
            gate(row(in: lib, bytes: 0), library: lib, connection: usb2), .warn(.slowDrive(estimateSeconds: nil)))
    }

    func testADenseModelAboveMemoryOnAFiveGigabitLinkDoesNotStream() {
        // Dense models are mapped and wired at load: only mixture-of-experts
        // streams. 5 Gb/s is not a slow link.
        let lib = library()
        XCTAssertEqual(gate(row(in: lib, bytes: 100 * gib), library: lib, connection: usb5, isMoE: false), .open)
    }

    // MARK: - Mixture-of-experts above memory (the streaming warning)

    func testAMixtureOfExpertsAboveMemoryOnAFiveGigabitLinkWarnsThatTheDriveLimitsGeneration() {
        let lib = library()
        XCTAssertEqual(
            gate(row(in: lib, bytes: 100 * gib, family: "gemma4"), library: lib, connection: usb5, isMoE: true),
            .warn(.streamsFromDrive))
    }

    func testAMixtureOfExpertsAboveMemoryOnATenGigabitLinkOpens() {
        let lib = library()
        XCTAssertEqual(
            gate(row(in: lib, bytes: 100 * gib), library: lib, connection: usb10, isMoE: true), .open)
    }

    func testAMeasuredFourHundredMegabytesPerSecondWarnsAMixtureOfExperts() {
        let lib = library()
        XCTAssertEqual(
            gate(row(in: lib, bytes: 100 * gib), library: lib, connection: usb10, measured: 400, isMoE: true),
            .warn(.streamsFromDrive))
    }

    func testAMixtureOfExpertsThatFitsInMemoryDoesNotStream() {
        let lib = library()
        // 40 GiB is under 64 GiB minus 4 GiB.
        XCTAssertEqual(gate(row(in: lib, bytes: 40 * gib), library: lib, connection: usb5, isMoE: true), .open)
    }

    func testStreamingOutranksTheSlowLoadWhenBothApply() {
        // A USB 2.0 drive holding an oversized mixture-of-experts model: the
        // sentence that matters is the one about generation.
        let lib = library()
        XCTAssertEqual(
            gate(row(in: lib, bytes: 100 * gib), library: lib, connection: usb2, isMoE: true),
            .warn(.streamsFromDrive))
    }

    func testAnUnknownLinkWithNothingMeasuredHasNothingToWarnAbout() {
        let lib = library()
        XCTAssertEqual(
            gate(row(in: lib, bytes: 500 * gib), library: lib, connection: unknownLink, isMoE: true), .open)
        XCTAssertEqual(gate(row(in: lib, bytes: 500 * gib), library: lib, connection: nil, isMoE: true), .open)
    }

    // MARK: - A model that cannot be loaded is never `warn`

    func testAModelOnADriveThatIsAwayIsNotConnectedNeverWarn() {
        let away = library("away-id", name: "Archive", available: false)
        let offline = row(in: away, bytes: 500 * gib, family: "gemma4", availability: .notConnected)

        // Even with a link that would warn for an available model.
        for connection in [usb2, usb5, nil] as [DriveConnection?] {
            XCTAssertEqual(
                gate(offline, library: away, connection: connection, measured: 10, isMoE: true),
                .notConnected(libraryName: "Archive"))
        }
    }

    func testAModelWhoseFolderIsGoneIsMissingNeverWarn() {
        let lib = library()
        let gone = row(in: lib, bytes: 500 * gib, family: "gemma4", availability: .missing)
        XCTAssertEqual(gate(gone, library: lib, connection: usb2, measured: 10, isMoE: true), .missing)
    }

    func testTheDriveToConnectComesFromTheRowThenTheLibrary() {
        let away = library("away-id", name: "Listed Name", available: false)
        let named = InstalledModel(
            alias: "m", repo: "x/m", path: "/p", family: "qwen", library: "away-id", libraryName: "Row Name",
            installId: "away-id:text/m", availability: .notConnected, locationKind: .library, connectName: "Connect Name")
        XCTAssertEqual(gate(named, library: away, connection: nil), .notConnected(libraryName: "Connect Name"))

        let noConnectName = InstalledModel(
            alias: "m", repo: "x/m", path: "/p", family: "qwen", library: "away-id", libraryName: "Row Name",
            installId: "away-id:text/m", availability: .notConnected, locationKind: .library, connectName: "")
        XCTAssertEqual(gate(noConnectName, library: away, connection: nil), .notConnected(libraryName: "Row Name"))

        let nothing = InstalledModel(
            alias: "m", repo: "x/m", path: "/p", family: "qwen", library: "away-id", libraryName: nil,
            installId: "away-id:text/m", availability: .notConnected, locationKind: .library, connectName: nil)
        XCTAssertEqual(gate(nothing, library: away, connection: nil), .notConnected(libraryName: "Listed Name"))
        XCTAssertEqual(gate(nothing, library: nil, connection: nil), .notConnected(libraryName: nil))
    }

    func testAnAvailableModelInALibraryTheListNoLongerCallsConnectedIsNotWarned() {
        // The two answers can come from different refreshes. With nothing to
        // read a link from there is nothing to say.
        let away = library(available: false)
        XCTAssertEqual(
            gate(row(in: away, bytes: 500 * gib), library: away, connection: usb2, isMoE: true), .open)
    }

    func testOnlyAWarnedOrOpenGateCanLoad() {
        XCTAssertTrue(LoadGate.open.canLoad)
        XCTAssertTrue(LoadGate.warn(.streamsFromDrive).canLoad, "advice never prevents a load (requirement 5.5)")
        XCTAssertTrue(LoadGate.warn(.slowDrive(estimateSeconds: 9)).canLoad)
        XCTAssertFalse(LoadGate.notConnected(libraryName: "T7").canLoad)
        XCTAssertFalse(LoadGate.missing.canLoad)
    }

    func testTheManifestIsOnlyReadForAModelThatCouldBeWarnedAbout() {
        // Answering "is it mixture-of-experts" reads a manifest from the drive.
        var asked = 0
        func moe() -> Bool { asked += 1; return true }
        let lib = library()
        let away = library("away-id", name: "Archive", available: false)

        _ = LoadGate.evaluate(
            model: row(bytes: 500 * gib), library: LibraryFixtures.internalLibrary, connection: usb2,
            measuredMBps: nil, isMoE: moe(), physicalMemoryBytes: memory)
        _ = LoadGate.evaluate(
            model: row(in: away, availability: .notConnected), library: away, connection: nil,
            measuredMBps: nil, isMoE: moe(), physicalMemoryBytes: memory)
        _ = LoadGate.evaluate(
            model: row(in: lib, availability: .missing), library: lib, connection: usb2,
            measuredMBps: nil, isMoE: moe(), physicalMemoryBytes: memory)
        XCTAssertEqual(asked, 0)

        _ = LoadGate.evaluate(
            model: row(in: lib, bytes: 100 * gib), library: lib, connection: usb5,
            measuredMBps: nil, isMoE: moe(), physicalMemoryBytes: memory)
        XCTAssertEqual(asked, 1)
    }

    // MARK: - The sentences

    func testEachStateHasItsOwnEnglishSentence() {
        XCTAssertEqual(
            LoadGateText.statement(for: .notConnected(libraryName: "Spare"), language: .english),
            "Connect Spare to load")
        XCTAssertEqual(
            LoadGateText.statement(for: .missing, language: .english), "The files for this model are missing.")
        XCTAssertNil(LoadGateText.statement(for: .open, language: .english))
        XCTAssertNil(LoadGateText.statement(for: .warn(.streamsFromDrive), language: .english))

        let dense = LoadGateText.warning(for: .warn(.slowDrive(estimateSeconds: 175)), language: .english)
        let three = InstalledModelRowModel.durationText(seconds: 175, language: .english)
        XCTAssertEqual(dense, "This drive's connection is slow, so loading will take about \(three).")
        XCTAssertEqual(
            LoadGateText.warning(for: .warn(.slowDrive(estimateSeconds: nil)), language: .english),
            "This drive's connection is slow, so loading will take a long time.")
        XCTAssertEqual(
            LoadGateText.warning(for: .warn(.streamsFromDrive), language: .english),
            "This model likely streams from the drive, so generation speed will be limited by the drive.")
        XCTAssertNil(LoadGateText.warning(for: .open, language: .english))
        XCTAssertNil(LoadGateText.warning(for: .missing, language: .english))
        XCTAssertNil(LoadGateText.warning(for: .notConnected(libraryName: "T7"), language: .english))
    }

    func testADurationThatEndsInAPeriodDoesNotDoubleTheFullStop() {
        // German abbreviates minutes as "Min." and Arabic with a period of its
        // own: the sentence ends with the duration and with its own full stop.
        for language in AppLanguage.allCases where language != .system {
            for seconds in [3, 45, 175, 600, 7_200] {
                let text = LoadGateText.text(for: .slowDrive(estimateSeconds: seconds), language: language)
                XCTAssertFalse(text.contains("..") || text.contains("\u{3002}\u{3002}"), "\(language.rawValue) \(seconds): \(text)")
                XCTAssertTrue(
                    text.contains(InstalledModelRowModel.durationText(seconds: seconds, language: language)),
                    "\(language.rawValue): the duration must stay in the sentence")
            }
        }
        let german = LoadGateText.text(for: .slowDrive(estimateSeconds: 175), language: .german)
        XCTAssertTrue(german.hasSuffix("Min."), german)
        XCTAssertFalse(german.hasSuffix("Min.."), german)
    }

    func testTheTwoWarningsSayDifferentThings() {
        let dense = LoadGateText.warning(for: .warn(.slowDrive(estimateSeconds: 90)), language: .english)!
        let streaming = LoadGateText.warning(for: .warn(.streamsFromDrive), language: .english)!
        XCTAssertTrue(dense.contains("loading will take"), dense)
        XCTAssertFalse(dense.contains("generation"), dense)
        XCTAssertTrue(streaming.contains("generation speed will be limited by the drive"), streaming)
        XCTAssertFalse(streaming.contains("loading will take"), streaming)
        XCTAssertNotEqual(dense, streaming)
    }

    func testADriveNameIsEscapedAndNeverRaw() {
        let hostile = LoadGateText.statement(
            for: .notConnected(libraryName: "T7\u{202E}gpj\u{0007}"), language: .english)!
        XCTAssertFalse(hostile.contains("\u{202E}"))
        XCTAssertFalse(hostile.contains("\u{0007}"))
        XCTAssertTrue(hostile.hasPrefix("Connect T7"))

        XCTAssertEqual(
            LoadGateText.statement(for: .notConnected(libraryName: nil), language: .english),
            "Connect Unknown library to load")
        XCTAssertEqual(
            LoadGateText.statement(for: .notConnected(libraryName: ""), language: .english),
            "Connect Unknown library to load")
    }

    func testTheConfirmationTitleNamesTheModelEscaped() {
        XCTAssertEqual(LoadGateText.confirmationTitle(alias: "qwen", language: .english), "Load qwen anyway?")
        XCTAssertFalse(
            LoadGateText.confirmationTitle(alias: "q\u{202E}wen", language: .english).contains("\u{202E}"))
        XCTAssertEqual(LoadGateText.loadAnywayTitle(language: .english), "Load anyway")
    }

    func testTheWarningGlyphIsItsOwnShape() {
        XCTAssertEqual(LoadGateText.warningGlyph, "speedometer")
        XCTAssertNotEqual(LoadGateText.warningGlyph, InstalledRowStatus.notConnected.glyph)
        XCTAssertNotEqual(LoadGateText.warningGlyph, InstalledRowStatus.missing.glyph)
    }

    // MARK: - The decision a load request makes

    func testTheDecisionTable() {
        let warn = LoadGate.warn(.streamsFromDrive)
        // An open model loads.
        XCTAssertEqual(LoadDecision.decide(gate: .open, isResident: false, confirmed: false), .load)
        // A warned model asks first, and loads once the user said so.
        XCTAssertEqual(
            LoadDecision.decide(gate: warn, isResident: false, confirmed: false), .confirm(.streamsFromDrive))
        XCTAssertEqual(LoadDecision.decide(gate: warn, isResident: false, confirmed: true), .load)
        // A model that is not there is refused whatever the user confirmed.
        XCTAssertEqual(
            LoadDecision.decide(gate: .missing, isResident: false, confirmed: true), .refuse(.missing))
        XCTAssertEqual(
            LoadDecision.decide(gate: .notConnected(libraryName: "T7"), isResident: false, confirmed: true),
            .refuse(.notConnected(libraryName: "T7")))
        // The model already in memory is never warned about again, in either wording.
        XCTAssertEqual(LoadDecision.decide(gate: warn, isResident: true, confirmed: false), .load)
        XCTAssertEqual(
            LoadDecision.decide(gate: .warn(.slowDrive(estimateSeconds: 60)), isResident: true, confirmed: false),
            .load)
        // Being in memory does not make a model whose files are gone loadable.
        XCTAssertEqual(LoadDecision.decide(gate: .missing, isResident: true, confirmed: false), .refuse(.missing))
    }
}
