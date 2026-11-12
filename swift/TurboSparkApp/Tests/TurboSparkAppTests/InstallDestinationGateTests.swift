import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.9 (requirement 2.3): the install gate asks the CHOSEN library how much
/// room it has. The old gate asked the retired Move Store root, so a model bound
/// for a drive was checked against the boot disk and the other way round.
@MainActor
final class InstallDestinationGateTests: XCTestCase {
    private let thisMac = LibraryFixtures.internalLibrary
    private let t7 = LibraryFixtures.external("t7-id", name: "T7")
    private let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)

    private func destinations(
        thisMacFree: UInt64?, t7Free: Int64?
    ) -> InstallDestinationModel {
        InstallDestinationModel(
            libraries: [thisMac, t7, spare],
            freeBytes: t7Free.map { ["t7-id": $0] } ?? [:],
            destinationDefault: DefaultInstallDestination(
                libraryId: "internal", library: thisMac, fallbackBecauseNotConnected: false,
                notConnectedLibraryName: nil),
            thisMacFreeBytes: thisMacFree, language: .english)
    }

    private func decide(
        _ destinations: InstallDestinationModel, library: String?, installBytes: UInt64 = 18_000_000_000
    ) -> ModelInstallDecision {
        ModelInstallGate.decide(
            probeRunnable: nil, refusedBecause: nil, verdict: .streams, installBytes: installBytes,
            to: destinations.gateDestination(library: library))
    }

    // MARK: The chosen library's room decides

    func testPlentyOnTheDriveAndLittleOnTheMacAllowsAnInstallToTheDrive() {
        let d = destinations(thisMacFree: 5_000_000_000, t7Free: 900_000_000_000)
        XCTAssertEqual(decide(d, library: "t7-id"), .allowed, "the boot disk's number must not count")
    }

    func testLittleOnTheDriveAsksEvenWhenTheMacHasPlenty() throws {
        let d = destinations(thisMacFree: 900_000_000_000, t7Free: 20_000_000_000)
        let decision = decide(d, library: "t7-id")
        XCTAssertFalse(decision.isBlocked)
        let reason = try XCTUnwrap(decision.reason, "a drive with too little room must ask")
        XCTAssertTrue(reason.contains("T7"), "it names the drive: \(reason)")
        XCTAssertTrue(reason.contains(MetricFormat.storage(20_000_000_000)), "the drive's own free space: \(reason)")
        XCTAssertTrue(reason.contains(MetricFormat.storage(18_000_000_000)), "the model's size: \(reason)")
        XCTAssertFalse(
            reason.contains(MetricFormat.storage(900_000_000_000)), "no boot-disk number: \(reason)")
    }

    func testThisMacBehavesAsItAlwaysDid() throws {
        let d = destinations(thisMacFree: 20_000_000_000, t7Free: 900_000_000_000)
        let decision = decide(d, library: nil)
        let reason = try XCTUnwrap(decision.reason)
        XCTAssertTrue(reason.contains("Verified download ranges"), reason)
        XCTAssertTrue(reason.contains("there is \(MetricFormat.storage(20_000_000_000)) free"), reason)
        XCTAssertFalse(reason.contains("T7"), reason)
        XCTAssertEqual(decide(d, library: "internal"), decision, "internal is This Mac")
    }

    func testThisMacHasPlentyAndAllowsIt() {
        let d = destinations(thisMacFree: 900_000_000_000, t7Free: 20_000_000_000)
        XCTAssertEqual(decide(d, library: nil), .allowed)
    }

    func testAnUnreadableDriveCapacityDoesNotBlock() {
        let d = destinations(thisMacFree: 5_000_000_000, t7Free: nil)
        XCTAssertEqual(
            decide(d, library: "t7-id", installBytes: 500_000_000_000), .allowed,
            "an unknown reading is not a full disk, and it is never the boot disk's reading")
    }

    func testTheDownloadScratchIsCountedForADriveToo() throws {
        let d = destinations(thisMacFree: 900_000_000_000, t7Free: 25_000_000_000)
        let decision = ModelInstallGate.decide(
            probeRunnable: true, refusedBecause: nil, verdict: .streams, installBytes: 12_000_000_000,
            downloadBytes: 10_000_000_000, to: d.gateDestination(library: "t7-id"))
        XCTAssertTrue(decision.reason?.contains(MetricFormat.storage(22_000_000_000)) ?? false)
    }

    // MARK: A destination that is not connected (requirement 2.4)

    func testADriveThatIsAwayBlocksAndSaysToConnectIt() throws {
        let d = destinations(thisMacFree: 900_000_000_000, t7Free: 900_000_000_000)
        let decision = decide(d, library: "spare-id")
        XCTAssertTrue(decision.isBlocked)
        let reason = try XCTUnwrap(decision.reason)
        XCTAssertTrue(reason.contains("Spare"), reason)
        XCTAssertTrue(reason.localizedCaseInsensitiveContains("connect"), reason)
        XCTAssertFalse(reason.contains("free"), "no number from another disk: \(reason)")
    }

    func testALibraryThatIsGoneBlocks() {
        let d = destinations(thisMacFree: 900_000_000_000, t7Free: 900_000_000_000)
        XCTAssertTrue(decide(d, library: "gone-id").isBlocked)
    }

    func testAProbeRefusalStillOutranksAnAwayDrive() {
        let d = destinations(thisMacFree: 900_000_000_000, t7Free: 900_000_000_000)
        let why = "no decode flow here"
        let decision = ModelInstallGate.decide(
            probeRunnable: false, refusedBecause: why, verdict: nil, installBytes: nil,
            to: d.gateDestination(library: "spare-id"))
        XCTAssertEqual(decision.reason, why, "connecting a drive does not give a model a decode flow")
    }

    func testTheBlockedSentenceIsTranslatedInEveryLanguage() throws {
        let english = InstallDestinationModel(
            libraries: [thisMac, spare], freeBytes: [:],
            destinationDefault: DefaultInstallDestination(
                libraryId: "internal", library: thisMac, fallbackBecauseNotConnected: false,
                notConnectedLibraryName: nil),
            thisMacFreeBytes: nil, language: .english)
        let reference = ModelInstallGate.notConnectedSentence(
            name: english.gateDestination(library: "spare-id").notConnectedName ?? "", language: .english)
        for language in AppLanguage.allCases where language != .system {
            let sentence = ModelInstallGate.notConnectedSentence(name: "Spare", language: language)
            XCTAssertTrue(sentence.contains("Spare"), "\(language.rawValue): \(sentence)")
            XCTAssertFalse(sentence.contains("%@"), "\(language.rawValue): \(sentence)")
            if language != .english { XCTAssertNotEqual(sentence, reference, language.rawValue) }
        }
    }

    func testTheLowSpaceSentenceIsTranslatedInEveryLanguage() throws {
        for language in AppLanguage.allCases where language != .system {
            let sentence = ModelInstallGate.lowSpaceSentence(
                need: "18 GB", free: "20 GB", name: "T7", language: language)
            XCTAssertTrue(sentence.contains("18 GB") && sentence.contains("20 GB") && sentence.contains("T7"),
                "\(language.rawValue): \(sentence)")
            // The three arguments are plain specifiers, filled in the order the
            // call passes them: what the install needs, the library, its room.
            // A translation that reordered them would say the drive needs 18 GB
            // and has "T7" free.
            let need = try XCTUnwrap(sentence.range(of: "18 GB"), language.rawValue)
            let name = try XCTUnwrap(sentence.range(of: "T7"), language.rawValue)
            let free = try XCTUnwrap(sentence.range(of: "20 GB"), language.rawValue)
            XCTAssertLessThan(need.lowerBound, name.lowerBound, "\(language.rawValue): \(sentence)")
            XCTAssertLessThan(name.lowerBound, free.lowerBound, "\(language.rawValue): \(sentence)")
            XCTAssertFalse(sentence.contains("%@"), "\(language.rawValue): \(sentence)")
            if language != .english {
                XCTAssertNotEqual(
                    sentence,
                    ModelInstallGate.lowSpaceSentence(
                        need: "18 GB", free: "20 GB", name: "T7", language: .english),
                    language.rawValue)
            }
        }
    }

    // MARK: The live This Mac reading

    func testThisMacFreeSpaceIsReadFromTheHomeNotFromAnyDrive() {
        // The reading is live, so only its shape is pinned: a number, or nil.
        // A retired store root under /Volumes would have given nil for a drive
        // that is away, which is the behavior this replaced.
        let free = ModelInstallGate.freeSpaceOnThisMac()
        XCTAssertNotNil(free, "the home folder's volume always answers")
        XCTAssertGreaterThan(free ?? 0, 0)
    }
}
