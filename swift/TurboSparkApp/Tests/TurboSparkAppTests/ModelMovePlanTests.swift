import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.11 (requirements 7.1 to 7.5): the pure decisions behind the move sheet.
///
/// **THE REFUSAL FUNCTION IS ONE TABLE.** Every reason the sheet can give before
/// a byte is copied comes out of `ModelMovePlan.refusal`, so each case, the order
/// when several apply, the exact required and available text and the free-space
/// figure the engine is handed are pinned here without a view or an engine.
final class ModelMovePlanTests: XCTestCase {
    private let gb: UInt64 = 1_000_000_000

    // MARK: Fixtures

    private let t7 = LibraryFixtures.external("t7-id", name: "T7", volume: "T7")
    private let spare = LibraryFixtures.external("spare-id", name: "Spare", volume: "Spare")
    private var thisMac: ModelLibrary { LibraryFixtures.internalLibrary }

    private func textRow(
        _ alias: String = "qwen", library: ModelLibrary? = nil, bytes: UInt64? = nil,
        availability: LibraryAvailability = .available
    ) -> InstalledModel {
        let bytes = bytes ?? 4 * gb
        let id = library?.id ?? "internal"
        let isInternal = library?.isInternal ?? true
        return InstalledModel(
            alias: alias, repo: "owner/\(alias)",
            path: isInternal
                ? "/Users/test/.turbospark/models/text/\(alias).gturbo"
                : "\(library?.root ?? "/Volumes/\(id)/TurboSpark")/text/\(alias).gturbo",
            family: "qwen", installBytes: bytes, library: id, libraryName: library?.name ?? "This Mac",
            installId: "\(id):text/\(alias)", availability: availability,
            locationKind: isInternal ? .thisMac : .library,
            connectName: availability == .notConnected ? library?.name : nil)
    }

    private func subject(_ row: InstalledModel? = nil) -> MoveSubject {
        MoveSubject.make(text: row ?? textRow())!
    }

    /// The destinations as `InstallDestinationModel` builds them from a library
    /// list and each library's free space.
    private func destinations(
        libraries: [ModelLibrary]? = nil, free: [String: Int64]? = nil, thisMacFree: UInt64? = 100 * 1_000_000_000,
        defaultId: String = "internal"
    ) -> InstallDestinationModel {
        let all = libraries ?? [thisMac, t7, spare]
        let chosen = all.first { $0.id == defaultId }
        return InstallDestinationModel(
            libraries: all, freeBytes: free ?? ["t7-id": Int64(900 * gb), "spare-id": Int64(500 * gb)],
            destinationDefault: DefaultInstallDestination(
                libraryId: defaultId, library: chosen, fallbackBecauseNotConnected: false,
                notConnectedLibraryName: nil),
            thisMacFreeBytes: thisMacFree, language: .english)
    }

    private func plan(
        _ subject: MoveSubject? = nil, size: MoveSize = .known(4 * 1_000_000_000),
        selection: String? = "t7-id", app: MoveAppState = MoveAppState(),
        destinations: InstallDestinationModel? = nil
    ) -> ModelMovePlan {
        ModelMovePlan.make(
            subject: subject ?? self.subject(), size: size, destinations: destinations ?? self.destinations(),
            selection: selection, app: app, language: .english)
    }

    private func connected(_ id: String, name: String, free: UInt64?) -> InstallDestinationResolution {
        .connected(
            InstallDestinationChoice(
                id: id, isThisMac: id == "internal", name: name, freeBytes: free,
                freeText: free.map { MetricFormat.storage($0) }, isConnected: true, menuTitle: name))
    }

    private func audioRecord(
        alias: String = "kokoro", library: String? = nil, installId: String? = nil, bytes: UInt64 = 300_000_000
    ) throws -> AudioInstalledRecord {
        func field(_ key: String, _ value: String?) -> String {
            guard let value else { return "" }
            return "\"\(key)\":\"\(value)\","
        }
        let json = """
            {\(field("library", library))\(field("installId", installId))"alias":"\(alias)",
             "path":"/Users/test/.turbospark/models/audio/\(alias)","family":"kokoro","status":"ready",
             "installBytes":\(bytes)}
            """
        return try JSONDecoder().decode(AudioInstalledRecord.self, from: Data(json.utf8))
    }

    // MARK: Which rows offer a move

    func testATextRowOnThisMacOrInALibraryCanBeMoved() throws {
        let local = try XCTUnwrap(MoveSubject.make(text: textRow()))
        XCTAssertEqual(local.kind, .text)
        XCTAssertEqual(local.installId, "internal:text/qwen")
        XCTAssertEqual(local.libraryId, "internal")
        XCTAssertTrue(local.isThisMac)
        XCTAssertEqual(local.knownBytes, 4 * gb)
        XCTAssertEqual(local.path, "/Users/test/.turbospark/models/text/qwen.gturbo")

        let onDrive = try XCTUnwrap(MoveSubject.make(text: textRow(library: t7)))
        XCTAssertEqual(onDrive.libraryId, "t7-id")
        XCTAssertFalse(onDrive.isThisMac)
        XCTAssertEqual(onDrive.libraryName, "T7")
    }

    func testRowsTheEngineDoesNotOwnOrCannotReachAreNotOffered() {
        // The memberwise default the app's own scanners build for another app's
        // folder: no library, This Mac kind, a placeholder id.
        let scanned = InstalledModel(alias: "lm", repo: "lm/lm", path: "/lm/lm.gguf", family: "qwen", installBytes: 1)
        XCTAssertNil(MoveSubject.make(text: scanned), "a scanned row has no registry install")

        let custom = InstalledModel(
            alias: "c", repo: "o/c", path: "/data/c.gturbo", family: "qwen", installBytes: 1,
            library: nil, libraryName: "custom location", locationKind: .custom)
        XCTAssertNil(MoveSubject.make(text: custom), "a custom folder is not in any library")

        let unknown = InstalledModel(
            alias: "u", repo: "o/u", path: "/x/u.gturbo", family: "qwen", installBytes: 1,
            library: "gone-id", libraryName: "unknown library", installId: "gone-id:text/u",
            locationKind: .unknown)
        XCTAssertNil(MoveSubject.make(text: unknown), "a library that is no longer listed has nowhere to move from")

        XCTAssertNil(MoveSubject.make(text: textRow(library: t7, availability: .notConnected)))
        XCTAssertNil(MoveSubject.make(text: textRow(library: t7, availability: .missing)))
    }

    func testARowWithNoRecordedSizeIsMeasuredNotTrustedAsZero() throws {
        let row = try XCTUnwrap(MoveSubject.make(text: textRow(bytes: 0)))
        XCTAssertNil(row.knownBytes, "zero bytes is 'not recorded', and a zero requirement would pass any free-space check")
    }

    func testAnImageRowNeedsAnInstallIdAndTheStore() throws {
        let image = ImageInstalledModel(
            alias: "z", modelID: "o/z", revision: "r", path: "/Users/test/.turbospark/models/image/z",
            width: 1, height: 1, schedulerSteps: 1, library: "internal", libraryName: "This Mac",
            installId: "internal:image/z")
        let subject = try XCTUnwrap(MoveSubject.make(image: image, isManaged: true))
        XCTAssertEqual(subject.kind, .image)
        XCTAssertEqual(subject.installId, "internal:image/z")
        XCTAssertNil(subject.knownBytes, "an image listing carries no size: the sheet measures it")
        XCTAssertTrue(subject.isThisMac)

        XCTAssertNil(MoveSubject.make(image: image, isManaged: false), "a scanned image folder is not the engine's")
        let noId = ImageInstalledModel(
            alias: "z", modelID: "o/z", revision: "r", path: "/p", width: 1, height: 1, schedulerSteps: 1)
        XCTAssertNil(MoveSubject.make(image: noId, isManaged: true))
        let onDrive = ImageInstalledModel(
            alias: "z", modelID: "o/z", revision: "r", path: "/Volumes/T7/TurboSpark/image/z",
            width: 1, height: 1, schedulerSteps: 1, library: "t7-id", libraryName: "T7",
            installId: "t7-id:image/z")
        let moved = try XCTUnwrap(MoveSubject.make(image: onDrive, isManaged: true))
        XCTAssertEqual(moved.libraryId, "t7-id")
        XCTAssertFalse(moved.isThisMac)
    }

    func testAnAudioRecordIsIdentifiedByLibraryAndAlias() throws {
        let local = try XCTUnwrap(MoveSubject.make(audio: audioRecord()))
        XCTAssertEqual(local.kind, .audio)
        XCTAssertEqual(local.installId, "internal:audio/kokoro", "a This Mac row an older engine wrote has no library")
        XCTAssertEqual(local.knownBytes, 300_000_000)
        XCTAssertTrue(local.isThisMac)

        let onDrive = try XCTUnwrap(MoveSubject.make(audio: audioRecord(library: "t7-id")))
        XCTAssertEqual(onDrive.installId, "t7-id:audio/kokoro")
        XCTAssertFalse(onDrive.isThisMac)

        let named = try XCTUnwrap(MoveSubject.make(audio: audioRecord(library: "t7-id", installId: "t7-id:audio/other")))
        XCTAssertEqual(named.installId, "t7-id:audio/other", "an id the engine sent wins over one built here")

        let empty = try XCTUnwrap(MoveSubject.make(audio: audioRecord(bytes: 0)))
        XCTAssertNil(empty.knownBytes)
    }

    // MARK: Refusals, one by one

    private func refusal(
        _ app: MoveAppState, size: MoveSize = .known(4 * 1_000_000_000),
        destination: InstallDestinationResolution? = nil, subject: MoveSubject? = nil
    ) -> MoveRefusal? {
        ModelMovePlan.refusal(
            subject: subject ?? self.subject(), size: size,
            destination: destination ?? connected("t7-id", name: "T7", free: 900 * 1_000_000_000), app: app)
    }

    func testAMoveThatNothingStandsInTheWayOfIsNotRefused() {
        XCTAssertNil(refusal(MoveAppState()))
    }

    func testEachAppStateHasItsOwnRefusal() {
        let cases: [(WritableKeyPath<MoveAppState, Bool>, MoveRefusal)] = [
            (\.moveRunning, .moveRunning),
            (\.installRunning, .installRunning),
            (\.loadedInChat, .loadedInChat),
            (\.loadedInImages, .loadedInImages),
            (\.attachedToServer, .attachedToServer),
            (\.usedByHumanize, .usedByHumanize),
            (\.residentInAudio, .residentInAudio),
            (\.turnRunning, .turnRunning),
            (\.modelOpening, .modelOpening),
        ]
        for (flag, expected) in cases {
            var app = MoveAppState()
            app[keyPath: flag] = true
            XCTAssertEqual(refusal(app), expected, "\(expected)")
        }
        XCTAssertEqual(Set(cases.map { "\($0.1)" }).count, cases.count, "no two states share a reason")
    }

    /// The order the reasons are tried in: a move or an install anywhere stops
    /// every move; then what holds THIS model (chat, the server, Humanize, Audio);
    /// then what is busy around it (a turn, a load); then the destination.
    func testWhenSeveralReasonsApplyTheFirstInOrderIsGiven() {
        let ordered: [(WritableKeyPath<MoveAppState, Bool>, MoveRefusal)] = [
            (\.moveRunning, .moveRunning),
            (\.installRunning, .installRunning),
            (\.loadedInChat, .loadedInChat),
            (\.loadedInImages, .loadedInImages),
            (\.attachedToServer, .attachedToServer),
            (\.usedByHumanize, .usedByHumanize),
            (\.residentInAudio, .residentInAudio),
            (\.turnRunning, .turnRunning),
            (\.modelOpening, .modelOpening),
        ]
        var app = MoveAppState()
        for (flag, _) in ordered { app[keyPath: flag] = true }
        // A destination that would also be refused must not outrank an app state.
        let away = InstallDestinationResolution.notConnected(name: "Spare")
        for (flag, expected) in ordered {
            XCTAssertEqual(refusal(app, destination: away), expected)
            app[keyPath: flag] = false
        }
        XCTAssertEqual(
            refusal(app, destination: away), .destinationNotConnected(name: "Spare"),
            "with the app quiet the destination is what is left")
    }

    // MARK: Refusals about the destination

    func testNoOtherLibraryIsAStatementNotADeadMenu() {
        let plan = self.plan(selection: nil, destinations: destinations(libraries: [thisMac], free: [:]))
        XCTAssertEqual(plan.refusal, .noDestination)
        XCTAssertFalse(plan.canMove)
        XCTAssertTrue(plan.choices.isEmpty)
        XCTAssertNil(plan.selectedId)
        XCTAssertNil(plan.freeBytesForEngine)
    }

    func testTheSameLibraryIsRefused() {
        let same = connected("internal", name: "This Mac", free: 100 * gb)
        XCTAssertEqual(refusal(MoveAppState(), destination: same), .sameLibrary)
        let onT7 = subject(textRow(library: t7))
        XCTAssertEqual(
            refusal(MoveAppState(), destination: connected("t7-id", name: "T7", free: 900 * gb), subject: onT7),
            .sameLibrary)
    }

    func testADestinationThatIsAwayOrUnlistedIsRefusedByName() {
        XCTAssertEqual(
            refusal(MoveAppState(), destination: .notConnected(name: "Spare")),
            .destinationNotConnected(name: "Spare"))
        XCTAssertEqual(
            refusal(MoveAppState(), destination: .unknown), .destinationNotConnected(name: nil),
            "a library the list no longer holds has no name to give")
    }

    func testFreeSpaceThatCannotBeReadIsARefusalNeverAZero() {
        XCTAssertEqual(
            refusal(MoveAppState(), destination: connected("t7-id", name: "T7", free: nil)),
            .freeSpaceUnknown(name: "T7"))
    }

    func testTooLittleFreeSpaceIsRefusedWithBothFigures() {
        XCTAssertEqual(
            refusal(MoveAppState(), destination: connected("t7-id", name: "T7", free: 3 * gb)),
            .insufficientSpace(required: 4 * gb, available: 3 * gb, name: "T7"))
        // The boundary is the model's size exactly: equal fits, one byte less does not.
        XCTAssertNil(refusal(MoveAppState(), destination: connected("t7-id", name: "T7", free: 4 * gb)))
        XCTAssertEqual(
            refusal(MoveAppState(), destination: connected("t7-id", name: "T7", free: 4 * gb - 1)),
            .insufficientSpace(required: 4 * gb, available: 4 * gb - 1, name: "T7"))
        XCTAssertEqual(
            refusal(MoveAppState(), destination: connected("t7-id", name: "T7", free: 0)),
            .insufficientSpace(required: 4 * gb, available: 0, name: "T7"))
    }

    func testTheDestinationChecksRunInOrder() {
        // Away outranks everything about its space, which it cannot have.
        XCTAssertEqual(
            refusal(MoveAppState(), size: .unavailable, destination: .notConnected(name: "Spare")),
            .destinationNotConnected(name: "Spare"))
        // An unknown size is reported before a free-space figure that depends on it.
        XCTAssertEqual(
            refusal(MoveAppState(), size: .unavailable, destination: connected("t7-id", name: "T7", free: nil)),
            .sizeUnavailable)
        XCTAssertEqual(
            refusal(MoveAppState(), size: .known(4 * gb), destination: connected("t7-id", name: "T7", free: nil)),
            .freeSpaceUnknown(name: "T7"))
    }

    func testASizeStillBeingMeasuredIsWaitedForNotRefused() {
        let measuring = plan(size: .measuring)
        XCTAssertNil(measuring.refusal)
        XCTAssertFalse(measuring.canMove, "Move waits for the size")
        XCTAssertNil(measuring.freeBytesForEngine)
        XCTAssertNil(measuring.requiredText)
        XCTAssertTrue(measuring.isMeasuring)

        let unavailable = plan(size: .unavailable)
        XCTAssertEqual(unavailable.refusal, .sizeUnavailable)
        XCTAssertFalse(unavailable.canMove)
    }

    // MARK: What the sheet shows

    func testRequiredAndAvailableAreTheExactStorageText() {
        let plan = self.plan()
        XCTAssertEqual(plan.requiredText, MetricFormat.storage(4 * gb))
        XCTAssertEqual(plan.availableText, MetricFormat.storage(900 * gb))
        XCTAssertNil(plan.refusal)
        XCTAssertTrue(plan.canMove)
        XCTAssertEqual(plan.selectedId, "t7-id")
    }

    func testAnUnreadableFreeSpaceReadsUnknownAndNeverZero() {
        let plan = self.plan(destinations: destinations(free: [:]))
        XCTAssertEqual(plan.availableText, AppStrings.localized("Unknown", language: .english))
        XCTAssertEqual(plan.refusal, .freeSpaceUnknown(name: "T7"))
        XCTAssertFalse(plan.canMove)
        XCTAssertNil(plan.freeBytesForEngine, "an unmeasured destination is never passed to the engine as a number")
    }

    func testTheRefusalSentenceCarriesTheFiguresTheRowsShow() {
        let plan = self.plan(destinations: destinations(free: ["t7-id": Int64(3 * gb)]))
        XCTAssertEqual(plan.refusal, .insufficientSpace(required: 4 * gb, available: 3 * gb, name: "T7"))
        let sentence = plan.refusalText ?? ""
        XCTAssertTrue(sentence.contains("T7"))
        XCTAssertTrue(sentence.contains(MetricFormat.storage(3 * gb)), sentence)
        XCTAssertTrue(sentence.contains(MetricFormat.storage(4 * gb)), sentence)
    }

    // MARK: What the engine is handed

    func testTheEngineGetsTheMeasuredFreeSpaceOfAConnectedDestinationOnly() {
        let ok = plan()
        XCTAssertEqual(ok.freeBytesForEngine, Int64(900 * gb))
        XCTAssertEqual(ok.destinationId, "t7-id")

        // An unplugged destination is never probed with a negative number.
        let away = plan(selection: "spare-id", destinations: destinations(libraries: [thisMac, t7, LibraryFixtures.external("spare-id", name: "Spare", available: false)]))
        XCTAssertEqual(away.refusal, .destinationNotConnected(name: "Spare"))
        XCTAssertNil(away.freeBytesForEngine)

        // Not even a refused plan for a connected one holds a figure.
        let tight = plan(destinations: destinations(free: ["t7-id": Int64(1 * gb)]))
        XCTAssertNotNil(tight.refusal)
        XCTAssertNil(tight.freeBytesForEngine)
    }

    // MARK: The destination menu

    func testTheMenuListsEveryOtherConnectedDestinationAndNeverTheSource() {
        let fromThisMac = plan(selection: nil)
        XCTAssertEqual(fromThisMac.choices.map(\.id), ["t7-id", "spare-id"])

        let fromT7 = plan(subject(textRow(library: t7)), selection: nil)
        XCTAssertEqual(fromT7.choices.map(\.id), ["internal", "spare-id"], "This Mac is a destination for a library row")
        XCTAssertFalse(fromT7.choices.contains { $0.id == "t7-id" })
    }

    func testThePreselectedDestinationIsTheDefaultUnlessItIsTheSource() {
        let fromThisMac = plan(selection: nil, destinations: destinations(defaultId: "internal"))
        XCTAssertEqual(fromThisMac.selectedId, "t7-id", "the default is the source: the first other destination")

        let toDefault = plan(subject(textRow(library: t7)), selection: nil, destinations: destinations(defaultId: "spare-id"))
        XCTAssertEqual(toDefault.selectedId, "spare-id")

        let chosen = plan(selection: "spare-id")
        XCTAssertEqual(chosen.selectedId, "spare-id", "an explicit choice wins")
    }

    func testADriveChosenAndThenPulledStaysChosenAndIsRefused() {
        let libraries = [thisMac, t7, LibraryFixtures.external("spare-id", name: "Spare", available: false)]
        let plan = self.plan(selection: "spare-id", destinations: destinations(libraries: libraries))
        XCTAssertTrue(plan.choices.contains { $0.id == "spare-id" && !$0.isConnected })
        XCTAssertEqual(plan.selectedId, "spare-id")
        XCTAssertEqual(plan.refusal, .destinationNotConnected(name: "Spare"))
        XCTAssertFalse(plan.canMove)
    }

    // MARK: Progress

    func testProgressIsIndeterminateUntilTheFirstCallback() {
        var progress = MoveProgress(
            installId: "internal:text/qwen", alias: "qwen", sourceName: "This Mac", destinationName: "T7")
        XCTAssertEqual(progress.phase, .starting)
        XCTAssertNil(progress.fraction)
        XCTAssertFalse(progress.hasProgress)

        progress.apply(done: 0, total: 100)
        XCTAssertEqual(progress.phase, .copying)
        XCTAssertEqual(progress.fraction, 0)

        progress.apply(done: 50, total: 100)
        XCTAssertEqual(progress.fraction, 0.5)
        progress.apply(done: 30, total: 100)
        XCTAssertEqual(progress.done, 50, "a late, smaller figure never moves the bar back")

        progress.apply(done: 100, total: 100)
        XCTAssertEqual(progress.phase, .finishing, "all the bytes are copied; the check and the removal follow")
        XCTAssertEqual(progress.fraction, 1)

        progress.cancelRequested = true
        XCTAssertEqual(progress.phase, .cancelling)
    }

    func testAZeroTotalNeverDividesAndKeepsTheBarIndeterminate() {
        var progress = MoveProgress(installId: "i", alias: "a", sourceName: "s", destinationName: "d")
        progress.apply(done: 0, total: 0)
        XCTAssertNil(progress.fraction)
        XCTAssertNotEqual(progress.phase, .finishing)
    }

    func testTheThrottleForwardsTheFirstTheLastAndOneEveryInterval() {
        final class Clock: @unchecked Sendable { var now: TimeInterval = 1_000 }
        let clock = Clock()
        let throttle = MoveProgressThrottle(minimumInterval: 0.1, now: { clock.now })
        XCTAssertTrue(throttle.shouldForward(done: 0, total: 100), "the first callback starts the bar")
        clock.now += 0.01
        XCTAssertFalse(throttle.shouldForward(done: 10, total: 100), "a flood is cut")
        clock.now += 0.02
        XCTAssertFalse(throttle.shouldForward(done: 20, total: 100))
        clock.now += 0.2
        XCTAssertTrue(throttle.shouldForward(done: 30, total: 100))
        clock.now += 0.001
        XCTAssertTrue(throttle.shouldForward(done: 100, total: 100), "the last figure is never dropped")
    }
}
