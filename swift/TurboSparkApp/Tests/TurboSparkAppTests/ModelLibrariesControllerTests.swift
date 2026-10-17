import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A speed test the test finishes by hand. Cancelling the calling task ends it
/// with `CancellationError`, as the real one does.
final class SpeedGate: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<DriveSpeedResult, Error>?
    private var recorded: [(url: URL, isInternal: Bool)] = []

    var calls: [(url: URL, isInternal: Bool)] { lock.lock(); defer { lock.unlock() }; return recorded }
    var isWaiting: Bool { lock.lock(); defer { lock.unlock() }; return continuation != nil }

    func run(_ url: URL, _ isInternal: Bool) async throws -> DriveSpeedResult {
        lock.lock()
        recorded.append((url, isInternal))
        lock.unlock()
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (waiter: CheckedContinuation<DriveSpeedResult, Error>) in
                lock.lock()
                if Task.isCancelled {
                    lock.unlock()
                    waiter.resume(throwing: CancellationError())
                    return
                }
                continuation = waiter
                lock.unlock()
            }
        } onCancel: {
            lock.lock()
            let waiter = continuation
            continuation = nil
            lock.unlock()
            waiter?.resume(throwing: CancellationError())
        }
    }

    func finish(_ result: Result<DriveSpeedResult, Error>) {
        lock.lock()
        let waiter = continuation
        continuation = nil
        lock.unlock()
        waiter?.resume(with: result)
    }
}

/// Task 6.2 (requirements 1.3 to 1.10, 5.6): the actions of the libraries
/// section, against the fake service. They change the list, refuse in one
/// sentence, never write the app error, and are disabled while an install runs.
@MainActor
final class ModelLibrariesControllerTests: LibraryStateTestCase {
    private var state: LibraryState!
    private var model: AppModel!
    private var controller: ModelLibrariesController!
    private var gate: SpeedGate!
    private var revealed: [String] = []
    private var announcements: [String] = []
    private var migration: MoveStoreMigration?
    private var baselineError: String?

    private let spare = "/Volumes/Spare"

    private func bootstrap(writes: Bool = true) async throws {
        service.with {
            $0.appliesWrites = writes
            $0.markers[spare] = nil
        }
        state = makeState()
        await state.refresh()
        model = await makeModel(using: state)
        baselineError = model.error
        let gate = SpeedGate()
        self.gate = gate
        revealed = []
        announcements = []
        controller = ModelLibrariesController(
            libraryState: state, model: model,
            migration: { [unowned self] in self.migration },
            speedTest: { url, isInternal in try await gate.run(url, isInternal) },
            reveal: { [unowned self] path in self.revealed.append(path) },
            announce: { [unowned self] text in self.announcements.append(text) },
            language: { .english })
    }

    private func scriptSpare() throws {
        let marker = try LibraryFixtures.marker("spare-id", name: "Spare", modelCount: 4)
        service.with { $0.markers[spare] = marker }
    }

    private func refuse(_ code: String, _ sentence: String = "x", status: TurboSparkError.Code = .invalidArgument) {
        service.with { $0.addError = TurboSparkError(code: status, message: "\(code): \(sentence)") }
    }

    // MARK: - Add

    func testAddingALibraryListsItAndSaysSo() async throws {
        try await bootstrap()
        try scriptSpare()
        await controller.add(path: spare)
        XCTAssertEqual(service.addCalls, [spare])
        XCTAssertTrue(state.libraries.contains { $0.id == "spare-id" }, "the add is followed by a refresh")
        XCTAssertEqual(controller.message?.kind, .info)
        XCTAssertEqual(controller.message?.text, "Added the library Spare.")
        XCTAssertEqual(announcements, ["Added the library Spare."], "completion is announced to VoiceOver")
        XCTAssertEqual(model.error, baselineError)
        XCTAssertNil(model.activeToast)
        XCTAssertEqual(service.mainThreadCalls, [], "the engine is never called on the main thread")
    }

    func testEveryRefusalShowsItsOwnSentenceInline() async throws {
        try await bootstrap()
        let cases: [(String, String)] = LibraryRefusal.allCases.map { ($0.rawValue, $0.rawValue) }
            + [("marker", "marker"), ("io", "io"), ("not_connected", "not_connected"), ("something_new", "other")]
        var seen = Set<String>()
        let before = state.libraries
        for (code, _) in cases {
            refuse(code, "/V/x overlaps the library Fast, which already covers it")
            await controller.add(path: spare)
            let message = try XCTUnwrap(controller.message, code)
            XCTAssertEqual(message.kind, .error, code)
            XCTAssertEqual(
                message.text,
                LibraryActionText.sentence(
                    for: TurboSparkError(code: .invalidArgument, message: "\(code): /V/x overlaps the library Fast, which already covers it"),
                    language: .english),
                code)
            if code != "something_new" { XCTAssertTrue(seen.insert(message.text).inserted, "\(code) repeats a sentence") }
            XCTAssertEqual(state.libraries, before, "a refusal changes nothing: \(code)")
        }
        XCTAssertEqual(model.error, baselineError, "an availability refusal is not an app error")
        XCTAssertNil(model.activeToast, "and it is not a toast either")
    }

    func testTheRefusalStaysUntilDismissedOrTheNextAdd() async throws {
        try await bootstrap()
        refuse("network")
        await controller.add(path: spare)
        let refusal = controller.message
        XCTAssertEqual(refusal?.kind, .error)

        await state.refresh()
        XCTAssertEqual(controller.message, refusal, "a refresh does not clear it")

        controller.dismissMessage()
        XCTAssertNil(controller.message)

        refuse("network")
        await controller.add(path: spare)
        XCTAssertNotNil(controller.message)
        service.with { $0.addError = nil }
        try scriptSpare()
        await controller.add(path: spare)
        XCTAssertEqual(controller.message?.kind, .info, "the next add replaces the refusal")
    }

    func testARefusalIsAnnouncedToVoiceOver() async throws {
        try await bootstrap()
        refuse("read_only")
        await controller.add(path: spare)
        XCTAssertEqual(announcements, [controller.message?.text])
    }

    func testTheEnginesBusyAnswerIsOneLocalizedSentence() async throws {
        try await bootstrap()
        refuse("busy", "an install is running", status: .busy)
        await controller.add(path: spare)
        XCTAssertEqual(
            controller.message?.text, "A model is installing or moving. Try again when it finishes.")
    }

    // MARK: - Busy

    func testNothingChangesWhileAnInstallRunsAndTheServiceIsNotAsked() async throws {
        try await bootstrap()
        try scriptSpare()
        for setBusy in [
            { self.model.isInstallingModel = true }, { self.model.isInstallingImageModel = true },
        ] {
            setBusy()
            XCTAssertTrue(controller.isChangeBlocked)
            await controller.add(path: spare)
            await controller.setDefault(id: "t7-id")
            await controller.remove(id: "t7-id")
            XCTAssertEqual(service.addCalls, [])
            XCTAssertEqual(service.writeCalls, [])
            XCTAssertEqual(
                controller.message?.text, "A model is installing or moving. Try again when it finishes.")
            XCTAssertEqual(controller.message?.kind, .error)
            model.isInstallingModel = false
            model.isInstallingImageModel = false
            controller.dismissMessage()
        }
        XCTAssertFalse(controller.isChangeBlocked)
    }

    // MARK: - Set as default

    func testSetAsDefaultChangesTheDefault() async throws {
        try await bootstrap()
        XCTAssertTrue(state.libraries.first { $0.isInternal }?.isDefault ?? false)
        await controller.setDefault(id: "t7-id")
        XCTAssertEqual(service.writeCalls, ["setDefault t7-id"])
        XCTAssertEqual(state.libraries.first { $0.id == "t7-id" }?.isDefault, true)
        XCTAssertEqual(state.libraries.filter(\.isDefault).count, 1, "exactly one default")
        XCTAssertEqual(state.destinationDefault.libraryId, "t7-id")
        XCTAssertEqual(controller.message?.kind, .info)
        XCTAssertEqual(controller.message?.text, "T7 is now the default library.")
        XCTAssertEqual(announcements, ["T7 is now the default library."])
        XCTAssertEqual(model.error, baselineError)
    }

    func testSettingThisMacAsDefaultNamesItInTheLanguage() async throws {
        try await bootstrap()
        await controller.setDefault(id: "t7-id")
        await controller.setDefault(id: "internal")
        XCTAssertEqual(controller.message?.text, "This Mac is now the default library.")
        XCTAssertEqual(state.destinationDefault.libraryId, "internal")
    }

    func testAFailedSetDefaultIsShownInline() async throws {
        try await bootstrap()
        service.with {
            $0.setDefaultError = TurboSparkError(code: .invalidArgument, message: "unknown_library: gone")
        }
        await controller.setDefault(id: "t7-id")
        XCTAssertEqual(controller.message?.text, "That library is no longer in the list.")
        XCTAssertEqual(model.error, baselineError)
    }

    // MARK: - Remove

    func testRemoveStopsListingTheLibrary() async throws {
        try await bootstrap()
        await controller.remove(id: "t7-id")
        XCTAssertEqual(service.writeCalls, ["remove t7-id"])
        XCTAssertFalse(state.libraries.contains { $0.id == "t7-id" })
        XCTAssertEqual(controller.message?.kind, .info)
        XCTAssertEqual(controller.message?.text, "Removed the library T7.")
        XCTAssertEqual(model.error, baselineError)
    }

    func testRemovingTheDefaultHandsItBackToThisMac() async throws {
        try await bootstrap()
        await controller.setDefault(id: "t7-id")
        await controller.remove(id: "t7-id")
        XCTAssertEqual(state.destinationDefault.libraryId, "internal")
        XCTAssertFalse(state.destinationDefault.fallbackBecauseNotConnected)
    }

    func testThisMacCannotBeRemovedAndTheEngineIsNeverAsked() async throws {
        try await bootstrap()
        await controller.remove(id: "internal")
        XCTAssertEqual(service.writeCalls, [], "removal of This Mac is refused before the service")
        XCTAssertEqual(controller.message?.text, "This Mac cannot be removed.")
        XCTAssertTrue(state.libraries.contains { $0.isInternal })
    }

    func testAFailedRemoveIsShownInlineAndTheListIsRefreshed() async throws {
        try await bootstrap()
        service.with {
            $0.removeError = TurboSparkError(code: .invalidArgument, message: "unknown_library: gone")
            $0.libraries.removeAll { $0.id == "t7-id" }
        }
        await controller.remove(id: "t7-id")
        XCTAssertEqual(controller.message?.text, "That library is no longer in the list.")
        XCTAssertFalse(
            state.libraries.contains { $0.id == "t7-id" }, "a stale list is refreshed after a refusal")
    }

    func testRemoveLeavesTheOfflineLibraryRowsWorkable() async throws {
        service.with { $0.libraries = [LibraryFixtures.internalLibrary, LibraryFixtures.external("t7-id", name: "T7", available: false)] }
        try await bootstrap()
        await controller.remove(id: "t7-id")
        XCTAssertEqual(state.libraries.map(\.id), ["internal"])
    }

    // MARK: - Offline libraries are state, not errors (requirement 3.4)

    func testAnOfflineLibraryRaisesNoErrorAndReadsNotConnectedWithItsDate() async throws {
        service.with {
            $0.libraries = [
                LibraryFixtures.internalLibrary, LibraryFixtures.external("t7-id", name: "T7", available: false),
            ]
        }
        try await bootstrap()
        XCTAssertEqual(model.error, baselineError)
        XCTAssertNil(model.activeToast)
        XCTAssertNil(controller.message)
        let rows = ModelLibraryRowModel.rows(from: state, language: .english)
        let offline = try XCTUnwrap(rows.first { $0.id == "t7-id" })
        XCTAssertEqual(offline.status, .notConnected)
        XCTAssertEqual(offline.statusText, "Not connected, last seen Oct 1, 2026")
        XCTAssertFalse(offline.statusGlyph.isEmpty)
    }

    func testAFailedListingRaisesNoErrorEither() async throws {
        try await bootstrap()
        service.with { $0.listError = FakeServiceError.scripted("down") }
        await state.refresh()
        XCTAssertEqual(model.error, baselineError)
        XCTAssertNil(controller.message)
        XCTAssertEqual(state.libraries.map(\.id), ["internal", "t7-id"], "the last good list stays")
    }

    // MARK: - Reveal

    func testRevealOpensOnlyAConnectedLibrary() async throws {
        service.with {
            $0.libraries = [
                LibraryFixtures.internalLibrary, self.t7,
                LibraryFixtures.external("away-id", name: "Away", available: false),
            ]
        }
        try await bootstrap()
        controller.revealInFinder(libraryId: "t7-id")
        controller.revealInFinder(libraryId: "away-id")
        controller.revealInFinder(libraryId: "nope")
        controller.revealInFinder(libraryId: "internal")
        XCTAssertEqual(revealed, ["/Volumes/T7/TurboSpark", "/Users/test/.turbospark/models"])
    }

    // MARK: - Test speed (requirement 5.6)

    func testASpeedTestShowsProgressThenStoresItsResult() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "t7-id")
        XCTAssertEqual(controller.speedTests["t7-id"], .running)
        let started = await eventually { self.gate.isWaiting }
        XCTAssertTrue(started)
        XCTAssertEqual(gate.calls.map { $0.url.path }, ["/Volumes/T7/TurboSpark"])
        XCTAssertEqual(gate.calls.map { $0.isInternal }, [false])

        // 1 GB in 2 s is 500 MB/s.
        gate.finish(.success(DriveSpeedResult(bytesRead: 1_000_000_000, seconds: 2)))
        await controller.waitForSpeedTest(libraryId: "t7-id")
        XCTAssertNil(controller.speedTests["t7-id"])
        XCTAssertEqual(state.measuredSpeeds["t7-id"], 500)
        XCTAssertEqual(announcements, ["Measured read speed: 500 MB/s"])
        XCTAssertEqual(model.error, baselineError)
    }

    func testASecondTestWhileOneRunsIsIgnored() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "t7-id")
        _ = await eventually { self.gate.isWaiting }
        controller.testSpeed(libraryId: "t7-id")
        XCTAssertEqual(gate.calls.count, 1)
        controller.cancelSpeedTest(libraryId: "t7-id")
        await controller.waitForSpeedTest(libraryId: "t7-id")
    }

    func testCancellingStopsTheTestAndShowsNothing() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "t7-id")
        _ = await eventually { self.gate.isWaiting }
        controller.cancelSpeedTest(libraryId: "t7-id")
        await controller.waitForSpeedTest(libraryId: "t7-id")
        XCTAssertNil(controller.speedTests["t7-id"], "a cancel is not a failure")
        XCTAssertNil(state.measuredSpeeds["t7-id"])
        XCTAssertEqual(announcements, [])
    }

    func testCancellingEveryTestWhenTheSectionLeavesTheScreen() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "t7-id")
        _ = await eventually { self.gate.isWaiting }
        controller.cancelAllSpeedTests()
        await controller.waitForSpeedTest(libraryId: "t7-id")
        XCTAssertNil(controller.speedTests["t7-id"])
    }

    func testThisMacIsRefusedWithASentenceAndNothingRuns() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "internal")
        XCTAssertEqual(controller.speedTests["internal"], .failed("This Mac does not need a speed test."))
        XCTAssertEqual(gate.calls.count, 0)
    }

    func testALibraryWithoutALargeFileGetsASentence() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "t7-id")
        _ = await eventually { self.gate.isWaiting }
        gate.finish(.failure(DriveSpeedTestError.noLargeFile))
        await controller.waitForSpeedTest(libraryId: "t7-id")
        XCTAssertEqual(
            controller.speedTests["t7-id"],
            .failed("This library has no large model file to read yet. Install a model on it, then test again."))
        XCTAssertNil(state.measuredSpeeds["t7-id"])
    }

    func testAnUnreadableDriveGetsASentenceWithTheSystemsReason() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "t7-id")
        _ = await eventually { self.gate.isWaiting }
        gate.finish(.failure(DriveSpeedTestError.unreadable("Input/output error")))
        await controller.waitForSpeedTest(libraryId: "t7-id")
        XCTAssertEqual(
            controller.speedTests["t7-id"], .failed("The speed test could not read this drive: Input/output error"))
    }

    func testANewTestClearsTheEarlierFailure() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "internal")
        XCTAssertNotNil(controller.speedTests["internal"])
        controller.dismissSpeedTestFailure(libraryId: "internal")
        XCTAssertNil(controller.speedTests["internal"])
    }

    func testAnOfflineOrUnknownLibraryIsNotTested() async throws {
        service.with {
            $0.libraries = [
                LibraryFixtures.internalLibrary, LibraryFixtures.external("away-id", name: "Away", available: false),
            ]
        }
        try await bootstrap()
        controller.testSpeed(libraryId: "away-id")
        controller.testSpeed(libraryId: "nope")
        XCTAssertEqual(gate.calls.count, 0)
        XCTAssertNil(controller.speedTests["away-id"])
    }

    func testAResultForADriveThatLeftMidTestIsDropped() async throws {
        try await bootstrap()
        controller.testSpeed(libraryId: "t7-id")
        _ = await eventually { self.gate.isWaiting }
        // The drive is unplugged and the next refresh sees it gone.
        service.with { $0.libraries = [LibraryFixtures.internalLibrary, LibraryFixtures.external("t7-id", name: "T7", available: false)] }
        await state.refresh()
        gate.finish(.success(DriveSpeedResult(bytesRead: 1_000_000_000, seconds: 2)))
        await controller.waitForSpeedTest(libraryId: "t7-id")
        XCTAssertNil(state.measuredSpeeds["t7-id"])
    }

    // MARK: - Stop waiting for a moved store (requirement 9.3)

    private func startPendingConversion() async throws -> FakeMovedStoreRecord {
        let record = FakeMovedStoreRecord("/Volumes/Old/Store")
        let pendingURL = directory.appendingPathComponent("moved_store_pending.json")
        let migration = MoveStoreMigration(
            service: service, libraryState: state, record: record, pendingURL: pendingURL,
            homeRoot: { "/Users/test/.turbospark" })
        self.migration = migration
        service.with {
            $0.importHandler = { _ in
                throw TurboSparkError(code: .invalidArgument, message: "not_connected: /Volumes/Old/Store is not connected")
            }
        }
        let (events, _) = AsyncStream<VolumeEvent>.makeStream()
        migration.start(events: events)
        await migration.waitUntilIdle()
        return record
    }

    func testStoppingTheWaitDropsThePlaceholderAndTheRecord() async throws {
        try await bootstrap()
        let record = try await startPendingConversion()
        XCTAssertEqual(state.displayLibraries.last?.id, LibraryState.pendingMovedStoreId)
        let pendingFile = directory.appendingPathComponent("moved_store_pending.json")
        XCTAssertTrue(FileManager.default.fileExists(atPath: pendingFile.path))

        controller.abandonConversion()

        XCTAssertNil(record.value, "the record is cleared")
        XCTAssertFalse(FileManager.default.fileExists(atPath: pendingFile.path), "the pending file is gone")
        XCTAssertNil(state.pendingMovedStore)
        XCTAssertEqual(state.displayLibraries.map(\.id), ["internal", "t7-id"])
        XCTAssertNil(controller.message)
        XCTAssertEqual(service.writeCalls, [], "no library is removed and no model is touched")
        XCTAssertEqual(model.error, baselineError)
    }

    func testStoppingTheWaitThatCannotClearTheRecordSaysSo() async throws {
        try await bootstrap()
        let record = try await startPendingConversion()
        record.failNextClears(1)
        controller.abandonConversion()
        XCTAssertEqual(
            controller.message?.text, "TurboSpark could not stop waiting for this drive. Try again.")
        XCTAssertEqual(controller.message?.kind, .error)
        XCTAssertNotNil(record.value, "the record is kept")
        XCTAssertNotNil(state.pendingMovedStore, "and so is the placeholder")

        controller.abandonConversion()
        XCTAssertNil(record.value, "a retry works")
        XCTAssertNil(state.pendingMovedStore)
    }

    func testStoppingTheWaitWithNothingPendingIsHarmless() async throws {
        try await bootstrap()
        controller.abandonConversion()
        XCTAssertNil(controller.message)
    }

    func testStoppingTheWaitIsNotBlockedByAnInstall() async throws {
        try await bootstrap()
        let record = try await startPendingConversion()
        model.isInstallingModel = true
        controller.abandonConversion()
        XCTAssertNil(record.value)
        model.isInstallingModel = false
    }
}
