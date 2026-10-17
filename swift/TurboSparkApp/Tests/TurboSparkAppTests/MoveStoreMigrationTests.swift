import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// A machine-level record the test controls. `clear` can be told to fail, which
/// is the shape of a crash between the import and clearing the record.
final class FakeMovedStoreRecord: MovedStoreRecord, @unchecked Sendable {
    private let lock = NSLock()
    private var root: String?
    private var failuresLeft = 0
    private var clears = 0
    /// Runs inside `clear`, so a test can look at the world at that moment.
    var onClear: (@Sendable () -> Void)?

    init(_ root: String?) { self.root = root }

    var value: String? { lock.lock(); defer { lock.unlock() }; return root }
    var clearCalls: Int { lock.lock(); defer { lock.unlock() }; return clears }

    /// Rewrites the record, as the legacy Move Store sheet does when it saves
    /// a new root.
    func replace(with newRoot: String?) {
        lock.lock(); defer { lock.unlock() }
        root = newRoot
    }

    func failNextClears(_ count: Int) {
        lock.lock(); defer { lock.unlock() }
        failuresLeft = count
    }

    func load() -> String? { value }

    func clear() -> Bool {
        onClear?()
        lock.lock(); defer { lock.unlock() }
        clears += 1
        if failuresLeft > 0 {
            failuresLeft -= 1
            return false
        }
        root = nil
        return true
    }
}

/// An ordered log that any thread can append to.
final class MigrationLog: @unchecked Sendable {
    private let lock = NSLock()
    private var items: [String] = []
    func add(_ entry: String) { lock.lock(); items.append(entry); lock.unlock() }
    var entries: [String] { lock.lock(); defer { lock.unlock() }; return items }
}

/// Task 5.10 (requirements 1.1, 9.2, 9.3): a moved store the user's record
/// names becomes a library once, without moving a model file.
///
/// Every case runs against `FakeModelLibraryService`: the app suites link the
/// release engine, which cannot build a moved store.
final class MoveStoreMigrationTests: LibraryStateTestCase {
    private let moved = "/Volumes/Old Drive/Models Store"
    private let home = "/Users/test/.turbospark"
    private let notConnectedSentence = "not_connected: /Volumes/Old Drive/Models Store is not connected"

    private var record: FakeMovedStoreRecord!
    private var state: LibraryState!
    private var migration: MoveStoreMigration!
    private var sink: AsyncStream<VolumeEvent>.Continuation!
    private var importedNotices = 0

    private var movedLibrary: ModelLibrary {
        ModelLibrary(
            id: "moved-id", name: "Models Store", root: moved + "/models",
            available: true, modelCount: 5)
    }

    private var pendingURL: URL { directory.appendingPathComponent("moved_store_pending.json") }

    override func setUpWithError() throws {
        try super.setUpWithError()
        record = FakeMovedStoreRecord(moved)
        importedNotices = 0
        AppJSONStore.clearLastReadError()
    }

    override func tearDownWithError() throws {
        // The read-failure latch is process-wide, and an `AppModel` built by a
        // later test shows what it holds as a toast.
        AppJSONStore.clearLastReadError()
        try super.tearDownWithError()
    }

    // MARK: - Harness

    /// Builds the migration and its event stream without starting it.
    private func make(homeRoot: String? = nil) -> AsyncStream<VolumeEvent> {
        state = makeState()
        let home = homeRoot ?? self.home
        migration = MoveStoreMigration(
            service: service, libraryState: state, record: record, pendingURL: pendingURL,
            homeRoot: { home }, onImported: { [unowned self] in self.importedNotices += 1 })
        let (events, continuation) = AsyncStream<VolumeEvent>.makeStream()
        sink = continuation
        return events
    }

    /// Starts the migration as the app does at launch and waits for the
    /// launch run to finish.
    private func launch(homeRoot: String? = nil) async {
        let events = make(homeRoot: homeRoot)
        migration.start(events: events)
        await migration.waitUntilIdle()
    }

    /// Sends one `mounted` event and waits for the run it may start.
    private func mount() async {
        let before = migration.handledEventCount
        sink.yield(.mounted(LibraryFixtures.volume("Old Drive")))
        let handled = await eventually { self.migration.handledEventCount == before + 1 }
        XCTAssertTrue(handled, "the mount event was not handled")
        await migration.waitUntilIdle()
    }

    /// The import completes and lists the moved store's library.
    private func importSucceeds(
        copyFailures: [(key: String, reason: String)] = [], log: MigrationLog? = nil
    ) throws {
        let imported = try LibraryFixtures.imported(
            movedLibrary, copyFailures: copyFailures, needsAdoption: ["music-a"])
        let library = movedLibrary
        let service = self.service!
        service.with {
            $0.importHandler = { _ in
                log?.add("import")
                service.with { state in
                    if !state.libraries.contains(where: { $0.id == library.id }) {
                        state.libraries.append(library)
                    }
                }
                return imported
            }
        }
    }

    private func importFails(_ message: String, code: TurboSparkError.Code = .invalidArgument) {
        service.with {
            $0.importHandler = { _ in throw TurboSparkError(code: code, message: message) }
        }
    }

    private func pendingObject() -> [String: Any]? {
        guard let data = try? Data(contentsOf: pendingURL) else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    private func writePendingFile(root: String, name: String = "Models Store", notConnected: Bool) throws {
        let object: [String: Any] = ["root": root, "name": name, "notConnected": notConnected]
        try JSONSerialization.data(withJSONObject: object).write(to: pendingURL)
    }

    private var pendingFileExists: Bool { FileManager.default.fileExists(atPath: pendingURL.path) }

    // MARK: - Nothing to do

    func testLaunchWithNoRecordDoesNothing() async {
        record = FakeMovedStoreRecord(nil)
        await launch()

        XCTAssertEqual(service.importRoots, [], "the engine is not called")
        XCTAssertEqual(service.listCalls, 0, "nothing was converted, so nothing refreshes")
        XCTAssertEqual(record.clearCalls, 0)
        XCTAssertFalse(pendingFileExists)
        XCTAssertNil(state.pendingMovedStore)
        XCTAssertEqual(importedNotices, 0)
    }

    func testALeftoverPendingFileWithNoRecordIsRemoved() async throws {
        // A crash between clearing the record and removing the file.
        try writePendingFile(root: moved, notConnected: true)
        record = FakeMovedStoreRecord(nil)
        await launch()

        XCTAssertFalse(pendingFileExists, "no record, so nothing is pending")
        XCTAssertNil(state.pendingMovedStore)
        XCTAssertEqual(service.importRoots, [])
    }

    func testARecordEqualToTheDefaultHomeDoesNothing() async {
        for spelling in [home, home + "/", home + "/./"] {
            record = FakeMovedStoreRecord(spelling)
            service.with { $0.importRoots = [] }
            await launch()

            XCTAssertEqual(service.importRoots, [], "\(spelling) is the home")
            XCTAssertEqual(record.clearCalls, 0)
            XCTAssertEqual(record.value, spelling, "the record is left alone")
            migration.stop()
        }
    }

    func testARecordThatIsNotAnAbsolutePathIsNeitherImportedNorCleared() async {
        // Handed to the engine it would resolve against the working directory.
        record = FakeMovedStoreRecord("models/elsewhere")
        await launch()

        XCTAssertEqual(service.importRoots, [])
        XCTAssertEqual(record.value, "models/elsewhere")
        XCTAssertEqual(record.clearCalls, 0)
    }

    // MARK: - A connected root

    func testAConnectedRootIsImportedOnceOffTheMainThreadThenTheRecordAndPendingFileGo() async throws {
        try writePendingFile(root: moved, notConnected: true)
        let log = MigrationLog()
        try importSucceeds(log: log)
        let pendingURL = self.pendingURL
        record.onClear = {
            log.add("clear (pending file \(FileManager.default.fileExists(atPath: pendingURL.path) ? "kept" : "gone"))")
        }
        await launch()

        XCTAssertEqual(service.importRoots, [moved], "one import, with the recorded root")
        XCTAssertFalse(
            service.mainThreadCalls.contains("importMovedRoot"),
            "the engine blocks, so the import runs off the main thread")
        XCTAssertEqual(
            log.entries, ["import", "clear (pending file kept)"],
            "import first, then the record, and the pending file only after that")
        XCTAssertNil(record.value, "a complete import clears the record")
        XCTAssertEqual(record.clearCalls, 1)
        XCTAssertFalse(pendingFileExists, "and removes the pending file")
        XCTAssertNil(state.pendingMovedStore)
        XCTAssertTrue(
            state.libraries.contains { $0.id == "moved-id" },
            "LibraryState was refreshed, so the converted library is listed")
        XCTAssertEqual(migration.lastImported?.needsAdoption, ["music-a"])
        XCTAssertTrue(migration.lastCopyFailures.isEmpty)
        XCTAssertFalse(migration.isPending)
        XCTAssertEqual(importedNotices, 1, "the model list is told once")
    }

    func testStartingLibraryMonitoringConvertsTheRecordedStoreThroughTheLiveRecord() async throws {
        // The launch path end to end with the machine record and pending file
        // the app really uses (under the test host's own state directory).
        let liveRecord = ModelStoreRootRecord.url
        let livePending = MoveStoreMigration.defaultPendingURL
        try? FileManager.default.removeItem(at: liveRecord)
        try? FileManager.default.removeItem(at: livePending)
        defer {
            try? FileManager.default.removeItem(at: liveRecord)
            try? FileManager.default.removeItem(at: livePending)
        }
        XCTAssertTrue(ModelStoreRootRecord.save(moved))
        try importSucceeds()

        let model = await makeModel(using: makeState())
        model.stopCronScheduler()
        let before = try TurboSparkCatalog.storeRoot()
        model.startLibraryMonitoring(events: { AsyncStream<VolumeEvent>.makeStream().0 })
        await model.moveStoreMigration?.waitUntilIdle()
        model.libraryState.stop()
        model.moveStoreMigration?.stop()

        XCTAssertEqual(service.importRoots, [moved], "starting library monitoring runs the conversion")
        XCTAssertNil(ModelStoreRootRecord.load(), "the live record is cleared")
        XCTAssertFalse(FileManager.default.fileExists(atPath: livePending.path))
        XCTAssertEqual(
            ModelStoreRootRecord.resolve(profileValue: moved), "",
            "and a profile that still names the old root does not bring it back")
        XCTAssertEqual(try TurboSparkCatalog.storeRoot(), before, "the engine root never moved")
    }

    func testACompleteImportRefreshesTheModelListThroughTheAppModel() async throws {
        // `AppModel.installed` was read at launch, before the import listed
        // the moved store's models, so it has to be read again.
        try importSucceeds()
        let model = await makeModel(using: makeState())
        model.stopCronScheduler()
        model.catalog = []
        let (events, _) = AsyncStream<VolumeEvent>.makeStream()
        model.startMoveStoreMigration(
            events: events, record: record, pendingURL: pendingURL, homeRoot: { "/Users/test/.turbospark" })
        await model.moveStoreMigration?.waitUntilIdle()

        XCTAssertNil(record.value)
        XCTAssertFalse(model.catalog.isEmpty, "refreshModels ran after the import")
    }

    // MARK: - Home copies that failed

    func testAnIncompleteImportKeepsTheRecordAndAPendingEntryUntilARetryCompletes() async throws {
        try importSucceeds(copyFailures: [(key: "embeddings/bge", reason: "no space left on device")])
        await launch()

        XCTAssertEqual(service.importRoots, [moved])
        XCTAssertEqual(record.value, moved, "the record stays: the failed copies are retried")
        XCTAssertEqual(record.clearCalls, 0, "an incomplete import never clears the record")
        XCTAssertEqual(pendingObject()?["root"] as? String, moved, "a pending entry stays")
        XCTAssertEqual(pendingObject()?["notConnected"] as? Bool, false)
        XCTAssertEqual(migration.lastCopyFailures.map(\.key), ["embeddings/bge"], "failures are state")
        XCTAssertTrue(migration.isPending)
        XCTAssertNil(
            state.pendingMovedStore,
            "the library is real and listed, so no not-connected placeholder stands beside it")
        XCTAssertTrue(state.libraries.contains { $0.id == "moved-id" })
        XCTAssertEqual(importedNotices, 1, "the models are listed, so the model list is told")

        // The next trigger retries, and now every copy lands.
        try importSucceeds()
        await mount()

        XCTAssertEqual(service.importRoots, [moved, moved], "the retry imports again")
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
        XCTAssertTrue(migration.lastCopyFailures.isEmpty)
        XCTAssertFalse(migration.isPending)
    }

    func testCopyFailuresAreNeverRaisedAsAnAppErrorOrAToast() async throws {
        try importSucceeds(copyFailures: [(key: "models.json", reason: "read only")])
        let model = await makeModel(using: makeState())
        model.stopCronScheduler()
        let baseline = model.error
        let (events, _) = AsyncStream<VolumeEvent>.makeStream()
        model.startMoveStoreMigration(
            events: events, record: record, pendingURL: pendingURL, homeRoot: { "/Users/test/.turbospark" })
        await model.moveStoreMigration?.waitUntilIdle()

        XCTAssertEqual(model.error, baseline, "a copy failure is state, not an app error")
        XCTAssertNil(model.activeToast)
        XCTAssertEqual(model.moveStoreMigration?.lastCopyFailures.count, 1)
    }

    // MARK: - An absent root

    func testAnAbsentRootWritesThePendingFileAndListsANotConnectedPlaceholder() async {
        importFails(notConnectedSentence)
        await launch()

        XCTAssertEqual(service.importRoots, [moved])
        XCTAssertEqual(record.value, moved, "the record stays")
        XCTAssertEqual(record.clearCalls, 0)
        XCTAssertEqual(importedNotices, 0)
        let pending = pendingObject()
        XCTAssertEqual(pending?["root"] as? String, moved)
        XCTAssertEqual(pending?["name"] as? String, "Models Store", "named after the folder")
        XCTAssertEqual(pending?["notConnected"] as? Bool, true)
        XCTAssertEqual(
            Set(pending?.keys.map { $0 } ?? []), ["root", "name", "notConnected"],
            "the file holds the root, the name and the flag, and nothing else")
        XCTAssertTrue(migration.isPending)

        XCTAssertEqual(state.pendingMovedStore, PendingMovedStore(root: moved, name: "Models Store"))
        let placeholder = state.displayLibraries.last
        XCTAssertEqual(placeholder?.id, LibraryState.pendingMovedStoreId)
        XCTAssertEqual(placeholder?.name, "Models Store")
        XCTAssertEqual(placeholder?.available, false, "listed as not connected")
        XCTAssertNil(placeholder?.root, "no root while it is not connected")
        XCTAssertEqual(placeholder?.isInternal, false)
        XCTAssertEqual(placeholder?.isDefault, false, "never the default")
        XCTAssertEqual(placeholder?.modelCount, 0)
    }

    func testThePlaceholderIsAnAppRowNotARegistryLibrary() async {
        importFails(notConnectedSentence)
        await launch()
        await state.refresh()

        guard let placeholder = state.displayLibraries.last else {
            return XCTFail("the placeholder is not listed")
        }
        XCTAssertTrue(LibraryState.isPendingMovedStore(placeholder))
        XCTAssertFalse(
            state.libraries.contains { $0.id == LibraryState.pendingMovedStoreId },
            "install pickers and every lookup read `libraries`, which never holds it")
        XCTAssertEqual(
            state.displayLibraries.dropLast().map(\.id), state.libraries.map(\.id),
            "the display list is the real libraries plus the placeholder at the end")
        XCTAssertEqual(state.destinationDefault.libraryId, "internal", "never a destination")
        XCTAssertFalse(state.destinationDefault.fallbackBecauseNotConnected)
        XCTAssertNil(state.library(containing: moved + "/models/text/x.gturbo"))
        XCTAssertNil(state.library(for: LibraryFixtures.row("any", library: placeholder)))
        XCTAssertFalse(
            LibraryState.isPendingMovedStore(LibraryFixtures.internalLibrary),
            "a real library is never taken for it")
        XCTAssertFalse(
            LibraryState.isPendingMovedStore(t7), "a real library is never taken for it")
        // A real library id is a v4 UUID, which this can never equal.
        XCTAssertNil(UUID(uuidString: LibraryState.pendingMovedStoreId))
    }

    func testAnAbsentRootRaisesNoAppErrorAndNoToast() async {
        importFails(notConnectedSentence)
        let model = await makeModel(using: makeState())
        model.stopCronScheduler()
        let baseline = model.error
        let (events, _) = AsyncStream<VolumeEvent>.makeStream()
        model.startMoveStoreMigration(
            events: events, record: record, pendingURL: pendingURL, homeRoot: { "/Users/test/.turbospark" })
        await model.moveStoreMigration?.waitUntilIdle()

        XCTAssertEqual(model.libraryState.pendingMovedStore?.name, "Models Store")
        XCTAssertEqual(model.error, baseline, "a drive that is away is not an app error")
        XCTAssertNil(model.activeToast, "and not a toast")
        XCTAssertEqual(record.value, moved)
    }

    func testALaterMountCompletesAPendingConversionAndTheRealLibraryReplacesThePlaceholder() async throws {
        importFails(notConnectedSentence)
        await launch()
        XCTAssertNotNil(state.pendingMovedStore)
        XCTAssertNotNil(pendingObject())

        // The drive is connected: the next mount imports it.
        try importSucceeds()
        await mount()

        XCTAssertEqual(service.importRoots, [moved, moved])
        XCTAssertNil(record.value, "the record is cleared")
        XCTAssertFalse(pendingFileExists, "the pending file is removed")
        XCTAssertNil(state.pendingMovedStore, "the placeholder is gone")
        XCTAssertEqual(
            state.displayLibraries.map(\.id), state.libraries.map(\.id),
            "nothing but real libraries is listed")
        XCTAssertTrue(state.libraries.contains { $0.id == "moved-id" }, "the real library is listed")
        XCTAssertFalse(migration.isPending)
        XCTAssertEqual(importedNotices, 1)
    }

    // MARK: - Other failures

    private func assertAFailureKeepsEverything(
        _ message: String, code: TurboSparkError.Code = .invalidArgument,
        file: StaticString = #filePath, line: UInt = #line
    ) async {
        importFails(message, code: code)
        await launch()

        XCTAssertEqual(service.importRoots, [moved], file: file, line: line)
        XCTAssertEqual(record.value, moved, "the record stays: \(message)", file: file, line: line)
        XCTAssertEqual(record.clearCalls, 0, "nothing is cleared: \(message)", file: file, line: line)
        XCTAssertEqual(pendingObject()?["root"] as? String, moved, "a pending entry is kept", file: file, line: line)
        XCTAssertEqual(pendingObject()?["notConnected"] as? Bool, false, file: file, line: line)
        XCTAssertNil(
            state.pendingMovedStore,
            "the root answered, so it is not shown as a drive that is away", file: file, line: line)
        XCTAssertNotNil(migration.lastFailure, "the failure is kept for a log", file: file, line: line)
        XCTAssertTrue(migration.isPending, file: file, line: line)
        XCTAssertEqual(importedNotices, 0, file: file, line: line)
    }

    func testANestedRefusalKeepsTheRecordAndThePendingEntry() async {
        await assertAFailureKeepsEverything("nested: the folder overlaps the library T7")
    }

    func testAMarkerFailureKeepsTheRecordAndThePendingEntry() async {
        await assertAFailureKeepsEverything("marker: library.json could not be read")
    }

    func testAnIoFailureKeepsTheRecordAndThePendingEntry() async {
        await assertAFailureKeepsEverything("io: reading installed.json: permission denied")
    }

    func testABusyEngineKeepsTheRecordAndThePendingEntry() async {
        await assertAFailureKeepsEverything("busy: an install is running", code: .busy)
    }

    func testAFailureThatIsNotALibraryFaultKeepsTheRecordToo() async {
        await assertAFailureKeepsEverything("the library reported success but wrote no result", code: .unknown)
    }

    func testABusyEngineRetriesOnTheNextTrigger() async throws {
        importFails("busy: an install is running", code: .busy)
        await launch()
        XCTAssertEqual(record.value, moved)

        try importSucceeds()
        await mount()

        XCTAssertEqual(service.importRoots.count, 2)
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
    }

    func testAFailureAfterAnAbsentRootReplacesThePlaceholder() async {
        importFails(notConnectedSentence)
        await launch()
        XCTAssertNotNil(state.pendingMovedStore)

        // The drive is there now but refuses the import.
        importFails("nested: the folder overlaps the library T7")
        await mount()

        XCTAssertNil(state.pendingMovedStore, "no longer shown as away")
        XCTAssertEqual(pendingObject()?["notConnected"] as? Bool, false)
        XCTAssertEqual(record.value, moved)
    }

    // MARK: - One import at a time

    func testTwoTriggersWhileTheFirstImportIsInFlightMakeOneCall() async throws {
        let gate = DispatchSemaphore(value: 0)
        let imported = try LibraryFixtures.imported(movedLibrary)
        service.with {
            $0.importHandler = { _ in
                gate.wait()
                return imported
            }
        }
        let record = self.record!
        let events = make()
        migration.start(events: events)
        // The launch import is now parked inside the engine.
        let entered = await eventually { self.service.importRoots.count == 1 }
        XCTAssertTrue(entered)

        let before = migration.handledEventCount
        sink.yield(.mounted(LibraryFixtures.volume("Old Drive")))
        sink.yield(.mounted(LibraryFixtures.volume("Spare")))
        let handled = await eventually { self.migration.handledEventCount == before + 2 }
        XCTAssertTrue(handled)
        XCTAssertEqual(service.importRoots.count, 1, "no second import starts while one is in flight")

        gate.signal()
        await migration.waitUntilIdle()

        // The first import completed and cleared the record, so the follow-up
        // the two mounts asked for finds nothing to do.
        XCTAssertEqual(service.importRoots.count, 1)
        XCTAssertNil(record.value)
    }

    func testAMountDuringAnImportThatFindsTheRootAbsentGetsOneFollowUp() async throws {
        // The drive was not mounted when the engine looked, and mounts right
        // after. Dropping that mount would leave the conversion waiting for
        // the next one.
        let gate = DispatchSemaphore(value: 0)
        let imported = try LibraryFixtures.imported(movedLibrary)
        let calls = MigrationLog()
        let sentence = notConnectedSentence
        service.with {
            $0.importHandler = { _ in
                calls.add("call")
                if calls.entries.count == 1 {
                    gate.wait()
                    throw TurboSparkError(code: .invalidArgument, message: sentence)
                }
                return imported
            }
        }
        let events = make()
        migration.start(events: events)
        let entered = await eventually { self.service.importRoots.count == 1 }
        XCTAssertTrue(entered)

        let before = migration.handledEventCount
        sink.yield(.mounted(LibraryFixtures.volume("Old Drive")))
        sink.yield(.mounted(LibraryFixtures.volume("Old Drive")))
        let handled = await eventually { self.migration.handledEventCount == before + 2 }
        XCTAssertTrue(handled)
        gate.signal()
        await migration.waitUntilIdle()

        XCTAssertEqual(service.importRoots.count, 2, "one follow-up, not one per mount")
        XCTAssertNil(record.value, "and the follow-up completed the conversion")
        XCTAssertNil(state.pendingMovedStore)
    }

    // MARK: - Which events run it

    func testOnlyAMountRunsTheImportAndOnlyWhileAConversionIsPending() async throws {
        importFails(notConnectedSentence)
        await launch()
        XCTAssertEqual(service.importRoots.count, 1)

        let events: [VolumeEvent] = [
            .willUnmount(LibraryFixtures.volume("Old Drive")),
            .didUnmount(LibraryFixtures.volume("Old Drive")),
            .renamed(old: LibraryFixtures.volume("Old Drive"), new: LibraryFixtures.volume("Renamed")),
        ]
        for event in events {
            let before = migration.handledEventCount
            sink.yield(event)
            _ = await eventually { self.migration.handledEventCount == before + 1 }
        }
        await migration.waitUntilIdle()
        XCTAssertEqual(service.importRoots.count, 1, "an unmount or a rename is not a reason to import")
    }

    func testAMountWithNothingPendingNeverCallsTheEngine() async throws {
        record = FakeMovedStoreRecord(nil)
        await launch()
        await mount()
        await mount()

        XCTAssertEqual(service.importRoots, [])
        XCTAssertEqual(service.listCalls, 0)
    }

    func testAMountAfterACompleteConversionDoesNothing() async throws {
        try importSucceeds()
        await launch()
        XCTAssertEqual(service.importRoots.count, 1)
        await mount()
        XCTAssertEqual(service.importRoots.count, 1)
    }

    func testStoppingEndsTheEventLoop() async throws {
        importFails(notConnectedSentence)
        await launch()
        migration.stop()
        try importSucceeds()

        sink.yield(.mounted(LibraryFixtures.volume("Old Drive")))
        await letTasksStart()
        await migration.waitUntilIdle()

        XCTAssertEqual(service.importRoots.count, 1, "a stopped migration reads no more events")
        XCTAssertEqual(record.value, moved)
    }

    func testStartingTwiceReadsOneStream() async throws {
        importFails(notConnectedSentence)
        let first = make()
        migration.start(events: first)
        await migration.waitUntilIdle()
        let (second, secondSink) = AsyncStream<VolumeEvent>.makeStream()
        migration.start(events: second)
        try importSucceeds()

        secondSink.yield(.mounted(LibraryFixtures.volume("Old Drive")))
        await letTasksStart()
        await migration.waitUntilIdle()
        XCTAssertEqual(service.importRoots.count, 1, "a second stream is ignored, like LibraryState's")
    }

    // MARK: - The pending file

    func testACorruptPendingFileIsQuarantinedAndTheMigrationRederivesFromTheRecord() async throws {
        try Data("{ this is not json".utf8).write(to: pendingURL)
        try importSucceeds()
        await launch()

        let leftovers = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertTrue(
            leftovers.contains { $0.hasPrefix("moved_store_pending.corrupt-") },
            "AppJSONStore moved the unreadable file aside instead of overwriting it: \(leftovers)")
        XCTAssertNotNil(AppJSONStore.lastReadError, "and reported it the way every store does")
        XCTAssertEqual(service.importRoots, [moved], "the record, not the file, is the input")
        XCTAssertNil(record.value, "so the conversion still completed")
        XCTAssertFalse(pendingFileExists)
    }

    func testACorruptPendingFileDoesNotHideAnAbsentRoot() async throws {
        try Data("[1, 2".utf8).write(to: pendingURL)
        importFails(notConnectedSentence)
        await launch()

        XCTAssertEqual(pendingObject()?["root"] as? String, moved, "re-derived from the record")
        XCTAssertEqual(state.pendingMovedStore?.name, "Models Store")
    }

    func testThePlaceholderShowsAtLaunchFromThePendingFileBeforeTheImportReturns() async throws {
        try writePendingFile(root: moved, notConnected: true)
        let gate = DispatchSemaphore(value: 0)
        let imported = try LibraryFixtures.imported(movedLibrary)
        service.with {
            $0.importHandler = { _ in
                gate.wait()
                return imported
            }
        }
        let events = make()
        migration.start(events: events)

        XCTAssertEqual(
            state.pendingMovedStore?.root, moved,
            "a drive that was away last time is listed as away at once")
        let entered = await eventually { self.service.importRoots.count == 1 }
        XCTAssertTrue(entered)
        XCTAssertNotNil(state.pendingMovedStore, "still shown while the engine is looking")

        gate.signal()
        await migration.waitUntilIdle()
        XCTAssertNil(state.pendingMovedStore, "and gone once the import lands")
    }

    func testAPendingFileForAnotherRootOrANonAwayStateShowsNoPlaceholder() async throws {
        let gate = DispatchSemaphore(value: 0)
        let imported = try LibraryFixtures.imported(movedLibrary)
        service.with {
            $0.importHandler = { _ in
                gate.wait()
                return imported
            }
        }

        try writePendingFile(root: "/Volumes/Other/Store", notConnected: true)
        let first = make()
        migration.start(events: first)
        XCTAssertNil(state.pendingMovedStore, "the file is for a root the record no longer names")
        gate.signal()
        await migration.waitUntilIdle()
        migration.stop()

        record = FakeMovedStoreRecord(moved)
        try writePendingFile(root: moved, notConnected: false)
        let second = make()
        migration.start(events: second)
        XCTAssertNil(state.pendingMovedStore, "a connected root was refused; it is not shown as away")
        gate.signal()
        await migration.waitUntilIdle()
    }

    func testTheFolderNameFollowsTheEngineRule() {
        XCTAssertEqual(MoveStoreMigration.folderName(of: "/Volumes/T7/.turbospark"), ".turbospark")
        XCTAssertEqual(MoveStoreMigration.folderName(of: "/Volumes/T7/Models Store"), "Models Store")
        XCTAssertEqual(MoveStoreMigration.folderName(of: "/Volumes/T7/Models Store/"), "Models Store")
        XCTAssertEqual(MoveStoreMigration.folderName(of: "/Volumes/T7"), "T7")
        XCTAssertEqual(MoveStoreMigration.folderName(of: "/"), "TurboSpark", "no folder name to use")
    }

    // MARK: - Crash safety

    func testACrashBetweenTheImportAndClearingTheRecordReimportsAndThenClears() async throws {
        try importSucceeds()
        record.failNextClears(1)
        await launch()

        // The import landed and the library is listed, but the record is still
        // there: the process would have died here.
        XCTAssertEqual(service.importRoots.count, 1)
        XCTAssertEqual(record.value, moved)
        XCTAssertEqual(record.clearCalls, 1)
        XCTAssertNotNil(pendingObject(), "the conversion stays pending until the record is gone")
        XCTAssertTrue(state.libraries.contains { $0.id == "moved-id" })
        XCTAssertTrue(migration.isPending)

        // The next launch or mount runs the import again (idempotent in the
        // engine) and this time clears.
        await mount()

        XCTAssertEqual(service.importRoots.count, 2, "re-imported")
        XCTAssertEqual(record.clearCalls, 2)
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
        XCTAssertFalse(migration.isPending)
        XCTAssertEqual(
            state.libraries.filter { $0.id == "moved-id" }.count, 1,
            "the library is listed once, not twice")
    }

    /// The legacy Move Store sheet can save a NEW root while an import of the
    /// old one is in flight. Clearing the record then would lose that newer
    /// store, so the record stays and a follow-up run imports the new root.
    func testARecordRewrittenDuringTheImportIsNotClearedAndTheNewRootIsImportedNext() async throws {
        let newer = "/Volumes/New Drive/Models Store"
        let imported = try LibraryFixtures.imported(movedLibrary)
        let record = self.record!
        let firstRoot = moved
        service.with {
            $0.importHandler = { root in
                if root == firstRoot { record.replace(with: newer) }
                return imported
            }
        }
        await launch()

        XCTAssertEqual(service.importRoots, [moved, newer], "the newer root is imported next")
        XCTAssertEqual(record.clearCalls, 1, "cleared once, after the newer root was imported")
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
        XCTAssertFalse(migration.isPending)
    }

    func testARelaunchAfterACrashAlsoCompletes() async throws {
        try importSucceeds()
        record.failNextClears(1)
        await launch()
        migration.stop()
        XCTAssertEqual(record.value, moved)

        // A new process: new state, same record and pending file.
        await launch()

        XCTAssertEqual(service.importRoots.count, 2)
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
    }
}
