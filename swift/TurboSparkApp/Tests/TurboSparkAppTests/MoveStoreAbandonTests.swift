import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.1 (requirement 9.3, and the 5.10 note on stuck conversions): the user
/// can stop waiting for a moved store. `abandon()` ends the conversion without
/// touching the drive or any model.
@MainActor
final class MoveStoreAbandonTests: LibraryStateTestCase {
    private let moved = "/Volumes/Old Drive/Models Store"
    private let home = "/Users/test/.turbospark"
    private let notConnectedSentence = "not_connected: /Volumes/Old Drive/Models Store is not connected"

    private var record: FakeMovedStoreRecord!
    private var state: LibraryState!
    private var migration: MoveStoreMigration!
    private var sink: AsyncStream<VolumeEvent>.Continuation!
    private var importedNotices = 0

    private var pendingURL: URL { directory.appendingPathComponent("moved_store_pending.json") }
    private var pendingFileExists: Bool { FileManager.default.fileExists(atPath: pendingURL.path) }

    private var movedLibrary: ModelLibrary {
        ModelLibrary(id: "moved-id", name: "Models Store", root: moved + "/models", available: true, modelCount: 5)
    }

    override func setUpWithError() throws {
        try super.setUpWithError()
        record = FakeMovedStoreRecord(moved)
        importedNotices = 0
        AppJSONStore.clearLastReadError()
    }

    override func tearDownWithError() throws {
        AppJSONStore.clearLastReadError()
        try super.tearDownWithError()
    }

    private func launch() async {
        let home = self.home
        state = makeState()
        migration = MoveStoreMigration(
            service: service, libraryState: state, record: record, pendingURL: pendingURL,
            homeRoot: { home }, onImported: { [unowned self] in self.importedNotices += 1 })
        let (events, continuation) = AsyncStream<VolumeEvent>.makeStream()
        sink = continuation
        migration.start(events: events)
        await migration.waitUntilIdle()
    }

    private func mount() async {
        let before = migration.handledEventCount
        sink.yield(.mounted(LibraryFixtures.volume("Old Drive")))
        _ = await eventually { self.migration.handledEventCount == before + 1 }
        await migration.waitUntilIdle()
    }

    private func importFails(_ message: String) {
        service.with {
            $0.importHandler = { _ in throw TurboSparkError(code: .invalidArgument, message: message) }
        }
    }

    // MARK: - The conversion waits for an absent drive

    func testAbandoningClearsTheRecordTheFileAndThePlaceholder() async throws {
        importFails(notConnectedSentence)
        await launch()
        XCTAssertNotNil(state.pendingMovedStore)
        XCTAssertTrue(pendingFileExists)
        XCTAssertTrue(migration.isPending)

        XCTAssertTrue(migration.abandon())

        XCTAssertNil(record.value, "the record no longer names the store")
        XCTAssertEqual(record.clearCalls, 1)
        XCTAssertFalse(pendingFileExists, "the pending file is discarded")
        XCTAssertNil(state.pendingMovedStore, "the placeholder is dropped")
        XCTAssertFalse(state.displayLibraries.contains { LibraryState.isPendingMovedStore($0) })
        XCTAssertFalse(migration.isPending)
        XCTAssertNil(migration.lastFailure)
        XCTAssertEqual(migration.lastCopyFailures, [])
        XCTAssertNil(migration.lastImported)
    }

    func testAbandoningTouchesNoLibraryAndNoModel() async throws {
        importFails(notConnectedSentence)
        await launch()
        let imports = service.importRoots
        let libraries = state.libraries

        migration.abandon()

        XCTAssertEqual(service.writeCalls, [], "nothing is removed, forgotten or re-defaulted")
        XCTAssertEqual(service.importRoots, imports, "and nothing is imported")
        XCTAssertEqual(state.libraries, libraries)
        XCTAssertEqual(service.mainThreadCalls, [])
    }

    func testAMountAfterAbandoningDoesNotConvertAnything() async throws {
        importFails(notConnectedSentence)
        await launch()
        let imports = service.importRoots.count
        migration.abandon()

        await mount()

        XCTAssertEqual(service.importRoots.count, imports, "no record, so nothing to import")
        XCTAssertNil(state.pendingMovedStore)
        XCTAssertFalse(pendingFileExists)
    }

    func testAbandoningIsIdempotent() async throws {
        importFails(notConnectedSentence)
        await launch()
        XCTAssertTrue(migration.abandon())
        XCTAssertTrue(migration.abandon(), "a second call finds nothing and still reports done")
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
        XCTAssertNil(state.pendingMovedStore)
    }

    func testAbandoningWithNothingPendingIsHarmless() async throws {
        record = FakeMovedStoreRecord(nil)
        await launch()
        XCTAssertTrue(migration.abandon())
        XCTAssertNil(record.value)
    }

    // MARK: - A record that will not clear

    func testARecordThatWillNotClearKeepsEverythingAndReportsFailure() async throws {
        importFails(notConnectedSentence)
        await launch()
        record.failNextClears(1)

        XCTAssertFalse(migration.abandon())

        XCTAssertEqual(record.value, moved, "still recorded")
        XCTAssertTrue(pendingFileExists, "still pending")
        XCTAssertNotNil(state.pendingMovedStore, "still listed")
        XCTAssertTrue(migration.isPending)

        XCTAssertTrue(migration.abandon(), "a retry succeeds")
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
        XCTAssertNil(state.pendingMovedStore)
    }

    // MARK: - A stuck conversion with no placeholder

    func testAStuckConversionCanBeAbandonedToo() async throws {
        importFails("io: could not read the folder")
        await launch()
        XCTAssertTrue(migration.isPending)
        XCTAssertNil(state.pendingMovedStore, "an io failure shows no placeholder")
        XCTAssertNotNil(migration.lastFailure)

        XCTAssertTrue(migration.abandon())

        XCTAssertFalse(migration.isPending)
        XCTAssertNil(migration.lastFailure, "the failure that was shown goes with the conversion")
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
    }

    func testIncompleteCopiesAreForgottenWithTheConversion() async throws {
        let imported = try LibraryFixtures.imported(
            movedLibrary, copyFailures: [(key: "embeddings/a", reason: "disk full")])
        service.with { $0.importHandler = { _ in imported } }
        await launch()
        XCTAssertEqual(migration.lastCopyFailures.count, 1)
        XCTAssertTrue(migration.isPending)

        XCTAssertTrue(migration.abandon())

        XCTAssertEqual(migration.lastCopyFailures, [])
        XCTAssertNil(migration.lastImported)
        XCTAssertFalse(migration.isPending)
        XCTAssertNil(record.value)
    }

    // MARK: - An import in flight

    func testAnImportThatFinishesAfterAbandoningDoesNotBringTheConversionBack() async throws {
        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let home = self.home
        let sentence = notConnectedSentence
        service.with {
            $0.importHandler = { _ in
                started.signal()
                release.wait()
                throw TurboSparkError(code: .invalidArgument, message: sentence)
            }
        }
        state = makeState()
        migration = MoveStoreMigration(
            service: service, libraryState: state, record: record, pendingURL: pendingURL,
            homeRoot: { home })
        let (events, continuation) = AsyncStream<VolumeEvent>.makeStream()
        sink = continuation
        migration.start(events: events)
        let running = await Task.detached { started.wait(timeout: .now() + 5) == .success }.value
        XCTAssertTrue(running, "the import never started")

        migration.abandon()
        release.signal()
        await migration.waitUntilIdle()

        XCTAssertNil(record.value)
        XCTAssertNil(state.pendingMovedStore, "the late not-connected answer does not list a placeholder")
        XCTAssertFalse(pendingFileExists, "nor write a pending file")
        XCTAssertFalse(migration.isPending)
    }

    func testAnImportThatCompletesAfterAbandoningStillRefreshesTheLibraries() async throws {
        let started = DispatchSemaphore(value: 0)
        let release = DispatchSemaphore(value: 0)
        let imported = try LibraryFixtures.imported(movedLibrary)
        let library = movedLibrary
        let home = self.home
        let service = self.service!
        service.with {
            $0.importHandler = { _ in
                started.signal()
                release.wait()
                service.with { $0.libraries.append(library) }
                return imported
            }
        }
        state = makeState()
        migration = MoveStoreMigration(
            service: service, libraryState: state, record: record, pendingURL: pendingURL,
            homeRoot: { home }, onImported: { [unowned self] in self.importedNotices += 1 })
        let (events, continuation) = AsyncStream<VolumeEvent>.makeStream()
        sink = continuation
        migration.start(events: events)
        let running = await Task.detached { started.wait(timeout: .now() + 5) == .success }.value
        XCTAssertTrue(running, "the import never started")

        migration.abandon()
        release.signal()
        await migration.waitUntilIdle()

        // The drive was connected after all, so the engine registered the
        // library. It is real and listed, and the abandoned record stays gone.
        XCTAssertTrue(state.libraries.contains { $0.id == "moved-id" })
        XCTAssertNil(record.value)
        XCTAssertFalse(pendingFileExists)
        XCTAssertFalse(migration.isPending)
        XCTAssertEqual(importedNotices, 1, "the picker list is read again")
    }
}
