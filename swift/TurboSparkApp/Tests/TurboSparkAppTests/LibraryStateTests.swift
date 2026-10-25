import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 5.6 (requirements 3.3, 3.4, 4.1, 4.6): library availability, link
/// speed and free space held in app state, refreshed off the main actor.
///
/// Every case runs against `FakeModelLibraryService`. The app suites link the
/// release engine, which cannot build fixture libraries.
final class LibraryStateTests: LibraryStateTestCase {
    private func volume(_ name: String, uuid: String? = nil, path: String? = nil) -> VolumeDescriptor {
        LibraryFixtures.volume(name, uuid: uuid, path: path)
    }

    // MARK: - Rows and the loadable list

    func testOfflineRowsStayInAllInstalledButNotInTheLoadableList() async {
        let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.t7, spare]
            $0.installed = [
                LibraryFixtures.row("local"),
                LibraryFixtures.row("on-t7", library: self.t7),
                LibraryFixtures.row("on-spare", library: spare, availability: .notConnected),
                LibraryFixtures.row("gone", library: self.t7, availability: .missing),
            ]
        }
        let state = makeState()
        await state.refresh()

        XCTAssertEqual(
            state.allInstalled.map(\.alias), ["local", "on-t7", "on-spare", "gone"],
            "an offline or missing row stays listed")
        XCTAssertEqual(
            state.loadableInstalled.map(\.alias), ["local", "on-t7"],
            "only available rows can be loaded")
        XCTAssertEqual(state.allInstalled.first { $0.alias == "on-spare" }?.connectName, "Spare")
    }

    func testRefreshPublishesLibrariesConnectionsAndFreeSpaceOfAvailableOnes() async {
        let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)
        service.with { $0.libraries = [LibraryFixtures.internalLibrary, self.t7, spare] }
        let state = makeState()
        await state.refresh()

        XCTAssertEqual(state.libraries.map(\.id), ["internal", "t7-id", "spare-id"])
        XCTAssertEqual(Set(state.connections.keys), ["internal", "t7-id"], "an offline library has no link")
        XCTAssertEqual(state.connections["t7-id"]?.bitsPerSecond, 10_000_000_000)
        XCTAssertEqual(state.freeBytes["t7-id"], 900_000_000_000)
        XCTAssertEqual(state.freeBytes["internal"], 100_000_000_000)
        XCTAssertNil(state.freeBytes["spare-id"], "an unplugged drive has no free space to report")
        XCTAssertEqual(
            Set(probe.connectionRequests), ["/Users/test/.turbospark/models", "/Volumes/T7/TurboSpark"],
            "the probe is asked about each available library root")
        XCTAssertEqual(state.connection(for: self.t7)?.volumeUUID, "UUID-T7")
        XCTAssertNil(state.connection(for: spare))
    }

    func testAFreeSpaceThatCannotBeReadIsLeftOutNotZero() async {
        let state = makeState(freeSpace: { _ in nil })
        await state.refresh()
        XCTAssertTrue(state.freeBytes.isEmpty)
    }

    // MARK: - Measured speed

    func testDidUnmountDropsTheLibrarysMeasuredSpeedAndForgetsTheProbe() async {
        let state = makeState()
        await state.refresh()
        state.recordMeasuredSpeed(412.5, for: "t7-id")
        XCTAssertEqual(state.measuredSpeed(for: "t7-id"), 412.5)

        await state.handle(.didUnmount(volume("T7", uuid: "UUID-T7")))

        // Before any refresh has run: the speed must not outlive the drive.
        XCTAssertNil(state.measuredSpeed(for: "t7-id"))
        XCTAssertEqual(probe.invalidations, ["/Volumes/T7"])
    }

    func testAnUnmountOfAnotherVolumeKeepsTheMeasuredSpeed() async {
        // T70's name starts with T7's: a prefix test without a path boundary
        // would drop the wrong library's speed.
        let t70 = LibraryFixtures.external("t70-id", name: "T70", volume: "T70")
        service.with { $0.libraries = [LibraryFixtures.internalLibrary, self.t7, t70] }
        let state = makeState()
        await state.refresh()
        state.recordMeasuredSpeed(412.5, for: "t7-id")
        state.recordMeasuredSpeed(200, for: "t70-id")

        await state.handle(.didUnmount(volume("Backup")))
        XCTAssertEqual(state.measuredSpeeds, ["t7-id": 412.5, "t70-id": 200])

        await state.handle(.didUnmount(volume("T7")))
        XCTAssertNil(state.measuredSpeed(for: "t7-id"))
        XCTAssertEqual(state.measuredSpeed(for: "t70-id"), 200, "only the unmounted volume's library loses its speed")
        XCTAssertEqual(probe.invalidations, ["/Volumes/Backup", "/Volumes/T7"])
    }

    func testAVolumeIsMatchedToItsLibraryByUUIDWhenThePathDiffers() async {
        let state = makeState()
        await state.refresh()
        state.recordMeasuredSpeed(300, for: "t7-id")

        // The mount path no longer prefixes the library root, but the volume
        // is the one the library's link was probed on.
        await state.handle(.didUnmount(volume("Renamed", uuid: "UUID-T7", path: "/Volumes/Elsewhere")))

        XCTAssertNil(state.measuredSpeed(for: "t7-id"))
    }

    func testWillUnmountChangesNothing() async {
        let state = makeState()
        await state.refresh()
        state.recordMeasuredSpeed(412.5, for: "t7-id")
        let calls = service.listCalls

        // It fires on every attempt, including one the approval dissents from.
        await state.handle(.willUnmount(volume("T7", uuid: "UUID-T7")))
        await letTasksStart()

        XCTAssertEqual(state.measuredSpeed(for: "t7-id"), 412.5)
        XCTAssertTrue(probe.invalidations.isEmpty)
        XCTAssertEqual(sleeper.sleepCount, 0, "no refresh is scheduled for it")
        XCTAssertEqual(service.listCalls, calls)
    }

    func testRecordedSpeedsAreKeptOnlyForAvailableLibrariesAndSaneValues() async {
        let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)
        service.with { $0.libraries = [LibraryFixtures.internalLibrary, self.t7, spare] }
        let state = makeState()
        await state.refresh()

        state.recordMeasuredSpeed(250, for: "spare-id")
        state.recordMeasuredSpeed(250, for: "nobody")
        state.recordMeasuredSpeed(.nan, for: "t7-id")
        state.recordMeasuredSpeed(0, for: "t7-id")
        state.recordMeasuredSpeed(-5, for: "t7-id")
        XCTAssertTrue(state.measuredSpeeds.isEmpty)

        state.recordMeasuredSpeed(250, for: "t7-id")
        XCTAssertEqual(state.measuredSpeeds, ["t7-id": 250])
    }

    func testARefreshThatFindsTheDriveGoneAlsoDropsItsSpeedAndLink() async {
        let state = makeState()
        await state.refresh()
        state.recordMeasuredSpeed(412.5, for: "t7-id")

        // An unmount event the app never saw (it was not running).
        service.with { $0.libraries = [LibraryFixtures.internalLibrary, LibraryFixtures.external("t7-id", name: "T7", available: false)] }
        await state.refresh()

        XCTAssertNil(state.measuredSpeed(for: "t7-id"))
        XCTAssertNil(state.connections["t7-id"])
        XCTAssertNil(state.freeBytes["t7-id"])
    }

    // MARK: - Events

    func testMountedOfAKnownLibraryMarksItAndItsModelsAvailable() async {
        let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)
        let spareHere = LibraryFixtures.external("spare-id", name: "Spare")
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, spare]
            $0.installed = [LibraryFixtures.row("on-spare", library: spare, availability: .notConnected)]
        }
        let state = makeState()
        await state.refresh()
        XCTAssertTrue(state.loadableInstalled.isEmpty)

        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, spareHere]
            $0.installed = [LibraryFixtures.row("on-spare", library: spareHere)]
        }
        await state.handle(.mounted(volume("Spare")))
        await fireDebounce(state)

        XCTAssertEqual(state.libraries.first { $0.id == "spare-id" }?.available, true)
        XCTAssertEqual(state.loadableInstalled.map(\.alias), ["on-spare"])
        XCTAssertNil(state.pendingOffer, "a known library is not offered")
    }

    func testRenamedReResolvesTheLibraryAtItsNewRoot() async {
        let state = makeState()
        await state.refresh()
        state.recordMeasuredSpeed(412.5, for: "t7-id")

        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, LibraryFixtures.external("t7-id", name: "T7", volume: "T7-Fast")]
        }
        await state.handle(.renamed(old: volume("T7", uuid: "UUID-T7"), new: volume("T7-Fast", uuid: "UUID-T7")))

        XCTAssertEqual(
            Set(probe.invalidations), ["/Volumes/T7", "/Volumes/T7-Fast"],
            "both the old and the new mount point are forgotten")
        XCTAssertNil(state.measuredSpeed(for: "t7-id"), "the speed was measured under the old name")

        await fireDebounce(state)
        XCTAssertEqual(state.libraries.first { $0.id == "t7-id" }?.root, "/Volumes/T7-Fast/TurboSpark")
        XCTAssertTrue(probe.connectionRequests.contains("/Volumes/T7-Fast/TurboSpark"))
    }

    func testEventsWithinTheDebounceWindowProduceOneRefresh() async {
        let state = makeState()
        await state.refresh()
        let baseline = service.listCalls

        await state.handle(.mounted(volume("A")))
        await state.handle(.didUnmount(volume("B")))
        await state.handle(.renamed(old: volume("C"), new: volume("D")))
        await state.waitForPendingWork()
        let armed = await eventually { self.sleeper.sleepCount >= 3 }
        XCTAssertTrue(armed, "each event starts a timer")
        await letTasksStart()

        XCTAssertEqual(service.listCalls, baseline, "nothing refreshes until the window closes")
        XCTAssertEqual(sleeper.sleepCount, 3)
        XCTAssertEqual(sleeper.pendingCount, 1, "each event supersedes the timer before it")
        XCTAssertEqual(sleeper.durations.last, .milliseconds(500))

        await fireDebounce(state)
        XCTAssertEqual(service.listCalls, baseline + 1)
        XCTAssertEqual(service.installedCalls, baseline + 1)
    }

    func testTheLaunchRefreshDoesNotWaitForTheDebounce() async {
        let (events, _) = AsyncStream<VolumeEvent>.makeStream()
        let state = makeState()
        state.start(events: events)

        let loaded = await eventually { !state.libraries.isEmpty }
        XCTAssertTrue(loaded)
        XCTAssertEqual(sleeper.sleepCount, 0)
        state.stop()
    }

    func testRefreshRunsOffTheMainThread() async {
        let state = makeState()
        await state.refresh()
        await state.handle(.mounted(volume("Somewhere")))
        await fireDebounce(state)
        await state.waitForPendingWork()

        XCTAssertGreaterThanOrEqual(service.listCalls, 2, "the refreshes actually ran")
        XCTAssertGreaterThanOrEqual(service.installedCalls, 2)
        XCTAssertEqual(service.mainThreadCalls, [], "the engine reads the mount table and probes drives")
    }

    func testEventsAreConsumedFromTheFirstStreamOnly() async {
        let (first, firstSink) = AsyncStream<VolumeEvent>.makeStream()
        let (second, secondSink) = AsyncStream<VolumeEvent>.makeStream()
        let state = makeState()
        state.start(events: first)
        state.start(events: second)

        firstSink.yield(.didUnmount(volume("One")))
        secondSink.yield(.didUnmount(volume("Two")))
        let handled = await eventually { state.handledEventCount >= 1 }
        XCTAssertTrue(handled)
        try? await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertEqual(state.handledEventCount, 1)
        XCTAssertEqual(probe.invalidations, ["/Volumes/One"])
        state.stop()
    }

    func testEventsFromTheStreamDriveTheState() async {
        let (events, sink) = AsyncStream<VolumeEvent>.makeStream()
        let state = makeState()
        state.start(events: events)
        let loaded = await eventually { state.measuredSpeeds.isEmpty && !state.libraries.isEmpty }
        XCTAssertTrue(loaded)
        state.recordMeasuredSpeed(300, for: "t7-id")

        sink.yield(.didUnmount(volume("T7", uuid: "UUID-T7")))

        let dropped = await eventually { state.measuredSpeed(for: "t7-id") == nil }
        XCTAssertTrue(dropped)
        state.stop()
    }

    func testStopEndsEventHandling() async {
        let (events, sink) = AsyncStream<VolumeEvent>.makeStream()
        let state = makeState()
        state.start(events: events)
        let loaded = await eventually { !state.libraries.isEmpty }
        XCTAssertTrue(loaded)

        state.stop()
        sink.yield(.didUnmount(volume("T7")))
        try? await Task.sleep(nanoseconds: 100_000_000)

        XCTAssertEqual(state.handledEventCount, 0)
        XCTAssertTrue(probe.invalidations.isEmpty)
    }

    // MARK: - No errors for availability (3.3, 3.4)

    func testAnOfflineLibraryRaisesNoAppErrorAndNoToast() async {
        let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)
        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, spare]
            $0.installed = [LibraryFixtures.row("on-spare", library: spare, availability: .notConnected)]
        }
        let state = makeState()
        let model = await makeModel(using: state)
        let errorBefore = model.error
        let toastBefore = model.activeToast

        await model.libraryState.refresh()
        await model.libraryState.handle(.didUnmount(volume("Spare")))
        await fireDebounce(model.libraryState)

        XCTAssertEqual(model.libraryState.allInstalled.map(\.alias), ["on-spare"])
        XCTAssertEqual(model.error, errorBefore)
        XCTAssertEqual(model.activeToast, toastBefore)
    }

    func testAThrowingServiceKeepsThePreviousStateAndRaisesNoAppError() async {
        let state = makeState()
        let model = await makeModel(using: state)
        await state.refresh()
        let libraries = state.libraries
        let installed = state.allInstalled
        let errorBefore = model.error
        let toastBefore = model.activeToast
        XCTAssertFalse(libraries.isEmpty)

        service.with {
            $0.listError = FakeServiceError.scripted("mount table unreadable")
            $0.installedError = FakeServiceError.scripted("registry unreadable")
        }
        await state.refresh()

        XCTAssertEqual(state.libraries, libraries, "a failed listing keeps what was known")
        XCTAssertEqual(state.allInstalled, installed)
        XCTAssertNotNil(state.lastRefreshFailure, "the failure is state, for diagnostics")
        XCTAssertEqual(model.error, errorBefore)
        XCTAssertEqual(model.activeToast, toastBefore)
    }

    func testAFailedInstalledListingDoesNotDiscardTheLibraryList() async {
        let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)
        let state = makeState()
        await state.refresh()
        let installed = state.allInstalled

        service.with {
            $0.libraries = [LibraryFixtures.internalLibrary, self.t7, spare]
            $0.installedError = FakeServiceError.scripted("registry unreadable")
        }
        await state.refresh()

        XCTAssertEqual(state.libraries.map(\.id), ["internal", "t7-id", "spare-id"])
        XCTAssertEqual(state.allInstalled, installed)
    }

    func testAnAppModelDoesNotStartWatchingVolumesByItself() async {
        // Launch wiring is the window's job (RootView.onAppear): a model built
        // under XCTest must not register with the workspace or call the engine.
        let model = AppModel()
        XCTAssertFalse(model.libraryState.isRunning)
        XCTAssertTrue(model.libraryState.libraries.isEmpty)
        await model.modelScanTask?.value
    }

    func testARefreshLeavesTheExistingInstalledListAlone() async {
        let state = makeState()
        let model = await makeModel(using: state)
        let kept = LibraryFixtures.row("kept")
        model.installed = [kept]

        await state.refresh()
        await state.waitForPendingWork()

        XCTAssertEqual(model.installed, [kept], "pickers still read the available-only list")
    }

    // MARK: - Lookups

    func testLookupsFindTheLibraryOfAModelAndOfAPath() async {
        let spare = LibraryFixtures.external("spare-id", name: "Spare", available: false)
        service.with { $0.libraries = [LibraryFixtures.internalLibrary, self.t7, spare] }
        let state = makeState()
        await state.refresh()

        XCTAssertEqual(state.library(for: LibraryFixtures.row("local"))?.id, "internal")
        XCTAssertEqual(state.library(for: LibraryFixtures.row("on-t7", library: t7))?.id, "t7-id")
        XCTAssertEqual(
            state.library(for: LibraryFixtures.row("on-spare", library: spare, availability: .notConnected))?.id,
            "spare-id", "an offline model still names its library")
        let custom = InstalledModel(
            alias: "x", repo: "o/x", path: "/tmp/x.gturbo", family: "qwen",
            library: nil, libraryName: "custom location", locationKind: .custom)
        XCTAssertNil(state.library(for: custom))

        XCTAssertEqual(state.library(containing: "/Volumes/T7/TurboSpark/text/a.gturbo")?.id, "t7-id")
        XCTAssertEqual(state.library(containing: "/Volumes/T7/TurboSpark")?.id, "t7-id")
        XCTAssertNil(state.library(containing: "/Volumes/T7/Other/a.gturbo"))
        XCTAssertNil(state.library(containing: "/Volumes/T7/TurboSparkX/a.gturbo"), "a prefix is not containment")
        XCTAssertEqual(
            state.library(containing: "/Users/test/.turbospark/models/text/a.gturbo")?.id, "internal")
        XCTAssertNil(state.library(containing: "/Volumes/Spare/TurboSpark/a"), "an offline library has no root")
    }
}
