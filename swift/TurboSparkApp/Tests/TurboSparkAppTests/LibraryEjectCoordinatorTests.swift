import Foundation
import XCTest

@testable import TurboSparkApp

/// Tasks 5.11 and 5.12 (requirements 6.1 to 6.4): a drive with loaded models
/// ejects in one action, and a drive pulled without ejecting ends the sessions
/// that held it.
///
/// Every case runs against fakes: a fake list of loaded models, a fake ejector
/// and a fake volume locator. No real disk, session or DiskArbitration object
/// is touched.
@MainActor
final class LibraryEjectCoordinatorTests: XCTestCase {
    private typealias Fx = EjectFixtures

    private struct Harness {
        let coordinator: LibraryEjectCoordinator
        let source: FakeLoadedModels
        let ejector: FakeEjector
        let log: EjectLog
        let notices: NoticeRecorder
    }

    private func makeHarness(
        _ models: [LoadedModel] = [], ejectResult: EjectResult = .ejected,
        locator: ModelVolumeLocating = FakeVolumeLocator()
    ) async -> Harness {
        let log = EjectLog()
        let source = FakeLoadedModels(models, log: log)
        let ejector = FakeEjector(result: ejectResult, log: log)
        let notices = NoticeRecorder(log: log)
        let coordinator = LibraryEjectCoordinator(
            source: source, ejector: ejector, locator: locator, present: { notices.add($0) })
        await coordinator.refreshSnapshot()
        return Harness(coordinator: coordinator, source: source, ejector: ejector, log: log, notices: notices)
    }

    // MARK: - Approval decisions (from the snapshot alone)

    func testWithNoLoadedModelEveryUnmountIsApproved() async {
        let h = await makeHarness()
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.spareDisk))
        XCTAssertEqual(h.coordinator.decisions.activeCycleCount, 0, "approving starts nothing")
        h.coordinator.stop()
    }

    func testAChatModelOnTheDiskDissents() async {
        let h = await makeHarness([Fx.chat])
        let answer = h.coordinator.unmountRequested(for: Fx.t7Disk)
        XCTAssertNotNil(answer, "a disk backing a loaded chat model is dissented from")
        XCTAssertEqual(answer?.reason, LibraryEjectCoordinator.dissentReason)
        await h.coordinator.waitForIdle()
    }

    func testAServerAttachedModelOnTheDiskDissents() async {
        let h = await makeHarness([Fx.server])
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
    }

    func testAnAudioModelOnTheDiskDissents() async {
        let h = await makeHarness([Fx.audio])
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
    }

    func testAModelOnAnotherDiskIsApproved() async {
        let h = await makeHarness([Fx.onSpare])
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.t7Disk), "the T7 holds nothing that is loaded")
        XCTAssertEqual(h.coordinator.decisions.activeCycleCount, 0)
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.spareDisk), "but the Spare does")
        await h.coordinator.waitForIdle()
    }

    func testThisMacsDiskIsNeverDissentedFromEvenWithAModelOnIt() async {
        let h = await makeHarness([Fx.onThisMac])
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.dataVolume))
        // A path under the home is also "within" the root volume by prefix, so
        // only the explicit rule keeps a request for / from matching.
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.rootVolume))
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        XCTAssertTrue(
            h.coordinator.decisions.snapshot.isEmpty,
            "a model on the internal disk is not even remembered as backing a drive")
    }

    func testAnInternalVolumeRequestIsApprovedEvenWhenTheModelsVolumeCouldNotBeRead() async {
        // The locator could not read the model's volume, so the snapshot cannot
        // tell it is internal and keeps it. The request side still knows the
        // root and the data volume are never a drive.
        struct Unreadable: ModelVolumeLocating {
            func locate(path: String) -> ModelVolume { .unresolved(path) }
        }
        let h = await makeHarness([Fx.onThisMac], locator: Unreadable())
        XCTAssertEqual(h.coordinator.decisions.snapshot.count, 1, "the unreadable entry is kept")
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.rootVolume))
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.dataVolume))
        XCTAssertEqual(h.coordinator.decisions.activeCycleCount, 0)
        h.coordinator.stop()
    }

    func testAModelMatchesByBSDNameWhenTheVolumeWasRenamed() async {
        let h = await makeHarness([Fx.chat])
        let renamed = DiskIdentity(
            bsdName: "disk5s1", wholeDiskBSDName: "disk5",
            volumeURL: URL(fileURLWithPath: "/Volumes/T7 renamed", isDirectory: true))
        XCTAssertNotNil(h.coordinator.unmountRequested(for: renamed), "the device is the same")
        await h.coordinator.waitForIdle()
    }

    func testAPathPrefixThatIsNotAPathBoundaryDoesNotMatch() async {
        // /Volumes/T70 is not inside /Volumes/T7.
        var locator = FakeVolumeLocator()
        locator.volumes["T70"] = ModelVolume(
            realPath: "", mountPoint: "/Volumes/T70", bsdName: "disk8s1", diskNames: ["disk8"])
        let model = LoadedModel(category: .chat, name: "q", path: "/Volumes/T70/TurboSpark/text/q.gturbo")
        let h = await makeHarness([model], locator: locator)
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        h.coordinator.stop()
    }

    func testAWholeDiskRequestMatchesThroughTheDisksTheVolumeSitsOn() async {
        let h = await makeHarness([Fx.chat])
        // A request that names the physical disk and no volume (the APFS store
        // under the synthesized container).
        let physical = DiskIdentity(bsdName: "disk4", wholeDiskBSDName: "disk4", volumeURL: nil)
        XCTAssertNotNil(h.coordinator.unmountRequested(for: physical))
        await h.coordinator.waitForIdle()

        let other = await makeHarness([Fx.chat])
        let unrelated = DiskIdentity(bsdName: "disk9", wholeDiskBSDName: "disk9", volumeURL: nil)
        XCTAssertNil(other.coordinator.unmountRequested(for: unrelated))
        other.coordinator.stop()
    }

    func testTheWholeDiskNameAloneDoesNotDecideForAVolumeRequest() async {
        // Another volume in the same APFS container reports the same
        // synthesized whole disk. Unmounting THAT volume does not touch the
        // model, so it is not dissented from.
        let h = await makeHarness([Fx.chat])
        let sibling = DiskIdentity(
            bsdName: "disk5s3", wholeDiskBSDName: "disk5",
            volumeURL: URL(fileURLWithPath: "/Volumes/Backup", isDirectory: true))
        XCTAssertNil(h.coordinator.unmountRequested(for: sibling))
        XCTAssertEqual(h.coordinator.decisions.activeCycleCount, 0)
        h.coordinator.stop()
    }

    func testTheDelegateAnswersWhileTheMainThreadIsBlocked() async {
        let h = await makeHarness([Fx.chat])
        let answer = EjectBox<UnmountDissent?>(nil)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            answer.value = h.coordinator.unmountRequested(for: Fx.t7Disk)
            done.signal()
        }
        // This test runs on the main thread. Block it, as a busy main thread
        // would be: a delegate that needs the main actor cannot answer.
        let result = blockingWait(done, seconds: 2)
        XCTAssertEqual(result, .success, "the decision waited for the main thread")
        XCTAssertNotNil(answer.value, "and it still dissented")
        await h.coordinator.waitForIdle()
    }

    func testTheMonitorGetsTheDissentInTimeWhileTheMainThreadIsBlocked() async {
        // The same through the real monitor: if the delegate needed main, the
        // monitor's 100 ms deadline would pass and the unmount would be approved.
        let h = await makeHarness([Fx.chat])
        let source = ScriptedApprovalSource()
        let monitor = VolumeMonitor(
            notificationCenter: NotificationCenter(), approvalSource: source,
            readDetails: { _ in VolumeDetails(bsdName: nil, uuid: nil) })
        monitor.delegate = h.coordinator
        monitor.start()
        let answer = EjectBox<UnmountDissent?>(nil)
        let done = DispatchSemaphore(value: 0)
        DispatchQueue.global().async {
            answer.value = source.requestUnmount(Fx.t7Disk)
            done.signal()
        }
        XCTAssertEqual(done.wait(timeout: .now() + 2), .success)
        XCTAssertNotNil(answer.value, "the monitor timed out and approved")
        monitor.stop()
        await h.coordinator.waitForIdle()
    }

    func testTheDecisionIsMadeInWellUnderTheApprovalTimeout() async {
        let h = await makeHarness([Fx.chat, Fx.server, Fx.audio, Fx.onSpare])
        var slowest: TimeInterval = 0
        // The first call is a dissent that opens a cycle; the rest are the
        // approve and the join paths. All of them are a lock and a scan.
        let disks = [Fx.spareDisk, Fx.t7Disk, Fx.dataVolume, Fx.t7DataDisk, Fx.spareDisk]
        for round in 0..<200 {
            let start = DispatchTime.now().uptimeNanoseconds
            _ = h.coordinator.decisions.decide(disks[round % disks.count])
            slowest = max(slowest, Double(DispatchTime.now().uptimeNanoseconds - start) / 1_000_000_000)
        }
        XCTAssertLessThan(slowest, VolumeMonitor.approvalTimeout / 4, "a decision is a lock and a scan")
    }

    func testPathsAreNormalizedWithoutTouchingTheFileSystem() {
        typealias State = EjectDecisionState
        XCTAssertEqual(State.normalized("/Volumes/T7/"), "/Volumes/T7")
        XCTAssertEqual(State.normalized("/Volumes//T7/./x/../y"), "/Volumes/T7/y")
        XCTAssertEqual(State.normalized("/"), "/")
        XCTAssertEqual(State.normalized("/.."), "/")
        // A /private prefix is kept as written: `standardizedFileURL` would
        // stat the path to decide whether to drop it.
        XCTAssertEqual(State.normalized("/private/var/run/x"), "/private/var/run/x")
    }

    func testASlowRefreshThatStartedEarlierCannotOverwriteANewerSnapshot() {
        let state = EjectDecisionState()
        let locator = FakeVolumeLocator()
        let older = state.nextTicket()
        let newer = state.nextTicket()
        state.apply(
            [EjectDecisionState.Entry(model: Fx.server, volume: locator.locate(path: Fx.server.path))],
            ticket: newer)
        // The earlier read finishes last.
        state.apply(
            [EjectDecisionState.Entry(model: Fx.chat, volume: locator.locate(path: Fx.chat.path))],
            ticket: older)
        XCTAssertEqual(state.snapshot.map(\.model), [Fx.server], "the newer snapshot stays")
    }

    func testARefreshThatLandsAfterStopDoesNotRearmTheCoordinator() async {
        let h = await makeHarness([Fx.chat])
        h.coordinator.stop()
        // A refresh scheduled before the stop finishes after it.
        await h.coordinator.refreshSnapshot()
        let locator = FakeVolumeLocator()
        h.coordinator.decisions.apply(
            [EjectDecisionState.Entry(model: Fx.chat, volume: locator.locate(path: Fx.chat.path))],
            ticket: h.coordinator.decisions.nextTicket())
        XCTAssertTrue(h.coordinator.decisions.snapshot.isEmpty, "a stopped state remembers nothing")
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
    }

    func testAStoppedCoordinatorApprovesEverything() async {
        let h = await makeHarness([Fx.chat])
        h.coordinator.stop()
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.t7Disk), "a locked app never dissents")
    }

    // MARK: - The eject sequence (requirements 6.1, 6.2)

    func testADissentedEjectStopsThenUnloadsThenEjectsOnceThenTellsTheUser() async {
        let h = await makeHarness([Fx.chat, Fx.server, Fx.audio])
        let answer = h.coordinator.unmountRequested(for: Fx.t7Disk)
        h.log.add("dissent")
        XCTAssertNotNil(answer)
        await h.coordinator.waitForIdle()

        XCTAssertEqual(
            h.log.entries,
            [
                "dissent",
                "stop:chat", "stop:server", "stop:audio",
                "unload:chat", "unload:server", "unload:audio",
                "eject",
                "notice:ejected",
            ],
            "dissent first, all generation stopped BEFORE any unload, one eject, then the notice")
        XCTAssertEqual(h.ejector.calls, [Fx.t7Disk], "the disk the request named")
        XCTAssertEqual(h.notices.all, [.ejected(models: ["qwen", "gemma", "speech"])])
        XCTAssertTrue(h.source.models.isEmpty)
    }

    func testEachSessionIsUnloadedExactlyOnceEvenWithSeveralModelsInIt() async {
        let second = LoadedModel(
            category: .server, name: "gemma2", path: "/Volumes/T7/TurboSpark/text/gemma2.gturbo")
        let h = await makeHarness([Fx.chat, Fx.server, second, Fx.audio])
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.source.unloaded, [.chat, .server, .audio], "one call per session kind")
        XCTAssertEqual(h.source.stopped, [.chat, .server, .audio])
        XCTAssertEqual(h.notices.all, [.ejected(models: ["qwen", "gemma", "gemma2", "speech"])])
    }

    func testOnlyTheRequestedDisksModelsAreUnloaded() async {
        let h = await makeHarness([Fx.chat, Fx.onSpare])
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.source.unloaded, [.chat])
        XCTAssertEqual(h.source.models, [Fx.onSpare], "the other drive's model stays loaded")
        XCTAssertEqual(h.ejector.calls, [Fx.t7Disk])
        XCTAssertEqual(h.notices.all, [.ejected(models: ["qwen"])])
    }

    func testModelsOnSiblingVolumesOfOneDriveAreUnloadedInOneCycle() async {
        let h = await makeHarness([Fx.chat, Fx.onSecondVolume])
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
        XCTAssertTrue(h.source.models.isEmpty, "the drive cannot eject while either volume is in use")
        XCTAssertEqual(h.ejector.calls.count, 1)
        XCTAssertEqual(h.notices.all, [.ejected(models: ["qwen", "phi"])])
    }

    func testAFailedReissueLeavesTheModelsUnloadedAndReportsTheSystemsReason() async {
        let h = await makeHarness([Fx.chat], ejectResult: .failed("Resource busy"))
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.ejector.calls.count, 1, "it does not retry")
        XCTAssertTrue(h.source.models.isEmpty, "the model stays unloaded")
        XCTAssertEqual(h.notices.all, [.ejectFailed(models: ["qwen"], reason: "Resource busy")])
    }

    func testASecondUserRequestStartsASecondCycleAndNothingMore() async {
        let h = await makeHarness([Fx.chat])
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.ejector.calls.count, 1)

        // The user loads a model again, sees the drive is still there, and
        // ejects once more: that is a new request with its own cycle.
        h.source.models = [Fx.chat]
        await h.coordinator.refreshSnapshot()
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.ejector.calls.count, 2, "one eject per request")
        XCTAssertEqual(h.source.unloaded, [.chat, .chat])
        XCTAssertEqual(h.notices.all.count, 2)
    }

    func testTheCoordinatorsOwnReissueIsApprovedNotDissentedFrom() async {
        let h = await makeHarness([Fx.chat, Fx.audio])
        let coordinator = h.coordinator
        let answers = EjectBox<[UnmountDissent?]>([])
        h.ejector.duringEject = { _ in
            // DiskArbitration asks again for the unmount the coordinator just
            // issued, for the volume, its sibling and the whole disk.
            for disk in [
                Fx.t7Disk, Fx.t7DataDisk,
                DiskIdentity(bsdName: "disk5", wholeDiskBSDName: "disk5", volumeURL: nil),
            ] {
                answers.value.append(coordinator.unmountRequested(for: disk))
            }
        }
        XCTAssertNotNil(coordinator.unmountRequested(for: Fx.t7Disk))
        await coordinator.waitForIdle()
        XCTAssertEqual(answers.value.count, 3)
        XCTAssertTrue(answers.value.allSatisfy { $0 == nil }, "dissenting from our own unmount would loop")
        XCTAssertEqual(h.ejector.calls.count, 1)
    }

    func testAModelLoadedDuringTheReissueDoesNotMakeTheCoordinatorDissentFromItself() async {
        let h = await makeHarness([Fx.chat])
        let coordinator = h.coordinator
        let source = h.source
        let answers = EjectBox<[UnmountDissent?]>([])
        h.ejector.duringEject = { _ in
            // A model lands on the drive while the unmount is in flight; the
            // snapshot now says the disk backs a loaded model again.
            await MainActor.run { source.models = [Fx.server] }
            await coordinator.refreshSnapshot()
            answers.value.append(coordinator.unmountRequested(for: Fx.t7Disk))
        }
        XCTAssertNotNil(coordinator.unmountRequested(for: Fx.t7Disk))
        await coordinator.waitForIdle()
        XCTAssertEqual(answers.value.count, 1)
        XCTAssertNil(answers.value.first ?? UnmountDissent(reason: "missing"), "our own re-issue is approved")
        XCTAssertEqual(h.ejector.calls.count, 1, "and it started no second cycle")
    }

    func testARequestForTheSameDriveWhileUnloadingJoinsTheCycleInsteadOfStartingAnother() async {
        let h = await makeHarness([Fx.chat, Fx.onSecondVolume])
        let gate = AsyncGate()
        h.source.unloadGate = gate
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        let parked = await eventuallyAsync { await gate.hasArrivals() }
        XCTAssertTrue(parked, "the cycle never reached the unload")

        // The second volume of the same drive asks while the first is parked:
        // still dissented from (it holds a model), but it is the same request.
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7DataDisk))
        XCTAssertEqual(h.coordinator.decisions.activeCycleCount, 1, "one cycle for one user request")
        await gate.release()
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.ejector.calls.count, 1)
        XCTAssertEqual(h.source.unloaded, [.chat, .server])
        XCTAssertEqual(h.notices.all.count, 1)
    }

    func testStoppingTheCoordinatorMidCycleAbandonsTheEjectAndShowsNothing() async {
        let h = await makeHarness([Fx.chat])
        let gate = AsyncGate()
        h.source.unloadGate = gate
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        let parked = await eventuallyAsync { await gate.hasArrivals() }
        XCTAssertTrue(parked)
        // The vault locked: this model is being replaced.
        h.coordinator.stop()
        await gate.release()
        await h.coordinator.waitForIdle()
        XCTAssertTrue(h.ejector.calls.isEmpty, "a replaced coordinator does not eject")
        XCTAssertTrue(h.notices.all.isEmpty)
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.t7Disk), "and it no longer dissents")
    }

    // MARK: - A drive removed without ejecting (requirement 6.3)

    func testDidUnmountWithALoadedModelTearsTheSessionDownAndAdvisesOnce() async {
        let h = await makeHarness([Fx.chat, Fx.server])
        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertEqual(
            h.log.entries,
            ["stop:chat", "stop:server", "unload:chat", "unload:server", "notice:removed"])
        XCTAssertEqual(h.notices.all, [.removedWithoutEject(models: ["qwen", "gemma"])])
        XCTAssertTrue(h.ejector.calls.isEmpty, "the drive is gone: there is nothing to eject")
        XCTAssertTrue(h.source.models.isEmpty)
    }

    func testDidUnmountWithNoLoadedModelOnTheVolumeDoesNothing() async {
        let h = await makeHarness([Fx.onSpare])
        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertTrue(h.log.entries.isEmpty)
        XCTAssertTrue(h.notices.all.isEmpty)
        XCTAssertEqual(h.source.models, [Fx.onSpare], "the model on the other drive is untouched")

        let nothing = await makeHarness()
        await nothing.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertTrue(nothing.notices.all.isEmpty)
    }

    func testARepeatedDidUnmountDoesNothing() async {
        let h = await makeHarness([Fx.chat])
        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        let after = h.log.entries
        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertEqual(h.log.entries, after, "nothing more happens")
        XCTAssertEqual(h.notices.all.count, 1)
    }

    func testWillUnmountNeverTearsAnythingDown() async {
        let h = await makeHarness([Fx.chat])
        // It fires on every attempt, including ones that are dissented from.
        await h.coordinator.handle(.willUnmount(Fx.t7Volume))
        XCTAssertTrue(h.log.entries.isEmpty)
        XCTAssertTrue(h.notices.all.isEmpty)
        XCTAssertEqual(h.source.models, [Fx.chat])
    }

    func testAModelIsFoundByBSDNameWhenTheMountPathChanged() async {
        let h = await makeHarness([Fx.chat])
        let renamed = VolumeDescriptor(
            url: URL(fileURLWithPath: "/Volumes/T7 renamed", isDirectory: true), name: "T7 renamed",
            bsdName: "disk5s1", uuid: "UUID-T7")
        await h.coordinator.handle(.didUnmount(renamed))
        XCTAssertEqual(h.notices.all, [.removedWithoutEject(models: ["qwen"])])
    }

    func testAnEjectTheCoordinatorCompletedShowsOnlyTheEjectNotice() async {
        // The model is listed again while the eject call is open (a stale
        // reading, or one loaded a moment ago), so only the coordinator's own
        // bookkeeping can tell this unmount from a pulled drive.
        let h = await makeHarness([Fx.chat])
        let coordinator = h.coordinator
        let source = h.source
        h.ejector.duringEject = { _ in
            // The OS reports the unmount while the eject call is still open.
            await MainActor.run { source.models = [Fx.chat] }
            await coordinator.handle(.didUnmount(Fx.t7Volume))
        }
        XCTAssertNotNil(coordinator.unmountRequested(for: Fx.t7Disk))
        await coordinator.waitForIdle()
        XCTAssertEqual(h.notices.all, [.ejected(models: ["qwen"])], "no advise-eject message beside it")
        XCTAssertEqual(h.source.unloaded, [.chat], "and nothing was torn down a second time")

        // The same event arriving after the cycle has finished, once the
        // session really is gone, finds nothing to tear down.
        h.source.models = []
        await coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertEqual(h.notices.all, [.ejected(models: ["qwen"])])
        XCTAssertEqual(h.source.unloaded, [.chat])
    }

    func testALateDidUnmountAfterTheCoordinatorsOwnEjectShowsNothing() async {
        let h = await makeHarness([Fx.chat])
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.notices.all, [.ejected(models: ["qwen"])])

        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertEqual(h.notices.all, [.ejected(models: ["qwen"])], "the eject notice stands alone")
        XCTAssertEqual(h.log.count(of: "unload:chat"), 1)
    }

    func testADriveRemovedWhileTheCycleIsStillUnloadingIsReportedAsRemoved() async {
        let h = await makeHarness([Fx.chat])
        let gate = AsyncGate()
        h.source.unloadGate = gate
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        let parked = await eventuallyAsync { await gate.hasArrivals() }
        XCTAssertTrue(parked)

        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertTrue(h.notices.all.isEmpty, "the cycle owns the teardown")
        await gate.release()
        await h.coordinator.waitForIdle()
        XCTAssertTrue(h.ejector.calls.isEmpty, "the drive is already gone")
        XCTAssertEqual(h.notices.all, [.removedWithoutEject(models: ["qwen"])])
        XCTAssertEqual(h.source.unloaded, [.chat], "unloaded once")
    }

    // MARK: - A model that stays held is not reported as unloaded

    func testAModelThatStaysHeldBlocksTheEjectAndTheNoticeNamesIt() async {
        let h = await makeHarness([Fx.chat])
        h.source.stubbornPaths = [Fx.chat.path]
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()

        XCTAssertTrue(h.ejector.calls.isEmpty, "ejecting a drive a session still holds is pointless")
        XCTAssertEqual(h.notices.all, [.unloadBlocked(models: ["qwen"])])
        XCTAssertFalse(h.log.entries.contains("eject"))
        XCTAssertFalse(h.log.entries.contains("notice:ejected"), "no 'Unloaded' for a model that is still loaded")
        XCTAssertEqual(h.source.models, [Fx.chat])
        XCTAssertEqual(h.coordinator.decisions.snapshot.map(\.model), [Fx.chat], "and it is still tracked")
        XCTAssertEqual(h.coordinator.decisions.activeCycleCount, 0, "the cycle ended")
    }

    func testWhenOneOfTwoModelsStaysHeldNoSuccessIsClaimedAndOnlyTheHeldOneIsNamed() async {
        let h = await makeHarness([Fx.chat, Fx.server])
        h.source.stubbornPaths = [Fx.server.path]
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()

        XCTAssertEqual(h.source.unloaded, [.chat, .server], "both were asked to go")
        XCTAssertEqual(h.source.models, [Fx.server])
        XCTAssertEqual(h.notices.all, [.unloadBlocked(models: ["gemma"])], "only the model still held is named")
        XCTAssertTrue(h.ejector.calls.isEmpty)
        XCTAssertEqual(h.coordinator.decisions.snapshot.map(\.model), [Fx.server])
    }

    func testAfterABlockedCycleTheNextRequestIsDissentedFromAndCanSucceed() async {
        let h = await makeHarness([Fx.chat])
        h.source.stubbornPaths = [Fx.chat.path]
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.notices.all, [.unloadBlocked(models: ["qwen"])])

        // The request that held it has finished.
        h.source.stubbornPaths = []
        XCTAssertNotNil(
            h.coordinator.unmountRequested(for: Fx.t7Disk),
            "the model is still tracked, so the drive is dissented from again")
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.ejector.calls, [Fx.t7Disk], "one eject, for the second request")
        XCTAssertEqual(h.notices.all, [.unloadBlocked(models: ["qwen"]), .ejected(models: ["qwen"])])
    }

    // MARK: - The release bracket (second fix pass, finding 1)
    //
    // The retry of a refused server detach is covered over the LIVE
    // `detachServerModels` in `ServerDetachTests`, not here over a copy.

    func testTheReleaseIsBracketedAroundTheStopStepTheUnloadAndTheCheck() async {
        let h = await makeHarness([Fx.chat, Fx.server])
        let before = h.source.trace.count
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.waitForIdle()

        let steps = Array(h.source.trace.dropFirst(before))
        let end = steps.firstIndex(of: "end")
        XCTAssertEqual(
            Array(steps.prefix((end ?? steps.count - 1) + 1)),
            ["begin", "stop:chat", "stop:server", "unload:chat", "unload:server", "read", "end"],
            "begun before the first stop, ended after the live set was read")
        XCTAssertEqual(h.source.beginCount, 1)
        XCTAssertEqual(h.source.holdDepth, 0, "balanced")
        XCTAssertEqual(h.ejector.calls, [Fx.t7Disk], "and the eject came after the release")
    }

    func testTheReleaseIsEndedEvenWhenTheCoordinatorIsStoppedPartWay() async {
        let h = await makeHarness([Fx.chat])
        let gate = AsyncGate()
        h.source.stopGate = gate
        XCTAssertNotNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        let arrived = await eventuallyAsync { await gate.hasArrivals() }
        XCTAssertTrue(arrived)
        XCTAssertEqual(h.source.holdDepth, 1, "held while the stop step is under way")

        h.coordinator.stop()
        await gate.release()
        await h.coordinator.waitForIdle()
        XCTAssertEqual(h.source.holdDepth, 0, "the early exit still ended it")
        XCTAssertTrue(h.ejector.calls.isEmpty)
    }

    func testASurpriseRemovalBracketsItsReleaseToo() async {
        let h = await makeHarness([Fx.chat])
        let before = h.source.trace.count
        await h.coordinator.handle(.didUnmount(Fx.t7Volume))

        let steps = Array(h.source.trace.dropFirst(before))
        let begin = steps.firstIndex(of: "begin")
        let unload = steps.firstIndex(of: "unload:chat")
        let end = steps.firstIndex(of: "end")
        XCTAssertNotNil(begin)
        XCTAssertNotNil(unload)
        XCTAssertNotNil(end)
        if let begin, let unload, let end {
            XCTAssertLessThan(begin, unload, "begun before the unload")
            XCTAssertLessThan(unload, end, "and ended after it")
        }
        XCTAssertEqual(h.source.beginCount, 1)
        XCTAssertEqual(h.source.holdDepth, 0)
    }

    func testNoReleaseIsBegunWhenNothingLoadedIsOnTheDrive() async {
        let h = await makeHarness([Fx.onSpare])
        XCTAssertNil(h.coordinator.unmountRequested(for: Fx.t7Disk))
        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertEqual(h.source.beginCount, 0)
    }

    func testASurpriseRemovalNamesOnlyTheModelsThatWereActuallyTornDown() async {
        let h = await makeHarness([Fx.chat, Fx.server])
        h.source.stubbornPaths = [Fx.server.path]
        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertEqual(h.notices.all, [.removedWithoutEject(models: ["qwen"])], "the server model is still held")
        XCTAssertEqual(h.source.models, [Fx.server])
        XCTAssertTrue(h.ejector.calls.isEmpty)
    }

    func testASurpriseRemovalWithEveryModelStillHeldClaimsNothing() async {
        let h = await makeHarness([Fx.chat])
        h.source.stubbornPaths = [Fx.chat.path]
        await h.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertTrue(h.notices.all.isEmpty, "nothing was torn down, so nothing is advised as torn down")
        XCTAssertEqual(h.source.models, [Fx.chat])
    }

    // MARK: - The event stream

    func testTheCoordinatorConsumesItsOwnEventStream() async {
        let h = await makeHarness([Fx.chat])
        let (events, sink) = AsyncStream<VolumeEvent>.makeStream()
        h.coordinator.start(events: events)
        sink.yield(.willUnmount(Fx.t7Volume))
        sink.yield(.didUnmount(Fx.t7Volume))
        let handled = await eventually { h.coordinator.handledEventCount == 2 }
        XCTAssertTrue(handled)
        XCTAssertEqual(h.notices.all, [.removedWithoutEject(models: ["qwen"])])
        h.coordinator.stop()
        sink.finish()
    }

    func testStartingTwiceDrainsOnlyTheFirstStream() async {
        let h = await makeHarness([Fx.chat])
        let (first, firstSink) = AsyncStream<VolumeEvent>.makeStream()
        let (second, secondSink) = AsyncStream<VolumeEvent>.makeStream()
        h.coordinator.start(events: first)
        h.coordinator.start(events: second)
        secondSink.yield(.didUnmount(Fx.t7Volume))
        await Task.yield()
        try? await Task.sleep(nanoseconds: 60_000_000)
        XCTAssertTrue(h.notices.all.isEmpty, "a second stream is ignored")
        firstSink.yield(.didUnmount(Fx.t7Volume))
        let handled = await eventually { h.coordinator.handledEventCount == 1 }
        XCTAssertTrue(handled)
        h.coordinator.stop()
        firstSink.finish()
        secondSink.finish()
    }
}

/// Plays the DiskArbitration side for the monitor: the test calls
/// `requestUnmount` as the callback would.
private final class ScriptedApprovalSource: UnmountApprovalSource, @unchecked Sendable {
    private let lock = NSLock()
    private var handler: (@Sendable (DiskIdentity) -> UnmountDissent?)?

    func start(_ handler: @escaping @Sendable (DiskIdentity) -> UnmountDissent?) {
        lock.lock(); self.handler = handler; lock.unlock()
    }

    func stop() { lock.lock(); handler = nil; lock.unlock() }

    func requestUnmount(_ disk: DiskIdentity) -> UnmountDissent? {
        lock.lock(); let current = handler; lock.unlock()
        return current?(disk)
    }
}

/// Sync on purpose: `DispatchSemaphore.wait` is unavailable in async code,
/// and this test means to block the calling thread.
private func blockingWait(_ semaphore: DispatchSemaphore, seconds: Double) -> DispatchTimeoutResult {
    semaphore.wait(timeout: .now() + seconds)
}
