import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Review findings B1 and B2 of tasks 5.11 and 5.12: "unload" must release the
/// model, not just drop one reference to it, and the app may say "unloaded"
/// only for what is really gone.
///
/// The real `TurboSparkSession` needs a multi-gigabyte checkpoint, so the
/// holders here are probe objects captured by real `AppModel` task slots
/// (`titleGenerationTask`, `memoryCaptureTask`, `backgroundAgentTasks`), which
/// are the very places the reviewer found holding the session. The release
/// code under test is the LIVE code: the chat session reaches it through a
/// `ChatSessionSlot` over a probe owner, never through a ledger entry the test
/// wrote itself.
final class SessionTeardownTests: LibraryStateTestCase {
    private typealias Fx = EjectFixtures

    private var holders: SessionHolders!

    override func setUp() async throws {
        try await super.setUp()
        holders = SessionHolders()
    }

    override func tearDown() async throws {
        holders.openAll()
        holders = nil
        try await super.tearDown()
    }

    // MARK: - The chat stop step (B1)

    func testTheChatStopStepCancelsAndAwaitsTitleMemoryAndBackgroundAgentWork() async {
        let model = await makeModel(using: makeState())
        let held = holders.hold(on: model)
        let source = AppModelLoadedModels(model: model, patience: .seconds(3))

        let started = Date()
        await source.stopGeneration(in: .chat, models: [Fx.chat])

        XCTAssertNil(model.titleGenerationTask, "the title task was cancelled and awaited")
        XCTAssertNil(model.memoryCaptureTask, "the memory task was cancelled and awaited")
        XCTAssertTrue(model.backgroundAgentTasks.isEmpty, "the background agent was stopped and awaited")
        XCTAssertEqual(model.backgroundAgentRuns["bga_t1"]?.status, "killed", "stopped the way Stop All stops it")
        XCTAssertLessThan(Date().timeIntervalSince(started), 2, "it did not wait out the patience")
        let released = await eventuallyAsync { held.value == nil }
        XCTAssertTrue(released, "nothing holds the session any more")
    }

    /// Pins the per-poll re-cancel: the title queue starts the next task from
    /// the tail of a cancelled one, so cancelling once leaves the successor
    /// holding the session for as long as the patience.
    func testTheChatStopStepCancelsASuccessorTheCancelledTitleTaskStarted() async {
        let model = await makeModel(using: makeState())
        let successors = EjectBox(0)
        let ref = holders.holdByTitleTaskWithSuccessor(on: model, successors: successors)
        let source = AppModelLoadedModels(model: model, patience: .seconds(3))

        let started = Date()
        await source.stopGeneration(in: .chat, models: [Fx.chat])

        XCTAssertEqual(successors.value, 1, "the first task did start a successor")
        XCTAssertNil(model.titleGenerationTask, "and the successor was cancelled too")
        XCTAssertLessThan(Date().timeIntervalSince(started), 2, "without waiting out the patience")
        let released = await eventuallyAsync { ref.value == nil }
        XCTAssertTrue(released)
    }

    func testTheChatStopStepGivesUpAfterThePatienceWhenSomethingIgnoresTheCancel() async {
        let model = await makeModel(using: makeState())
        let gate = EjectBox(false)
        let held = holders.hold(on: model, ignoringCancelUntil: gate)
        let source = AppModelLoadedModels(model: model, patience: .milliseconds(250))

        let started = Date()
        await source.stopGeneration(in: .chat, models: [Fx.chat])
        XCTAssertGreaterThanOrEqual(Date().timeIntervalSince(started), 0.2, "it waited for the patience")
        XCTAssertLessThan(Date().timeIntervalSince(started), 3, "and no longer")
        XCTAssertNotNil(held.value, "the holder is still there; the caller must verify")

        gate.value = true
        let ended = await eventuallyAsync { held.value == nil }
        XCTAssertTrue(ended)
    }

    /// `Task.sleep` throws at once in a cancelled task, so a loop without the
    /// `Task.isCancelled` guard would spin hot for the whole patience.
    func testACancelledCallerOfTheChatStopStepStillCancelsOnceAndDoesNotSpin() async {
        let model = await makeModel(using: makeState())
        let gate = EjectBox(false)
        _ = holders.hold(on: model, ignoringCancelUntil: gate)
        let source = AppModelLoadedModels(model: model, patience: .seconds(3))

        let caller = Task { @MainActor in await source.stopGeneration(in: .chat, models: [Fx.chat]) }
        caller.cancel()
        let started = Date()
        await caller.value

        XCTAssertLessThan(Date().timeIntervalSince(started), 1, "a cancelled caller does not wait out the patience")
        XCTAssertEqual(model.killedBackgroundAgentIDs, ["bga_t1"], "but the cancel pass ran once")
        gate.value = true
    }

    func testTheStopStepForOtherCategoriesLeavesTheChatSessionsWorkAlone() async {
        let model = await makeModel(using: makeState())
        let gate = EjectBox(false)
        let held = holders.hold(on: model, ignoringCancelUntil: gate)
        let source = AppModelLoadedModels(model: model, patience: .milliseconds(100))

        await source.stopGeneration(in: .server, models: [Fx.server])
        await source.stopGeneration(in: .audio, models: [Fx.audio])
        XCTAssertNotNil(model.titleGenerationTask, "an audio or server eject must not cancel the chat's helpers")
        XCTAssertNotNil(model.memoryCaptureTask)
        XCTAssertFalse(model.backgroundAgentTasks.isEmpty)
        XCTAssertTrue(model.killedBackgroundAgentIDs.isEmpty)

        gate.value = true
        let ended = await eventuallyAsync { held.value == nil }
        XCTAssertTrue(ended)
    }

    func testCancelSessionBoundWorkSaysWhetherAnythingIsStillRunning() async {
        let model = await makeModel(using: makeState())
        XCTAssertFalse(model.cancelSessionBoundWork(), "nothing running")
        let held = holders.hold(on: model)
        XCTAssertTrue(model.cancelSessionBoundWork(), "it was running when asked")
        let ended = await eventuallyAsync { !model.cancelSessionBoundWork() && held.value == nil }
        XCTAssertTrue(ended)
    }

    // MARK: - Waiting until the session is really gone (B1)

    func testDropAndWaitReturnsNilOnceTheLastHolderEnds() async {
        let owner = ProbeHolder()
        let gate = EjectBox(false)
        let task = holders.holdProbe(in: owner, gate: gate)
        let result = await SessionRelease.dropAndWait(
            get: { owner.thing }, clear: { owner.thing = nil }, patience: .seconds(3),
            settle: { gate.value = true })
        XCTAssertNil(result, "the holder ended, so the object is gone")
        XCTAssertNil(owner.thing, "and the owner let go")
        gate.value = true  // never leave a holder running if the assertions above failed
        await task.value
    }

    func testDropAndWaitReportsWhatIsStillHoldingOnAfterThePatience() async {
        let owner = ProbeHolder()
        let gate = EjectBox(false)
        let task = holders.holdProbe(in: owner, gate: gate)
        let result = await SessionRelease.dropAndWait(
            get: { owner.thing }, clear: { owner.thing = nil }, patience: .milliseconds(150))
        XCTAssertNil(owner.thing, "the owner let go either way")
        XCTAssertNotNil(result?.value, "but something else still holds the object")

        gate.value = true
        await task.value
        let ended = await eventuallyAsync { result?.value == nil }
        XCTAssertTrue(ended)
    }

    func testWaitUntilGoneSettlesBeforeEveryLookSoNewWorkIsCancelledToo() async {
        let owner = ProbeHolder()
        let gate = EjectBox(false)
        let task = holders.holdProbe(in: owner, gate: gate)
        let ref = WeakObject(owner.thing)
        owner.thing = nil
        let settled = EjectBox(0)
        let gone = await SessionRelease.waitUntilGone(ref, patience: .milliseconds(120), interval: .milliseconds(10)) {
            settled.value += 1
        }
        XCTAssertFalse(gone)
        XCTAssertGreaterThan(settled.value, 2, "settled again on each poll")
        gate.value = true
        await task.value
    }

    func testTheInventoryListsARetiredChatOnceEvenWhenTheSamePathIsLoadedAgain() {
        let path = "/Volumes/T7/TurboSpark/text/qwen.gturbo"
        let models = LoadedModelInventory.make(
            chat: path, retiredChats: [path, "/Volumes/T7/TurboSpark/text/./qwen.gturbo"],
            serverSessions: [], serverAudio: [], audio: nil,
            name: LoadedModelInventory.fallbackName(forPath:))
        XCTAssertEqual(models.map(\.path), [path])
    }

    func testTheInventoryListsARetiredAudioSessionOnceAndAsAudio() {
        let path = "/Volumes/T7/TurboSpark/audio/speech.gturbo"
        let models = LoadedModelInventory.make(
            chat: nil, serverSessions: [], serverAudio: [], audio: path,
            retiredAudio: [path, "/Volumes/T7/TurboSpark/audio/./speech.gturbo"],
            name: LoadedModelInventory.fallbackName(forPath:))
        XCTAssertEqual(models.map(\.path), [path])
        XCTAssertEqual(models.map(\.category), [.audio])
    }

    // MARK: - The bounded retry (B2)

    func testTheRetryTriesAgainUntilTheStepWorks() async {
        let attempts = EjectBox<[String: Int]>([:])
        let left = await BoundedRetry.drain(["a"], patience: .seconds(2), interval: .milliseconds(5)) { id in
            attempts.value[id, default: 0] += 1
            return attempts.value[id, default: 0] > 2
        }
        XCTAssertEqual(left, [])
        XCTAssertEqual(attempts.value["a"], 3, "refused twice, worked on the third")
    }

    func testTheRetryRetriesEachIdOnlyUntilItWorks() async {
        let attempts = EjectBox<[String: Int]>([:])
        let left = await BoundedRetry.drain(["a", "b"], patience: .seconds(2), interval: .milliseconds(5)) { id in
            attempts.value[id, default: 0] += 1
            return id == "a" || attempts.value[id, default: 0] > 2
        }
        XCTAssertEqual(left, [])
        XCTAssertEqual(attempts.value["a"], 1, "an id that worked is not tried again")
        XCTAssertEqual(attempts.value["b"], 3)
    }

    func testTheRetryIsBoundedAndReturnsWhatNeverWorked() async {
        let attempts = EjectBox(0)
        let started = Date()
        let left = await BoundedRetry.drain(["a"], patience: .milliseconds(100), interval: .milliseconds(10)) { _ in
            attempts.value += 1
            return false
        }
        XCTAssertEqual(left, ["a"])
        XCTAssertGreaterThan(attempts.value, 2)
        XCTAssertLessThan(attempts.value, 40)
        XCTAssertLessThan(Date().timeIntervalSince(started), 2)
    }

    func testTheRetryAlwaysMakesTheFirstAttempt() async {
        let attempts = EjectBox(0)
        let left = await BoundedRetry.drain(["a"], patience: .zero, interval: .milliseconds(10)) { _ in
            attempts.value += 1
            return true
        }
        XCTAssertEqual(left, [])
        XCTAssertEqual(attempts.value, 1)
    }

    // MARK: - The bounded await (audio close)

    func testABoundedAwaitReturnsTrueWhenTheWorkFinishesInTime() async {
        let ran = EjectBox(false)
        let finished = await BoundedAwait.finishes(within: .seconds(2)) { ran.value = true }
        XCTAssertTrue(finished)
        XCTAssertTrue(ran.value)
    }

    func testABoundedAwaitReturnsFalseWhenTheWorkOutlastsThePatienceAndLetsItRun() async {
        let gate = EjectBox(false)
        holders.gateForCleanup(gate)
        let done = EjectBox(false)
        let started = Date()
        let finished = await BoundedAwait.finishes(within: .milliseconds(150)) {
            while !gate.value { await SessionHolders.nap() }
            done.value = true
        }
        XCTAssertFalse(finished)
        XCTAssertLessThan(Date().timeIntervalSince(started), 2, "the wait is bounded")
        XCTAssertFalse(done.value, "the work was not cut short")
        gate.value = true
        let ended = await eventuallyAsync { done.value }
        XCTAssertTrue(ended, "and it carries on after the wait gave up")
    }

    // MARK: - The whole path, over the live source (B1, 5.11 and 5.12)

    private struct Rig {
        let coordinator: LibraryEjectCoordinator
        let source: AppModelLoadedModels
        let ejector: FakeEjector
        let notices: NoticeRecorder
        let log: EjectLog
    }

    private func makeRig(_ model: AppModel, owner: ProbeHolder, patience: Duration) async -> Rig {
        let log = EjectLog()
        let source = AppModelLoadedModels(
            model: model, patience: patience,
            seams: .init(chatSession: SessionHolders.slot(over: owner, path: Fx.chat.path)))
        let ejector = FakeEjector(log: log)
        let notices = NoticeRecorder(log: log)
        let coordinator = LibraryEjectCoordinator(
            source: source, ejector: ejector, locator: FakeVolumeLocator(), present: { notices.add($0) })
        await coordinator.refreshSnapshot()
        return Rig(coordinator: coordinator, source: source, ejector: ejector, notices: notices, log: log)
    }

    func testAnEjectWhileBackgroundWorkHoldsTheChatModelCancelsItAndEjectsOnce() async {
        let model = await makeModel(using: makeState())
        let owner = ProbeHolder()
        let held = holders.hold(on: model, owner: owner)
        let rig = await makeRig(model, owner: owner, patience: .seconds(3))
        XCTAssertEqual(rig.coordinator.decisions.snapshot.map(\.model.path), [Fx.chat.path])

        XCTAssertNotNil(rig.coordinator.unmountRequested(for: Fx.t7Disk))
        await rig.coordinator.waitForIdle(timeout: 10)

        XCTAssertEqual(rig.ejector.calls, [Fx.t7Disk], "one eject")
        XCTAssertEqual(rig.notices.all, [.ejected(models: ["qwen"])])
        XCTAssertNil(held.value, "the session is really gone")
        XCTAssertNil(owner.thing, "and the owner let go")
        XCTAssertNil(model.titleGenerationTask)
        XCTAssertNil(model.memoryCaptureTask)
        XCTAssertTrue(model.backgroundAgentTasks.isEmpty)
    }

    func testAnEjectWhileTheChatModelStaysHeldDoesNotEjectAndSaysSo() async {
        let model = await makeModel(using: makeState())
        let owner = ProbeHolder()
        let gate = EjectBox(false)
        let held = holders.hold(on: model, owner: owner, ignoringCancelUntil: gate)
        let rig = await makeRig(model, owner: owner, patience: .milliseconds(200))

        XCTAssertNotNil(rig.coordinator.unmountRequested(for: Fx.t7Disk))
        await rig.coordinator.waitForIdle(timeout: 10)

        XCTAssertTrue(rig.ejector.calls.isEmpty, "the model is still mapped: no eject")
        XCTAssertEqual(rig.notices.all, [.unloadBlocked(models: ["qwen"])])
        XCTAssertFalse(rig.log.entries.contains("notice:ejected"))
        XCTAssertEqual(rig.coordinator.decisions.snapshot.map(\.model.path), [Fx.chat.path], "still tracked")
        XCTAssertNil(owner.thing, "the app itself did let go")
        XCTAssertEqual(rig.source.loadedModels().map(\.path), [Fx.chat.path], "but the model is still listed as loaded")

        gate.value = true
        let ended = await eventuallyAsync { held.value == nil }
        XCTAssertTrue(ended)
    }

    func testASurpriseRemovalCancelsTheBackgroundWorkToo() async {
        let model = await makeModel(using: makeState())
        let owner = ProbeHolder()
        let held = holders.hold(on: model, owner: owner)
        let rig = await makeRig(model, owner: owner, patience: .seconds(3))

        await rig.coordinator.handle(.didUnmount(Fx.t7Volume))

        XCTAssertEqual(rig.notices.all, [.removedWithoutEject(models: ["qwen"])])
        XCTAssertTrue(rig.ejector.calls.isEmpty)
        XCTAssertNil(model.titleGenerationTask)
        XCTAssertNil(model.memoryCaptureTask)
        XCTAssertTrue(model.backgroundAgentTasks.isEmpty)
        let released = await eventuallyAsync { held.value == nil }
        XCTAssertTrue(released)
    }

    func testASurpriseRemovalWithTheChatModelStillHeldDoesNotClaimItWasTornDown() async {
        let model = await makeModel(using: makeState())
        let owner = ProbeHolder()
        let gate = EjectBox(false)
        let held = holders.hold(on: model, owner: owner, ignoringCancelUntil: gate)
        let rig = await makeRig(model, owner: owner, patience: .milliseconds(200))

        await rig.coordinator.handle(.didUnmount(Fx.t7Volume))
        XCTAssertTrue(rig.notices.all.isEmpty, "nothing was torn down, so nothing is claimed")

        gate.value = true
        let ended = await eventuallyAsync { held.value == nil }
        XCTAssertTrue(ended)
    }
}
