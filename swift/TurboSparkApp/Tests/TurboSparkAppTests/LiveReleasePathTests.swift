import Foundation
import XCTest

@testable import TurboSparkApp

/// Second fix pass, finding 2 of tasks 5.11 and 5.12: the live `unload(.chat)`
/// and `unload(.audio)` run against probe objects, so a session that outlives
/// the wait is provably kept in the retired ledger and listed as loaded, and an
/// audio close that never returns cannot hang the eject.
final class LiveReleasePathTests: LibraryStateTestCase {
    private typealias Fx = EjectFixtures

    private var holders: SessionHolders!
    /// Opened at the end of every test so no close or holder outlives it.
    private var closeGates: [EjectBox<Bool>] = []

    override func setUp() async throws {
        try await super.setUp()
        holders = SessionHolders()
    }

    override func tearDown() async throws {
        holders.openAll()
        for gate in closeGates { gate.value = true }
        closeGates = []
        holders = nil
        try await super.tearDown()
    }

    // MARK: - Chat

    func testAChatSessionTheAppDroppedButATaskStillHoldsStaysListedAsLoaded() async {
        let model = await makeModel(using: makeState())
        let owner = ProbeHolder()
        let gate = EjectBox(false)
        let held = holders.hold(on: model, owner: owner, ignoringCancelUntil: gate)
        let source = AppModelLoadedModels(
            model: model, patience: .milliseconds(200),
            seams: .init(chatSession: SessionHolders.slot(over: owner, path: Fx.chat.path)))
        XCTAssertEqual(source.loadedModels().map(\.path), [Fx.chat.path], "loaded before the release")

        await source.unload(.chat, models: [Fx.chat])

        XCTAssertNil(owner.thing, "the app let go of its own reference")
        XCTAssertNotNil(held.value, "but a task still holds the session")
        XCTAssertEqual(source.retired.alive(.chat).map(\.path), [Fx.chat.path], "so the ledger remembers it")
        XCTAssertEqual(
            source.loadedModels().map(\.path), [Fx.chat.path],
            "and it is still listed as loaded, so no one is told it is unloaded")

        gate.value = true
        let gone = await eventuallyAsync { source.loadedModels().isEmpty }
        XCTAssertTrue(gone, "once the holder ends, it is no longer listed")
    }

    func testAChatSessionNothingElseHoldsIsReleasedAndNotRemembered() async {
        let model = await makeModel(using: makeState())
        let owner = ProbeHolder()
        let held = holders.hold(on: model, owner: owner)
        let source = AppModelLoadedModels(
            model: model, patience: .seconds(3),
            seams: .init(chatSession: SessionHolders.slot(over: owner, path: Fx.chat.path)))

        await source.unload(.chat, models: [Fx.chat])

        XCTAssertNil(owner.thing)
        XCTAssertNil(held.value, "the release cancelled the helpers that held it")
        XCTAssertTrue(source.retired.alive().isEmpty)
        XCTAssertTrue(source.loadedModels().isEmpty)
    }

    func testASecondUnloadWaitsAgainForARetiredSessionAndCancelsWhatHoldsIt() async {
        let model = await makeModel(using: makeState())
        let owner = ProbeHolder()
        let gate = EjectBox(false)
        let held = holders.hold(on: model, owner: owner, ignoringCancelUntil: gate)
        let source = AppModelLoadedModels(
            model: model, patience: .milliseconds(150),
            seams: .init(chatSession: SessionHolders.slot(over: owner, path: Fx.chat.path)))
        await source.unload(.chat, models: [Fx.chat])
        XCTAssertNotNil(held.value)

        // The request that held it has finished.
        gate.value = true
        await source.unload(.chat, models: [Fx.chat])
        XCTAssertNil(held.value)
        XCTAssertTrue(source.loadedModels().isEmpty)
    }

    func testAnUnloadOfAnotherPathLeavesTheChatSessionAlone() async {
        let model = await makeModel(using: makeState())
        let owner = ProbeHolder()
        _ = holders.hold(on: model, owner: owner)
        let source = AppModelLoadedModels(
            model: model, patience: .milliseconds(100),
            seams: .init(chatSession: SessionHolders.slot(over: owner, path: Fx.chat.path)))

        await source.unload(.chat, models: [Fx.onThisMac])
        XCTAssertNotNil(owner.thing, "a model on another drive is not this session")
        XCTAssertEqual(source.loadedModels().map(\.path), [Fx.chat.path])
    }

    // MARK: - Audio

    private func audioSlot(resident: ProbeHolder, closeGate: EjectBox<Bool>) -> AudioSessionSlot {
        closeGates.append(closeGate)
        return AudioSessionSlot(
            residentPath: { resident.thing == nil ? nil : Fx.audio.path },
            detach: {
                guard let probe = resident.thing else { return nil }
                resident.thing = nil
                return DetachedSession {
                    // A close that waits for the native job to return.
                    while !closeGate.value { await SessionHolders.nap() }
                    withExtendedLifetime(probe) {}
                }
            })
    }

    private func residentAudio() -> ProbeHolder {
        let resident = ProbeHolder()
        resident.thing = SessionProbe()
        return resident
    }

    func testAnAudioSessionThatClosesIsReleasedAndNotRemembered() async {
        let model = await makeModel(using: makeState())
        let resident = residentAudio()
        let source = AppModelLoadedModels(
            model: model, patience: .seconds(3),
            seams: .init(audioSession: audioSlot(resident: resident, closeGate: EjectBox(true))))
        XCTAssertEqual(source.loadedModels().map(\.path), [Fx.audio.path])

        await source.unload(.audio, models: [Fx.audio])
        XCTAssertNil(resident.thing)
        XCTAssertTrue(source.retired.alive().isEmpty)
        XCTAssertTrue(source.loadedModels().isEmpty)
    }

    func testAnAudioCloseThatNeverReturnsIsBoundedAndTheModelStaysListedAsLoaded() async {
        let model = await makeModel(using: makeState())
        let resident = residentAudio()
        let closeGate = EjectBox(false)
        let source = AppModelLoadedModels(
            model: model, patience: .milliseconds(200),
            seams: .init(audioSession: audioSlot(resident: resident, closeGate: closeGate)))

        let returned = EjectBox(false)
        let started = Date()
        let unloading = Task { @MainActor in
            await source.unload(.audio, models: [Fx.audio])
            returned.value = true
        }
        let bounded = await eventuallyAsync(timeout: 2) { returned.value }
        XCTAssertTrue(bounded, "the unload returned although the close did not")
        XCTAssertLessThan(Date().timeIntervalSince(started), 2)
        let listed = source.loadedModels()
        closeGate.value = true  // always let the close end, whatever the assertions say
        await unloading.value

        XCTAssertEqual(listed.map(\.path), [Fx.audio.path], "a close still running is the model still held")
        XCTAssertEqual(listed.map(\.category), [.audio])
        let gone = await eventuallyAsync { source.loadedModels().isEmpty }
        XCTAssertTrue(gone, "and it is no longer listed once the close returns")
    }

    func testAnEjectOfAnAudioModelWhoseCloseHangsDoesNotEjectAndSaysSo() async {
        let model = await makeModel(using: makeState())
        let resident = residentAudio()
        let closeGate = EjectBox(false)
        let log = EjectLog()
        let source = AppModelLoadedModels(
            model: model, patience: .milliseconds(200),
            seams: .init(audioSession: audioSlot(resident: resident, closeGate: closeGate)))
        let ejector = FakeEjector(log: log)
        let notices = NoticeRecorder(log: log)
        let coordinator = LibraryEjectCoordinator(
            source: source, ejector: ejector, locator: FakeVolumeLocator(), present: { notices.add($0) })
        await coordinator.refreshSnapshot()

        XCTAssertNotNil(coordinator.unmountRequested(for: Fx.t7Disk))
        let idle = EjectBox(false)
        let waiting = Task { @MainActor in
            await coordinator.waitForIdle(timeout: 8)
            idle.value = true
        }
        let finished = await eventuallyAsync(timeout: 3) { idle.value }
        closeGate.value = true
        await waiting.value

        XCTAssertTrue(finished, "the eject cycle ended although the close hung")
        XCTAssertTrue(ejector.calls.isEmpty, "the drive is not ejected under a session still closing")
        XCTAssertEqual(notices.all, [.unloadBlocked(models: ["speech"])])
    }

    func testASecondUnloadWaitsForARetiredAudioSession() async {
        let model = await makeModel(using: makeState())
        let resident = residentAudio()
        let closeGate = EjectBox(false)
        let source = AppModelLoadedModels(
            model: model, patience: .milliseconds(150),
            seams: .init(audioSession: audioSlot(resident: resident, closeGate: closeGate)))
        let first = Task { @MainActor in await source.unload(.audio, models: [Fx.audio]) }
        let returned = await returnsWithin(first)
        closeGate.value = true  // always let the close end, whatever the assertions say
        await first.value
        XCTAssertTrue(returned, "the first unload gave up on the close instead of waiting for it")
        XCTAssertEqual(source.retired.alive(.audio).count, 1)

        await source.unload(.audio, models: [Fx.audio])
        let gone = await eventuallyAsync { source.loadedModels().isEmpty }
        XCTAssertTrue(gone)
    }

    /// True when `task` finished within two seconds. A step that waits without
    /// limit shows up as false rather than as a hung suite.
    private func returnsWithin(_ task: Task<Void, Never>) async -> Bool {
        let done = EjectBox(false)
        let watcher = Task { await task.value; done.value = true }
        let finished = await eventuallyAsync(timeout: 2) { done.value }
        if !finished { watcher.cancel() }
        return finished
    }
}
