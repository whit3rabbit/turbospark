import Foundation
import XCTest

@testable import TurboSparkApp

/// Fans one test-controlled event out to every stream the app model asks for
/// (the library state, the store migration and the eject coordinator each
/// drain their own).
private final class StreamFan: @unchecked Sendable {
    private let lock = NSLock()
    private var sinks: [AsyncStream<VolumeEvent>.Continuation] = []

    func make() -> AsyncStream<VolumeEvent> {
        let (stream, sink) = AsyncStream<VolumeEvent>.makeStream()
        lock.lock(); sinks.append(sink); lock.unlock()
        return stream
    }

    func send(_ event: VolumeEvent) {
        lock.lock(); let all = sinks; lock.unlock()
        for sink in all { sink.yield(event) }
    }

    func finish() {
        lock.lock(); let all = sinks; sinks = []; lock.unlock()
        for sink in all { sink.finish() }
    }
}

/// Tasks 5.11 and 5.12 through the app model: the coordinator is created by
/// `startLibraryMonitoring`, registered as the monitor's approval delegate,
/// kept current from the app's own state, and removed by `shutdown()` without
/// the monitor being stopped.
///
/// Everything runs against fakes: a fake monitor host, a fake ejector and (for
/// most cases) a fake list of loaded models. No `AppModel` here registers with
/// the workspace or DiskArbitration.
final class EjectCoordinationWiringTests: LibraryStateTestCase {
    private typealias Fx = EjectFixtures

    private struct Rig {
        let model: AppModel
        let host: FakeApprovalHost
        let ejector: FakeEjector
        let log: EjectLog
        let fan: StreamFan
        let source: FakeLoadedModels?
    }

    private var rigs: [Rig] = []

    override func tearDownWithError() throws {
        for rig in rigs {
            rig.fan.finish()
        }
        rigs = []
        AppJSONStore.clearLastReadError()
        try super.tearDownWithError()
    }

    /// An app model with library monitoring started on fakes. With `models` the
    /// coordinator reads a fake list; without, the live source over the model.
    private func start(
        models: [LoadedModel]? = nil, host: FakeApprovalHost = FakeApprovalHost(),
        state: LibraryState? = nil
    ) async -> Rig {
        let model = await makeModel(using: state ?? makeState())
        model.stopCronScheduler()
        let log = EjectLog()
        let source: FakeLoadedModels? = await MainActor.run { models.map { FakeLoadedModels($0, log: log) } }
        let ejector = FakeEjector(log: log)
        let fan = StreamFan()
        await model.startLibraryMonitoring(
            events: { fan.make() }, approvalHost: { host }, ejector: ejector,
            volumeLocator: FakeVolumeLocator(), loadedModels: source)
        let rig = Rig(model: model, host: host, ejector: ejector, log: log, fan: fan, source: source)
        rigs.append(rig)
        return rig
    }

    @MainActor
    private func coordinator(of rig: Rig) -> LibraryEjectCoordinator? {
        rig.model.ejectCoordination?.coordinator
    }

    // MARK: - Registration

    func testStartingLibraryMonitoringRegistersACoordinatorAsTheApprovalDelegate() async {
        let rig = await start(models: [])
        XCTAssertTrue(rig.host.delegate is LibraryEjectCoordinator)

        // Starting again changes nothing: one model, one coordinator.
        let first = rig.host.delegate
        let other = FakeApprovalHost()
        await rig.model.startLibraryMonitoring(events: { rig.fan.make() }, approvalHost: { other })
        XCTAssertTrue(rig.host.delegate === first)
        XCTAssertNil(other.delegate, "the second call did not register a second coordinator")
        await rig.model.shutdown()
    }

    func testShutdownDeregistersTheCoordinatorWithoutStoppingTheMonitor() async {
        let rig = await start(models: [Fx.chat])
        let coordinator = await self.coordinator(of: rig)
        await coordinator?.refreshSnapshot()
        XCTAssertNotNil(coordinator?.unmountRequested(for: Fx.t7Disk), "it dissents while it is the live one")
        await coordinator?.waitForIdle()

        await rig.model.shutdown()
        XCTAssertNil(rig.host.delegate, "the registration is removed")
        XCTAssertEqual(rig.host.stopCount, 0, "and the monitor is never stopped")
        let removed = await rig.model.ejectCoordination
        XCTAssertNil(removed)
        // A request that still reaches the old coordinator is approved.
        await rig.source?.setModels([Fx.chat])
        XCTAssertNil(coordinator?.unmountRequested(for: Fx.t7Disk), "a locked app never dissents")
    }

    func testAReplacementModelKeepsTheRegistrationWhenTheOldOneShutsDownAfterIt() async {
        let host = FakeApprovalHost()
        let old = await start(models: [Fx.chat], host: host)
        let oldCoordinator = await coordinator(of: old)

        // The vault unlocks: a new model registers before the old one is torn down.
        let new = await start(models: [Fx.chat], host: host)
        let newCoordinator = await coordinator(of: new)
        XCTAssertTrue(host.delegate === newCoordinator)

        await old.model.shutdown()
        XCTAssertTrue(host.delegate === newCoordinator, "the old model must not remove the new registration")
        XCTAssertNil(oldCoordinator?.unmountRequested(for: Fx.t7Disk), "the old coordinator approves")

        await new.model.shutdown()
        XCTAssertNil(host.delegate)
        XCTAssertEqual(host.stopCount, 0)
    }

    func testAModelThatShutDownAndStartsAgainRegistersAFreshCoordinator() async {
        let rig = await start(models: [])
        let first = rig.host.delegate
        await rig.model.shutdown()
        XCTAssertNil(rig.host.delegate)

        await rig.model.startLibraryMonitoring(
            events: { rig.fan.make() }, approvalHost: { rig.host }, ejector: rig.ejector,
            volumeLocator: FakeVolumeLocator(), loadedModels: nil)
        XCTAssertNotNil(rig.host.delegate)
        XCTAssertFalse(rig.host.delegate === first, "a new coordinator, not the stopped one")
        await rig.model.shutdown()
    }

    // MARK: - Through the app model

    func testAnEjectThroughTheAppModelUnloadsEjectsAndTellsTheUser() async {
        let rig = await start(models: [Fx.chat, Fx.server])
        let coordinator = await self.coordinator(of: rig)
        await coordinator?.refreshSnapshot()

        XCTAssertNotNil(rig.host.delegate?.unmountRequested(for: Fx.t7Disk))
        await coordinator?.waitForIdle()

        XCTAssertEqual(rig.ejector.calls, [Fx.t7Disk])
        let expected = EjectNoticeText.message(for: .ejected(models: ["qwen", "gemma"]))
        let shown = await rig.model.activeToast
        XCTAssertEqual(shown?.message, expected)
        XCTAssertEqual(shown?.style, .success)
        await rig.model.shutdown()
    }

    func testASurpriseRemovalShowsTheAdviseMessageAndLeavesEveryRecordAlone() async {
        let state = makeState()
        let rig = await start(models: [Fx.chat], state: state)
        await state.refresh()
        let librariesBefore = await state.libraries
        let rowsBefore = await state.allInstalled
        let installedBefore = await rig.model.installed
        let writesBefore = service.writeCalls

        // The drive is pulled: the service now lists it as not connected, which
        // is what the next refresh sees. Both consumers get the same event.
        service.with {
            $0.libraries = [
                LibraryFixtures.internalLibrary,
                LibraryFixtures.external("t7-id", name: "T7", volume: "T7", available: false),
            ]
            $0.installed = [
                LibraryFixtures.row("local"),
                LibraryFixtures.row("on-t7", library: self.t7, availability: .notConnected),
            ]
        }
        rig.fan.send(.didUnmount(Fx.t7Volume))
        let coordinator = await self.coordinator(of: rig)
        let handled = await eventually { coordinator?.handledEventCount == 1 }
        XCTAssertTrue(handled)
        await fireDebounce(state)

        let shown = await rig.model.activeToast
        XCTAssertEqual(
            shown?.message, EjectNoticeText.message(for: .removedWithoutEject(models: ["qwen"])))
        XCTAssertEqual(shown?.style, .warning)
        XCTAssertTrue(rig.ejector.calls.isEmpty, "there is nothing to eject")
        let sources = await rig.source?.models
        XCTAssertEqual(sources, [], "the session was torn down")

        // Requirement 6.4: the library list and every model record are intact.
        let librariesAfter = await state.libraries
        let rowsAfter = await state.allInstalled
        let installedAfter = await rig.model.installed
        XCTAssertEqual(librariesAfter.map(\.id), librariesBefore.map(\.id), "the library is still listed")
        XCTAssertEqual(librariesAfter.first { $0.id == "t7-id" }?.available, false, "and marked not connected")
        XCTAssertEqual(rowsAfter.map(\.installId), rowsBefore.map(\.installId), "no model row was dropped")
        XCTAssertEqual(rowsAfter.first { $0.alias == "on-t7" }?.availability, .notConnected)
        XCTAssertEqual(installedAfter, installedBefore, "the picker list was not touched")
        XCTAssertEqual(service.writeCalls, writesBefore, "no registry write of any kind")
        XCTAssertEqual(service.writeCalls, [])
        await rig.model.shutdown()
    }

    // MARK: - The live source over the app model

    func testTheSnapshotFollowsTheServersAudioModelsAndTheLiveSourceReleasesThem() async {
        let rig = await start()
        let path = "/Volumes/T7/TurboSpark/audio/speech.gturbo"
        await MainActor.run { rig.model.serverAudioModelPaths = ["speech.gturbo": path] }

        let coordinator = await self.coordinator(of: rig)
        let learned = await eventuallyAsync { coordinator?.decisions.snapshot.map(\.model.path) == [path] }
        XCTAssertTrue(learned, "a change to the served audio models refreshes the snapshot by itself")
        XCTAssertEqual(coordinator?.decisions.snapshot.first?.model.name, "speech")
        XCTAssertEqual(coordinator?.decisions.snapshot.first?.model.category, .server)

        XCTAssertNotNil(rig.host.delegate?.unmountRequested(for: Fx.t7Disk))
        await coordinator?.waitForIdle()

        let remaining = await rig.model.serverAudioModelPaths
        XCTAssertTrue(remaining.isEmpty, "the live unload path released the served model")
        XCTAssertEqual(rig.ejector.calls, [Fx.t7Disk])
        let shown = await rig.model.activeToast
        XCTAssertEqual(shown?.message, EjectNoticeText.message(for: .ejected(models: ["speech"])))
        await rig.model.shutdown()
    }

    func testAModelReleasedElsewhereStopsDissentingOnceTheSnapshotFollows() async {
        let rig = await start()
        let path = "/Volumes/T7/TurboSpark/audio/speech.gturbo"
        await MainActor.run { rig.model.serverAudioModelPaths = ["speech.gturbo": path] }
        let coordinator = await self.coordinator(of: rig)
        let learned = await eventuallyAsync { coordinator?.decisions.snapshot.count == 1 }
        XCTAssertTrue(learned)

        // The user detaches it in the Server pane.
        await MainActor.run { rig.model.serverAudioModelPaths = [:] }
        let forgot = await eventuallyAsync { coordinator?.decisions.snapshot.isEmpty == true }
        XCTAssertTrue(forgot, "the snapshot did not follow the change")
        XCTAssertNil(rig.host.delegate?.unmountRequested(for: Fx.t7Disk), "an unused drive ejects as ever")
        await rig.model.shutdown()
    }

    func testTheLiveSourceSeesNothingOnAFreshModel() async {
        let rig = await start()
        let coordinator = await self.coordinator(of: rig)
        await coordinator?.refreshSnapshot()
        XCTAssertEqual(coordinator?.decisions.snapshot.count, 0)
        XCTAssertNil(rig.host.delegate?.unmountRequested(for: Fx.t7Disk))
        await rig.model.shutdown()
    }

    // MARK: - The inventory

    func testTheInventoryListsChatThenServerByIdThenAudioAndSkipsTheServedChatSession() {
        let models = LoadedModelInventory.make(
            chat: "/Volumes/T7/TurboSpark/text/qwen.gturbo",
            serverSessions: [
                .init(id: "b.gturbo", path: "/Volumes/T7/TurboSpark/text/b.gturbo", isChatSession: false),
                .init(id: "qwen.gturbo", path: "/Volumes/T7/TurboSpark/text/qwen.gturbo", isChatSession: true),
                .init(id: "a.gturbo", path: "/Volumes/T7/TurboSpark/text/a.gturbo", isChatSession: false),
            ],
            serverAudio: [(id: "tts", path: "/Volumes/T7/TurboSpark/audio/tts.gturbo")],
            audio: "/Volumes/T7/TurboSpark/audio/stt.gturbo",
            name: LoadedModelInventory.fallbackName(forPath:))
        XCTAssertEqual(models.map(\.name), ["qwen", "a", "b", "tts", "stt"])
        XCTAssertEqual(models.map(\.category), [.chat, .server, .server, .server, .audio])
    }

    func testTheFallbackNameDropsTheModelExtension() {
        XCTAssertEqual(LoadedModelInventory.fallbackName(forPath: "/Volumes/T7/x/gemma4.gturbo"), "gemma4")
        XCTAssertEqual(LoadedModelInventory.fallbackName(forPath: "/Volumes/T7/x/plain-folder"), "plain-folder")
    }
}

@MainActor
extension FakeLoadedModels {
    func setModels(_ new: [LoadedModel]) { models = new }
}
