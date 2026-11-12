import Combine
import Foundation
import XCTest

@testable import TurboSparkApp

/// One stream per consumer, fed by hand (the library state, the store migration
/// and the eject coordinator each drain their own).
private final class OfferStreamFan: @unchecked Sendable {
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

/// Task 6.6 (requirement 4.3): an offer is announced by exactly ONE info toast,
/// once per marker id per process, when it becomes the pending offer.
///
/// The trigger lives in `AppModel` because `LibraryState` has no toast reference
/// by design. Every case reads `activeToast` through a recorder, since a second
/// toast replaces the first.
@MainActor
final class LibraryOfferAnnouncerTests: LibraryStateTestCase {
    private var state: LibraryState!
    private var model: AppModel!
    private var toasts: [AppToast] = []
    private var recorder: AnyCancellable?
    private var baselineError: String?
    private var fans: [OfferStreamFan] = []

    override func tearDownWithError() throws {
        for fan in fans { fan.finish() }
        fans = []
        recorder = nil
        AppJSONStore.clearLastReadError()
        try super.tearDownWithError()
    }

    private func script(_ name: String, id: String, models: Int = 4) {
        let marker = try! LibraryFixtures.marker(id, name: name, modelCount: models)
        service.with { $0.markers["/Volumes/\(name)"] = marker }
    }

    private func mount(_ name: String) async {
        await state.handle(.mounted(LibraryFixtures.volume(name)))
        await state.waitForPendingWork()
    }

    /// A model on `state` whose announcer uses `ledger`, recording every toast.
    private func bootstrap(ledger: LibraryOfferAnnouncementLedger? = nil) async {
        state = makeState()
        await state.refresh()
        model = await makeModel(using: state)
        baselineError = model.error
        toasts = []
        recorder = model.$activeToast.compactMap { $0 }.sink { [unowned self] toast in self.toasts.append(toast) }
        model.startLibraryOfferAnnouncements(ledger: ledger ?? LibraryOfferAnnouncementLedger())
    }

    // MARK: - One info toast per offer

    func testAnOfferIsAnnouncedByOneInfoToastWithItsName() async {
        script("Spare", id: "spare-id")
        await bootstrap()

        await mount("Spare")

        XCTAssertEqual(toasts.count, 1)
        XCTAssertEqual(toasts.first?.style, .info)
        XCTAssertEqual(
            toasts.first?.message,
            LibraryOfferBannerModel.announcement(
                for: state.pendingOffer!, language: AppStrings.currentLanguage))
        XCTAssertTrue(toasts.first?.message.contains("Spare") ?? false)
        XCTAssertEqual(model.error, baselineError, "an offer is never an app error")
    }

    func testTheToastStaysLongEnoughToReadTwoLines() async {
        script("Spare", id: "spare-id")
        await bootstrap()
        await mount("Spare")
        XCTAssertGreaterThanOrEqual(toasts.first?.duration ?? 0, 6)
    }

    func testTheSameOfferIsNotAnnouncedAgainWhenTheDriveIsReplugged() async {
        script("Spare", id: "spare-id")
        await bootstrap()
        await mount("Spare")
        XCTAssertEqual(toasts.count, 1)

        // The drive is pulled and put back while the offer is still pending.
        await state.handle(.didUnmount(LibraryFixtures.volume("Spare")))
        await mount("Spare")
        await mount("Spare")

        XCTAssertEqual(state.pendingOffer?.markerId, "spare-id", "the offer is still the one on screen")
        XCTAssertEqual(toasts.count, 1, "a re-publish of the same offer is not a new offer")
    }

    func testARefreshThatRepublishesTheQueueDoesNotAnnounceAgain() async {
        script("Spare", id: "spare-id")
        await bootstrap()
        await mount("Spare")

        await state.refresh()
        await state.refresh()

        XCTAssertEqual(toasts.count, 1)
    }

    func testASecondOfferIsAnnouncedWhenItBecomesPendingNotWhenItIsQueued() async throws {
        script("Alpha", id: "alpha-id")
        script("Beta", id: "beta-id")
        await bootstrap()

        await mount("Alpha")
        await mount("Beta")
        XCTAssertEqual(state.queuedOffers.map(\.markerId), ["beta-id"])
        XCTAssertEqual(toasts.count, 1, "a queued offer waits for its turn without a toast")
        XCTAssertTrue(toasts[0].message.contains("Alpha"))

        state.dismiss(try XCTUnwrap(state.pendingOffer))

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id")
        XCTAssertEqual(toasts.count, 2, "Beta is announced the moment it becomes the offer on screen")
        XCTAssertTrue(toasts[1].message.contains("Beta"))
        XCTAssertEqual(toasts.map(\.style), [.info, .info])
    }

    func testAQueuedOfferPromotedByAnAddIsAnnouncedToo() async throws {
        script("Alpha", id: "alpha-id")
        script("Beta", id: "beta-id")
        await bootstrap()
        await mount("Alpha")
        await mount("Beta")

        try await state.accept(try XCTUnwrap(state.pendingOffer))

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id")
        XCTAssertEqual(toasts.count, 2)
        XCTAssertTrue(toasts[1].message.contains("Beta"))
    }

    // MARK: - None for nothing pending

    func testNothingPendingPostsNoToast() async {
        await bootstrap()

        await mount("Camera")
        await state.refresh()

        XCTAssertNil(state.pendingOffer)
        XCTAssertEqual(toasts.count, 0)
    }

    func testAnswerRemovingTheLastOfferPostsNoToast() async throws {
        script("Spare", id: "spare-id")
        await bootstrap()
        await mount("Spare")
        XCTAssertEqual(toasts.count, 1)

        state.dismiss(try XCTUnwrap(state.pendingOffer))

        XCTAssertNil(state.pendingOffer)
        XCTAssertEqual(toasts.count, 1)
    }

    func testADismissedOfferIsNeverAnnouncedAgain() async throws {
        script("Spare", id: "spare-id")
        await bootstrap()
        await mount("Spare")
        state.dismiss(try XCTUnwrap(state.pendingOffer))

        await state.handle(.didUnmount(LibraryFixtures.volume("Spare")))
        await mount("Spare")

        XCTAssertNil(state.pendingOffer, "5.7 owns this: a dismissed library is not offered again")
        XCTAssertEqual(toasts.count, 1)
    }

    // MARK: - Once per offer id per process

    func testTheLedgerIsSharedSoANewModelDoesNotAnnounceTheSameIdAgain() async {
        script("Spare", id: "spare-id")
        let ledger = LibraryOfferAnnouncementLedger()
        await bootstrap(ledger: ledger)
        await mount("Spare")
        XCTAssertEqual(toasts.count, 1)

        // A vault lock builds a new model (and state); the same id turns up.
        let secondState = makeState()
        await secondState.refresh()
        let second = await makeModel(using: secondState)
        var secondToasts: [AppToast] = []
        let secondRecorder = second.$activeToast.compactMap { $0 }.sink { secondToasts.append($0) }
        second.startLibraryOfferAnnouncements(ledger: ledger)
        await secondState.handle(.mounted(LibraryFixtures.volume("Spare")))
        await secondState.waitForPendingWork()

        XCTAssertEqual(secondState.pendingOffer?.markerId, "spare-id", "the banner still shows")
        XCTAssertEqual(secondToasts.count, 0, "but this process already announced that id")
        secondRecorder.cancel()
    }

    func testTheLedgerRecordsAnIdOnceAndEachIdIndependently() {
        let ledger = LibraryOfferAnnouncementLedger()
        XCTAssertTrue(ledger.record("a"))
        XCTAssertFalse(ledger.record("a"))
        XCTAssertTrue(ledger.record("b"))
        XCTAssertFalse(ledger.record("b"))
        XCTAssertTrue(LibraryOfferAnnouncementLedger().record("a"), "a new ledger remembers nothing")
        XCTAssertTrue(LibraryOfferAnnouncementLedger.process === LibraryOfferAnnouncementLedger.process)
    }

    // MARK: - An offer already pending when the announcer starts

    func testAnOfferAlreadyPendingWhenTheAnnouncerStartsIsAnnouncedOnce() async {
        script("Spare", id: "spare-id")
        state = makeState()
        await state.refresh()
        await mount("Spare")
        model = await makeModel(using: state)
        toasts = []
        recorder = model.$activeToast.compactMap { $0 }.sink { [unowned self] in self.toasts.append($0) }

        model.startLibraryOfferAnnouncements(ledger: LibraryOfferAnnouncementLedger())
        model.startLibraryOfferAnnouncements(ledger: LibraryOfferAnnouncementLedger())

        XCTAssertEqual(toasts.count, 1, "starting twice must not subscribe twice")
    }

    func testStoppingTheAnnouncerEndsItsToasts() async {
        script("Spare", id: "spare-id")
        await bootstrap()
        model.libraryOfferAnnouncer?.stop()

        await mount("Spare")

        XCTAssertEqual(toasts.count, 0)
    }

    // MARK: - Through startLibraryMonitoring and shutdown

    func testLibraryMonitoringStartsTheAnnouncerAndShutdownEndsIt() async {
        script("Spare", id: "spare-id")
        let fan = OfferStreamFan()
        fans.append(fan)
        state = makeState()
        model = await makeModel(using: state)
        model.stopCronScheduler()
        toasts = []
        recorder = model.$activeToast.compactMap { $0 }.sink { [unowned self] in self.toasts.append($0) }
        let log = EjectLog()

        model.startLibraryMonitoring(
            events: { fan.make() }, approvalHost: { FakeApprovalHost() }, ejector: FakeEjector(log: log),
            volumeLocator: FakeVolumeLocator(), loadedModels: FakeLoadedModels([], log: log),
            offerLedger: LibraryOfferAnnouncementLedger())
        XCTAssertNotNil(model.libraryOfferAnnouncer, "monitoring starts the announcer")
        await state.refresh()

        fan.send(.mounted(LibraryFixtures.volume("Spare")))
        let offered = await eventually { self.state.pendingOffer != nil }
        XCTAssertTrue(offered)
        XCTAssertEqual(toasts.count, 1)
        XCTAssertEqual(toasts.first?.style, .info)

        model.shutdown()
        XCTAssertNil(model.libraryOfferAnnouncer, "shutdown drops it")
        XCTAssertEqual(toasts.count, 1)
    }
}
