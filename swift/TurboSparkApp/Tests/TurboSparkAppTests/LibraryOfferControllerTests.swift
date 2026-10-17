import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.6 (requirements 4.3 and 4.4): what the two buttons of the add-library
/// offer do, against the fake engine. Add runs off the main thread and shows a
/// refusal inline while keeping the offer; Not now persists the dismissal; both
/// are disabled, not hidden, while an install runs or the add is in flight; and
/// nothing here writes the app error or posts a toast.
@MainActor
final class LibraryOfferControllerTests: LibraryStateTestCase {
    private var state: LibraryState!
    private var model: AppModel!
    private var controller: LibraryOfferController!
    private var announcements: [String] = []
    private var baselineError: String?
    private var baselineToast: AppToast?

    private func volume(_ name: String) -> VolumeDescriptor { LibraryFixtures.volume(name) }

    private func script(_ name: String, id: String, models: Int = 4, readOnly: Bool = false) {
        let marker = try! LibraryFixtures.marker(id, name: name, modelCount: models, readOnly: readOnly)
        service.with { $0.markers["/Volumes/\(name)"] = marker }
    }

    private func mount(_ name: String) async {
        await state.handle(.mounted(volume(name)))
        await state.waitForPendingWork()
    }

    /// A state with one pending offer for "Spare" and a controller on it.
    private func bootstrap(offers names: [String] = ["Spare"]) async {
        for name in names { script(name, id: "\(name.lowercased())-id") }
        state = makeState()
        await state.refresh()
        for name in names { await mount(name) }
        model = await makeModel(using: state)
        baselineError = model.error
        baselineToast = model.activeToast
        announcements = []
        let model = self.model!
        controller = LibraryOfferController(
            libraryState: state,
            isBusy: { model.isInstallOrMoveRunning },
            announce: { [unowned self] text in self.announcements.append(text) },
            language: { .english })
    }

    private func pending() throws -> LibraryOffer { try XCTUnwrap(state.pendingOffer) }

    private func assertNoAppErrorOrToast(_ message: String = "", file: StaticString = #filePath, line: UInt = #line) {
        XCTAssertEqual(model.error, baselineError, "the app error moved. \(message)", file: file, line: line)
        XCTAssertEqual(model.activeToast, baselineToast, "a toast was posted. \(message)", file: file, line: line)
    }

    // MARK: - Add

    func testAddCallsAcceptOffTheMainThreadAndTheLibraryAppears() async throws {
        await bootstrap()
        let offer = try pending()

        await controller.add(offer)

        XCTAssertEqual(service.addCalls, ["/Volumes/Spare"], "accept asked the engine to add the volume")
        XCTAssertEqual(service.addOnMainThread, [false], "the engine call blocks: never on the main thread")
        XCTAssertEqual(service.mainThreadCalls, [])
        XCTAssertTrue(state.libraries.contains { $0.id == "spare-id" }, "the library was added and listed")
        XCTAssertNil(state.pendingOffer, "the offer left")
        XCTAssertNil(controller.failure)
        XCTAssertFalse(controller.isAdding)
        assertNoAppErrorOrToast()
    }

    func testAddingAnnouncesTheOutcomeToVoiceOverWithoutAToast() async throws {
        await bootstrap()
        await controller.add(try pending())
        XCTAssertEqual(announcements, ["Added the library Spare."])
        assertNoAppErrorOrToast()
    }

    func testTheNextQueuedOfferTakesItsPlaceWhenTheFirstIsAdded() async throws {
        await bootstrap(offers: ["Alpha", "Beta"])
        XCTAssertEqual(state.pendingOffer?.markerId, "alpha-id")
        XCTAssertEqual(state.queuedOffers.map(\.markerId), ["beta-id"])

        await controller.add(try pending())

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id")
        XCTAssertTrue(state.queuedOffers.isEmpty)
        XCTAssertEqual(service.addCalls, ["/Volumes/Alpha"])
    }

    // MARK: - A refusal stays inline and keeps the offer

    func testARefusalShowsOneSentenceInlineAndKeepsTheOffer() async throws {
        let cases: [(code: String, status: TurboSparkError.Code, sentence: String)] = [
            ("read_only", .invalidArgument, "That drive is read-only. A library needs a drive TurboSpark can write to."),
            ("io", .invalidArgument, "TurboSpark could not use that location because of a file system error."),
            ("busy", .busy, "A model is installing or moving. Try again when it finishes."),
        ]
        for refusal in cases {
            await bootstrap()
            let shown = try pending()
            service.with {
                $0.addError = TurboSparkError(code: refusal.status, message: "\(refusal.code): the engine's own words")
            }

            await controller.add(shown)

            XCTAssertEqual(controller.failure, refusal.sentence, refusal.code)
            XCTAssertEqual(state.pendingOffer, shown, "\(refusal.code): the offer stays so the user can try again")
            XCTAssertFalse(controller.isAdding, refusal.code)
            XCTAssertEqual(announcements, [refusal.sentence], "\(refusal.code): failure is announced")
            XCTAssertFalse(
                state.libraries.contains { $0.id == "spare-id" }, "\(refusal.code): nothing was added")
            assertNoAppErrorOrToast(refusal.code)
            // The next case starts from a fresh state and a clean engine.
            service.with { $0.addError = nil }
        }
    }

    func testTheInlineSentenceNeverShowsTheEnginesOwnWordsOrAnUnknownError() async throws {
        await bootstrap()
        service.with { $0.addError = FakeServiceError.scripted("secret path /Users/x") }
        await controller.add(try pending())
        XCTAssertEqual(controller.failure, "TurboSpark could not change the library list.")
        XCTAssertFalse(controller.failure?.contains("secret") ?? true)
    }

    func testARetryAfterARefusalClearsTheSentenceAndAdds() async throws {
        await bootstrap()
        let shown = try pending()
        service.with {
            $0.addError = TurboSparkError(code: .invalidArgument, message: "read_only: x")
        }
        await controller.add(shown)
        XCTAssertNotNil(controller.failure)

        service.with { $0.addError = nil }
        await controller.add(shown)

        XCTAssertNil(controller.failure)
        XCTAssertNil(state.pendingOffer)
        XCTAssertEqual(service.addCalls, ["/Volumes/Spare", "/Volumes/Spare"])
    }

    func testNotNowClearsAnEarlierSentenceAndDismisses() async throws {
        await bootstrap()
        let shown = try pending()
        service.with { $0.addError = TurboSparkError(code: .invalidArgument, message: "io: x") }
        await controller.add(shown)
        XCTAssertNotNil(controller.failure)

        controller.notNow(shown)

        XCTAssertNil(controller.failure)
        XCTAssertNil(state.pendingOffer)
    }

    // MARK: - Busy and in flight

    func testChangesAreBlockedWhileAnInstallOrMoveRuns() async throws {
        await bootstrap()
        XCTAssertFalse(controller.isChangeBlocked)

        model.isInstallingModel = true
        XCTAssertTrue(controller.isChangeBlocked)
        model.isInstallingModel = false
        XCTAssertFalse(controller.isChangeBlocked)

        model.isInstallingImageModel = true
        XCTAssertTrue(controller.isChangeBlocked)
        model.isInstallingImageModel = false
        XCTAssertFalse(controller.isChangeBlocked)
    }

    func testAddIsRefusedInTheBusySentenceWhileAnInstallRunsAndTheEngineIsNotAsked() async throws {
        await bootstrap()
        let shown = try pending()
        model.isInstallingModel = true

        await controller.add(shown)

        XCTAssertEqual(controller.failure, "A model is installing or moving. Try again when it finishes.")
        XCTAssertEqual(service.addCalls, [], "the engine would refuse; the app does not ask")
        XCTAssertEqual(state.pendingOffer, shown)
        assertNoAppErrorOrToast()
    }

    func testNotNowDoesNothingWhileAnInstallRunsBecauseItsButtonIsDisabled() async throws {
        await bootstrap()
        let shown = try pending()
        model.isInstallingModel = true

        controller.notNow(shown)

        XCTAssertEqual(state.pendingOffer, shown)
        XCTAssertFalse(FileManager.default.fileExists(atPath: dismissalsURL.path))
    }

    func testWhileTheAddIsInFlightBothButtonsAreBlockedAndASecondAddIsIgnored() async throws {
        await bootstrap()
        let shown = try pending()
        let gate = DispatchSemaphore(value: 0)
        service.with { $0.addGate = gate }

        let first = Task { @MainActor in await self.controller.add(shown) }
        let entered = await eventually { self.service.addCalls.count == 1 }
        XCTAssertTrue(entered, "the add never reached the engine")

        XCTAssertTrue(controller.isAdding)
        XCTAssertTrue(controller.isChangeBlocked)
        await controller.add(shown)
        controller.notNow(shown)
        XCTAssertEqual(service.addCalls.count, 1, "a second add during the first is ignored")
        XCTAssertEqual(state.pendingOffer, shown, "Not now during the add changed nothing")
        XCTAssertFalse(FileManager.default.fileExists(atPath: dismissalsURL.path))

        gate.signal()
        await first.value

        XCTAssertFalse(controller.isAdding)
        XCTAssertNil(state.pendingOffer)
        XCTAssertEqual(service.addCalls.count, 1)
    }

    func testAnOfferAnsweredInTheOtherPlacementIsNotAddedOrDismissedAgain() async throws {
        await bootstrap()
        let shown = try pending()
        state.dismiss(shown)
        let dismissalsBefore = try Data(contentsOf: dismissalsURL)

        await controller.add(shown)
        controller.notNow(shown)

        XCTAssertEqual(service.addCalls, [], "the other placement already answered it")
        XCTAssertNil(controller.failure)
        XCTAssertEqual(try Data(contentsOf: dismissalsURL), dismissalsBefore)
    }

    // MARK: - Not now

    func testNotNowDismissesPersistsTheMarkerIdAndTheNextOfferAppears() async throws {
        await bootstrap(offers: ["Alpha", "Beta"])
        let shown = try pending()

        controller.notNow(shown)

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id", "the next queued offer takes its place")
        XCTAssertEqual(
            try JSONDecoder().decode([String].self, from: Data(contentsOf: dismissalsURL)), ["alpha-id"])
        XCTAssertEqual(service.addCalls, [], "Not now never adds")
        assertNoAppErrorOrToast()
    }

    func testADismissedLibraryIsNotOfferedAgainAfterTheStoreIsReloaded() async throws {
        await bootstrap()
        controller.notNow(try pending())
        XCTAssertNil(state.pendingOffer)

        // A new process: nothing carried over but the file.
        let reloaded = makeState()
        await reloaded.refresh()
        await reloaded.handle(.mounted(volume("Spare")))
        await reloaded.waitForPendingWork()

        XCTAssertNil(reloaded.pendingOffer, "a dismissed library is never offered again")
        XCTAssertEqual(service.discoverRoots.count, 2, "the drive was looked at again and still not offered")
    }

    func testAddingDoesNotRecordADismissal() async throws {
        await bootstrap()
        await controller.add(try pending())
        XCTAssertFalse(FileManager.default.fileExists(atPath: dismissalsURL.path))
    }
}
