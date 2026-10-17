import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 5.7 (requirements 4.3, 4.4): one offer per library found on a newly
/// mounted drive, kept until it is added or dismissed, and never repeated
/// after a dismissal.
///
/// The offer is published state only. The toast the design mentions needs a
/// new string, which belongs to the UI task (6.6), so none is posted here.
final class LibraryOfferTests: LibraryStateTestCase {
    private func volume(_ name: String) -> VolumeDescriptor { LibraryFixtures.volume(name) }

    private func script(_ root: String, id: String, name: String = "Spare", models: Int = 4, readOnly: Bool = false) {
        let marker = try! LibraryFixtures.marker(id, name: name, modelCount: models, readOnly: readOnly)
        service.with { $0.markers[root] = marker }
    }

    /// Mounts `name` and waits for the discovery it starts.
    private func mount(_ name: String, on state: LibraryState) async {
        await state.handle(.mounted(volume(name)))
        await state.waitForPendingWork()
    }

    private func offer(_ id: String, name: String = "Spare", root: String, models: Int = 4) -> LibraryOffer {
        LibraryOffer(volumeRoot: root, markerId: id, name: name, modelCount: models, readOnly: false)
    }

    private func persistedDismissals() throws -> [String] {
        try JSONDecoder().decode([String].self, from: Data(contentsOf: dismissalsURL))
    }

    // MARK: - Publishing

    func testAnUnknownLibraryOnAMountedDriveProducesOneOffer() async {
        script("/Volumes/Spare", id: "spare-id", name: "Spare", models: 4)
        let state = makeState()
        await state.refresh()

        await mount("Spare", on: state)

        XCTAssertEqual(state.pendingOffer, offer("spare-id", root: "/Volumes/Spare", models: 4))
        XCTAssertEqual(service.discoverRoots, ["/Volumes/Spare"])
        XCTAssertEqual(service.mainThreadCalls, [], "discovery reads a drive: never on the main thread")
    }

    func testTheOfferCarriesTheReadOnlyFlag() async {
        script("/Volumes/Spare", id: "spare-id", readOnly: true)
        let state = makeState()
        await state.refresh()
        await mount("Spare", on: state)
        XCTAssertEqual(state.pendingOffer?.readOnly, true)
    }

    func testAMountedDriveWithNoLibraryProducesNoOffer() async {
        let state = makeState()
        await state.refresh()
        await mount("Camera", on: state)
        XCTAssertNil(state.pendingOffer)
        XCTAssertEqual(service.discoverRoots, ["/Volumes/Camera"])
    }

    func testAKnownLibraryUnderANewVolumeNameProducesNoOffer() async {
        // T7 comes back as "T7 1": its marker id is already listed.
        script("/Volumes/T7 1", id: "t7-id", name: "T7")
        let state = makeState()
        await state.refresh()

        await mount("T7 1", on: state)

        XCTAssertNil(state.pendingOffer)
        XCTAssertEqual(service.discoverRoots, ["/Volumes/T7 1"], "an unlisted volume is looked at")
    }

    func testTheVolumeOfAnAvailableLibraryIsNotEvenDiscovered() async {
        script("/Volumes/T7", id: "t7-id", name: "T7")
        let state = makeState()
        await state.refresh()

        await mount("T7", on: state)

        XCTAssertNil(state.pendingOffer)
        XCTAssertTrue(service.discoverRoots.isEmpty, "a listed library's volume needs no marker read")
    }

    func testANetworkVolumeIsNotDiscovered() async {
        script("/Volumes/Share", id: "share-id")
        probe.setProvider { url in
            DriveConnection(kind: .other("Network"), bitsPerSecond: nil, volumeUUID: nil, isLocal: false)
        }
        let state = makeState()
        await state.refresh()

        await mount("Share", on: state)

        XCTAssertNil(state.pendingOffer)
        XCTAssertTrue(service.discoverRoots.isEmpty)
    }

    func testADiscoverErrorProducesNoOfferAndNoAppError() async {
        service.with { $0.discoverError = FakeServiceError.scripted("marker unreadable") }
        let state = makeState()
        let model = await makeModel(using: state)
        let errorBefore = model.error
        let toastBefore = model.activeToast
        await state.refresh()

        await mount("Spare", on: state)

        XCTAssertNil(state.pendingOffer)
        XCTAssertEqual(service.discoverRoots, ["/Volumes/Spare"])
        XCTAssertEqual(model.error, errorBefore)
        XCTAssertEqual(model.activeToast, toastBefore)
    }

    func testNothingIsOfferedWhileTheListedLibrariesAreUnknown() async {
        // Without the list, a library the user already added could not be told
        // from a new one.
        script("/Volumes/Spare", id: "spare-id")
        service.with { $0.listError = FakeServiceError.scripted("registry unreadable") }
        let state = makeState()

        await mount("Spare", on: state)

        XCTAssertNil(state.pendingOffer)
    }

    func testTheSameLibraryMountedTwiceIsOfferedOnce() async {
        script("/Volumes/Spare", id: "spare-id")
        let state = makeState()
        await state.refresh()

        await mount("Spare", on: state)
        await mount("Spare", on: state)

        XCTAssertEqual(state.pendingOffer?.markerId, "spare-id")
        XCTAssertTrue(state.queuedOffers.isEmpty, "the same marker does not queue behind itself")
    }

    // MARK: - Two unknown libraries

    func testASecondUnknownLibraryQueuesBehindTheFirstAndTakesItsPlaceOnDismiss() async throws {
        script("/Volumes/Alpha", id: "alpha-id", name: "Alpha")
        script("/Volumes/Beta", id: "beta-id", name: "Beta")
        let state = makeState()
        await state.refresh()

        await mount("Alpha", on: state)
        await mount("Beta", on: state)

        XCTAssertEqual(state.pendingOffer?.markerId, "alpha-id", "the offer on screen is never replaced")
        XCTAssertEqual(state.queuedOffers.map(\.markerId), ["beta-id"])

        state.dismiss(state.pendingOffer!)

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id")
        XCTAssertTrue(state.queuedOffers.isEmpty)
        XCTAssertEqual(try persistedDismissals(), ["alpha-id"])
    }

    func testAcceptingTheFirstOfferPromotesTheQueuedOne() async throws {
        script("/Volumes/Alpha", id: "alpha-id", name: "Alpha")
        script("/Volumes/Beta", id: "beta-id", name: "Beta")
        let state = makeState()
        await state.refresh()
        await mount("Alpha", on: state)
        await mount("Beta", on: state)

        try await state.accept(state.pendingOffer!)

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id")
        XCTAssertTrue(state.queuedOffers.isEmpty)
    }

    // MARK: - Dismissal

    func testDismissClearsTheOfferAndPersistsTheMarkerId() async throws {
        script("/Volumes/Spare", id: "spare-id")
        let state = makeState()
        await state.refresh()
        await mount("Spare", on: state)
        let shown = try XCTUnwrap(state.pendingOffer)

        state.dismiss(shown)

        XCTAssertNil(state.pendingOffer)
        XCTAssertEqual(try persistedDismissals(), ["spare-id"])
    }

    func testADismissedLibraryIsNotOfferedAgainAfterAReload() async throws {
        script("/Volumes/Spare", id: "spare-id")
        let first = makeState()
        await first.refresh()
        await mount("Spare", on: first)
        first.dismiss(try XCTUnwrap(first.pendingOffer))

        // A new process: nothing carried over but the file.
        let reloaded = makeState()
        await reloaded.refresh()
        await mount("Spare", on: reloaded)

        XCTAssertNil(reloaded.pendingOffer)
        XCTAssertEqual(service.discoverRoots.count, 2, "the drive was looked at again, and still not offered")
    }

    func testADismissedLibraryIsNotOfferedAgainInTheSameRun() async throws {
        script("/Volumes/Spare", id: "spare-id")
        let state = makeState()
        await state.refresh()
        await mount("Spare", on: state)
        state.dismiss(try XCTUnwrap(state.pendingOffer))

        await mount("Spare", on: state)

        XCTAssertNil(state.pendingOffer)
    }

    func testDismissingOneLibraryDoesNotSilenceAnother() async throws {
        script("/Volumes/Alpha", id: "alpha-id", name: "Alpha")
        script("/Volumes/Beta", id: "beta-id", name: "Beta")
        let state = makeState()
        await state.refresh()
        await mount("Alpha", on: state)
        state.dismiss(try XCTUnwrap(state.pendingOffer))

        await mount("Beta", on: state)

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id")
    }

    func testACorruptDismissalsFileIsSetAsideAndOffersStillWork() async throws {
        try Data("{ not json".utf8).write(to: dismissalsURL)
        script("/Volumes/Spare", id: "spare-id")
        let state = makeState()
        await state.refresh()

        await mount("Spare", on: state)
        XCTAssertEqual(state.pendingOffer?.markerId, "spare-id")
        state.dismiss(try XCTUnwrap(state.pendingOffer))

        let names = try FileManager.default.contentsOfDirectory(atPath: directory.path)
        XCTAssertTrue(
            names.contains { $0.contains("corrupt") },
            "AppJSONStore quarantines the unreadable file rather than overwriting it: \(names)")
        XCTAssertEqual(try persistedDismissals(), ["spare-id"])
        AppJSONStore.clearLastReadError()
    }

    // MARK: - Accepting

    func testAcceptAddsTheLibraryRefreshesAndClearsTheOffer() async throws {
        script("/Volumes/Spare", id: "spare-id", name: "Spare")
        let state = makeState()
        await state.refresh()
        await mount("Spare", on: state)
        let shown = try XCTUnwrap(state.pendingOffer)
        XCTAssertFalse(state.libraries.contains { $0.id == "spare-id" })

        try await state.accept(shown)

        XCTAssertEqual(service.addCalls, ["/Volumes/Spare"])
        XCTAssertNil(state.pendingOffer)
        XCTAssertTrue(state.libraries.contains { $0.id == "spare-id" }, "the list was refreshed")
        XCTAssertFalse(
            FileManager.default.fileExists(atPath: dismissalsURL.path), "adding is not dismissing")
    }

    func testAFailedAddKeepsTheOfferAndTellsTheCaller() async throws {
        script("/Volumes/Spare", id: "spare-id")
        let state = makeState()
        await state.refresh()
        await mount("Spare", on: state)
        let shown = try XCTUnwrap(state.pendingOffer)
        service.with { $0.addError = FakeServiceError.scripted("nested") }

        do {
            try await state.accept(shown)
            XCTFail("accept must surface the engine's refusal")
        } catch {
            XCTAssertEqual(error as? FakeServiceError, .scripted("nested"))
        }

        XCTAssertEqual(state.pendingOffer, shown, "the user can try again or dismiss")
    }

    func testAnOfferIsDroppedWhenTheLibraryIsListedByOtherMeans() async {
        script("/Volumes/Alpha", id: "alpha-id", name: "Alpha")
        script("/Volumes/Beta", id: "beta-id", name: "Beta")
        let state = makeState()
        await state.refresh()
        await mount("Alpha", on: state)
        await mount("Beta", on: state)

        // The user added Alpha from settings while its offer was up.
        service.with {
            $0.libraries.append(LibraryFixtures.external("alpha-id", name: "Alpha"))
        }
        await state.refresh()

        XCTAssertEqual(state.pendingOffer?.markerId, "beta-id")
        XCTAssertTrue(state.queuedOffers.isEmpty)
    }
}
