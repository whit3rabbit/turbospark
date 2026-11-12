import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 5.8 (requirement 2.2): the default install destination is the default
/// library when it is connected, otherwise This Mac with a flag that says the
/// default is not connected. No UI is involved.
final class LibraryDestinationTests: LibraryStateTestCase {
    private func library(_ id: String, name: String, volume: String? = nil, available: Bool = true, isDefault: Bool = false) -> ModelLibrary {
        LibraryFixtures.external(id, name: name, volume: volume, available: available, isDefault: isDefault)
    }

    private func internalLibrary(isDefault: Bool) -> ModelLibrary {
        ModelLibrary(
            id: "internal", name: "This Mac", root: "/Users/test/.turbospark/models",
            available: true, modelCount: 2, isDefault: isDefault, isInternal: true)
    }

    private func destination(_ libraries: [ModelLibrary]) async -> DefaultInstallDestination {
        service.with { $0.libraries = libraries }
        let state = makeState()
        await state.refresh()
        return state.destinationDefault
    }

    func testAnAvailableDefaultLibraryIsTheDestination() async {
        let t7 = library("t7-id", name: "T7", isDefault: true)
        let result = await destination([internalLibrary(isDefault: false), t7])

        XCTAssertEqual(result.libraryId, "t7-id")
        XCTAssertEqual(result.library, t7)
        XCTAssertFalse(result.fallbackBecauseNotConnected)
        XCTAssertNil(result.notConnectedLibraryName)
    }

    func testAnOfflineDefaultFallsBackToThisMacWithTheFlagAndTheName() async {
        let t7 = library("t7-id", name: "T7", available: false, isDefault: true)
        let result = await destination([internalLibrary(isDefault: false), t7])

        XCTAssertEqual(result.libraryId, "internal")
        XCTAssertEqual(result.library?.isInternal, true)
        XCTAssertTrue(result.fallbackBecauseNotConnected)
        XCTAssertEqual(result.notConnectedLibraryName, "T7", "the message names the drive to connect")
    }

    func testThisMacAsTheDefaultIsNotAFallback() async {
        let result = await destination([internalLibrary(isDefault: true), library("t7-id", name: "T7")])

        XCTAssertEqual(result.libraryId, "internal")
        XCTAssertFalse(result.fallbackBecauseNotConnected)
        XCTAssertNil(result.notConnectedLibraryName)
    }

    func testNoDefaultSetMeansThisMacWithoutTheFlag() async {
        let result = await destination([internalLibrary(isDefault: false), library("t7-id", name: "T7")])

        XCTAssertEqual(result.libraryId, "internal")
        XCTAssertEqual(result.library?.id, "internal")
        XCTAssertFalse(result.fallbackBecauseNotConnected)
    }

    func testBeforeTheFirstRefreshTheDestinationIsThisMacWithoutTheFlag() {
        let state = makeState()
        let result = state.destinationDefault

        XCTAssertEqual(result.libraryId, "internal")
        XCTAssertNil(result.library, "the list has not loaded yet")
        XCTAssertFalse(result.fallbackBecauseNotConnected)
    }

    func testTheDestinationFollowsTheDriveWhenItReconnects() async {
        let offline = library("t7-id", name: "T7", available: false, isDefault: true)
        let online = library("t7-id", name: "T7", isDefault: true)
        service.with { $0.libraries = [self.internalLibrary(isDefault: false), offline] }
        let state = makeState()
        await state.refresh()
        XCTAssertTrue(state.destinationDefault.fallbackBecauseNotConnected)

        service.with { $0.libraries = [self.internalLibrary(isDefault: false), online] }
        await state.handle(.mounted(LibraryFixtures.volume("T7")))
        await fireDebounce(state)

        XCTAssertEqual(state.destinationDefault.libraryId, "t7-id")
        XCTAssertFalse(state.destinationDefault.fallbackBecauseNotConnected)
    }
}
