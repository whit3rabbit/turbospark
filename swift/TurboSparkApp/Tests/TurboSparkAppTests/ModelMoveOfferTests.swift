import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 6.11: which rows offer "Move to...", and the pure favorite rewrite a
/// move uses.
final class ModelMoveOfferTests: XCTestCase {
    private let t7 = LibraryFixtures.external("t7-id", name: "T7", volume: "T7")

    private func plan(
        _ model: InstalledModel, library: ModelLibrary?, loaded: Bool = false
    ) -> InstalledModelActionPlan {
        let row = InstalledModelRowModel(model: model, library: library, connection: nil, measuredMBps: nil)
        return InstalledModelActionPlan(rowModel: row, isCurrentlyLoaded: loaded)
    }

    private func engineRow(
        library: ModelLibrary?, availability: LibraryAvailability = .available
    ) -> InstalledModel {
        let id = library?.id ?? "internal"
        let isInternal = library?.isInternal ?? true
        return InstalledModel(
            alias: "qwen", repo: "o/qwen", path: "/p/qwen.gturbo", family: "qwen", installBytes: 4,
            library: id, libraryName: library?.name ?? "This Mac", installId: "\(id):text/qwen",
            availability: availability, locationKind: isInternal ? .thisMac : .library)
    }

    // MARK: The action bar

    func testARowOnThisMacOrOnAConnectedDriveOffersMove() {
        XCTAssertTrue(plan(engineRow(library: nil), library: LibraryFixtures.internalLibrary).offersMove)
        XCTAssertTrue(plan(engineRow(library: t7), library: t7).offersMove)
    }

    func testARowThatCannotBeMovedOffersNoMove() {
        // Another app's folder: no registry install to move.
        let scanned = InstalledModel(alias: "lm", repo: "o/lm", path: "/lm/lm.gguf", family: "qwen", installBytes: 1)
        XCTAssertFalse(plan(scanned, library: nil).offersMove)
        let custom = InstalledModel(
            alias: "c", repo: "o/c", path: "/c.gturbo", family: "qwen", installBytes: 1, library: nil,
            libraryName: "custom location", locationKind: .custom)
        XCTAssertFalse(plan(custom, library: nil).offersMove)
        let unknown = InstalledModel(
            alias: "u", repo: "o/u", path: "/u.gturbo", family: "qwen", installBytes: 1, library: "gone",
            libraryName: "unknown library", installId: "gone:text/u", locationKind: .unknown)
        XCTAssertFalse(plan(unknown, library: nil).offersMove)
        XCTAssertFalse(plan(engineRow(library: t7, availability: .notConnected), library: t7).offersMove)
        XCTAssertFalse(plan(engineRow(library: t7, availability: .missing), library: t7).offersMove)
    }

    func testALoadedRowStillOffersMoveAndTheSheetGivesTheReason() {
        XCTAssertTrue(plan(engineRow(library: nil), library: LibraryFixtures.internalLibrary, loaded: true).offersMove)
    }

    func testTheActionPlanAgreesWithTheSubjectBuilder() {
        let rows: [InstalledModel] = [
            engineRow(library: nil), engineRow(library: t7),
            engineRow(library: t7, availability: .notConnected), engineRow(library: t7, availability: .missing),
            InstalledModel(alias: "lm", repo: "o/lm", path: "/lm", family: "qwen", installBytes: 1),
        ]
        for row in rows {
            let library = row.library == "internal" ? LibraryFixtures.internalLibrary : (row.library == "t7-id" ? t7 : nil)
            XCTAssertEqual(
                plan(row, library: library).offersMove, MoveSubject.make(text: row) != nil,
                "\(row.installId) \(row.availability)")
        }
    }

    // MARK: Server favorites

    private func favorite(paths: [String], ids: [String] = []) -> ServerFavorite {
        ServerFavorite(name: "f", host: "h", port: 1, modelPaths: paths, modelInstallIds: ids, contextTokens: 1)
    }

    func testAFavoriteThatStoredTheOldIdGetsTheNewOne() {
        let moved = favorite(paths: ["/a", "/old"], ids: ["internal:text/other", "internal:text/qwen"])
            .followingMove(from: "internal:text/qwen", oldPath: "/old", to: "t7-id:text/qwen")
        XCTAssertEqual(moved?.modelInstallIds, ["internal:text/other", "t7-id:text/qwen"])
        XCTAssertEqual(moved?.modelPaths, ["/a", "/old"], "paths are never rewritten (the older app's key)")
    }

    func testAPathOnlyEntryForTheModelGainsTheIdInItsOwnSlot() {
        let moved = favorite(paths: ["/a", "/old", "/b"], ids: [])
            .followingMove(from: "internal:text/qwen", oldPath: "/old", to: "t7-id:text/qwen")
        XCTAssertEqual(moved?.modelInstallIds, ["", "t7-id:text/qwen"], "aligned by index, trailing blanks dropped")
    }

    func testAFavoriteOfOtherModelsIsLeftAlone() {
        let other = favorite(paths: ["/a"], ids: ["internal:text/other"])
        XCTAssertNil(other.followingMove(from: "internal:text/qwen", oldPath: "/old", to: "t7-id:text/qwen"))
        let alreadyMoved = favorite(paths: ["/old"], ids: ["t7-id:text/qwen"])
        XCTAssertNil(
            alreadyMoved.followingMove(from: "internal:text/qwen", oldPath: "/old", to: "t7-id:text/qwen"),
            "an entry that already has an id of its own is not claimed by a path")
        XCTAssertNil(favorite(paths: []).followingMove(from: "a", oldPath: "/p", to: "b"))
    }

    func testTheMovedFavoriteResolvesToTheRowAtItsNewPlace() {
        let moved = LibraryFixtures.row("qwen", library: t7)
        let updated = favorite(paths: ["/old"], ids: ["internal:text/qwen"])
            .followingMove(from: "internal:text/qwen", oldPath: "/old", to: moved.installId)
        guard case .available(let found)? = updated?.resolve(in: [moved]).first else {
            return XCTFail("did not resolve")
        }
        XCTAssertEqual(found.installId, moved.installId)
    }

    // MARK: The manager's selection

    func testTheManagerFollowsASelectedModelToItsNewPath() {
        let event = MovedModelEvent(
            oldInstallId: "internal:text/qwen", newInstallId: "t7-id:text/qwen", oldPath: "/old",
            newPath: "/new")
        XCTAssertEqual(event.follow(selectedPath: "/old"), "/new")
        XCTAssertEqual(event.follow(selectedPath: "/elsewhere"), "/elsewhere")
        XCTAssertNil(event.follow(selectedPath: nil))
        let unknownPlace = MovedModelEvent(
            oldInstallId: "a", newInstallId: "b", oldPath: "/old", newPath: nil)
        XCTAssertEqual(unknownPlace.follow(selectedPath: "/old"), "/old", "no new path: the selection is left as it is")
    }
}
