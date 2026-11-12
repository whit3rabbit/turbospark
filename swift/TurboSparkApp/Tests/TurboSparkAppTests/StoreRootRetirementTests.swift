import Foundation
import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Task 5.10 (requirements 1.1, 9.2): the app no longer redirects the engine to
/// a user store root. The recorded root is the migration's input and nothing
/// else.
///
/// **THE ENGINE ROOT IS PROCESS-WIDE STATE.** Before this task a launch with a
/// record called `TurboSparkCatalog.setStoreRoot(record)`, so the CLI-visible
/// split the model-libraries work removes came back inside the app process. The
/// test builds a real `AppModel` (its `init` is the launch path) with a record
/// on disk and asks the engine what its root is.
@MainActor
final class StoreRootRetirementTests: XCTestCase {
    private let moved = "/Volumes/Retired Drive/.turbospark"

    override func setUp() async throws {
        try await super.setUp()
        try? TurboSparkCatalog.setStoreRoot(nil)
        try? FileManager.default.removeItem(at: ModelStoreRootRecord.url)
    }

    override func tearDown() async throws {
        // A failing run must not leave the process redirected for the suites
        // that run after it.
        try? TurboSparkCatalog.setStoreRoot(nil)
        try? FileManager.default.removeItem(at: ModelStoreRootRecord.url)
        try await super.tearDown()
    }

    func testLaunchWithARecordedRootDoesNotRedirectTheEngine() throws {
        let before = try TurboSparkCatalog.storeRoot()
        XCTAssertNotEqual(before, moved, "the engine starts at its default root")
        XCTAssertTrue(ModelStoreRootRecord.save(moved))

        let model = AppModel()
        model.stopCronScheduler()

        XCTAssertEqual(
            try TurboSparkCatalog.storeRoot(), before,
            "launching with a recorded root must leave the engine at the home")
        XCTAssertEqual(
            ModelStoreRootRecord.load(), moved,
            "the record is still there: it is the migration's input")
    }

    func testALegacyProfileRootIsStillPromotedToTheRecordWithoutRedirectingTheEngine() throws {
        let before = try TurboSparkCatalog.storeRoot()
        // `resolve` is what the launch path calls with the profile's value.
        XCTAssertEqual(ModelStoreRootRecord.resolve(profileValue: moved), moved)
        XCTAssertEqual(ModelStoreRootRecord.load(), moved, "promoted so the conversion sees it")
        XCTAssertEqual(try TurboSparkCatalog.storeRoot(), before)
    }

    func testWritingTheRecordAtRuntimeDoesNotChangeTheEngineRoot() throws {
        let before = try TurboSparkCatalog.storeRoot()
        XCTAssertTrue(ModelStoreRootRecord.save(moved))
        XCTAssertEqual(try TurboSparkCatalog.storeRoot(), before)
    }

    // MARK: - Clearing the record must stick

    func testAClearedRecordIsNotBroughtBackByAProfileThatStillNamesTheOldRoot() {
        XCTAssertTrue(ModelStoreRootRecord.save(moved))
        XCTAssertTrue(ModelStoreRootRecord.clear())
        XCTAssertNil(ModelStoreRootRecord.load(), "cleared means no root is recorded")

        // `settings.json` of a profile still holds the old root. Promoting it
        // again would re-import the library at every launch, and bring back
        // one the user removed.
        XCTAssertEqual(ModelStoreRootRecord.resolve(profileValue: moved), "")
        XCTAssertNil(ModelStoreRootRecord.load())
        XCTAssertEqual(ModelStoreRootRecord.resolve(profileValue: "/old/place"), "")
    }

    func testAWhitespaceOnlyRecordIsNoRecordAndStillBlocksTheLegacyPromotion() throws {
        try "  \n\t ".write(to: ModelStoreRootRecord.url, atomically: true, encoding: .utf8)

        XCTAssertNil(ModelStoreRootRecord.load(), "blank is not a path")
        XCTAssertEqual(
            ModelStoreRootRecord.resolve(profileValue: moved), "",
            "the file exists, so a stale profile value is not promoted")
    }

    func testARootRecordedAfterAClearIsHonoredAgain() {
        XCTAssertTrue(ModelStoreRootRecord.save(moved))
        XCTAssertTrue(ModelStoreRootRecord.clear())
        XCTAssertTrue(ModelStoreRootRecord.save("/Volumes/Other/.turbospark"))
        XCTAssertEqual(ModelStoreRootRecord.load(), "/Volumes/Other/.turbospark")
        XCTAssertEqual(ModelStoreRootRecord.resolve(profileValue: ""), "/Volumes/Other/.turbospark")
    }

    func testClearingWithNoRecordIsHarmlessAndAnExistingProfileValueStillMigrates() {
        // Only a record that was CLEARED blocks the legacy promotion; a
        // machine that never had one still migrates the profile's value.
        XCTAssertFalse(FileManager.default.fileExists(atPath: ModelStoreRootRecord.url.path))
        XCTAssertEqual(ModelStoreRootRecord.resolve(profileValue: moved), moved)
        XCTAssertEqual(ModelStoreRootRecord.load(), moved)
    }
}
