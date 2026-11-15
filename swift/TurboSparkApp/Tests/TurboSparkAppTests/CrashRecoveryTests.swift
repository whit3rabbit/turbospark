import XCTest
@testable import TurboSparkApp

final class CrashRecoveryTests: XCTestCase {
    private var directory: URL!

    override func setUpWithError() throws {
        directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("CrashRecoveryTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }

    override func tearDownWithError() throws {
        try? FileManager.default.removeItem(at: directory)
    }

    private var markerURL: URL { directory.appendingPathComponent(CrashRecoveryStore.fileName) }

    func testFirstRunIsCleanAndWritesMarker() {
        let store = CrashRecoveryStore(directory: directory)
        XCTAssertEqual(store.beginSession(), .clean)
        XCTAssertTrue(FileManager.default.fileExists(atPath: markerURL.path))
    }

    func testOrderlyExitLeavesNextRunClean() {
        let first = CrashRecoveryStore(directory: directory)
        first.beginSession()
        first.endSession()
        XCTAssertFalse(FileManager.default.fileExists(atPath: markerURL.path))
        XCTAssertEqual(CrashRecoveryStore(directory: directory).beginSession(), .clean)
    }

    // A fresh instance stands for the next process: nothing but the file
    // carries state across.
    func testAbnormalExitSurvivesIntoNextRun() {
        let started = Date(timeIntervalSince1970: 1_800_000_000)
        CrashRecoveryStore(directory: directory).beginSession(now: started)
        let next = CrashRecoveryStore(directory: directory)
        XCTAssertEqual(next.beginSession(), .previousEndedAbnormally(startedAt: started))
    }

    func testNoticeIsOncePerMarker() {
        CrashRecoveryStore(directory: directory).beginSession()
        let second = CrashRecoveryStore(directory: directory)
        XCTAssertNotEqual(second.beginSession(), .clean)
        // The second run did not exit in order either, but the first notice
        // was consumed: the marker now describes the second run.
        let third = CrashRecoveryStore(directory: directory)
        XCTAssertNotEqual(third.beginSession(), .clean)
        third.endSession()
        XCTAssertEqual(CrashRecoveryStore(directory: directory).beginSession(), .clean)
    }

    func testSecondBeginInSameProcessDoesNotReadOwnMarker() {
        let store = CrashRecoveryStore(directory: directory)
        XCTAssertEqual(store.beginSession(), .clean)
        XCTAssertEqual(store.beginSession(), .clean)
        // And it must not have rewritten or cleared the marker.
        XCTAssertTrue(FileManager.default.fileExists(atPath: markerURL.path))
    }

    func testCorruptMarkerStillCountsAsAbnormal() throws {
        try Data("not json".utf8).write(to: markerURL)
        XCTAssertEqual(
            CrashRecoveryStore(directory: directory).beginSession(),
            .previousEndedAbnormally(startedAt: nil))
    }

    func testMarkerCarriesOnlyVersionAndTime() throws {
        CrashRecoveryStore(directory: directory).beginSession()
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: markerURL))
        let keys = Set((object as? [String: Any] ?? [:]).keys)
        XCTAssertEqual(keys, ["version", "startedAt"])
    }

    func testUnwritableDirectoryDoesNotThrowOrBlock() {
        let missing = directory.appendingPathComponent("absent/deeper", isDirectory: true)
        let store = CrashRecoveryStore(directory: missing)
        XCTAssertEqual(store.beginSession(), .clean)
        store.endSession()
    }
}
