import Foundation
import XCTest

@testable import TurboSparkApp

final class WorkflowHistoricalProfileReaderTests: XCTestCase {
    func testHistoricalInitializerReadsAndWritesCoreRecords() throws {
        guard let path = ProcessInfo.processInfo.environment["WORKFLOW_COMPATIBILITY_ROOT"] else {
            throw XCTSkip("This test is injected by the cross-version compatibility script")
        }
        let root = URL(fileURLWithPath: path, isDirectory: true)
        let key = Data(repeating: 0x42, count: 32)
        let database = try ProfileDatabase(
            url: root.appendingPathComponent("workflow-bearing-profile.sqlite3"),
            key: key)
        defer { database.close() }

        XCTAssertEqual(
            try database.loadRecord(key: "compatibility:sentinel"),
            Data("core-record-before-old-reader".utf8))
        XCTAssertNil(try database.loadRecord(key: "compatibility:historical-write"))
        try database.saveRecord(
            key: "compatibility:historical-write",
            payload: Data("written-by-historical-reader".utf8))
        XCTAssertEqual(
            try database.loadRecord(key: "compatibility:historical-write"),
            Data("written-by-historical-reader".utf8))
    }
}
