import Foundation
import XCTest

@testable import TurboSparkApp

@MainActor
final class SparkleUpdateControllerTests: XCTestCase {
    func testHarnessLogKeepsEveryUpdateEvent() throws {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(
            at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }

        let logURL = directory.appendingPathComponent("update.log")
        try AutoInstallUserDriver.appendHarnessLog("detected\n", to: logURL)
        try AutoInstallUserDriver.appendHarnessLog("downloaded\n", to: logURL)
        try AutoInstallUserDriver.appendHarnessLog("relaunched\n", to: logURL)

        XCTAssertEqual(
            try String(contentsOf: logURL, encoding: .utf8),
            "detected\ndownloaded\nrelaunched\n")
    }
}
