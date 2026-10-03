import Foundation
import XCTest
@testable import TurboSparkApp

final class REPLWorkerEntryTests: XCTestCase {
    func testNormalArgumentsDoNotSelectWorkerMode() {
        XCTAssertFalse(REPLWorkerMain.isWorkerInvocation(["TurboSparkApp"]))
        XCTAssertTrue(REPLWorkerMain.isWorkerInvocation([
            "TurboSparkApp",
            REPLWorkerMain.workerModeArgument
        ]))
    }

    func testPackagedExecutableRunsWorkerSmokeWithoutStartingTheApp() throws {
        let packageRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        let executable = try XCTUnwrap([
            ".build/out/Products/Debug/TurboSparkApp",
            ".build/out/Products/Release/TurboSparkApp",
            ".build/arm64-apple-macosx/debug/TurboSparkApp",
            ".build/arm64-apple-macosx/release/TurboSparkApp",
            ".build/debug/TurboSparkApp",
            ".build/release/TurboSparkApp"
        ]
        .map { packageRoot.appendingPathComponent($0, isDirectory: false) }
        .first { FileManager.default.isExecutableFile(atPath: $0.path) },
            "The TurboSparkApp executable should be available in the SwiftPM build products")

        let process = Process()
        process.executableURL = executable
        process.arguments = [REPLWorkerMain.workerModeArgument, "--smoke"]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice

        try process.run()
        let deadline = Date().addingTimeInterval(10)
        while process.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.025)
        }
        guard !process.isRunning else {
            process.terminate()
            process.waitUntilExit()
            XCTFail("Worker mode should exit instead of starting the app")
            return
        }
        process.waitUntilExit()

        let data = output.fileHandleForReading.readDataToEndOfFile()
        let text = String(decoding: data, as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
        XCTAssertEqual(process.terminationStatus, 0)
        XCTAssertEqual(text, "worker:2")
    }
}
