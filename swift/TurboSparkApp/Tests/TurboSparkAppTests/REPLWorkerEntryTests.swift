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

    /// A request line that never ends must be refused once the buffer passes
    /// the transport limit, before any newline arrives, and the loop must
    /// resync on the next newline without treating the tail as a request.
    func testServeRefusesUnterminatedOversizedLineBeforeNewline() throws {
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
        process.arguments = [REPLWorkerMain.workerModeArgument, REPLWorkerMain.serveArgument]
        let input = Pipe()
        let output = Pipe()
        process.standardInput = input
        process.standardOutput = output
        process.standardError = FileHandle.nullDevice
        try process.run()
        defer {
            if process.isRunning { process.terminate() }
            process.waitUntilExit()
        }

        // Collect stdout lines on a background queue so a missing reply
        // times out instead of hanging the test.
        let lines = LineCollector()
        let reader = output.fileHandleForReading
        DispatchQueue.global().async {
            var pending = Data()
            while true {
                let chunk = reader.availableData
                if chunk.isEmpty { break }
                pending.append(chunk)
                while let newline = pending.firstIndex(of: 0x0A) {
                    let line = pending[pending.startIndex..<newline]
                    pending.removeSubrange(pending.startIndex...newline)
                    lines.append(String(decoding: line, as: UTF8.self))
                }
            }
        }

        // Over the limit, with no newline anywhere.
        let writer = input.fileHandleForWriting
        let block = Data(repeating: 0x78, count: 1_024 * 1_024)
        let blocks = REPLWorkerMain.maximumTransportBytes / block.count + 1
        let sender = DispatchQueue.global()
        sender.async {
            for _ in 0..<blocks { writer.write(block) }
        }

        let refusal = try XCTUnwrap(
            lines.waitForLine(index: 0, timeout: 20),
            "the worker must refuse an unterminated oversized line before a newline arrives")
        XCTAssertTrue(refusal.contains("byte transport limit"), refusal)

        // Finish the refused line, then send a normal bad line: only the
        // latter may produce a reply.
        writer.write(Data("tail\ngarbage\n".utf8))
        let second = try XCTUnwrap(lines.waitForLine(index: 1, timeout: 20))
        XCTAssertTrue(second.contains("not a JSON object"), second)
        XCTAssertNil(lines.waitForLine(index: 2, timeout: 0.5))
        try? writer.close()
    }
}

private final class LineCollector: @unchecked Sendable {
    private let lock = NSLock()
    private var lines: [String] = []

    func append(_ line: String) {
        lock.lock()
        lines.append(line)
        lock.unlock()
    }

    func waitForLine(index: Int, timeout: TimeInterval) -> String? {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            lock.lock()
            let value = index < lines.count ? lines[index] : nil
            lock.unlock()
            if let value { return value }
            Thread.sleep(forTimeInterval: 0.02)
        }
        return nil
    }
}
