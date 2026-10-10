import Darwin
import XCTest

@testable import TurboSpark

final class StderrCaptureTests: XCTestCase {
    private var capture = StderrCapture()

    override func tearDown() { capture.stop() }

    private final class Lines: @unchecked Sendable {
        private let lock = NSLock()
        private var storage: [String] = []
        func add(_ s: String) { lock.lock(); storage.append(s); lock.unlock() }
        var all: [String] { lock.lock(); defer { lock.unlock() }; return storage }
    }

    private func wait(for lines: Lines, containing text: String) {
        let deadline = Date().addingTimeInterval(3)
        while !lines.all.contains(where: { $0.contains(text) }), Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
    }

    func testWritesToStderrReachTheHandlerOneLineAtATime() throws {
        let lines = Lines()
        try capture.start { lines.add($0) }
        fputs("vision: auto -- first\nsecond line\n", stderr)
        fflush(stderr)
        wait(for: lines, containing: "second line")
        XCTAssertEqual(lines.all, ["vision: auto -- first", "second line"])
    }

    func testAnUnterminatedLineIsHeldUntilItsNewlineArrives() throws {
        let lines = Lines()
        try capture.start { lines.add($0) }
        fputs("half", stderr); fflush(stderr)
        Thread.sleep(forTimeInterval: 0.15)
        XCTAssertTrue(lines.all.isEmpty, "no line until the newline")
        fputs(" and half\n", stderr); fflush(stderr)
        wait(for: lines, containing: "half and half")
        XCTAssertEqual(lines.all, ["half and half"])
    }

    func testASecondStartIsRefusedAndStopRestoresStderr() throws {
        try capture.start { _ in }
        XCTAssertTrue(capture.isRunning)
        XCTAssertThrowsError(try capture.start { _ in })
        capture.stop()
        XCTAssertFalse(capture.isRunning)
        // After stop, a fresh capture works again and sees only new output.
        let lines = Lines()
        try capture.start { lines.add($0) }
        fputs("after restart\n", stderr); fflush(stderr)
        wait(for: lines, containing: "after restart")
        XCTAssertEqual(lines.all, ["after restart"])
    }

    func testHandlerCanStopCaptureWithoutDeadlocking() throws {
        let activeCapture = capture
        let stopped = DispatchSemaphore(value: 0)
        try activeCapture.start { _ in
            activeCapture.stop()
            stopped.signal()
        }
        fputs("stop from handler\n", stderr); fflush(stderr)

        XCTAssertEqual(stopped.wait(timeout: .now() + 3), .success)
        activeCapture.stop() // Wait for the callback-initiated asynchronous drain to finish.
        XCTAssertFalse(activeCapture.isRunning)
    }

    func testStopDeliversAnUnterminatedFinalLine() throws {
        let lines = Lines()
        try capture.start { lines.add($0) }
        fputs("final fragment", stderr); fflush(stderr)

        capture.stop()

        XCTAssertEqual(lines.all, ["final fragment"])
    }

    func testStopDrainsAndTeesOutputQueuedBehindAHandler() throws {
        let activeCapture = capture
        var template = Array("/tmp/turbospark-stderr-XXXXXX".utf8CString)
        let teeDescriptor = template.withUnsafeMutableBufferPointer { mkstemp($0.baseAddress!) }
        guard teeDescriptor >= 0 else {
            XCTFail("could not create stderr tee file: errno \(errno)")
            return
        }
        let teePath = String(cString: template)
        let savedStderr = dup(STDERR_FILENO)
        guard savedStderr >= 0 else {
            close(teeDescriptor)
            unlink(teePath)
            XCTFail("could not save stderr: errno \(errno)")
            return
        }
        guard dup2(teeDescriptor, STDERR_FILENO) >= 0 else {
            close(savedStderr)
            close(teeDescriptor)
            unlink(teePath)
            XCTFail("could not redirect test stderr: errno \(errno)")
            return
        }
        defer {
            _ = dup2(savedStderr, STDERR_FILENO)
            close(savedStderr)
            close(teeDescriptor)
            unlink(teePath)
        }

        let lines = Lines()
        let firstLineEntered = DispatchSemaphore(value: 0)
        let releaseFirstLine = DispatchSemaphore(value: 0)
        let stopped = DispatchSemaphore(value: 0)
        try activeCapture.start { line in
            lines.add(line)
            if line == "first" {
                firstLineEntered.signal()
                releaseFirstLine.wait()
            }
        }
        defer {
            releaseFirstLine.signal()
            activeCapture.stop()
        }

        fputs("first\n", stderr); fflush(stderr)
        XCTAssertEqual(firstLineEntered.wait(timeout: .now() + 3), .success)
        fputs("second-before-stop\n", stderr); fflush(stderr)
        DispatchQueue.global().async {
            activeCapture.stop()
            stopped.signal()
        }

        let deadline = Date().addingTimeInterval(3)
        while activeCapture.isRunning && Date() < deadline {
            Thread.sleep(forTimeInterval: 0.01)
        }
        XCTAssertFalse(activeCapture.isRunning, "stop should restore stderr before waiting for the callback queue")
        releaseFirstLine.signal()
        XCTAssertEqual(stopped.wait(timeout: .now() + 3), .success)
        XCTAssertEqual(lines.all, ["first", "second-before-stop"])

        XCTAssertGreaterThanOrEqual(lseek(teeDescriptor, 0, SEEK_SET), 0)
        var teeBytes = [UInt8](repeating: 0, count: 4096)
        let teeCount = Darwin.read(teeDescriptor, &teeBytes, teeBytes.count)
        XCTAssertGreaterThan(teeCount, 0)
        let teedText = String(decoding: teeBytes.prefix(max(0, teeCount)), as: UTF8.self)
        XCTAssertTrue(teedText.contains("second-before-stop"))
    }
}
