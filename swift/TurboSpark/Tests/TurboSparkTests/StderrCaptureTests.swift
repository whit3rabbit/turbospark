import Darwin
import XCTest

@testable import TurboSpark

final class StderrCaptureTests: XCTestCase {
    override func tearDown() { StderrCapture.shared.stop() }

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
        try StderrCapture.shared.start { lines.add($0) }
        fputs("vision: auto -- first\nsecond line\n", stderr)
        fflush(stderr)
        wait(for: lines, containing: "second line")
        XCTAssertEqual(lines.all, ["vision: auto -- first", "second line"])
    }

    func testAnUnterminatedLineIsHeldUntilItsNewlineArrives() throws {
        let lines = Lines()
        try StderrCapture.shared.start { lines.add($0) }
        fputs("half", stderr); fflush(stderr)
        Thread.sleep(forTimeInterval: 0.15)
        XCTAssertTrue(lines.all.isEmpty, "no line until the newline")
        fputs(" and half\n", stderr); fflush(stderr)
        wait(for: lines, containing: "half and half")
        XCTAssertEqual(lines.all, ["half and half"])
    }

    func testASecondStartIsRefusedAndStopRestoresStderr() throws {
        try StderrCapture.shared.start { _ in }
        XCTAssertTrue(StderrCapture.shared.isRunning)
        XCTAssertThrowsError(try StderrCapture.shared.start { _ in })
        StderrCapture.shared.stop()
        XCTAssertFalse(StderrCapture.shared.isRunning)
        // After stop, a fresh capture works again and sees only new output.
        let lines = Lines()
        try StderrCapture.shared.start { lines.add($0) }
        fputs("after restart\n", stderr); fflush(stderr)
        wait(for: lines, containing: "after restart")
        XCTAssertEqual(lines.all, ["after restart"])
    }
}
