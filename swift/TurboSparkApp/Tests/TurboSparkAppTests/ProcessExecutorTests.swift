import XCTest
@testable import TurboSparkApp

/// Regression tests for the run_command / hook-runner deadlock class of bug
/// (state#2, T2, T12): `Process.waitUntilExit()` followed by
/// `readDataToEndOfFile()` hangs forever once a child writes more than one
/// pipe buffer (~64KB) before exiting, because nothing drains the pipe while
/// the caller waits. `ProcessExecutor.run` fixes this by draining stdout,
/// stderr, and stdin concurrently with the wait; these tests would hang
/// (and eventually fail on the harness's own timeout) against the previous
/// implementation.
final class ProcessExecutorTests: XCTestCase {
    func testLargeOutputDoesNotDeadlock() async throws {
        // `yes | head -c 200000` produces ~200KB on stdout, well past a
        // single 64KB pipe buffer, with nothing reading it until the whole
        // pipeline finishes writing under the old synchronous drain.
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "yes | head -c 200000"],
            timeoutSeconds: 10
        )
        XCTAssertFalse(result.timedOut, "A large-output command must complete well within its timeout, not hang.")
        XCTAssertGreaterThan(result.stdout.count, 190_000)
    }

    func testTimeoutActuallyTerminatesALongRunningCommand() async throws {
        let start = Date()
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "sleep 30"],
            timeoutSeconds: 0.5
        )
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertTrue(result.timedOut)
        XCTAssertLessThan(elapsed, 5.0, "A 0.5s timeout must terminate the child well before its own 30s sleep completes.")
    }

    func testLargeStdinIsDeliveredWithoutBlocking() async throws {
        // `cat` echoes stdin back on stdout. A payload larger than one pipe
        // buffer, written synchronously BEFORE the child is being drained,
        // is exactly the shape that blocked the hook runner's payload write
        // (T12): the write call itself never returned.
        let payload = Data(repeating: UInt8(ascii: "a"), count: 300_000)
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/cat"),
            arguments: [],
            stdin: payload,
            timeoutSeconds: 10
        )
        XCTAssertFalse(result.timedOut)
        XCTAssertEqual(result.stdout.count, payload.count)
    }

    func testOutputIsCappedRatherThanUnbounded() async throws {
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "yes | head -c 50000"],
            timeoutSeconds: 10,
            outputCapBytes: 1000
        )
        XCTAssertFalse(result.timedOut)
        XCTAssertLessThan(result.stdout.count, 1100, "Captured output should respect the cap plus the truncation note.")
        XCTAssertTrue(result.stdout.contains("truncated"))
    }

    func testOrdinaryCommandStillWorks() async throws {
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/echo"),
            arguments: ["hello"],
            timeoutSeconds: 5
        )
        XCTAssertFalse(result.timedOut)
        XCTAssertEqual(result.exitCode, 0)
        XCTAssertEqual(result.stdout.trimmingCharacters(in: .whitespacesAndNewlines), "hello")
    }
}

/// Exercises the full `run_command` tool path (AppToolRegistry.execute),
/// which is the actual caller most model turns go through.
final class RunCommandToolTests: XCTestCase {
    func testRunCommandHandlesLargeOutputWithoutHanging() async throws {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "yes | head -c 200000"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: nil)
        XCTAssertFalse(result.isError)
        XCTAssertGreaterThan(result.output.count, 190_000)
    }

    func testRunCommandHonorsAModelSuppliedTimeout() async throws {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "sleep 30", "timeout": "300"],
            category: .terminal
        )
        let start = Date()
        let result = await AppToolRegistry.execute(call: call, in: nil)
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 5.0, "A 300ms timeout argument must be honored, not the 120s default.")
        XCTAssertTrue(result.output.contains("timed out"))
    }
}
