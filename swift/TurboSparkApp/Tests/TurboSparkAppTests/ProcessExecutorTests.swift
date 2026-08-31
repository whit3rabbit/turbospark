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

/// **CANCELLING A RUNNING COMMAND MUST STOP IT, NOT SPIN ON IT.**
///
/// The wait loop paced itself with `try? await Task.sleep`, which returns
/// IMMEDIATELY once the surrounding task is cancelled. Swallowing that with
/// `try?` left `while process.isRunning` unchanged, so a cancelled
/// `run_command` burned a core at full speed for the whole remaining deadline
/// (120 s by default) while the child ran to completion behind it. Neither
/// half announced itself.
final class ProcessExecutorCancellationTests: XCTestCase {

    func testCancellingTheTaskKillsTheChildPromptly() async throws {
        let started = Date()
        let task = Task {
            try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "sleep 60"],
                // A deadline far past the test's own patience: if
                // cancellation is not honored, this is how long the old code
                // spun for.
                timeoutSeconds: 120
            )
        }

        // Let the child actually start before cancelling, or the test proves
        // only that a not-yet-running process exits.
        try await Task.sleep(nanoseconds: 300_000_000)
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("a cancelled run must throw rather than return an ordinary result")
        } catch is CancellationError {
            // Expected.
        } catch {
            XCTFail("expected CancellationError, got \(error)")
        }

        let elapsed = Date().timeIntervalSince(started)
        XCTAssertLessThan(
            elapsed, 10.0,
            "cancellation returned after \(elapsed)s; the 120s deadline was being waited out")
    }

    /// The child must be dead, not merely abandoned. A `sleep` that survives
    /// its canceller is an orphan holding a pipe nobody reads.
    func testTheChildProcessIsNotLeftRunningAfterCancellation() async throws {
        let marker = "turbospark-cancel-marker-\(UUID().uuidString)"
        let task = Task {
            try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "sleep 60 # \(marker)"],
                timeoutSeconds: 120
            )
        }
        try await Task.sleep(nanoseconds: 300_000_000)
        task.cancel()
        _ = try? await task.value

        // Give the SIGTERM/SIGKILL ladder a moment to be reaped.
        try await Task.sleep(nanoseconds: 500_000_000)

        let check = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "ps ax | grep -c '[\(marker.prefix(1))]\(marker.dropFirst())' || true"],
            timeoutSeconds: 10
        )
        let survivors = Int(check.stdout.trimmingCharacters(in: .whitespacesAndNewlines)) ?? 0
        XCTAssertEqual(survivors, 0, "the cancelled child is still running")
    }

    /// The timeout path must keep working: cancellation is a NEW exit from
    /// this loop and must not have displaced the one that was there.
    func testTheTimeoutPathStillFiresAndStillReportsItself() async throws {
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "sleep 30"],
            timeoutSeconds: 1
        )
        XCTAssertTrue(result.timedOut, "a command past its deadline must report timing out")
    }
}

/// Exercises the full `run_command` tool path (AppToolRegistry.execute),
/// which is the actual caller most model turns go through.
final class RunCommandToolTests: XCTestCase {
    /// A project rooted at a real temporary directory.
    ///
    /// Both cases below used to pass `in: nil`, which reached `execute`'s old
    /// fallback of rooting a projectless chat at the user's home directory.
    /// That fallback is gone -- a tool that resolves a path or spawns a
    /// process is now refused outright without a workspace -- so these have
    /// to name one. The test that asserts the refusal itself lives in
    /// `ToolPermissionsTests`.
    private func temporaryProject() -> AppProject {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("turbospark-tests-\(UUID().uuidString)")
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        return AppProject(name: "Test", rootDirectoryPath: root.path)
    }

    func testRunCommandHandlesLargeOutputWithoutHanging() async throws {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "yes | head -c 200000"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: temporaryProject())
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
        let result = await AppToolRegistry.execute(call: call, in: temporaryProject())
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertLessThan(elapsed, 5.0, "A 300ms timeout argument must be honored, not the 120s default.")
        XCTAssertTrue(result.output.contains("timed out"))
    }
}
