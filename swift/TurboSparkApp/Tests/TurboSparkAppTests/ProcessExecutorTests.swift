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

    func testOutputReportsWhenTheCaptureIsTruncated() async throws {
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "printf 123456789"],
            timeoutSeconds: 10,
            outputCapBytes: 4
        )
        XCTAssertEqual(result.exitCode, 0)
        XCTAssertTrue(result.outputTruncated)
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

    func testEarlyExitWhileStdinWriterIsScheduledDoesNotRaiseSIGPIPE() async throws {
        // Command hooks always receive a JSON payload. A hook that exits
        // immediately can close its stdin before the detached writer starts;
        // that must become a contained EPIPE, never SIGPIPE for XCTest.
        let payload = Data(repeating: UInt8(ascii: "x"), count: 300_000)
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "exit 0"],
            stdin: payload,
            timeoutSeconds: 5
        )
        XCTAssertFalse(result.timedOut)
        XCTAssertEqual(result.exitCode, 0)
    }

    func testPartialStdinConsumerExitDoesNotRaiseSIGPIPE() async throws {
        // A hook can inspect just its first input byte and exit. The writer
        // may already have filled the pipe when that close arrives, so this
        // exercises EPIPE after a real partial read rather than only `exit`.
        let payload = Data(repeating: UInt8(ascii: "x"), count: 1_000_000)
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "head -c 1 >/dev/null; exit 0"],
            stdin: payload,
            timeoutSeconds: 5
        )
        XCTAssertFalse(result.timedOut)
        XCTAssertEqual(result.exitCode, 0)
    }

    func testConcurrentEarlyExitsWhileWritingStdinDoNotRaiseSIGPIPE() async throws {
        // The descriptor-level suppression must apply to every independent
        // process, not accidentally rely on a process-wide signal setting.
        let payload = Data(repeating: UInt8(ascii: "x"), count: 300_000)
        let completions = try await withThrowingTaskGroup(of: Bool.self) { group in
            for _ in 0..<12 {
                group.addTask {
                    let result = try await ProcessExecutor.run(
                        executableURL: URL(fileURLWithPath: "/bin/sh"),
                        arguments: ["-c", "exit 0"],
                        stdin: payload,
                        timeoutSeconds: 5
                    )
                    return !result.timedOut && result.exitCode == 0
                }
            }

            var completed: [Bool] = []
            for try await value in group { completed.append(value) }
            return completed
        }
        XCTAssertEqual(completions.count, 12)
        XCTAssertTrue(completions.allSatisfy { $0 })
    }

    func testTimeoutDuringBlockedStdinWriteIsContained() async throws {
        // This child never drains stdin, so the background writer is blocked
        // when the timeout closes its pipe. It must not turn that EPIPE into
        // a signal for the app process.
        let payload = Data(repeating: UInt8(ascii: "x"), count: 1_000_000)
        let start = Date()
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "sleep 30"],
            stdin: payload,
            timeoutSeconds: 0.5
        )
        XCTAssertTrue(result.timedOut)
        XCTAssertLessThan(Date().timeIntervalSince(start), 5.0)
    }

    func testLargeStdoutAndStderrDrainTogether() async throws {
        // Separate readers must keep both streams moving. Draining only one
        // would leave the other full and recreate the pre-wait deadlock.
        let result = try await ProcessExecutor.run(
            executableURL: URL(fileURLWithPath: "/bin/sh"),
            arguments: ["-c", "yes out | head -c 200000; yes err | head -c 200000 >&2"],
            timeoutSeconds: 10
        )
        XCTAssertFalse(result.timedOut)
        XCTAssertGreaterThan(result.stdout.count, 190_000)
        XCTAssertGreaterThan(result.stderr.count, 190_000)
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

    func testCancellationDuringBlockedStdinWriteIsContained() async throws {
        // The cancellation path closes stdin while the detached writer waits
        // on a full pipe. The task must still complete as cancellation, not
        // leak SIGPIPE or wait for the original deadline.
        let payload = Data(repeating: UInt8(ascii: "x"), count: 1_000_000)
        let started = Date()
        let task = Task {
            try await ProcessExecutor.run(
                executableURL: URL(fileURLWithPath: "/bin/sh"),
                arguments: ["-c", "sleep 60"],
                stdin: payload,
                timeoutSeconds: 120
            )
        }
        try await Task.sleep(nanoseconds: 300_000_000)
        task.cancel()

        do {
            _ = try await task.value
            XCTFail("a cancelled run must throw rather than return an ordinary result")
        } catch is CancellationError {
            // Expected.
        }

        XCTAssertLessThan(Date().timeIntervalSince(started), 10.0)
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
        // The model-facing cap bounds what a runaway command spends of the
        // context window: head and tail kept, the middle replaced by a
        // marker. The whole stream no longer reaches the model (it used to,
        // all 200KB of it).
        XCTAssertLessThanOrEqual(result.output.count, 31_000)
        XCTAssertTrue(result.output.contains("chars truncated"), result.output)
    }

    /// The head/tail shape is the point of the compaction: build failures
    /// summarize at the END of their output, so a head-only cut hides
    /// exactly the lines that explain the failure.
    func testLargeOutputKeepsHeadAndTailAroundTheMarker() async throws {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "printf 'HEADSTART\\n'; yes x | head -n 25000; printf 'TAILMARK\\n'"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: temporaryProject())
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("HEADSTART"), result.output)
        XCTAssertTrue(result.output.contains("TAILMARK"), result.output)
        XCTAssertTrue(result.output.contains("chars truncated"), result.output)
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
        // A timeout is work that did not finish, so it must not read as a
        // success the model can build on.
        XCTAssertTrue(result.isError, "A timed-out command must report isError, got: \(result.output)")
    }

    /// The whole point of merged streams: relative order between stdout and
    /// stderr survives, where the old two-buffer join could not preserve it.
    func testInterleavedOutputPreservesArrivalOrder() async throws {
        let call = AppToolCall(
            name: "run_command",
            arguments: [
                "command": "printf 'a1\\n'; sleep 0.1; printf 'e1\\n' >&2; "
                    + "sleep 0.1; printf 'a2\\n'; sleep 0.1; printf 'e2\\n' >&2"
            ],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: temporaryProject())
        XCTAssertFalse(result.isError)
        let ranges = ["a1", "e1", "a2", "e2"].map { result.output.range(of: $0) }
        XCTAssertNotNil(ranges[0]); XCTAssertNotNil(ranges[1])
        XCTAssertNotNil(ranges[2]); XCTAssertNotNil(ranges[3])
        let positions = ranges.map { $0!.lowerBound }
        XCTAssertEqual(positions, positions.sorted(), "lines must appear in the order they were written")
    }

    func testANSIEscapeSequencesAreStripped() async throws {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "printf '\\033[31mred-text\\033[0m plain\\n'"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: temporaryProject())
        XCTAssertFalse(result.isError)
        XCTAssertTrue(result.output.contains("red-text plain"))
        XCTAssertFalse(result.output.contains("\u{1B}"), "escape bytes must not reach the model")
    }

    func testShellEnvironmentPreventsEditorAndPagerHangs() async throws {
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "printf '%s\\n' \"$GIT_EDITOR\" \"$GIT_PAGER\" \"$PAGER\""],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: temporaryProject())
        XCTAssertFalse(result.isError)
        let lines = result.output.split(separator: "\n").map(String.init)
        XCTAssertEqual(lines, ["true", "cat", "cat"], result.output)
    }

    func testModelTimeoutIsClampedToTheMaximum() {
        XCTAssertEqual(ShellCommandRunner.clampedTimeoutSeconds(timeoutMs: nil), 120)
        XCTAssertEqual(ShellCommandRunner.clampedTimeoutSeconds(timeoutMs: 300), 0.3)
        XCTAssertEqual(
            ShellCommandRunner.clampedTimeoutSeconds(timeoutMs: 3_600_000), 600,
            "an hour, in milliseconds, must clamp to the 600s ceiling")
    }

    /// grep exit 1 is the command ANSWERING ("nothing matched"). Reporting
    /// it as an error invited the model to retry a question already
    /// answered.
    func testGrepNoMatchIsReportedAsAnAnswerNotAnError() async throws {
        let project = temporaryProject()
        let rootURL = try XCTUnwrap(project.rootDirectoryURL)
        try "apple\nbanana\n".write(
            to: rootURL.appendingPathComponent("fruit.txt"),
            atomically: true, encoding: .utf8)
        let call = AppToolCall(
            name: "run_command",
            arguments: ["command": "grep zebra fruit.txt"],
            category: .terminal
        )
        let result = await AppToolRegistry.execute(call: call, in: project)
        XCTAssertFalse(result.isError, "grep exit 1 means no matches: an answer, not a failure")
        XCTAssertTrue(result.output.contains("No matches found"), result.output)
    }

    func testWorkingDirectoryPersistsBetweenCalls() async throws {
        ShellCwdTracker.shared.resetForTests()
        defer { ShellCwdTracker.shared.resetForTests() }
        let project = temporaryProject()

        func run(_ command: String) async -> AppToolResult {
            await AppToolRegistry.execute(
                call: AppToolCall(name: "run_command", arguments: ["command": command], category: .terminal),
                in: project)
        }

        // The tool's own `pwd -P` spelling of the root is the baseline:
        // resolving the root path through a URL API here resolves symlinks
        // differently (/var vs /private/var) and tests the wrong thing.
        let rootPwd = await run("pwd")
        XCTAssertFalse(rootPwd.isError, rootPwd.output)
        let physicalRoot = rootPwd.output.trimmingCharacters(in: .whitespacesAndNewlines)

        let cd = await run("mkdir -p sub && cd sub")
        XCTAssertFalse(cd.isError, cd.output)

        let subPwd = await run("pwd")
        XCTAssertFalse(subPwd.isError, subPwd.output)
        XCTAssertEqual(
            subPwd.output.trimmingCharacters(in: .whitespacesAndNewlines),
            physicalRoot + "/sub",
            "the next call must start where the previous one left off")
    }

    func testLeavingTheProjectResetsTheWorkingDirectory() async throws {
        ShellCwdTracker.shared.resetForTests()
        defer { ShellCwdTracker.shared.resetForTests() }
        let project = temporaryProject()

        func run(_ command: String) async -> AppToolResult {
            await AppToolRegistry.execute(
                call: AppToolCall(name: "run_command", arguments: ["command": command], category: .terminal),
                in: project)
        }

        let rootPwd = await run("pwd")
        XCTAssertFalse(rootPwd.isError, rootPwd.output)
        let physicalRoot = rootPwd.output.trimmingCharacters(in: .whitespacesAndNewlines)

        let escape = await run("cd /tmp")
        XCTAssertFalse(escape.isError, escape.output)
        XCTAssertTrue(
            escape.output.contains("Shell cwd was reset"),
            "the model must be told its cd did not stick: \(escape.output)")

        let checkPwd = await run("pwd")
        XCTAssertFalse(checkPwd.isError, checkPwd.output)
        XCTAssertEqual(
            checkPwd.output.trimmingCharacters(in: .whitespacesAndNewlines),
            physicalRoot,
            "the next call must run at the project root again")
    }
}
