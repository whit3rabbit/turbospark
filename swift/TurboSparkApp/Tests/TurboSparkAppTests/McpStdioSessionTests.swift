import XCTest
@testable import TurboSparkApp

/// Hand-advanced idle clock: `sleep` suspends until `advance` passes its deadline.
private final class ManualClock: McpSessionClock, @unchecked Sendable {
    private struct Waiter {
        let deadline: TimeInterval
        let continuation: CheckedContinuation<Void, Error>
    }
    private let lock = NSLock()
    private var now: TimeInterval = 0
    private var waiters: [UUID: Waiter] = [:]

    var waiterCount: Int { lock.lock(); defer { lock.unlock() }; return waiters.count }

    func sleep(seconds: TimeInterval) async throws {
        let id = UUID()
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { (cont: CheckedContinuation<Void, Error>) in
                lock.lock()
                waiters[id] = Waiter(deadline: now + seconds, continuation: cont)
                lock.unlock()
            }
        } onCancel: {
            lock.lock()
            let waiter = waiters.removeValue(forKey: id)
            lock.unlock()
            waiter?.continuation.resume(throwing: CancellationError())
        }
    }

    func advance(by seconds: TimeInterval) {
        lock.lock()
        now += seconds
        let due = waiters.filter { $0.value.deadline <= now }
        for id in due.keys { waiters.removeValue(forKey: id) }
        lock.unlock()
        for (_, waiter) in due { waiter.continuation.resume() }
    }
}

final class McpStdioSessionTests: XCTestCase {
    private var dir: URL!
    private var script: URL!
    private var log: URL!

    /// A minimal MCP server. Each request is handled on its own thread so
    /// slow calls overlap; every spawn and every `notifications/cancelled` is
    /// appended to MCP_LOG so tests can count processes without timing.
    private static let serverSource = """
        import sys, os, json, time, threading, subprocess
        log = os.environ["MCP_LOG"]
        log_lock = threading.Lock()
        def note(s):
            with log_lock:
                with open(log, "a") as f: f.write(s + "\\n")
        note("spawn %d" % os.getpid())
        out = threading.Lock()
        def send(o):
            with out:
                sys.stdout.write(json.dumps(o) + "\\n"); sys.stdout.flush()
        def handle(m):
            if m["method"] == "initialize":
                send({"jsonrpc": "2.0", "id": m["id"], "result": {"protocolVersion": "2024-11-05", "capabilities": {}, "serverInfo": {"name": "fake", "version": "1"}}})
            elif m["method"] == "tools/call":
                p = m["params"]; a = p.get("arguments", {})
                if p["name"] == "crash":
                    sys.stderr.write("boom from fake server\\n"); sys.stderr.flush()
                    os._exit(3)
                if p["name"] == "slow":
                    marker = a.get("marker")
                    if marker:
                        note("started %s" % marker)
                        code = "import pathlib,sys,time; time.sleep(float(sys.argv[2])); pathlib.Path(sys.argv[1]).write_text('side effect')"
                        subprocess.Popen([sys.executable, "-c", code, marker, str(a.get("sideEffectDelay", 1))])
                        note("spawned child %s" % marker)
                    time.sleep(float(a.get("delay", 1)))
                send({"jsonrpc": "2.0", "id": m["id"], "result": {"content": [{"type": "text", "text": "%s:%s" % (os.getpid(), a.get("text", ""))}]}})
        for line in sys.stdin:
            m = json.loads(line)
            if m.get("method") == "notifications/cancelled":
                note("cancelled %s" % m["params"]["requestId"])
            elif "id" in m:
                threading.Thread(target=handle, args=(m,), daemon=True).start()
        """

    override func setUpWithError() throws {
        dir = FileManager.default.temporaryDirectory
            .appendingPathComponent("mcp-session-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        script = dir.appendingPathComponent("server.py")
        try Self.serverSource.write(to: script, atomically: true, encoding: .utf8)
        log = dir.appendingPathComponent("log.txt")
    }

    override func tearDown() async throws {
        await McpClientEngine.shared.shutdownAllSessions()
        try? FileManager.default.removeItem(at: dir)
    }

    private func config(name: String = "fake", extraArg: String? = nil) -> McpServerConfig {
        McpServerConfig(
            name: name,
            transport: .stdio(
                command: "/usr/bin/python3",
                args: ["-u", script.path] + (extraArg.map { [$0] } ?? []),
                env: ["MCP_LOG": log.path]))
    }

    private func logLines() -> [String] {
        ((try? String(contentsOf: log, encoding: .utf8)) ?? "")
            .split(separator: "\n").map(String.init)
    }

    private var spawnCount: Int { logLines().filter { $0.hasPrefix("spawn ") }.count }

    private func eventually(_ what: String, timeout: TimeInterval = 5, _ condition: () async -> Bool) async {
        let deadline = Date().addingTimeInterval(timeout)
        while Date() < deadline {
            if await condition() { return }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
        XCTFail("timed out waiting for: \(what)")
    }

    func testSequentialCallsReuseOneProcess() async throws {
        let engine = McpClientEngine()
        defer { Task { await engine.shutdownAllSessions() } }
        var outputs: [String] = []
        for i in 0..<5 {
            outputs.append(try await engine.callTool(
                config: config(), toolName: "echo", arguments: ["text": "n\(i)"], timeoutSeconds: 10))
        }
        XCTAssertEqual(spawnCount, 1, "five calls must share one server process")
        XCTAssertEqual(outputs.map { String($0.split(separator: ":").last!) }, ["n0", "n1", "n2", "n3", "n4"])
        XCTAssertEqual(Set(outputs.map { $0.split(separator: ":").first! }).count, 1, "one pid answered all calls")
    }

    func testConcurrentCallsMultiplexOverOneSession() async throws {
        let engine = McpClientEngine()
        defer { Task { await engine.shutdownAllSessions() } }
        let cfg = config()
        let start = Date()
        let results = try await withThrowingTaskGroup(of: String.self) { group in
            for i in 0..<4 {
                group.addTask {
                    try await engine.callTool(
                        config: cfg, toolName: "slow",
                        arguments: ["delay": 0.5, "text": "c\(i)"], timeoutSeconds: 10)
                }
            }
            var all: [String] = []
            for try await r in group { all.append(r) }
            return all
        }
        let elapsed = Date().timeIntervalSince(start)
        XCTAssertEqual(Set(results.map { String($0.split(separator: ":").last!) }), ["c0", "c1", "c2", "c3"],
                       "each response must reach the call that asked for it")
        XCTAssertEqual(spawnCount, 1)
        // Serialized, four 0.5s calls need 2s plus the handshake.
        XCTAssertLessThan(elapsed, 1.9, "calls must overlap on one session")
    }

    func testCrashFailsTheCallWithStderrAndTheNextCallRestarts() async throws {
        let engine = McpClientEngine()
        defer { Task { await engine.shutdownAllSessions() } }
        do {
            _ = try await engine.callTool(config: config(), toolName: "crash", arguments: [:], timeoutSeconds: 10)
            XCTFail("a crashing server must fail the call")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("boom from fake server"),
                          "the error must carry the stderr tail: \(error.localizedDescription)")
        }
        let output = try await engine.callTool(
            config: config(), toolName: "echo", arguments: ["text": "again"], timeoutSeconds: 10)
        XCTAssertTrue(output.hasSuffix(":again"))
        XCTAssertEqual(spawnCount, 2, "the crashed session is replaced, not reused")
    }

    func testConfigChangeRetiresTheOldSession() async throws {
        let engine = McpClientEngine()
        defer { Task { await engine.shutdownAllSessions() } }
        _ = try await engine.callTool(config: config(), toolName: "echo", arguments: [:], timeoutSeconds: 10)
        _ = try await engine.callTool(config: config(extraArg: "changed"), toolName: "echo", arguments: [:], timeoutSeconds: 10)
        XCTAssertEqual(spawnCount, 2, "a changed config must spawn a new process")
        await eventually("old session retired") { await engine.sessionCount == 1 }
    }

    func testChangedWorkingDirectoryIsADifferentSession() async throws {
        let engine = McpClientEngine()
        defer { Task { await engine.shutdownAllSessions() } }
        _ = try await engine.callTool(config: config(), toolName: "echo", arguments: [:], workingDirectory: dir, timeoutSeconds: 10)
        _ = try await engine.callTool(config: config(), toolName: "echo", arguments: [:], workingDirectory: dir, timeoutSeconds: 10)
        XCTAssertEqual(spawnCount, 1)
        let other = dir.appendingPathComponent("sub")
        try FileManager.default.createDirectory(at: other, withIntermediateDirectories: true)
        _ = try await engine.callTool(config: config(), toolName: "echo", arguments: [:], workingDirectory: other, timeoutSeconds: 10)
        XCTAssertEqual(spawnCount, 2, "a project change must not reuse the old project's server")
    }

    func testInvalidateKillsTheProcessAndTheNextCallRespawns() async throws {
        let engine = McpClientEngine()
        defer { Task { await engine.shutdownAllSessions() } }
        let first = try await engine.callTool(config: config(), toolName: "echo", arguments: [:], timeoutSeconds: 10)
        let pid = pid_t(first.split(separator: ":").first.flatMap { Int($0) } ?? 0)
        XCTAssertGreaterThan(pid, 0)
        await engine.invalidateSessions(serverName: "FAKE")
        await eventually("child exited") { kill(pid, 0) != 0 }
        _ = try await engine.callTool(config: config(), toolName: "echo", arguments: [:], timeoutSeconds: 10)
        XCTAssertEqual(spawnCount, 2)
    }

    func testCancellingCallKillsSessionTreeAndFailsOtherCalls() async throws {
        let engine = McpClientEngine()
        defer { Task { await engine.shutdownAllSessions() } }
        let cfg = config()
        let first = try await engine.callTool(config: cfg, toolName: "echo", arguments: [:], timeoutSeconds: 10)
        let originalPID = pid_t(first.split(separator: ":").first.flatMap { Int($0) } ?? 0)
        XCTAssertGreaterThan(originalPID, 0)
        let firstMarker = dir.appendingPathComponent("cancelled-side-effect.txt")
        let secondMarker = dir.appendingPathComponent("sibling-side-effect.txt")
        let slow = Task {
            try await engine.callTool(config: cfg, toolName: "slow", arguments: [
                "delay": 2, "text": "cancelled", "marker": firstMarker.path, "sideEffectDelay": 0.8
            ], timeoutSeconds: 10)
        }
        let sibling = Task {
            try await engine.callTool(config: cfg, toolName: "slow", arguments: [
                "delay": 2, "text": "sibling", "marker": secondMarker.path, "sideEffectDelay": 0.8
            ], timeoutSeconds: 10)
        }
        await eventually("both calls reached the MCP server") {
            let lines = self.logLines()
            return lines.contains("spawned child \(firstMarker.path)")
                && lines.contains("spawned child \(secondMarker.path)")
        }
        slow.cancel()
        do {
            _ = try await slow.value
            XCTFail("a cancelled call must throw")
        } catch is CancellationError {
        } catch {
            XCTFail("expected CancellationError, got \(error)")
        }
        do {
            _ = try await sibling.value
            XCTFail("terminating a shared session must fail its other in-flight calls")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("session closed"), error.localizedDescription)
        }
        try await Task.sleep(nanoseconds: 1_200_000_000)
        XCTAssertFalse(FileManager.default.fileExists(atPath: firstMarker.path), "cancelled work must not finish later")
        XCTAssertFalse(FileManager.default.fileExists(atPath: secondMarker.path), "sibling work must not outlive its session")
        await eventually("cancelled server process exited") { kill(originalPID, 0) != 0 }

        let output = try await engine.callTool(config: cfg, toolName: "echo", arguments: ["text": "after"], timeoutSeconds: 10)
        XCTAssertTrue(output.hasSuffix(":after"))
        let replacementPID = pid_t(output.split(separator: ":").first.flatMap { Int($0) } ?? 0)
        XCTAssertNotEqual(replacementPID, originalPID)
        XCTAssertEqual(spawnCount, 2, "a cancelled session must be replaced on the next call")
    }

    func testTimeoutKillsSessionTreeAndTheNextCallRespawns() async throws {
        let engine = McpClientEngine()
        defer { Task { await engine.shutdownAllSessions() } }
        let cfg = config()
        let first = try await engine.callTool(config: cfg, toolName: "echo", arguments: [:], timeoutSeconds: 10)
        let originalPID = pid_t(first.split(separator: ":").first.flatMap { Int($0) } ?? 0)
        XCTAssertGreaterThan(originalPID, 0)
        let marker = dir.appendingPathComponent("timeout-side-effect.txt")
        let timedOut = Task {
            try await engine.callTool(config: cfg, toolName: "slow", arguments: [
                "delay": 2, "marker": marker.path, "sideEffectDelay": 0.8
            ], timeoutSeconds: 0.4)
        }
        await eventually("timeout side-effect subprocess spawned") {
            self.logLines().contains("spawned child \(marker.path)")
        }
        do {
            _ = try await timedOut.value
            XCTFail("expected a timeout")
        } catch {
            XCTAssertTrue(error.localizedDescription.contains("Timed out"), error.localizedDescription)
        }
        try await Task.sleep(nanoseconds: 1_200_000_000)
        XCTAssertFalse(FileManager.default.fileExists(atPath: marker.path), "timed-out work must not finish later")
        await eventually("timed-out server process exited") { kill(originalPID, 0) != 0 }

        let output = try await engine.callTool(config: cfg, toolName: "echo", arguments: ["text": "ok"], timeoutSeconds: 10)
        XCTAssertTrue(output.hasSuffix(":ok"))
        let replacementPID = pid_t(output.split(separator: ":").first.flatMap { Int($0) } ?? 0)
        XCTAssertNotEqual(replacementPID, originalPID)
        XCTAssertEqual(spawnCount, 2)
    }

    func testIdleSessionIsTornDownAndRespawnedOnDemand() async throws {
        let clock = ManualClock()
        let engine = McpClientEngine(idleTimeout: 300, clock: clock)
        defer { Task { await engine.shutdownAllSessions() } }
        let cfg = config()
        _ = try await engine.callTool(config: cfg, toolName: "echo", arguments: [:], timeoutSeconds: 10)
        await eventually("idle timer armed") { clock.waiterCount > 0 }

        clock.advance(by: 299)
        try await Task.sleep(nanoseconds: 100_000_000)
        let live = await engine.sessionCount
        XCTAssertEqual(live, 1, "not idle long enough yet")

        clock.advance(by: 2)
        await eventually("idle teardown") { await engine.sessionCount == 0 }
        _ = try await engine.callTool(config: cfg, toolName: "echo", arguments: [:], timeoutSeconds: 10)
        XCTAssertEqual(spawnCount, 2)
    }

    func testAnInFlightCallPreventsIdleTeardown() async throws {
        let clock = ManualClock()
        let engine = McpClientEngine(idleTimeout: 300, clock: clock)
        defer { Task { await engine.shutdownAllSessions() } }
        let cfg = config()
        _ = try await engine.callTool(config: cfg, toolName: "echo", arguments: [:], timeoutSeconds: 10)
        await eventually("idle timer armed") { clock.waiterCount > 0 }
        let slow = Task {
            try await engine.callTool(config: cfg, toolName: "slow", arguments: ["delay": 0.6, "text": "x"], timeoutSeconds: 10)
        }
        try await Task.sleep(nanoseconds: 200_000_000)
        clock.advance(by: 1000)
        let output = try await slow.value
        XCTAssertTrue(output.hasSuffix(":x"), "the stale idle timer must not kill a busy session")
        XCTAssertEqual(spawnCount, 1)
    }

    func testShutdownKillsEveryTrackedChild() async throws {
        let engine = McpClientEngine()
        let first = try await engine.callTool(config: config(), toolName: "echo", arguments: [:], timeoutSeconds: 10)
        let pid = pid_t(first.split(separator: ":").first.flatMap { Int($0) } ?? 0)
        XCTAssertGreaterThan(pid, 0)
        XCTAssertGreaterThan(McpSessionProcessRegistry.count, 0)
        McpClientEngine.killAllSessionProcessesNow()
        await eventually("child killed") { kill(pid, 0) != 0 }
        XCTAssertEqual(McpSessionProcessRegistry.count, 0)
    }
}
