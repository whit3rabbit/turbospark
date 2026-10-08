import Foundation

/// Identity of a persistent MCP stdio session. Any field changing means a
/// different process must be spawned, so the key doubles as the config
/// fingerprint. It lives in memory only (env values may be secrets).
struct McpSessionKey: Hashable, Sendable {
    let serverName: String
    let command: String
    let args: [String]
    let env: [String: String]
    let envPassthrough: [String]
    let workingDirectory: String?

    var normalizedName: String { McpServerConfig.normalizedName(serverName) }
}

enum McpSessionError: Error {
    /// The session was already dead when the request was about to be written,
    /// so nothing reached the server and a fresh session may retry.
    case closedBeforeSend
}

/// Sleep source for the idle timer, injectable so tests need not wait minutes.
protocol McpSessionClock: Sendable {
    func sleep(seconds: TimeInterval) async throws
}

struct SystemMcpSessionClock: McpSessionClock {
    func sleep(seconds: TimeInterval) async throws {
        try await Task.sleep(nanoseconds: UInt64(max(0, seconds) * 1_000_000_000))
    }
}

/// Live session children, tracked outside the actors so the synchronous quit
/// path (`stopAllBackgroundWorkForShutdown`) can kill them without awaiting.
enum McpSessionProcessRegistry {
    private static let lock = NSLock()
    private static var pids = Set<pid_t>()

    static func add(_ pid: pid_t) {
        lock.lock(); pids.insert(pid); lock.unlock()
    }

    static func remove(_ pid: pid_t) {
        lock.lock(); pids.remove(pid); lock.unlock()
    }

    static var count: Int {
        lock.lock(); defer { lock.unlock() }
        return pids.count
    }

    static func killAllNow() {
        lock.lock()
        let snapshot = Array(pids)
        pids.removeAll()
        lock.unlock()
        for pid in snapshot { ProcessExecutor.killTreeNow(pid) }
    }
}

/// Resumes a continuation exactly once, whether the answer arrives before or
/// after the awaiting task installs it.
private final class McpResponseWaiter: @unchecked Sendable {
    private let lock = NSLock()
    private var continuation: CheckedContinuation<Data, Error>?
    private var result: Result<Data, Error>?

    func install(_ continuation: CheckedContinuation<Data, Error>) {
        lock.lock()
        if let result {
            lock.unlock()
            continuation.resume(with: result)
        } else {
            self.continuation = continuation
            lock.unlock()
        }
    }

    func fulfil(_ outcome: Result<Data, Error>) {
        lock.lock()
        guard result == nil else { lock.unlock(); return }
        result = outcome
        let waiting = continuation
        continuation = nil
        lock.unlock()
        waiting?.resume(with: outcome)
    }
}

/// One long-lived MCP stdio server process shared by every call to that
/// server. A single reader task owns stdout and routes each response to the
/// request that is waiting on its JSON-RPC id, so concurrent calls multiplex
/// over one pipe.
actor McpStdioSession {
    private struct Pending {
        let waiter: McpResponseWaiter
        let timeout: Task<Void, Never>
    }

    private let key: McpSessionKey
    private let workingDirectory: URL?
    private let idleTimeout: TimeInterval
    private let clock: any McpSessionClock

    // Closed state is read synchronously by the engine's get-or-create.
    private nonisolated let closedFlag = ClosedFlag()
    private final class ClosedFlag: @unchecked Sendable {
        private let lock = NSLock()
        private var value = false
        var isSet: Bool { lock.lock(); defer { lock.unlock() }; return value }
        func set() { lock.lock(); value = true; lock.unlock() }
    }
    nonisolated var isClosed: Bool { closedFlag.isSet }

    private var process: Process?
    private var stdin: Pipe?
    private var buffer: LineBuffer?
    private var stderrTail: StderrTail?
    private var signal: AsyncStream<Void>.Continuation?
    private var readerTask: Task<Void, Never>?
    private var startTask: Task<Void, Error>?
    private var started = false
    private var retiring = false
    private var nextID = 1
    private var pending: [Int: Pending] = [:]
    private var idleTask: Task<Void, Never>?
    private var idleGeneration = 0

    init(key: McpSessionKey, workingDirectory: URL?, idleTimeout: TimeInterval, clock: any McpSessionClock) {
        self.key = key
        self.workingDirectory = workingDirectory
        self.idleTimeout = idleTimeout
        self.clock = clock
    }

    // MARK: Lifecycle

    /// Spawns the child and completes the MCP handshake once; later callers
    /// share the same result.
    func ensureStarted(timeoutSeconds: TimeInterval) async throws {
        if closedFlag.isSet { throw McpSessionError.closedBeforeSend }
        if started { return }
        if startTask == nil {
            startTask = Task { try await self.performStart(timeoutSeconds: timeoutSeconds) }
        }
        try await startTask!.value
    }

    private func performStart(timeoutSeconds: TimeInterval) async throws {
        let (stream, continuation) = AsyncStream<Void>.makeStream(bufferingPolicy: .bufferingNewest(1))
        let lines = LineBuffer()
        lines.onActivity = { continuation.yield() }
        let spawned: (process: Process, stdinPipe: Pipe, stdoutBuffer: LineBuffer, stderrTail: StderrTail)
        do {
            spawned = try McpClientEngine.spawnStdioServer(
                serverName: key.serverName, command: key.command, args: key.args,
                env: key.env, envPassthrough: key.envPassthrough,
                workingDirectory: workingDirectory, errorCode: 2,
                stdoutBuffer: lines,
                onTermination: { continuation.yield() })
        } catch {
            continuation.finish()
            closedFlag.set()
            throw error
        }
        process = spawned.process
        stdin = spawned.stdinPipe
        buffer = spawned.stdoutBuffer
        stderrTail = spawned.stderrTail
        signal = continuation
        McpSessionProcessRegistry.add(spawned.process.processIdentifier)
        readerTask = Task { await self.runReader(stream) }

        do {
            let initData = try await request(
                method: "initialize",
                params: [
                    "protocolVersion": "2024-11-05",
                    "capabilities": ["tools": [:] as [String: Any]],
                    "clientInfo": ["name": "TurboSpark", "version": "1.0.0"]
                ],
                timeoutSeconds: timeoutSeconds)
            if let response = try? JSONSerialization.jsonObject(with: initData) as? [String: Any] {
                try McpClientEngine.throwIfRpcError(response, serverName: key.serverName, step: "initialize")
            }
            try send(["jsonrpc": "2.0", "method": "notifications/initialized", "params": [:] as [String: Any]])
            started = true
            scheduleIdleIfQuiet()
        } catch {
            teardown(reason: "the handshake failed")
            throw error
        }
    }

    /// Stops accepting new work; the process exits once in-flight calls finish.
    func retire() {
        retiring = true
        if pending.isEmpty { teardown(reason: "the server configuration changed") }
    }

    func close(reason: String) {
        teardown(reason: reason)
    }

    // MARK: Requests

    /// Sends one JSON-RPC request and returns the raw response line. Honors the
    /// timeout and Task cancellation without disturbing other in-flight calls.
    func request(method: String, params: [String: Any], timeoutSeconds: TimeInterval) async throws -> Data {
        try Task.checkCancellation()
        guard !closedFlag.isSet, process != nil else { throw McpSessionError.closedBeforeSend }
        idleTask?.cancel()
        idleGeneration += 1

        let id = nextID
        nextID += 1
        let waiter = McpResponseWaiter()
        let timeoutTask = Task { [weak self] in
            try? await Task.sleep(nanoseconds: UInt64(max(0, timeoutSeconds) * 1_000_000_000))
            guard !Task.isCancelled else { return }
            await self?.timeOut(id: id, seconds: timeoutSeconds)
        }
        pending[id] = Pending(waiter: waiter, timeout: timeoutTask)
        do {
            try send(["jsonrpc": "2.0", "id": id, "method": method, "params": params])
        } catch {
            pending.removeValue(forKey: id)
            timeoutTask.cancel()
            scheduleIdleIfQuiet()
            // A failed write means the pipe is gone: nothing was delivered.
            teardown(reason: "its stdin closed")
            throw McpSessionError.closedBeforeSend
        }
        return try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { waiter.install($0) }
        } onCancel: {
            Task { await self.cancel(id: id) }
        }
    }

    private func send(_ object: [String: Any]) throws {
        guard let pipe = stdin else { throw McpSessionError.closedBeforeSend }
        let data = try JSONSerialization.data(withJSONObject: object, options: [])
        // `write(contentsOf:)` throws on EPIPE; the legacy overload would raise
        // an uncatchable Objective-C exception and crash the app.
        try pipe.fileHandleForWriting.write(contentsOf: data + Data("\n".utf8))
    }

    private func cancel(id: Int) {
        guard let entry = pending.removeValue(forKey: id) else { return }
        entry.timeout.cancel()
        entry.waiter.fulfil(.failure(CancellationError()))
        try? send([
            "jsonrpc": "2.0", "method": "notifications/cancelled",
            "params": ["requestId": id, "reason": "cancelled by client"] as [String: Any]
        ])
        afterSettle()
    }

    private func timeOut(id: Int, seconds: TimeInterval) {
        guard let entry = pending.removeValue(forKey: id) else { return }
        let tail = stderrTail?.text ?? ""
        entry.waiter.fulfil(.failure(NSError(domain: "McpClientEngine", code: 5, userInfo: [
            NSLocalizedDescriptionKey: "Timed out waiting for MCP response (id: \(id))."
                + (tail.isEmpty ? "" : " Server stderr: \(tail)")])))
        try? send([
            "jsonrpc": "2.0", "method": "notifications/cancelled",
            "params": ["requestId": id, "reason": "timed out after \(Int(seconds))s"] as [String: Any]
        ])
        afterSettle()
    }

    // MARK: Reader

    private func runReader(_ stream: AsyncStream<Void>) async {
        for await _ in stream {
            drainLines()
            guard let process, let buffer else { break }
            if buffer.isClosed || !process.isRunning {
                // The exit can be observed a few ticks before the last stdout
                // chunk is delivered by the handler thread; drain once more.
                if !buffer.isClosed {
                    try? await Task.sleep(nanoseconds: 150_000_000)
                    drainLines()
                }
                handleDeath()
                break
            }
        }
    }

    private func drainLines() {
        guard let buffer else { return }
        while let line = buffer.popLine() { handleLine(line) }
    }

    private func handleLine(_ line: Data) {
        guard let json = try? JSONSerialization.jsonObject(with: line) as? [String: Any] else { return }
        let id = json["id"] as? Int
        if let method = json["method"] as? String {
            // A request FROM the server. `ping` is answered; anything else gets
            // method-not-found so the server is not left waiting on this client.
            guard let id else { return }
            if method == "ping" {
                try? send(["jsonrpc": "2.0", "id": id, "result": [:] as [String: Any]])
            } else {
                try? send(["jsonrpc": "2.0", "id": id,
                           "error": ["code": -32601, "message": "Method not found"] as [String: Any]])
            }
            return
        }
        guard let id, let entry = pending.removeValue(forKey: id) else { return }
        entry.timeout.cancel()
        entry.waiter.fulfil(.success(line))
        afterSettle()
    }

    // MARK: Death, teardown, idle

    private func handleDeath() {
        // `terminationStatus` raises an Objective-C exception while the child
        // is still running (stdout can reach EOF, or a kill be mid-flight,
        // before the exit is reaped), so only read it once it has exited.
        let statusText: String
        if let process, !process.isRunning {
            statusText = "exited (status \(process.terminationStatus))"
        } else {
            statusText = "closed its output"
        }
        let tail = stderrTail?.text ?? ""
        let message = "MCP server '\(key.serverName)' \(statusText) before answering."
            + (tail.isEmpty ? "" : " stderr: \(tail)")
        failAll(NSError(domain: "McpClientEngine", code: 7, userInfo: [NSLocalizedDescriptionKey: message]))
        teardown(reason: "it exited")
    }

    private func failAll(_ error: Error) {
        let all = pending
        pending.removeAll()
        for (_, entry) in all {
            entry.timeout.cancel()
            entry.waiter.fulfil(.failure(error))
        }
    }

    private func teardown(reason: String) {
        let wasClosed = closedFlag.isSet
        closedFlag.set()
        idleTask?.cancel()
        idleGeneration += 1
        if !pending.isEmpty {
            let tail = stderrTail?.text ?? ""
            failAll(NSError(domain: "McpClientEngine", code: 7, userInfo: [
                NSLocalizedDescriptionKey: "MCP server '\(key.serverName)' session closed: \(reason)."
                    + (tail.isEmpty ? "" : " stderr: \(tail)")]))
        }
        guard !wasClosed else { return }
        signal?.finish()
        signal = nil
        readerTask = nil
        if let process {
            McpSessionProcessRegistry.remove(process.processIdentifier)
            // The ladder sleeps (SIGTERM, then SIGKILL after 2 s); keep it off
            // the actor so other sessions are not stalled.
            if process.isRunning {
                DispatchQueue.global(qos: .utility).async { ProcessExecutor.terminateAndReap(process) }
            }
        }
        stdin = nil
    }

    private func afterSettle() {
        guard pending.isEmpty else { return }
        if retiring { teardown(reason: "the server configuration changed"); return }
        scheduleIdleIfQuiet()
    }

    private func scheduleIdleIfQuiet() {
        guard pending.isEmpty, !closedFlag.isSet, started || process != nil else { return }
        idleTask?.cancel()
        idleGeneration += 1
        let generation = idleGeneration
        let clock = self.clock
        let timeout = idleTimeout
        idleTask = Task { [weak self] in
            do { try await clock.sleep(seconds: timeout) } catch { return }
            await self?.idleFired(generation: generation)
        }
    }

    private func idleFired(generation: Int) {
        // A request started (or another idle timer replaced this one) since.
        guard generation == idleGeneration, pending.isEmpty else { return }
        teardown(reason: "it was idle")
    }
}
