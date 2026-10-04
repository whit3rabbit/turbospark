import Foundation

/// Supervises one REPL worker child process (task 3.1). The worker is the
/// packaged app executable launched with the internal worker arguments; it
/// speaks newline-delimited JSON on stdin and stdout: one init line at
/// startup carrying the limits and artifact directory, then one evaluate
/// request per line with exactly one result per request.
///
/// Serialization contract (3.5): the worker's serve loop evaluates one
/// request at a time in write order, and concurrent `evaluate` calls are
/// FIFO-ordered by the mailbox before their lines are written, so a later
/// call for the same session executes only after the earlier call finishes
/// and every caller receives exactly its own result. Scripts over the
/// configured request-size bound are rejected locally without reaching the
/// worker. Termination semantics (deadline, cancellation, join-before-
/// replacement, session reset) belong to task 3.2; here `terminateAndWait`
/// only stops the child.
final class REPLWorkerProcess: @unchecked Sendable {
    private struct InitMessage: Codable {
        var type = "init"
        var limits: REPLLimits
        var artifactDirectory: URL
    }

    private struct EvaluateMessage: Codable {
        var type = "evaluate"
        var id: Int
        var code: String
        var settlementTimeout: TimeInterval
    }

    private struct ResultMessage: Codable {
        var id: Int
        var result: REPLCallResult
    }

    private struct PendingRequest {
        var line: Data
        var id: Int
        var continuation: CheckedContinuation<REPLCallResult, Never>
    }

    static let serveArguments = [REPLWorkerMain.workerModeArgument, REPLWorkerMain.serveArgument]

    /// The packaged app executable, which re-enters itself in worker mode.
    /// Tests inject an explicit build product instead.
    static func defaultExecutableURL() -> URL? {
        Bundle.main.executableURL
    }

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "com.turbospark.repl.worker-write")
    private let lock = NSLock()
    private var pending: [Int: CheckedContinuation<REPLCallResult, Never>] = [:]
    private var nextIdentifier = 0
    private var didTerminate = false
    private var stdoutBuffer = Data()
    private let limits: REPLLimits
    private var mailboxContinuation: AsyncStream<PendingRequest>.Continuation?

    init(
        executableURL: URL,
        configuration: REPLSessionConfiguration,
        limits: REPLLimits = REPLLimits()
    ) throws {
        self.limits = limits
        process.executableURL = executableURL
        process.arguments = Self.serveArguments
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = FileHandle.nullDevice

        let (mailbox, continuation) = AsyncStream<PendingRequest>.makeStream()
        self.mailboxContinuation = continuation

        try process.run()

        // A closed pipe must surface as an error, not kill the supervisor.
        signal(SIGPIPE, SIG_IGN)

        let encoder = JSONEncoder()
        let initLine = try encoder.encode(InitMessage(
            limits: limits,
            artifactDirectory: configuration.artifactDirectory))
        try writeLine(initLine)

        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else {
                self?.handleReaderClose(handle)
                return
            }
            self?.receive(chunk)
        }
        process.terminationHandler = { [weak self] _ in
            self?.handleTermination()
        }

        Task { [weak self] in
            for await request in mailbox {
                self?.perform(request)
            }
        }
    }

    var isRunning: Bool {
        process.isRunning
    }

    /// Sends one bounded evaluate request and resolves with exactly one
    /// result. Concurrent callers are FIFO-serialized by the mailbox and
    /// never overlap inside the worker.
    func evaluate(
        code: String,
        settlementTimeout: TimeInterval = REPLWorkerContext.defaultSettlementDeadline
    ) async -> REPLCallResult {
        let bytes = Data(code.utf8)
        guard bytes.count <= limits.maximumRequestBytes else {
            return Self.localFailure(
                "script of \(bytes.count) bytes exceeds the "
                    + "\(limits.maximumRequestBytes) byte request limit")
        }

        let identifier = lock.withLock { () -> Int in
            nextIdentifier += 1
            return nextIdentifier
        }
        guard let line = try? JSONEncoder().encode(EvaluateMessage(
            id: identifier,
            code: code,
            settlementTimeout: settlementTimeout))
        else {
            return Self.localFailure("the evaluate request could not be encoded")
        }

        lock.lock()
        if didTerminate || !process.isRunning {
            lock.unlock()
            return Self.localFailure("the worker process is not running")
        }
        let mailbox = mailboxContinuation
        lock.unlock()
        guard let mailbox else {
            return Self.localFailure("the worker supervisor is shut down")
        }

        return await withCheckedContinuation { continuation in
            mailbox.yield(PendingRequest(
                line: line,
                id: identifier,
                continuation: continuation))
        }
    }

    /// Stops the child: stdin closes so the serve loop exits on EOF, with a
    /// short grace period before SIGTERM. All parked callers fail fast.
    func terminateAndWait(grace: TimeInterval = 2) {
        let shouldStop = lock.withLock { () -> Bool in
            if didTerminate { return false }
            didTerminate = true
            return true
        }
        guard shouldStop else { return }

        mailboxContinuation?.finish()
        failAllPending()
        stdoutPipe.fileHandleForReading.readabilityHandler = nil
        try? stdinPipe.fileHandleForWriting.close()

        if process.isRunning {
            let deadline = Date().addingTimeInterval(grace)
            while process.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.02)
            }
            if process.isRunning {
                process.terminate()
            }
        }
        process.waitUntilExit()
        process.terminationHandler = nil
    }

    deinit {
        if !didTerminate {
            process.terminationHandler = nil
            if process.isRunning {
                process.terminate()
            }
        }
    }

    // MARK: Internals

    /// Sends one request and hands the caller's continuation to the
    /// response reader. The continuation is registered before the line is
    /// written so a fast worker reply can never race past registration.
    private func perform(_ request: PendingRequest) {
        lock.lock()
        if didTerminate {
            lock.unlock()
            request.continuation.resume(returning: Self.localFailure(
                "the worker terminated before the evaluation returned"))
            return
        }
        pending[request.id] = request.continuation
        lock.unlock()

        do {
            try writeLine(request.line)
        } catch {
            lock.lock()
            let parked = pending.removeValue(forKey: request.id)
            lock.unlock()
            parked?.resume(returning: Self.localFailure(
                "the request could not be delivered to the worker: "
                    + error.localizedDescription))
        }
    }

    private func receive(_ chunk: Data) {
        var lines: [Data] = []
        lock.lock()
        stdoutBuffer.append(chunk)
        while let newlineIndex = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = Data(stdoutBuffer[stdoutBuffer.startIndex..<newlineIndex])
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newlineIndex)
            lines.append(lineData)
        }
        lock.unlock()

        for line in lines where !line.isEmpty {
            guard let message = try? JSONDecoder().decode(
                ResultMessage.self, from: line)
            else { continue }
            lock.lock()
            var continuation = pending.removeValue(forKey: message.id)
            if continuation == nil, let oldest = pending.keys.min() {
                // A line-level worker failure (transport bound, malformed
                // line, unknown type) carries no request id. The worker
                // serves one request at a time in write order, so the
                // oldest pending caller is the best-effort match and must
                // never stay parked on an unmatched id.
                continuation = pending.removeValue(forKey: oldest)
            }
            lock.unlock()
            continuation?.resume(returning: message.result)
        }
    }

    private func handleReaderClose(_ handle: FileHandle) {
        handle.readabilityHandler = nil
    }

    private func handleTermination() {
        failAllPending()
    }

    private func failAllPending() {
        lock.lock()
        let continuations = Array(pending.values)
        pending.removeAll()
        lock.unlock()
        for continuation in continuations {
            continuation.resume(returning: Self.localFailure(
                "the worker process exited before the evaluation returned"))
        }
    }

    private func writeLine(_ line: Data) throws {
        var framed = line
        framed.append(0x0A)
        try writeQueue.sync {
            try stdinPipe.fileHandleForWriting.write(contentsOf: framed)
        }
    }

    private static func localFailure(_ text: String) -> REPLCallResult {
        REPLCallResult(
            status: .failed,
            outputEvents: [],
            consoleText: "",
            errorText: text,
            completionText: nil,
            images: [],
            truncated: false,
            sessionCreated: false,
            sessionReset: false)
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}
