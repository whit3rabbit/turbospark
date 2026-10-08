import Foundation

/// Host-side supervisor for one codemode execution: launches the packaged
/// executable in codemode serve mode (one process per execution, the same
/// isolation pi gets from one worker per `execute()`), streams output items
/// back, routes nested-call lines to the injected call handler, and owns
/// the deadline. Timeout and cancellation terminate the process with
/// `SIGTERM`, which is the only reliable preempt for a script spinning in
/// JavaScriptCore (there is no public in-process interrupt API).
///
/// The call handler receives the tool's advertised name and its JSON
/// arguments string, and returns the tool's text output or an error
/// message. Payloads are JSON-encoded onto the wire here, keeping the
/// worker boundary JSON-strings-only: the supervisor never builds
/// structured values from worker data.
final class CodemodeWorkerSupervisor: @unchecked Sendable {
    /// Executes one re-gated nested call. `(name, argumentsJSON)`. Success
    /// carries the tool's text output; failure carries the rejection
    /// message the script sees.
    typealias CallHandler = @Sendable (String, String?) async -> Result<String, CodemodeCallError>

    private let process = Process()
    private let stdinPipe = Pipe()
    private let stdoutPipe = Pipe()
    private let stderrPipe = Pipe()
    private let writeQueue = DispatchQueue(label: "com.turbospark.codemode.worker-write")
    private let lock = NSLock()
    private let request: CodemodeWire.RunRequest
    private let callHandler: CallHandler
    private let timeout: TimeInterval
    private let limits: CodemodeLimits

    /// A nested call the host has accepted and not yet answered. The task is
    /// kept so the call can be CANCELLED when the script ends without
    /// awaiting it (or is killed): `McpClientEngine` honors task cancellation
    /// and reaps its server process, so a cancelled call stops instead of
    /// completing host-side after the model was told it was cancelled.
    private struct PendingCall {
        var name: String
        var startDate: Date
        var task: Task<Void, Never>?
    }

    private var pendingCalls: [Int: PendingCall] = [:]
    /// Calls accepted over the whole run, against `limits.maximumCalls`.
    private var acceptedCallCount = 0
    /// Calls currently holding a concurrency slot, against
    /// `limits.maximumConcurrentCalls`.
    private var activeCallCount = 0
    private var calls: [CodemodeCallRecord] = []
    private var outputs: [CodemodeOutputItem] = []
    private var done: CodemodeWire.Done?
    private var crashMessage: String?
    private var didTerminate = false
    private var stdoutBuffer = Data()
    /// Set by the reader thread once it has read the worker's stdout to EOF
    /// and handed every line to `receive`. A worker that exited is only
    /// treated as having died without a result after this is true.
    private var stdoutClosed = false
    /// The last few KiB of the worker's stderr and whether it hit EOF. A
    /// worker that dies before reporting (a JavaScriptCore abort, an
    /// out-of-memory kill) says why only here, so it is kept and attached to
    /// the "worker exited" failure instead of being sent to /dev/null.
    private var stderrTail = Data()
    private var stderrClosed = false
    static let maximumStderrTailBytes = 4_096
    /// When `completedResult` first saw the worker gone, to bound the wait
    /// for EOF if something else (a grandchild) holds the pipe open.
    private var workerExitObservedAt: Date?

    /// A broken pipe must surface as a write error, not kill the app with
    /// SIGPIPE. Process-global, so it is set once, not per supervisor.
    private static let ignoreBrokenPipes: Void = {
        signal(SIGPIPE, SIG_IGN)
    }()

    init(
        request: CodemodeWire.RunRequest,
        executableURL: URL,
        arguments: [String]? = nil,
        callHandler: @escaping CallHandler,
        timeout: TimeInterval,
        limits: CodemodeLimits
    ) throws {
        self.request = request
        self.callHandler = callHandler
        self.timeout = timeout
        self.limits = limits
        process.executableURL = executableURL
        // Injectable so a test can run a fake worker as `/bin/sh -c '...'`. A
        // fresh script FILE would be held at first exec while macOS assesses
        // it, which under load took minutes and made those tests flaky.
        process.arguments = arguments
            ?? [REPLWorkerMain.workerModeArgument, REPLWorkerMain.serveCodemodeArgument]
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = stderrPipe
        // The worker is this app's own binary and a script never needs the
        // app's environment: API keys and tokens in the launching shell
        // would otherwise sit in the process that runs model-written code.
        // Same allowlist (and the same helper) MCP children get.
        process.environment = McpClientEngine.childEnvironment(
            parent: ProcessInfo.processInfo.environment, passthrough: [], declared: [:])

        _ = Self.ignoreBrokenPipes
        try process.run()

        if let line = CodemodeWire.encodeLine(request) {
            writeLine(line)
        }

        // Deliberately present though it does nothing: the first version of the
        // stdout fix dropped it, and under CPU load threads then sat in
        // `waitUntilExit` for minutes after their child was gone. Exit
        // detection stays on the same path the committed code used. Reading
        // stdout is the reader thread's job, never this handler's.
        process.terminationHandler = { _ in }
        startReader()
        startStderrReader()
    }

    /// Keeps the last `maximumStderrTailBytes` of the worker's stderr. A
    /// dedicated blocking reader like stdout's, and for the same reason: it
    /// cannot lose the tail the way a handler cleared at termination can.
    private func startStderrReader() {
        let handle = stderrPipe.fileHandleForReading
        let thread = Thread { [weak self] in
            var chunk = [UInt8](repeating: 0, count: 4_096)
            while true {
                let count = read(handle.fileDescriptor, &chunk, chunk.count)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { break }
                self?.appendStderr(Data(chunk[0..<count]))
            }
            guard let self else { return }
            self.lock.withLock { self.stderrClosed = true }
        }
        thread.name = "com.turbospark.codemode.worker-stderr"
        thread.qualityOfService = .utility
        thread.start()
    }

    private func appendStderr(_ data: Data) {
        lock.withLock {
            stderrTail.append(data)
            if stderrTail.count > Self.maximumStderrTailBytes {
                stderrTail.removeFirst(stderrTail.count - Self.maximumStderrTailBytes)
            }
        }
    }

    /// Reads the worker's stdout on one dedicated thread until EOF.
    ///
    /// **THIS REPLACES A `readabilityHandler` PLUS A `terminationHandler`
    /// THAT CLEARED IT.** The worker writes its terminal `done` line and
    /// exits at once; the termination handler could fire before the handler
    /// had read that line, so a script that SUCCEEDED was reported as "the
    /// worker exited before the script finished" (about 1 run in 10 under CPU
    /// load, reproduced on the committed sources). A blocking read to EOF
    /// cannot lose the tail: the kernel keeps the bytes until they are read,
    /// and EOF arrives only after the last of them. The thread captures the
    /// read handle so its descriptor cannot be closed and reused underneath
    /// it.
    private func startReader() {
        let handle = stdoutPipe.fileHandleForReading
        let thread = Thread { [weak self] in
            var chunk = [UInt8](repeating: 0, count: 65_536)
            while true {
                let count = read(handle.fileDescriptor, &chunk, chunk.count)
                if count < 0 && errno == EINTR { continue }
                guard count > 0 else { break }
                self?.receive(Data(chunk[0..<count]))
            }
            guard let self else { return }
            self.lock.withLock { self.stdoutClosed = true }
        }
        thread.name = "com.turbospark.codemode.worker-read"
        thread.qualityOfService = .utility
        thread.start()
    }

    var isRunning: Bool {
        process.isRunning
    }

    /// Runs to completion, the deadline, or cancellation of the enclosing
    /// task, and assembles the final result. Partial output and completed
    /// call records survive a timeout or an abort; calls still in flight
    /// when the script ends are reported `cancelled`.
    func run() async -> CodemodeResult {
        let deadline = Date().addingTimeInterval(timeout)
        var tick = 0
        while true {
            if Task.isCancelled {
                return finish(kind: .aborted, message: "the codemode run was cancelled.")
            }
            if let result = completedResult() {
                return result
            }
            // About every 100 ms; the syscall is cheap but there is no reason
            // to make it on every 20 ms tick.
            tick += 1
            if tick % 5 == 0, let breach = memoryBreachMessage() {
                return finish(kind: .sandbox, message: breach)
            }
            if Date() >= deadline {
                return finish(
                    kind: .timeout,
                    message: String(
                        format: "the script exceeded its %.0f second deadline and was terminated.",
                        timeout))
            }
            try? await Task.sleep(nanoseconds: 20_000_000)
        }
    }

    /// Stops the child: stdin closes so the serve loop exits on EOF, with a
    /// short grace before SIGTERM. In-flight nested calls are reported
    /// `cancelled`.
    func terminateAndWait(grace: TimeInterval = 2) {
        let shouldStop = lock.withLock { () -> Bool in
            if didTerminate { return false }
            didTerminate = true
            for pending in pendingCalls.values { pending.task?.cancel() }
            return true
        }
        guard shouldStop else { return }
        try? stdinPipe.fileHandleForWriting.close()
        if process.isRunning {
            let deadline = Date().addingTimeInterval(grace)
            while process.isRunning && Date() < deadline {
                Thread.sleep(forTimeInterval: 0.02)
            }
            if process.isRunning {
                // SIGTERM, then SIGKILL after 2 s, then a tree kill, all
                // bounded. `McpClientEngine` reaps its children the same way;
                // an unbounded `waitUntilExit` here could block a cooperative
                // thread forever on a child that ignores SIGTERM.
                ProcessExecutor.terminateAndReap(process)
            }
        }
    }

    // MARK: Internals

    /// A result is ready when the worker reported `done` or `crash`, or the
    /// process exited without either.
    private func completedResult() -> CodemodeResult? {
        lock.lock()
        defer { lock.unlock() }
        if let done {
            return assemble(done: done, crash: nil)
        }
        if let crashMessage {
            return assemble(done: nil, crash: crashMessage)
        }
        if !process.isRunning {
            // Gone, but its last lines (a `done` written just before exit)
            // may not have been read yet. Wait for the reader to hit EOF
            // before concluding it died without a result; the grace bounds
            // the wait if something else holds the pipe's write end open.
            let observedAt = workerExitObservedAt ?? Date()
            workerExitObservedAt = observedAt
            if (stdoutClosed && stderrClosed) || Date().timeIntervalSince(observedAt) > 2 {
                return assemble(done: nil, crash: workerExitDescription())
            }
        }
        return nil
    }

    /// Why a worker that exited without a result died, as far as the host can
    /// tell: how it ended and the tail of what it said on stderr. Called with
    /// the lock held and only once the process is no longer running.
    private func workerExitDescription() -> String {
        var text = "the codemode worker exited before the script finished"
        if process.terminationReason == .uncaughtSignal {
            text += " (terminated by signal \(process.terminationStatus))"
        } else {
            text += " (exit status \(process.terminationStatus))"
        }
        let tail = String(decoding: stderrTail, as: UTF8.self)
            .trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty {
            text += ". Worker stderr: " + String(tail.suffix(500))
        }
        return text + "."
    }

    /// Non-nil once the worker's physical footprint is over the limit. Reads
    /// the kernel's accounting for the child, so it works when the script is
    /// stuck in a synchronous loop and nothing inside the VM could check.
    private func memoryBreachMessage() -> String? {
        let cap = limits.maximumWorkerMemoryBytes
        guard cap > 0, process.isRunning,
              let used = Self.physicalFootprint(of: process.processIdentifier),
              used > UInt64(cap)
        else { return nil }
        return "the codemode worker used \(used / 1_048_576) MiB of memory, over its "
            + "\(cap / 1_048_576) MiB limit, and was terminated."
    }

    /// Physical memory footprint of a process in bytes, or nil when the
    /// kernel will not say (the process is gone). `ri_phys_footprint` is the
    /// figure Activity Monitor and jetsam use, so it counts compressed and
    /// dirty pages that plain RSS misses.
    static func physicalFootprint(of pid: pid_t) -> UInt64? {
        var info = rusage_info_v4()
        let status = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
                proc_pid_rusage(pid, RUSAGE_INFO_V4, $0)
            }
        }
        return status == 0 ? info.ri_phys_footprint : nil
    }

    private func finish(kind: CodemodeErrorKind, message: String) -> CodemodeResult {
        terminateAndWait()
        lock.lock()
        defer { lock.unlock() }
        return assemble(
            done: nil,
            crash: message,
            forcedKind: kind)
    }

    /// Builds the final result under an already-held lock. Worker-reported
    /// errors carry the worker's own JSON; supervisor-side endings (timeout,
    /// abort, process death) force the error kind and message.
    private func assemble(done: CodemodeWire.Done?, crash: String?, forcedKind: CodemodeErrorKind? = nil) -> CodemodeResult {
        var recordedCalls = calls
        // A call still in flight when the script ends is reported cancelled
        // AND cancelled for real: leaving its task running would let the tool
        // finish (with side effects) after the model was told it did not.
        for pending in pendingCalls.values { pending.task?.cancel() }
        let pendingNames = pendingCalls.values.map(\.name)
        pendingCalls.removeAll()
        for name in pendingNames {
            recordedCalls.append(CodemodeCallRecord(name: name, status: .cancelled, durationMs: 0))
        }

        let error: CodemodeError?
        var ok = false
        var valueJSON: String?
        var storeWrites: [CodemodeStoreWrite] = []

        if let done {
            ok = done.ok
            valueJSON = done.value
            if done.ok {
                storeWrites = Self.decodeWrites(done.writes)
                error = nil
            } else {
                error = Self.decodeError(done.error) ?? CodemodeError(
                    kind: .script, name: "Error", message: "the script failed.", stack: nil)
            }
        } else {
            valueJSON = nil
            error = CodemodeError(
                kind: forcedKind ?? .sandbox,
                name: nil,
                message: crash ?? "the codemode sandbox failed.",
                stack: nil)
        }

        return CodemodeResult(
            ok: ok,
            valueJSON: valueJSON,
            output: outputs,
            calls: recordedCalls,
            error: error,
            storeWrites: ok ? storeWrites : [])
    }

    private func receive(_ chunk: Data) {
        var lines: [Data] = []
        var oversized = false
        lock.lock()
        // Whatever is left in the buffer from earlier chunks has no newline
        // (complete lines are removed as they are found), so only the bytes
        // just appended need scanning. Rescanning the whole buffer on every
        // chunk is quadratic when a line is long or never ends.
        var searchStart = stdoutBuffer.endIndex
        stdoutBuffer.append(chunk)
        while let newlineIndex = stdoutBuffer[searchStart...].firstIndex(of: 0x0A) {
            let lineData = Data(stdoutBuffer[stdoutBuffer.startIndex..<newlineIndex])
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newlineIndex)
            // The remainder is all new bytes; scan it from its start.
            searchStart = stdoutBuffer.startIndex
            if lineData.count > CodemodeWire.maximumLineBytes {
                oversized = true
            } else {
                lines.append(lineData)
            }
        }
        // A worker that never ends a line must not grow this buffer without
        // bound. Output and the return value are budgeted inside the VM, so
        // a line this long means the worker is broken or compromised: fail
        // the run and let the caller kill it.
        if oversized || stdoutBuffer.count > CodemodeWire.maximumLineBytes {
            crashMessage = "the codemode worker wrote a line over the "
                + "\(CodemodeWire.maximumLineBytes) byte limit and was stopped."
            stdoutBuffer.removeAll(keepingCapacity: false)
        }
        lock.unlock()

        for line in lines where !line.isEmpty {
            guard let text = String(data: line, encoding: .utf8),
                  let message = CodemodeWire.decodeWorkerMessage(text)
            else { continue }
            switch message {
            case let .call(call):
                handleCall(call)
            case let .output(output):
                lock.withLock {
                    let kind = output.kind == "console" ? CodemodeOutputItem.Kind.console : .text
                    outputs.append(CodemodeOutputItem(
                        kind: kind, text: output.text, level: output.level))
                }
            case let .done(done):
                lock.withLock { self.done = done }
            case let .crash(crash):
                lock.withLock { self.crashMessage = crash.message }
            }
        }
    }

    private enum CallAdmission {
        case admit
        /// Over `maximumCalls`: answered with a rejection the script sees.
        case refuse(String)
        /// The run is already settled or being torn down; nothing to answer.
        case ignore
    }

    private func handleCall(_ call: CodemodeWire.Call) {
        let identifier = call.id
        let handler = callHandler
        let resultCap = limits.maximumCallResultBytes

        lock.lock()
        let admission: CallAdmission
        if done != nil || crashMessage != nil || didTerminate {
            // A call that arrives after the result settled must not run its
            // tool. The prelude stops a script from issuing one; this is the
            // host-side backstop for a call already on the wire.
            admission = .ignore
        } else if acceptedCallCount >= limits.maximumCalls {
            admission = .refuse(
                "codemode allows at most \(limits.maximumCalls) tool calls per script. "
                    + "Filter earlier, fetch less, or split the work across several codemode runs.")
        } else {
            acceptedCallCount += 1
            admission = .admit
            let task = Task<Void, Never>.detached(priority: .utility) { [weak self] in
                guard let self else { return }
                await self.runAdmittedCall(
                    call, handler: handler, resultCap: resultCap)
            }
            // Registered under the same lock hold that created the task, so
            // the task's own completion (which takes this lock) always finds
            // the entry.
            pendingCalls[identifier] = PendingCall(
                name: call.name, startDate: Date(), task: task)
        }
        lock.unlock()

        if case let .refuse(message) = admission {
            sendCallResult(
                CodemodeWire.CallResult(
                    type: "result", id: identifier, ok: false, payload: message))
        }
    }

    /// Runs one admitted call: waits for a concurrency slot, invokes the
    /// re-gated handler, and answers the worker. Cancellation (script ended,
    /// deadline, abort) stops the wait or the handler and sends nothing.
    private func runAdmittedCall(
        _ call: CodemodeWire.Call,
        handler: CallHandler,
        resultCap: Int
    ) async {
        let identifier = call.id
        // Queue rather than reject: `Promise.all` over a list is the normal
        // shape of a batch, and failing most of it would defeat codemode.
        while !tryAcquireCallSlot() {
            if Task.isCancelled { return }
            try? await Task.sleep(nanoseconds: 10_000_000)
        }
        defer { releaseCallSlot() }
        if Task.isCancelled { return }

        let startDate = Date()
        let outcome = await handler(call.name, call.args)
        let durationMs = Int(Date().timeIntervalSince(startDate) * 1000)
        // A call already reaped by the completion path (script ended
        // without awaiting it) is reported cancelled there; a late
        // completion must not add a second record for it.
        let stillPending = lock.withLock { () -> Bool in
            guard !didTerminate else { return false }
            return pendingCalls.removeValue(forKey: identifier) != nil
        }
        guard stillPending else { return }

        switch outcome {
        case let .success(text):
            if Self.resultExceedsLimit(text, cap: resultCap) {
                // Reject, never truncate: a script that parses or counts a
                // clipped result computes a wrong answer with no error.
                lock.withLock {
                    calls.append(CodemodeCallRecord(
                        name: call.name, status: .error, durationMs: durationMs))
                }
                sendCallResult(
                    CodemodeWire.CallResult(
                        type: "result", id: identifier, ok: false,
                        payload: "tool '\(call.name)' returned \(text.utf8.count) bytes, over the "
                            + "\(resultCap) byte limit for one nested result. Ask the tool for less "
                            + "(filters, fewer fields, pagination) so the script gets a complete result."))
                return
            }
            lock.withLock {
                calls.append(CodemodeCallRecord(
                    name: call.name, status: .ok, durationMs: durationMs))
            }
            // The worker JSON-parses the payload into the resolved value, so
            // the tool's text output crosses JSON-encoded.
            let encoded = (try? JSONEncoder().encode(text))
                .flatMap { String(data: $0, encoding: .utf8) }
            sendCallResult(
                CodemodeWire.CallResult(
                    type: "result", id: identifier, ok: true,
                    payload: encoded ?? "\"\""))
        case let .failure(message):
            lock.withLock {
                calls.append(CodemodeCallRecord(
                    name: call.name, status: .error, durationMs: durationMs))
            }
            sendCallResult(
                CodemodeWire.CallResult(
                    type: "result", id: identifier, ok: false, payload: message.message))
        }
    }

    private func tryAcquireCallSlot() -> Bool {
        lock.withLock {
            guard activeCallCount < limits.maximumConcurrentCalls else { return false }
            activeCallCount += 1
            return true
        }
    }

    private func releaseCallSlot() {
        lock.withLock { activeCallCount -= 1 }
    }

    private func sendCallResult(_ result: CodemodeWire.CallResult) {
        guard let line = CodemodeWire.encodeLine(result) else { return }
        writeLine(line)
    }

    private func writeLine(_ line: Data) {
        writeQueue.sync {
            do {
                try stdinPipe.fileHandleForWriting.write(contentsOf: line)
            } catch {
                // A closed pipe means the worker is gone; its pending calls
                // are reaped by the completion path.
            }
        }
    }

    /// Whether one nested tool result is too large to hand into the VM.
    /// Measured in UTF-8 bytes (O(1) on a native string) because the bound
    /// protects memory and the wire line limit, not the output budget: a
    /// result only counts against that if the script prints or returns it.
    /// Callers REJECT an oversized result. Clipping it would hand the script
    /// a string that fails `JSON.parse` or, worse, parses to a wrong count.
    static func resultExceedsLimit(_ text: String, cap: Int) -> Bool {
        cap > 0 && text.utf8.count > cap
    }

    private static func decodeWrites(_ json: String?) -> [CodemodeStoreWrite] {
        guard let json, let data = json.data(using: .utf8),
              let entries = try? JSONSerialization.jsonObject(with: data) as? [[Any]]
        else { return [] }
        var writes: [CodemodeStoreWrite] = []
        for entry in entries {
            guard entry.count >= 1, let key = entry[0] as? String else { continue }
            if entry.count >= 2, let value = entry[1] as? String {
                writes.append(CodemodeStoreWrite(key: key, valueJSON: value))
            } else {
                writes.append(CodemodeStoreWrite(key: key, valueJSON: nil))
            }
        }
        return writes
    }

    private static func decodeError(_ json: String?) -> CodemodeError? {
        guard let json, let data = json.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        let name = object["name"] as? String
        let message = object["message"] as? String ?? "the script failed."
        let stack = object["stack"] as? String
        // An explicit line (a syntax error) wins; otherwise the first stack
        // frame that belongs to the script. The prelude's frames were already
        // dropped in the VM, so an error the prelude built for a bad call
        // (an unknown tool) still points at the line that made the call.
        let line = (object["line"] as? Int) ?? stack.flatMap(Self.firstScriptLine(in:))
        return CodemodeError(
            kind: .script, name: name, message: message, stack: stack, line: line)
    }

    /// The line of the first `codemode.js:<line>:<column>` frame in a
    /// JavaScriptCore stack, or nil when there is none.
    static func firstScriptLine(in stack: String) -> Int? {
        let marker = CodemodePrelude.scriptURL + ":"
        guard let range = stack.range(of: marker) else { return nil }
        let digits = stack[range.upperBound...].prefix { $0.isASCII && $0.isNumber }
        return Int(digits)
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}
