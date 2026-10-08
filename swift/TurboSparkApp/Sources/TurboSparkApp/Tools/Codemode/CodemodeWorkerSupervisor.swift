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
    private let writeQueue = DispatchQueue(label: "com.turbospark.codemode.worker-write")
    private let lock = NSLock()
    private let request: CodemodeWire.RunRequest
    private let callHandler: CallHandler
    private let timeout: TimeInterval
    private let limits: CodemodeLimits

    private var pendingCalls: [Int: (name: String, startDate: Date)] = [:]
    private var calls: [CodemodeCallRecord] = []
    private var outputs: [CodemodeOutputItem] = []
    private var done: CodemodeWire.Done?
    private var crashMessage: String?
    private var didTerminate = false
    private var stdoutBuffer = Data()

    init(
        request: CodemodeWire.RunRequest,
        executableURL: URL,
        callHandler: @escaping CallHandler,
        timeout: TimeInterval,
        limits: CodemodeLimits
    ) throws {
        self.request = request
        self.callHandler = callHandler
        self.timeout = timeout
        self.limits = limits
        process.executableURL = executableURL
        process.arguments = [REPLWorkerMain.workerModeArgument, REPLWorkerMain.serveCodemodeArgument]
        process.standardInput = stdinPipe
        process.standardOutput = stdoutPipe
        process.standardError = FileHandle.nullDevice

        try process.run()
        signal(SIGPIPE, SIG_IGN)

        if let line = CodemodeWire.encodeLine(request) {
            writeLine(line)
        }

        stdoutPipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let chunk = handle.availableData
            guard !chunk.isEmpty else {
                handle.readabilityHandler = nil
                return
            }
            self?.receive(chunk)
        }
        process.terminationHandler = { [weak self] _ in
            self?.stdoutPipe.fileHandleForReading.readabilityHandler = nil
        }
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
        while true {
            if Task.isCancelled {
                return finish(kind: .aborted, message: "the codemode run was cancelled.")
            }
            if let result = completedResult() {
                return result
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
            return true
        }
        guard shouldStop else { return }
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
            return assemble(
                done: nil,
                crash: "the codemode worker exited before the script finished.")
        }
        return nil
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
        lock.lock()
        stdoutBuffer.append(chunk)
        while let newlineIndex = stdoutBuffer.firstIndex(of: 0x0A) {
            let lineData = Data(stdoutBuffer[stdoutBuffer.startIndex..<newlineIndex])
            stdoutBuffer.removeSubrange(stdoutBuffer.startIndex...newlineIndex)
            lines.append(lineData)
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

    private func handleCall(_ call: CodemodeWire.Call) {
        let identifier = call.id
        lock.lock()
        let startDate = Date()
        pendingCalls[identifier] = (call.name, startDate)
        lock.unlock()

        let handler = callHandler
        let payloadCap = limits.maximumCallPayloadCharacters
        Task.detached(priority: .utility) { [weak self] in
            let outcome = await handler(call.name, call.args)
            guard let self else { return }
            let durationMs = Int(Date().timeIntervalSince(startDate) * 1000)
            // A call already reaped by the completion path (script ended
            // without awaiting it) is reported cancelled there; a late
            // completion must not add a second record for it.
            let stillPending = self.lock.withLock { () -> Bool in
                guard !self.didTerminate else { return false }
                return self.pendingCalls.removeValue(forKey: identifier) != nil
            }
            guard stillPending else { return }

            switch outcome {
            case let .success(text):
                self.lock.withLock {
                    self.calls.append(CodemodeCallRecord(
                        name: call.name, status: .ok, durationMs: durationMs))
                }
                // The worker JSON-parses the payload into the resolved
                // value, so the tool's text output crosses JSON-encoded.
                // Truncation happens on the raw text, before encoding.
                let truncated = Self.truncatedPayload(text, cap: payloadCap)
                let encoded = (try? JSONEncoder().encode(truncated))
                    .flatMap { String(data: $0, encoding: .utf8) }
                self.sendCallResult(
                    CodemodeWire.CallResult(
                        type: "result", id: identifier, ok: true,
                        payload: encoded ?? "\"\""))
            case let .failure(message):
                self.lock.withLock {
                    self.calls.append(CodemodeCallRecord(
                        name: call.name, status: .error, durationMs: durationMs))
                }
                self.sendCallResult(
                    CodemodeWire.CallResult(
                        type: "result", id: identifier, ok: false, payload: message.message))
            }
        }
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

    /// Caps one nested tool result handed into the VM. A single huge tool
    /// response must not be able to exhaust the script's whole output
    /// budget by itself; the head and tail survive with a marker between.
    static func truncatedPayload(_ text: String, cap: Int) -> String {
        guard cap > 0, text.count > cap else { return text }
        let head = cap * 3 / 4
        let tail = cap / 4
        let omitted = text.count - head - tail
        return String(text.prefix(head))
            + "\n... [\(omitted) characters truncated] ...\n"
            + String(text.suffix(tail))
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
        return CodemodeError(kind: .script, name: name, message: message, stack: stack)
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}
