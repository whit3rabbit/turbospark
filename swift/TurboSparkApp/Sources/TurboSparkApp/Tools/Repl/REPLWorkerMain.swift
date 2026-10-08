import Foundation

enum REPLWorkerMain {
    static let workerModeArgument = "--turbospark-js-repl-worker"
    private static let smokeArgument = "--smoke"
    static let serveArgument = "--serve"
    static let serveCodemodeArgument = "--codemode-serve"
    /// Transport-level bound for one request line, independent of the
    /// script-size limit so the JSON envelope never pushes a small script
    /// over the configured request bound.
    static let maximumTransportBytes = 4 * 1_024 * 1_024

    private struct SmokeSnapshot: Sendable {
        var completionText: String?
        var ranOnMainThread: Bool
    }

    private final class SmokeResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value = SmokeSnapshot(completionText: nil, ranOnMainThread: true)

        func store(completionText: String?, ranOnMainThread: Bool) {
            lock.lock()
            value = SmokeSnapshot(
                completionText: completionText,
                ranOnMainThread: ranOnMainThread)
            lock.unlock()
        }

        func load() -> SmokeSnapshot {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    static func isWorkerInvocation(_ arguments: [String]) -> Bool {
        arguments.dropFirst().contains(workerModeArgument)
    }

    /// Handles the internal worker entry before SwiftUI services are created.
    /// The smoke option is used by the package test to exercise the linked
    /// JavaScriptCore runtime in the packaged executable. The serve option
    /// runs the request loop that the app-side supervisor speaks to: one
    /// init line carrying the limits and artifact directory, then one
    /// newline-delimited JSON evaluate request per line on stdin, with one
    /// newline-delimited JSON result per request on stdout. Requests over
    /// the configured request-size bound are rejected without evaluation.
    @discardableResult
    static func runIfRequested(
        arguments: [String] = CommandLine.arguments,
        writeOutput: (String) -> Void = { print($0) }
    ) -> Bool {
        guard isWorkerInvocation(arguments) else { return false }
        if arguments.contains(serveArgument) {
            serveRequests()
            return true
        }
        if arguments.contains(serveCodemodeArgument) {
            serveCodemodeRequests()
            return true
        }
        guard arguments.contains(smokeArgument) else { return true }

        let result = SmokeResultBox()
        let finished = DispatchSemaphore(value: 0)
        let worker = REPLWorkerContext()
        Task.detached {
            defer { finished.signal() }
            let evaluation = await worker.evaluateWithThreadStatus(code: "1 + 1")
            result.store(
                completionText: evaluation.result.completionText,
                ranOnMainThread: evaluation.ranOnMainThread)
        }

        finished.wait()
        let snapshot = result.load()
        guard !snapshot.ranOnMainThread else {
            FileHandle.standardError.write(Data("JavaScriptCore smoke evaluation ran on the main thread\n".utf8))
            exit(EXIT_FAILURE)
        }
        guard let result = snapshot.completionText else {
            FileHandle.standardError.write(Data("JavaScriptCore smoke evaluation failed\n".utf8))
            exit(EXIT_FAILURE)
        }

        writeOutput("worker:\(result)")
        return true
    }

    // MARK: Serve loop

    private struct InitMessage: Codable {
        var limits: REPLLimits
        var artifactDirectory: URL
    }

    private struct EvaluateMessage: Codable {
        var id: Int
        var code: String
        var settlementTimeout: TimeInterval
    }

    private struct ResultMessage: Codable {
        var id: Int
        var result: REPLCallResult
    }

    /// Reads stdin to EOF, serving one request at a time in arrival order.
    /// The worker owns a single REPLWorkerContext for its lifetime, so
    /// bindings persist across requests.
    private static func serveRequests() {
        let encoder = JSONEncoder()
        let decoder = JSONDecoder()
        var worker: REPLWorkerContext?
        var limits = REPLLimits()
        var buffer = Data()

        func reply(_ message: ResultMessage) {
            guard let line = try? encoder.encode(message) else { return }
            var framed = line
            framed.append(0x0A)
            FileHandle.standardOutput.write(framed)
        }

        func localFailure(id: Int, _ text: String) {
            reply(ResultMessage(id: id, result: REPLCallResult(
                status: .failed,
                outputEvents: [],
                consoleText: "",
                errorText: text,
                completionText: nil,
                images: [],
                truncated: false,
                sessionCreated: worker == nil,
                sessionReset: false)))
        }

        while true {
            // Foundation's read(upToCount:) blocks until the full length or
            // EOF on a pipe, so interactive request lines would never
            // arrive; the raw POSIX read returns as soon as any bytes are
            // available, which the request loop needs.
            var incoming = [UInt8](repeating: 0, count: 65_536)
            let byteCount = read(STDIN_FILENO, &incoming, incoming.count)
            guard byteCount > 0 else { break }
            buffer.append(contentsOf: incoming[0..<byteCount])

            while let newlineIndex = buffer.firstIndex(of: 0x0A) {
                let lineData = buffer[buffer.startIndex..<newlineIndex]
                buffer.removeSubrange(buffer.startIndex...newlineIndex)
                guard !lineData.isEmpty else { continue }

                if lineData.count > Self.maximumTransportBytes {
                    localFailure(
                        id: -1,
                        "request of \(lineData.count) bytes exceeds the "
                            + "\(Self.maximumTransportBytes) byte transport limit")
                    continue
                }

                guard String(data: Data(lineData), encoding: .utf8) != nil else {
                    localFailure(id: -1, "request is not valid UTF-8")
                    continue
                }
                guard let lineObject = try? decoder.decode(
                    [String: REPLWorkerProbeValue].self, from: Data(lineData)),
                    let type = lineObject["type"]?.stringValue
                else {
                    localFailure(id: -1, "request is not a JSON object with a type")
                    continue
                }

                switch type {
                case "init":
                    guard let message = try? decoder.decode(
                        InitMessage.self, from: Data(lineData))
                    else {
                        localFailure(id: -1, "init request is malformed")
                        continue
                    }
                    limits = message.limits
                    worker = REPLWorkerContext(
                        limits: message.limits,
                        configuration: REPLSessionConfiguration(
                            artifactDirectory: message.artifactDirectory))
                case "evaluate":
                    guard let message = try? decoder.decode(
                        EvaluateMessage.self, from: Data(lineData))
                    else {
                        localFailure(id: -1, "evaluate request is malformed")
                        continue
                    }
                    // The request bound applies to the script, never to the
                    // wire line, so the JSON envelope never pushes a small
                    // script over the limit.
                    if Data(message.code.utf8).count > limits.maximumRequestBytes {
                        localFailure(
                            id: message.id,
                            "script of \(Data(message.code.utf8).count) bytes exceeds the "
                                + "\(limits.maximumRequestBytes) byte request limit")
                        continue
                    }
                    guard let worker else {
                        localFailure(
                            id: message.id,
                            "the worker received an evaluate request before init")
                        continue
                    }
                    // One request at a time: block this loop until the
                    // evaluation finishes, so responses keep request order.
                    let finished = DispatchSemaphore(value: 0)
                    let box = ResultBox()
                    Task.detached {
                        let result = await worker.evaluate(
                            code: message.code,
                            settlementTimeout: message.settlementTimeout)
                        box.store(result)
                        finished.signal()
                    }
                    finished.wait()
                    reply(ResultMessage(id: message.id, result: box.load()))
                default:
                    localFailure(id: -1, "unknown request type \(type)")
                }
            }
        }
    }

    // MARK: Codemode serve loop

    /// Thread-safe newline writer for worker stdout. Codemode writes come
    /// from both the serve thread (nothing) and the execution thread (call,
    /// output, and terminal lines), so the write itself is lock guarded.
    private final class StdoutLineWriter: @unchecked Sendable {
        private let lock = NSLock()

        func write(_ line: Data?) {
            guard var framed = line else { return }
            framed.append(0x0A)
            lock.lock()
            defer { lock.unlock() }
            FileHandle.standardOutput.write(framed)
        }
    }

    /// Serves exactly one codemode script execution per process lifetime.
    /// The request line carries the whole run (code, tool entries, store
    /// snapshot, limits); while it executes, the loop keeps reading stdin
    /// so nested-call results can arrive, then writes one terminal
    /// `done` or `crash` line and exits the process.
    private static func serveCodemodeRequests() {
        let writer = StdoutLineWriter()
        let decoder = JSONDecoder()
        var buffer = Data()
        var context: CodemodeWorkerContext?
        var executionStarted = false

        func writeCrash(_ message: String) {
            writer.write(CodemodeWire.encodeLine(CodemodeWire.Crash(type: "crash", message: message)))
        }

        while true {
            // Same raw-read requirement as serveRequests: Foundation's
            // read(upToCount:) blocks to full length or EOF on a pipe.
            var incoming = [UInt8](repeating: 0, count: 65_536)
            let byteCount = read(STDIN_FILENO, &incoming, incoming.count)
            guard byteCount > 0 else { break }
            buffer.append(contentsOf: incoming[0..<byteCount])

            // Codemode lines carry whole nested tool results, so they use the
            // codemode wire bound rather than the REPL's smaller one. A line
            // that never ends must not grow this buffer without limit.
            if buffer.count > CodemodeWire.maximumLineBytes, !buffer.contains(0x0A) {
                writeCrash("codemode request exceeds the "
                    + "\(CodemodeWire.maximumLineBytes) byte transport limit")
                buffer.removeAll(keepingCapacity: false)
                continue
            }

            while let newlineIndex = buffer.firstIndex(of: 0x0A) {
                let lineData = buffer[buffer.startIndex..<newlineIndex]
                buffer.removeSubrange(buffer.startIndex...newlineIndex)
                guard !lineData.isEmpty else { continue }
                if lineData.count > CodemodeWire.maximumLineBytes {
                    writeCrash("codemode request of \(lineData.count) bytes exceeds the "
                        + "\(CodemodeWire.maximumLineBytes) byte transport limit")
                    continue
                }
                guard let lineObject = try? decoder.decode(
                    [String: REPLWorkerProbeValue].self, from: Data(lineData)),
                    let type = lineObject["type"]?.stringValue
                else {
                    writeCrash("codemode request is not a JSON object with a type")
                    continue
                }

                switch type {
                case "codemode":
                    guard !executionStarted else {
                        writeCrash("this worker already served its one codemode execution")
                        continue
                    }
                    guard let request = try? decoder.decode(
                        CodemodeWire.RunRequest.self, from: Data(lineData))
                    else {
                        writeCrash("codemode request is malformed")
                        continue
                    }
                    executionStarted = true
                    let workerContext = CodemodeWorkerContext(
                        request: request, sendLine: { writer.write($0) })
                    context = workerContext
                    Task.detached(priority: .utility) {
                        let outcome = workerContext.run()
                        switch outcome {
                        case .done(let done):
                            writer.write(CodemodeWire.encodeLine(done))
                        case .crashed(let message):
                            writeCrash(message)
                        }
                        // The terminal line is the process's last word; the
                        // supervisor kills us anyway, but exiting here keeps
                        // a finished worker from lingering on its pipes.
                        exit(EXIT_SUCCESS)
                    }
                case "result":
                    guard let result = try? decoder.decode(
                        CodemodeWire.CallResult.self, from: Data(lineData))
                    else { continue }
                    context?.deliver(result: result)
                default:
                    // Unknown lines are ignored: this worker serves exactly
                    // one protocol and its supervisor sends only that.
                    continue
                }
            }
        }
    }

    private final class ResultBox: @unchecked Sendable {
        private let lock = NSLock()
        private var value = REPLCallResult(
            status: .failed,
            outputEvents: [],
            consoleText: "",
            errorText: "the worker result was never stored",
            completionText: nil,
            images: [],
            truncated: false,
            sessionCreated: false,
            sessionReset: false)

        func store(_ result: REPLCallResult) {
            lock.lock()
            value = result
            lock.unlock()
        }

        func load() -> REPLCallResult {
            lock.lock()
            defer { lock.unlock() }
            return value
        }
    }

    /// A minimal dynamic-JSON probe: the worker only needs to read the
    /// "type" discriminator before decoding the concrete message.
    private struct REPLWorkerProbeValue: Codable {
        var stringValue: String?

        init(from decoder: Decoder) throws {
            let container = try decoder.singleValueContainer()
            stringValue = try? container.decode(String.self)
        }

        func encode(to encoder: Encoder) throws {
            var container = encoder.singleValueContainer()
            if let stringValue {
                try container.encode(stringValue)
            } else {
                try container.encodeNil()
            }
        }
    }
}
