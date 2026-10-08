import Foundation
import JavaScriptCore

/// Runs one codemode script inside one fresh JavaScriptCore context, on the
/// worker side of the process boundary. One instance serves exactly one
/// execution; the process exits after its terminal message.
///
/// Threading model (the pattern proven by REPLWorkerContext and
/// REPLHostFacade): evaluation, the bridges, and the settlement poll loop
/// all run on the caller's thread; nested-call results arrive from the
/// stdin reader thread through `deliver(result:)`, and JavaScriptCore
/// serializes context access internally so the callback invocation is safe.
/// The pending-callback table is the one shared structure and is lock
/// guarded.
final class CodemodeWorkerContext: @unchecked Sendable {
    enum Outcome {
        case done(CodemodeWire.Done)
        case crashed(String)
    }

    private let request: CodemodeWire.RunRequest
    private let sendLine: (Data) -> Void
    private let lock = NSLock()
    private var pendingCallbacks: [Int: JSValue] = [:]
    private var nextCallID = 0
    private var outputItems: [CodemodeOutputItem] = []
    private var capturedException: String?
    private let outcomeBox = CodemodeOutcomeBox()

    /// Backstop deadline for the settlement poll loop. The supervisor owns
    /// the real deadline and terminates the process; this only guarantees
    /// the worker cannot spin forever if that never arrives.
    static let settlementBackstop: TimeInterval = 900

    init(request: CodemodeWire.RunRequest, sendLine: @escaping (Data) -> Void) {
        self.request = request
        self.sendLine = sendLine
    }

    /// Delivers one nested-call result from the stdin reader thread.
    func deliver(result: CodemodeWire.CallResult) {
        let callback: JSValue? = lock.withLock {
            pendingCallbacks.removeValue(forKey: result.id)
        }
        guard let callback else { return }
        let errorArgument: Any = result.ok ? NSNull() : (result.payload ?? "the tool call failed")
        let payloadArgument: Any = result.ok ? (result.payload ?? NSNull()) : NSNull()
        callback.call(withArguments: [errorArgument, payloadArgument])
    }

    /// Evaluates the prelude and the script and blocks until the script
    /// settles, stalls, or hits the backstop. Never returns a thrown Swift
    /// error: every failure becomes an outcome.
    func run() -> Outcome {
        guard let context = JSContext() else {
            return .crashed("JavaScriptCore could not create the codemode context.")
        }
        context.exceptionHandler = { [weak self] _, exception in
            // With a handler installed, JavaScriptCore hands uncaught
            // exceptions here instead of leaving them on context.exception
            // (the REPLWorkerContext pattern). Script-level errors reach
            // the done bridge through the wrapper; what arrives here is a
            // prelude/compile failure or a stray uncaught throw.
            self?.capturedException = exception?.toString()
        }

        installBridges(into: context)
        context.evaluateScript(CodemodePrelude.source(limits: request.limits))
        if let failure = currentException(context) {
            return .crashed("the codemode prelude failed: \(failure)")
        }

        context.evaluateScript(CodemodePrelude.wrapper(code: request.code))
        if let failure = currentException(context) {
            // A syntax error or a synchronous top-level throw surfaces here
            // as a script failure; everything after the first await reports
            // through the done bridge instead.
            context.exception = nil
            return .done(scriptError(failure))
        }

        pollUntilSettled(context: context)
        if let outcome = outcomeBox.result {
            return .done(outcome)
        }
        return .done(CodemodeWire.Done(
            ok: false,
            value: nil,
            error: encodeErrorJSON(
                name: "Error",
                message: "the script did not settle within "
                    + "\(Int(Self.settlementBackstop)) seconds.",
                stack: nil),
            writes: nil))
    }

    // MARK: Internals

    private func installBridges(into context: JSContext) {
        let entriesData = (try? JSONEncoder().encode(request.tools)) ?? Data("[]".utf8)
        let entriesJSON = String(data: entriesData, encoding: .utf8) ?? "[]"
        let snapshotJSON = request.storeSnapshot

        let toolsInit: @convention(block) () -> String = { entriesJSON }
        context.setObject(toolsInit, forKeyedSubscript: "__codemodeToolsInit" as NSString)

        let storeInit: @convention(block) () -> String = { snapshotJSON }
        context.setObject(storeInit, forKeyedSubscript: "__codemodeStoreInit" as NSString)

        let output: @convention(block) (String, String, JSValue) -> Void =
            { [weak self] kind, text, level in
                guard let self else { return }
                let levelText = (level.isNull || level.isUndefined) ? nil : level.toString()
                let item = CodemodeOutputItem(
                    kind: kind == "console" ? .console : .text,
                    text: text,
                    level: levelText)
                self.outputItems.append(item)
                if let line = CodemodeWire.encodeLine(CodemodeWire.Output(
                    type: "output", kind: item.kind.rawValue, text: text, level: levelText))
                {
                    self.sendLine(line)
                }
            }
        context.setObject(output, forKeyedSubscript: "__codemodeOutput" as NSString)

        let done: @convention(block) (Bool, JSValue, JSValue, JSValue) -> Void =
            { [weak self] ok, value, error, writes in
                let valueText = (value.isNull || value.isUndefined) ? nil : value.toString()
                let errorText = (error.isNull || error.isUndefined) ? nil : error.toString()
                let writesText = (writes.isNull || writes.isUndefined) ? nil : writes.toString()
                self?.outcomeBox.record(
                    ok: ok, value: valueText, error: errorText, writes: writesText)
            }
        context.setObject(done, forKeyedSubscript: "__codemodeDone" as NSString)

        let call: @convention(block) (String, JSValue, JSValue) -> Void =
            { [weak self] name, arguments, callback in
                guard let self else { return }
                let argsText: String?
                if arguments.isNull || arguments.isUndefined {
                    argsText = nil
                } else {
                    argsText = arguments.toString()
                }
                let identifier = self.lock.withLock { () -> Int in
                    self.nextCallID += 1
                    return self.nextCallID
                }
                self.lock.withLock {
                    self.pendingCallbacks[identifier] = callback
                }
                if let line = CodemodeWire.encodeLine(CodemodeWire.Call(
                    type: "call", id: identifier, name: name, args: argsText))
                {
                    self.sendLine(line)
                }
            }
        context.setObject(call, forKeyedSubscript: "__codemodeCall" as NSString)
    }

    /// Drains microtasks and watches for settlement. A script that is not
    /// settled, has no pending tool call, and stays that way across two
    /// consecutive drains is waiting on a promise nothing in this VM can
    /// ever resolve (there are no timers), so it fails fast instead of
    /// burning the backstop.
    private func pollUntilSettled(context: JSContext) {
        var emptyObservations = 0
        let deadline = Date().addingTimeInterval(Self.settlementBackstop)
        while !outcomeBox.isSettled {
            _ = context.evaluateScript("0")
            context.exception = nil
            if outcomeBox.isSettled { break }

            let pendingCount = lock.withLock { pendingCallbacks.count }
            if pendingCount == 0 {
                emptyObservations += 1
                if emptyObservations >= 2 {
                    context.evaluateScript(
                        "globalThis.__codemodeReportStalled && globalThis.__codemodeReportStalled()")
                    context.exception = nil
                    if outcomeBox.isSettled { break }
                }
            } else {
                emptyObservations = 0
            }

            if Date() >= deadline { break }
            Thread.sleep(forTimeInterval: 0.005)
        }
    }

    private func currentException(_ context: JSContext) -> String? {
        let captured = capturedException ?? context.exception?.toString()
        capturedException = nil
        context.exception = nil
        return captured
    }

    private func scriptError(_ description: String) -> CodemodeWire.Done {
        CodemodeWire.Done(
            ok: false,
            value: nil,
            error: encodeErrorJSON(name: "SyntaxError", message: description, stack: nil),
            writes: nil)
    }

    private func encodeErrorJSON(name: String, message: String, stack: String?) -> String {
        var object: [String: Any] = ["name": name, "message": message]
        if let stack, !stack.isEmpty { object["stack"] = stack }
        guard let data = try? JSONSerialization.data(withJSONObject: object),
              let text = String(data: data, encoding: .utf8)
        else { return "{\"name\":\"Error\",\"message\":\"the script failed\"}" }
        return text
    }
}

/// First-write-wins settlement record. The done bridge may fire once more
/// after a failure (an output-cap breach followed by a caught unwind), and
/// the first report must stand.
final class CodemodeOutcomeBox: @unchecked Sendable {
    private let lock = NSLock()
    private var outcome: CodemodeWire.Done?

    func record(ok: Bool, value: String?, error: String?, writes: String?) {
        lock.lock()
        defer { lock.unlock() }
        guard outcome == nil else { return }
        outcome = CodemodeWire.Done(
            ok: ok, value: value, error: error, writes: writes)
    }

    var isSettled: Bool {
        lock.lock()
        defer { lock.unlock() }
        return outcome != nil
    }

    var result: CodemodeWire.Done? {
        lock.lock()
        defer { lock.unlock() }
        return outcome
    }
}

private extension NSLock {
    func withLock<T>(_ body: () -> T) -> T {
        lock()
        defer { unlock() }
        return body()
    }
}
