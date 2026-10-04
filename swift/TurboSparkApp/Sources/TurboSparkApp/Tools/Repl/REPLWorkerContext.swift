import Foundation
import JavaScriptCore

struct REPLWorkerEvaluation: Sendable {
    var result: REPLCallResult
    var ranOnMainThread: Bool
}

/// Owns the persistent JavaScriptCore context inside one worker process.
/// All context creation, facade installation, evaluation, and result rendering
/// happen on the same private serial queue.
///
/// Decision flow per call, implementing the top-level-await strategy proven
/// by REPLTopLevelAwaitPrototype (task 1.5) through the shared machinery in
/// REPLScriptLowering.swift:
/// 1. JSCheckScriptSyntax probes the code as a classic script. Valid classic
///    scripts evaluate natively with evaluateScript and keep native lexical
///    declaration semantics; the value returned by JavaScriptCore is the
///    completion value.
/// 2. Classic scripts cannot contain top-level await, so a failed probe hands
///    the source to the bounded token-based parser shared with the prototype.
/// 3. A program that uses top-level await and stays inside the supported
///    subset is lowered to an async wrapper whose top-level declarations are
///    rewritten to persistent assignments on the global object; declarations
///    are never moved into a wrapper scope.
/// 4. Anything outside the subset returns unsupportedSyntax before any
///    evaluation runs. Parse errors, unsupported syntax, and ordinary
///    exceptions all preserve the worker session; only the supervisor's
///    wedge-class handling (task 3.2) may end it.
final class REPLWorkerContext: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.turbospark.repl.worker-context")
    private var context: JSContext?
    private var completionRenderer: JSValue?
    private var outputEvents: [REPLTextOutputEvent] = []
    private var capturedException: String?
    private var hasCreatedSession = false
    private let outcomeBox = REPLSettlementBox()
    private let limits: REPLLimits

    init(limits: REPLLimits = REPLLimits()) {
        self.limits = limits
    }

    /// Deadline the lowered path waits for the wrapper promise to settle.
    /// The per-call timeout owned by the parent supervisor terminates the
    /// worker process (task 3.2); this deadline only bounds the wait here.
    static let defaultSettlementDeadline: TimeInterval = 30

    func evaluate(
        code: String,
        settlementTimeout: TimeInterval = REPLWorkerContext.defaultSettlementDeadline
    ) async -> REPLCallResult {
        await evaluateWithThreadStatus(code: code, settlementTimeout: settlementTimeout).result
    }

    func evaluateWithThreadStatus(
        code: String,
        settlementTimeout: TimeInterval = REPLWorkerContext.defaultSettlementDeadline
    ) async -> REPLWorkerEvaluation {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                let result = evaluateOnWorkerQueue(
                    code: code,
                    settlementTimeout: settlementTimeout)
                continuation.resume(returning: REPLWorkerEvaluation(
                    result: result,
                    ranOnMainThread: Thread.isMainThread))
            }
        }
    }

    private func evaluateOnWorkerQueue(
        code: String,
        settlementTimeout: TimeInterval
    ) -> REPLCallResult {
        let sessionCreated = !hasCreatedSession
        outputEvents = []
        capturedException = nil

        guard let context = context ?? makeContext() else {
            hasCreatedSession = true
            return makeResult(
                status: .failed,
                completionText: nil,
                errorText: "JavaScriptCore could not create an evaluation context.",
                sessionCreated: sessionCreated)
        }
        self.context = context
        hasCreatedSession = true
        context.exception = nil

        if code.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return makeResult(
                status: .parseError,
                completionText: nil,
                errorText: "the script is empty",
                sessionCreated: sessionCreated)
        }

        let probe = REPLClassicScriptSyntax.check(context: context, code: code)
        if probe.ok {
            return evaluateNatively(context: context, code: code, sessionCreated: sessionCreated)
        }
        let nativeMessage = probe.message ?? "unknown syntax error"

        switch REPLBoundedScriptParser.parse(source: code) {
        case let .lexFailed(reason):
            return makeResult(
                status: .parseError,
                completionText: nil,
                errorText: "\(nativeMessage) (bounded scanner: \(reason))",
                sessionCreated: sessionCreated)
        case let .unsupported(reason, usesTopLevelAwait):
            if usesTopLevelAwait {
                return makeResult(
                    status: .unsupportedSyntax,
                    completionText: nil,
                    errorText: "unsupported top-level syntax: \(reason). "
                        + "The session was not changed.",
                    sessionCreated: sessionCreated)
            }
            return makeResult(
                status: .parseError,
                completionText: nil,
                errorText: nativeMessage,
                sessionCreated: sessionCreated)
        case let .program(program):
            guard program.usesTopLevelAwait else {
                return makeResult(
                    status: .parseError,
                    completionText: nil,
                    errorText: nativeMessage,
                    sessionCreated: sessionCreated)
            }
            return evaluateLowered(
                context: context,
                program: program,
                source: code,
                settlementTimeout: settlementTimeout,
                sessionCreated: sessionCreated)
        }
    }

    private func evaluateNatively(
        context: JSContext,
        code: String,
        sessionCreated: Bool
    ) -> REPLCallResult {
        let value = context.evaluateScript(code)
        let failure = capturedException ?? context.exception?.toString()
        context.exception = nil
        if let failure {
            return makeResult(
                status: .failed,
                completionText: nil,
                errorText: failure,
                sessionCreated: sessionCreated)
        }
        return makeResult(
            status: .completed,
            completionText: renderCompletion(value),
            errorText: nil,
            sessionCreated: sessionCreated)
    }

    private func evaluateLowered(
        context: JSContext,
        program: REPLBoundedProgram,
        source: String,
        settlementTimeout: TimeInterval,
        sessionCreated: Bool
    ) -> REPLCallResult {
        let wrapper = REPLLoweredWrapperBuilder.build(
            program: program,
            source: source,
            settleGlobal: "__turbosparkReplSettled",
            renderGlobal: "__turbosparkReplRender")
        outcomeBox.reset()
        capturedException = nil
        context.exception = nil
        _ = context.evaluateScript(wrapper)

        if let compileError = capturedException ?? context.exception?.toString() {
            context.exception = nil
            return makeResult(
                status: .parseError,
                completionText: nil,
                errorText: "\(compileError) (from the lowered wrapper; the session was not changed)",
                sessionCreated: sessionCreated)
        }

        let deadline = Date().addingTimeInterval(settlementTimeout)
        while true {
            if outcomeBox.isSettled { break }
            drainMicrotasks(context: context)
            if outcomeBox.isSettled { break }
            if Date() >= deadline {
                return makeResult(
                    status: .timedOut,
                    completionText: nil,
                    errorText: "top-level await did not settle within \(settlementTimeout) s; "
                        + "this evaluation path cannot terminate the context in-process",
                    sessionCreated: sessionCreated)
            }
            Thread.sleep(forTimeInterval: 0.005)
        }

        guard let outcome = outcomeBox.outcome else {
            return makeResult(
                status: .failed,
                completionText: nil,
                errorText: "settlement loop ended without an outcome",
                sessionCreated: sessionCreated)
        }
        if outcome.fulfilled {
            return makeResult(
                status: .completed,
                completionText: outcome.text,
                errorText: nil,
                sessionCreated: sessionCreated)
        }
        return makeResult(
            status: .failed,
            completionText: nil,
            errorText: outcome.text,
            sessionCreated: sessionCreated)
    }

    /// Explicit microtask checkpoint for the settlement loop. Script
    /// evaluation drains microtasks on its own, but the loop keeps this
    /// checkpoint instead of depending on call-side draining.
    private func drainMicrotasks(context: JSContext) {
        _ = context.evaluateScript("0")
        context.exception = nil
    }

    private func makeContext() -> JSContext? {
        guard let context = JSContext() else { return nil }
        context.exceptionHandler = { [weak self] _, exception in
            self?.capturedException = exception?.toString()
        }
        completionRenderer = context.evaluateScript(Self.completionRendererScript)

        let box = outcomeBox
        let settled: @convention(block) (Bool, String) -> Void = { fulfilled, text in
            box.record(fulfilled: fulfilled, text: text)
        }
        context.setObject(settled, forKeyedSubscript: "__turbosparkReplSettledBridge" as NSString)
        context.evaluateScript(Self.settleInstallationScript)

        REPLHostFacade { [weak self] event in
            self?.outputEvents.append(event)
        }
        .installConsole(into: context)
        context.exception = nil
        return context
    }

    private func renderCompletion(_ value: JSValue?) -> String {
        guard let value else { return "undefined" }
        return completionRenderer?.call(withArguments: [value])?.toString() ?? value.toString()
    }

    private func makeResult(
        status: REPLCallResult.Status,
        completionText: String?,
        errorText: String?,
        sessionCreated: Bool
    ) -> REPLCallResult {
        // The captured event sequence is the ordering contract (6.1); the
        // summaries below are derived from the compacted sequence so the
        // output cap bounds every text channel the result carries (6.3).
        let compacted = Self.compactOutputEvents(
            outputEvents,
            cap: limits.maximumOutputCharacters)
        let consoleText = compacted.events
            .filter { [.log, .info, .debug].contains($0.level) }
            .map(\.text)
            .joined(separator: "\n")
        let capturedErrors = compacted.events
            .filter { [.warn, .error].contains($0.level) }
            .map(\.text)
        let combinedErrorText = (capturedErrors + [errorText].compactMap { $0 })
            .joined(separator: "\n")

        return REPLCallResult(
            status: status,
            outputEvents: compacted.events,
            consoleText: consoleText,
            errorText: combinedErrorText.isEmpty ? nil : combinedErrorText,
            completionText: completionText,
            images: [],
            truncated: compacted.truncated,
            sessionCreated: sessionCreated,
            sessionReset: false)
    }

    /// Head-and-tail compaction of one call's captured console stream,
    /// mirroring the shell output policy (ShellOutputFormatting): the head
    /// keeps two thirds of the cap, the tail one quarter, and a marker event
    /// names the characters removed from the middle. Events survive whole
    /// when they fit inside a cut and are sliced when a cut lands inside
    /// them, so the surviving sequence keeps exact cross-level order and
    /// every event keeps its level; a sliced warn or error event still feeds
    /// the error channel. The truncation flag is computed on the Swift side
    /// after evaluation and is never exposed to scripts, so scripts cannot
    /// tamper with it.
    static func compactOutputEvents(
        _ events: [REPLTextOutputEvent],
        cap: Int
    ) -> (events: [REPLTextOutputEvent], truncated: Bool) {
        guard !events.isEmpty else { return (events, false) }
        let streamLength = events.reduce(0) { $0 + $1.text.count } + events.count - 1
        let boundedCap = max(cap, 0)
        guard streamLength > boundedCap else { return (events, false) }

        let headCut = min(boundedCap * 2 / 3, streamLength)
        let tailCut = min(boundedCap / 4, streamLength - headCut)

        var kept: [REPLTextOutputEvent] = []
        var position = 0
        for event in events {
            let length = event.text.count
            if position >= headCut { break }
            if position + length <= headCut {
                kept.append(event)
            } else {
                kept.append(REPLTextOutputEvent(
                    level: event.level,
                    text: String(event.text.prefix(headCut - position))))
                break
            }
            position += length + 1
        }

        var tail: [REPLTextOutputEvent] = []
        var endPosition = streamLength
        let tailStart = streamLength - tailCut
        for event in events.reversed() {
            let length = event.text.count
            let start = endPosition - length
            if start >= tailStart {
                tail.append(event)
            } else if start + length > tailStart {
                tail.append(REPLTextOutputEvent(
                    level: event.level,
                    text: String(event.text.suffix(start + length - tailStart))))
                break
            } else {
                break
            }
            endPosition = start - 1
        }

        kept.append(REPLTextOutputEvent(
            level: .log,
            text: "... [\(streamLength - headCut - tailCut) chars truncated] ..."))
        kept.append(contentsOf: tail.reversed())
        return (kept, true)
    }

    /// Installs the completion renderer and exposes it to the lowered
    /// wrapper under a fixed non-enumerable global name. The renderer
    /// captures JSON.stringify and String at install time so later mutation
    /// of those globals cannot corrupt rendered output.
    private static let completionRendererScript = """
    (() => {
      const stringify = JSON.stringify;
      const stringValue = String;
      const render = value => {
        if (typeof value === "object" && value !== null) {
          try {
            const encoded = stringify(value);
            if (encoded !== undefined) return encoded;
          } catch (_) {}
        }
        return stringValue(value);
      };
      Object.defineProperty(globalThis, "__turbosparkReplRender", {
        value: render,
        writable: false,
        enumerable: false,
        configurable: false
      });
      return render;
    })()
    """

    /// Promotes the Swift settle bridge to a fixed non-enumerable, frozen
    /// property so scripts cannot replace or enumerate it.
    private static let settleInstallationScript = """
    (() => {
      Object.defineProperty(globalThis, "__turbosparkReplSettled", {
        value: globalThis.__turbosparkReplSettledBridge,
        writable: false,
        enumerable: false,
        configurable: false
      });
      delete globalThis.__turbosparkReplSettledBridge;
    })()
    """
}
