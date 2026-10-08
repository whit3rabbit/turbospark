import Foundation
import JavaScriptCore

struct REPLWorkerEvaluation: Sendable {
    var result: REPLCallResult
    var ranOnMainThread: Bool
}

/// One call's captured console events, images and exception text.
///
/// Lock-guarded because file-response callbacks run script code on the
/// facade's response queue while the worker queue resets and reads this state
/// between calls: an unguarded Array append against a read is a data race.
/// Events are also bounded AS THEY ARRIVE (head plus a rolling tail and a
/// dropped-character count), so a tight `console.log` loop cannot grow the
/// worker until it runs out of memory before the final head-and-tail cap in
/// `makeResult` gets to run.
final class REPLCallCapture: @unchecked Sendable {
    private let lock = NSLock()
    private let cap: Int
    private var head: [REPLTextOutputEvent] = []
    private var headCharacters = 0
    private var tail: [REPLTextOutputEvent] = []
    private var tailCharacters = 0
    private var droppedCharacters = 0
    private var images: [REPLEmittedImage] = []
    private var exception: String?

    init(cap: Int) {
        self.cap = max(cap, 0)
    }

    func reset() {
        lock.lock(); defer { lock.unlock() }
        head = []; headCharacters = 0
        tail = []; tailCharacters = 0
        droppedCharacters = 0
        images = []
        exception = nil
    }

    func append(_ event: REPLTextOutputEvent) {
        lock.lock(); defer { lock.unlock() }
        let weight = event.text.count + 1
        if headCharacters < cap {
            head.append(event)
            headCharacters += weight
            return
        }
        tail.append(event)
        tailCharacters += weight
        // Keep at least the newest event so a single huge line still leaves
        // a tail; the final compaction slices it to size.
        while tailCharacters > cap, tail.count > 1 {
            let removed = tail.removeFirst()
            let removedWeight = removed.text.count + 1
            tailCharacters -= removedWeight
            droppedCharacters += removedWeight
        }
    }

    func append(_ image: REPLEmittedImage) {
        lock.lock(); defer { lock.unlock() }
        images.append(image)
    }

    var capturedException: String? {
        get { lock.lock(); defer { lock.unlock() }; return exception }
        set { lock.lock(); defer { lock.unlock() }; exception = newValue }
    }

    /// The bounded event sequence, with a marker event where the dropped
    /// middle was, and whether anything was dropped.
    func snapshot() -> (events: [REPLTextOutputEvent], images: [REPLEmittedImage], dropped: Bool) {
        lock.lock(); defer { lock.unlock() }
        var events = head
        if droppedCharacters > 0 {
            events.append(REPLTextOutputEvent(
                level: .log,
                text: "... [\(droppedCharacters) chars truncated] ..."))
        }
        events.append(contentsOf: tail)
        return (events, images, droppedCharacters > 0)
    }
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
    private let capture: REPLCallCapture
    private var capturedException: String? {
        get { capture.capturedException }
        set { capture.capturedException = newValue }
    }
    private var hasCreatedSession = false
    private let outcomeBox = REPLSettlementBox()
    private let limits: REPLLimits
    private let configuration: REPLSessionConfiguration
    private let fileChannel: any REPLFileRequestChannel

    /// `fileChannel` is the bounded request channel for `repl.fs`. Omitting
    /// it binds the fail-closed channel, so a worker created without an
    /// app-side broker denies every file operation instead of reaching the
    /// filesystem (5.5). The channel never carries a grant list.
    ///
    /// `configuration.artifactDirectory` receives the images scripts emit
    /// through `repl.emitImage`. The default parks emissions under the
    /// store root's "repl-artifacts" directory (redirected automatically
    /// for test hosts by AppStorageRoot) until the tool wiring (task 5.x)
    /// supplies the per-chat directory. Nothing constructs a REPL session
    /// from the tool path before that wiring exists, so the subsystem is
    /// inert without a feature flag.
    init(
        limits: REPLLimits = REPLLimits(),
        configuration: REPLSessionConfiguration = REPLSessionConfiguration(
            artifactDirectory: AppStorageRoot.subdirectory("repl-artifacts")),
        fileChannel: (any REPLFileRequestChannel)? = nil
    ) {
        self.limits = limits
        self.capture = REPLCallCapture(cap: limits.maximumOutputCharacters)
        self.configuration = configuration
        self.fileChannel = fileChannel ?? REPLNoAccessFileChannel()
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
        capture.reset()

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

        let capture = self.capture
        let facade = REPLHostFacade { event in
            capture.append(event)
        }
        facade.installConsole(into: context)
        // The capability bridges install first and the surface seals once,
        // after every piece is present, because a sealed `repl` object can
        // never be extended.
        facade.installFileRequestBridge(into: context, channel: fileChannel)
        facade.installImageEmitterBridge(
            into: context,
            config: configuration,
            limits: limits
        ) { image in
            capture.append(image)
        }
        facade.sealReplSurface(into: context)
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
        let captured = capture.snapshot()
        let compacted = Self.compactOutputEvents(
            captured.events,
            cap: limits.maximumOutputCharacters)
        let consoleText = compacted.events
            .filter { [.log, .info, .debug].contains($0.level) }
            .map(\.text)
            .joined(separator: "\n")
        let capturedErrors = compacted.events
            .filter { [.warn, .error].contains($0.level) }
            .map(\.text)
        // The completion and error text cross the same IPC and land in the
        // model context, so they get the same head-and-tail bound.
        let boundedCompletion = completionText.map {
            Self.boundText($0, cap: limits.maximumOutputCharacters)
        }
        let boundedError = errorText.map {
            Self.boundText($0, cap: limits.maximumOutputCharacters)
        }
        let combinedErrorText = (capturedErrors + [boundedError?.text].compactMap { $0 })
            .joined(separator: "\n")

        return REPLCallResult(
            status: status,
            outputEvents: compacted.events,
            consoleText: consoleText,
            errorText: combinedErrorText.isEmpty ? nil : combinedErrorText,
            completionText: boundedCompletion?.text,
            images: captured.images,
            truncated: compacted.truncated || captured.dropped
                || (boundedCompletion?.truncated ?? false) || (boundedError?.truncated ?? false),
            sessionCreated: sessionCreated,
            sessionReset: false)
    }

    /// Head-and-tail bound for one text value, same proportions as
    /// `compactOutputEvents`: the head keeps two thirds of the cap, the tail
    /// one quarter, and a marker names the characters removed.
    static func boundText(_ text: String, cap: Int) -> (text: String, truncated: Bool) {
        let boundedCap = max(cap, 0)
        guard text.utf8.count > boundedCap, text.count > boundedCap else { return (text, false) }
        let headCut = boundedCap * 2 / 3
        let tailCut = boundedCap / 4
        let removed = text.count - headCut - tailCut
        return (
            String(text.prefix(headCut))
                + "\n... [\(removed) chars truncated] ...\n"
                + String(text.suffix(tailCut)),
            true)
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
