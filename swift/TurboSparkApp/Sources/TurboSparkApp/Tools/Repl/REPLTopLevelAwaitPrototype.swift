import Foundation
import JavaScriptCore

/// Isolated proof prototype for task 1.5 of the js-repl-tool spec.
///
/// Decision flow per call, all on one private serial queue:
/// 1. JSCheckScriptSyntax probes the code as a classic script. Valid classic
///    scripts are evaluated natively with evaluateScript and keep native
///    lexical declaration semantics (no transform, design point 1).
/// 2. Classic scripts cannot contain top-level await, so a failed probe hands
///    the source to the bounded token-based parser shared with the production
///    worker context (REPLScriptLowering.swift; a real lexer and top-level
///    statement scanner, never a regex over source text).
/// 3. If the parser finds top-level await and every top-level statement is
///    inside the supported subset, the program is lowered to an async wrapper
///    whose top-level declarations are rewritten to persistent assignments on
///    the global object. The wrapper's promise settles through native
///    callbacks, with a microtask-drain loop plus a native timer wheel so
///    host-delayed promises settle too.
/// 4. Anything outside the subset returns the dedicated unsupportedSyntax
///    status before any evaluation runs, leaving the session unchanged.
///
/// Recorded bounds of the proven strategy (inherited unchanged by the
/// production path in REPLWorkerContext):
/// - Declarations persist as writable global properties (var-like). const
///   enforcement, temporal dead zones, and lexical redeclaration errors are
///   not preserved, and a native lexical binding of the same name shadows a
///   lowered property assignment (see the divergence proof test).
/// - awaits inside template substitutions are not detected and surface as
///   parseError; awaits inside shorthand class or object methods may be
///   overcounted, which stays safe because valid classic scripts never reach
///   the parser.
/// - A final compound statement completes as undefined because the lowering
///   does not capture block completion values.
/// - The settlement deadline cannot terminate the context in-process; the
///   production supervisor terminates the worker process instead (task 3.2),
///   which also prevents a late settlement from a timed-out call recording
///   into the next call's outcome.
final class REPLTopLevelAwaitPrototype: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.turbospark.repl.tla-proof")
    private var context: JSContext?
    private var renderer: JSValue?
    private var capturedException: String?
    private var hasCreatedSession = false
    private var hostTimers: [HostTimer] = []
    private let outcomeBox = REPLSettlementBox()

    private struct HostTimer {
        var fireDate: Date
        var value: JSValue
        var resolve: JSValue
    }

    // MARK: Public proof surface

    /// Evaluates code through the decision flow described in the class docs.
    func evaluate(code: String, settlementTimeout: TimeInterval = 5) async -> REPLCallResult {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                continuation.resume(
                    returning: evaluateOnQueue(code: code, settlementTimeout: settlementTimeout))
            }
        }
    }

    /// Probes the code with the public JSCheckScriptSyntax entry point.
    func classicSyntaxProbe(code: String) async -> REPLClassicSyntaxProbe {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                continuation.resume(returning: classicSyntaxProbeOnQueue(code: code))
            }
        }
    }

    /// Runs a native probe closure on the prototype queue so tests can touch
    /// the JavaScriptCore C API without breaking single-thread confinement.
    func runProbe<T: Sendable>(_ body: @escaping (JSContext) -> T) async -> T {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                guard let context = context ?? makeContext() else {
                    preconditionFailure("JavaScriptCore context unavailable")
                }
                continuation.resume(returning: body(context))
            }
        }
    }

    // MARK: Queue-confined implementation

    private func evaluateOnQueue(code: String, settlementTimeout: TimeInterval) -> REPLCallResult {
        let sessionCreated = !hasCreatedSession
        capturedException = nil
        hostTimers.removeAll()

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
            settleGlobal: "__replProofSettled",
            renderGlobal: "__replProofRender")
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
            fireDueHostTimers(context: context)
            drainMicrotasks(context: context)
            if outcomeBox.isSettled { break }
            if Date() >= deadline {
                hostTimers.removeAll()
                return makeResult(
                    status: .timedOut,
                    completionText: nil,
                    errorText: "top-level await did not settle within \(settlementTimeout) s; "
                        + "the in-process prototype cannot terminate the context",
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

    private func classicSyntaxProbeOnQueue(code: String) -> REPLClassicSyntaxProbe {
        guard let context = context ?? makeContext() else {
            return REPLClassicSyntaxProbe(ok: false, message: "context unavailable")
        }
        self.context = context
        return REPLClassicScriptSyntax.check(context: context, code: code)
    }

    private func fireDueHostTimers(context: JSContext) {
        let now = Date()
        let due = hostTimers.filter { $0.fireDate <= now }
        guard !due.isEmpty else { return }
        hostTimers.removeAll { $0.fireDate <= now }
        for timer in due {
            _ = timer.resolve.call(withArguments: [timer.value])
        }
        context.exception = nil
    }

    /// Any script evaluation drains the microtask queue at its end, which is
    /// the settlement mechanism this prototype relies on. Mutation evidence:
    /// JSValue.call also drains microtasks after a native resolve, so this
    /// explicit drain is a defensive checkpoint rather than the sole
    /// load-bearing step; production keeps an explicit checkpoint instead of
    /// depending on call-side draining.
    private func drainMicrotasks(context: JSContext) {
        _ = context.evaluateScript("0")
        context.exception = nil
    }

    private func makeContext() -> JSContext? {
        guard let context = JSContext() else { return nil }
        context.exceptionHandler = { [weak self] _, exception in
            self?.capturedException = exception?.toString()
        }

        let box = outcomeBox
        let settled: @convention(block) (Bool, String) -> Void = { fulfilled, text in
            box.record(fulfilled: fulfilled, text: text)
        }
        context.setObject(settled, forKeyedSubscript: "__replProofSettled" as NSString)

        let schedule: @convention(block) (Double, JSValue, JSValue) -> Void = { [weak self]
            milliseconds, value, resolve in
            guard let self else { return }
            self.hostTimers.append(
                HostTimer(
                    fireDate: Date().addingTimeInterval(milliseconds / 1000),
                    value: value,
                    resolve: resolve))
        }
        context.setObject(schedule, forKeyedSubscript: "__replProofScheduleDelay" as NSString)

        context.evaluateScript(Self.hostSetupScript)
        context.evaluateScript("globalThis.__replProofRender = \(Self.rendererScript);")
        renderer = context.evaluateScript("globalThis.__replProofRender")
        context.exception = nil
        return context
    }

    private func renderCompletion(_ value: JSValue?) -> String {
        guard let value else { return "undefined" }
        return renderer?.call(withArguments: [value])?.toString() ?? value.toString()
    }

    private func makeResult(
        status: REPLCallResult.Status,
        completionText: String?,
        errorText: String?,
        sessionCreated: Bool
    ) -> REPLCallResult {
        REPLCallResult(
            status: status,
            outputEvents: [],
            consoleText: "",
            errorText: errorText,
            completionText: completionText,
            images: [],
            truncated: false,
            sessionCreated: sessionCreated,
            sessionReset: false)
    }

    // MARK: Installed host scripts

    private static let hostSetupScript = """
    globalThis.__replProofDelayedResolve = (milliseconds, value) =>
      new Promise(resolve => __replProofScheduleDelay(milliseconds, value, resolve));
    """

    private static let rendererScript = """
    (() => {
      const stringify = JSON.stringify;
      const stringValue = String;
      return value => {
        if (typeof value === "object" && value !== null) {
          try {
            const encoded = stringify(value);
            if (encoded !== undefined) return encoded;
          } catch (_) {}
        }
        return stringValue(value);
      };
    })()
    """
}
