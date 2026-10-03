import Foundation
import JavaScriptCore

struct REPLWorkerEvaluation: Sendable {
    var result: REPLCallResult
    var ranOnMainThread: Bool
}

/// Owns the persistent JavaScriptCore context inside one worker process.
/// All context creation, facade installation, evaluation, and result rendering
/// happen on the same private serial queue.
final class REPLWorkerContext: @unchecked Sendable {
    private let queue = DispatchQueue(label: "com.turbospark.repl.worker-context")
    private var context: JSContext?
    private var completionRenderer: JSValue?
    private var outputEvents: [REPLTextOutputEvent] = []
    private var capturedException: String?
    private var hasCreatedSession = false

    func evaluate(code: String) async -> REPLCallResult {
        await evaluateWithThreadStatus(code: code).result
    }

    func evaluateWithThreadStatus(code: String) async -> REPLWorkerEvaluation {
        await withCheckedContinuation { continuation in
            queue.async { [self] in
                let result = evaluateOnWorkerQueue(code: code)
                continuation.resume(returning: REPLWorkerEvaluation(
                    result: result,
                    ranOnMainThread: Thread.isMainThread))
            }
        }
    }

    private func evaluateOnWorkerQueue(code: String) -> REPLCallResult {
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

        let value = context.evaluateScript(code)
        let exceptionText = capturedException ?? context.exception?.toString()
        context.exception = nil

        guard exceptionText == nil else {
            return makeResult(
                status: .failed,
                completionText: nil,
                errorText: exceptionText,
                sessionCreated: sessionCreated)
        }

        return makeResult(
            status: .completed,
            completionText: renderCompletion(value),
            errorText: nil,
            sessionCreated: sessionCreated)
    }

    private func makeContext() -> JSContext? {
        guard let context = JSContext() else { return nil }
        context.exceptionHandler = { [weak self] _, exception in
            self?.capturedException = exception?.toString()
        }
        completionRenderer = context.evaluateScript(Self.completionRendererScript)
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
        let consoleText = outputEvents
            .filter { [.log, .info, .debug].contains($0.level) }
            .map(\.text)
            .joined(separator: "\n")
        let capturedErrors = outputEvents
            .filter { [.warn, .error].contains($0.level) }
            .map(\.text)
        let combinedErrorText = (capturedErrors + [errorText].compactMap { $0 })
            .joined(separator: "\n")

        return REPLCallResult(
            status: status,
            outputEvents: outputEvents,
            consoleText: consoleText,
            errorText: combinedErrorText.isEmpty ? nil : combinedErrorText,
            completionText: completionText,
            images: [],
            truncated: false,
            sessionCreated: sessionCreated,
            sessionReset: false)
    }

    private static let completionRendererScript = """
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
