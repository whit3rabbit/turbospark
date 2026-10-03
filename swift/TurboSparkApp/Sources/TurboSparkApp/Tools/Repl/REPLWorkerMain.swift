import Foundation
import JavaScriptCore

enum REPLWorkerMain {
    static let workerModeArgument = "--turbospark-js-repl-worker"
    private static let smokeArgument = "--smoke"

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
    /// JavaScriptCore runtime in the packaged executable.
    @discardableResult
    static func runIfRequested(
        arguments: [String] = CommandLine.arguments,
        writeOutput: (String) -> Void = { print($0) }
    ) -> Bool {
        guard isWorkerInvocation(arguments) else { return false }
        guard arguments.contains(smokeArgument) else { return true }

        let result = SmokeResultBox()
        let finished = DispatchSemaphore(value: 0)
        DispatchQueue(label: "com.turbospark.repl.worker-smoke").async {
            defer { finished.signal() }
            guard let context = JSContext(),
                  let value = context.evaluateScript("1 + 1"),
                  context.exception == nil else {
                result.store(completionText: nil, ranOnMainThread: Thread.isMainThread)
                return
            }
            result.store(
                completionText: value.toString(),
                ranOnMainThread: Thread.isMainThread)
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
}
