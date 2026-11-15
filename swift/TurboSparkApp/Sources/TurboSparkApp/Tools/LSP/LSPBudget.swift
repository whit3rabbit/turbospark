import Foundation

/// OC-10: a hard time cap on an async operation. Unlike a task group, the
/// caller is released the moment the budget expires even when the operation
/// ignores cancellation (a cold server start awaits an unstructured task), so
/// a slow language server can never extend a turn past its budget. The
/// operation keeps running in the background; its late results land in the
/// client's diagnostics store.
enum LSPBudget {
    private final class State<T: Sendable>: @unchecked Sendable {
        private let lock = NSLock()
        private var continuation: CheckedContinuation<T?, Never>?
        private var outcome: T??
        private var tasks: [Task<Void, Never>] = []
        private var done = false

        func install(_ continuation: CheckedContinuation<T?, Never>) {
            lock.lock()
            if let outcome {
                lock.unlock()
                continuation.resume(returning: outcome)
            } else {
                self.continuation = continuation
                lock.unlock()
            }
        }

        func adopt(_ task: Task<Void, Never>) {
            lock.lock()
            let finished = done
            if !finished { tasks.append(task) }
            lock.unlock()
            if finished { task.cancel() }
        }

        func finish(_ value: T?) {
            lock.lock()
            guard !done else { lock.unlock(); return }
            done = true
            let waiting = continuation
            continuation = nil
            if waiting == nil { outcome = .some(value) }
            let all = tasks
            tasks.removeAll()
            lock.unlock()
            for task in all { task.cancel() }
            waiting?.resume(returning: value)
        }
    }

    /// Runs `operation`, returning its value, or nil when `seconds` elapse (or
    /// the caller is cancelled) first.
    static func run<T: Sendable>(
        seconds: TimeInterval, _ operation: @escaping @Sendable () async -> T
    ) async -> T? {
        let state = State<T>()
        return await withTaskCancellationHandler {
            await withCheckedContinuation { (continuation: CheckedContinuation<T?, Never>) in
                state.install(continuation)
                state.adopt(Task { state.finish(await operation()) })
                state.adopt(Task {
                    try? await Task.sleep(nanoseconds: UInt64(max(0, seconds) * 1_000_000_000))
                    state.finish(nil)
                })
            }
        } onCancel: {
            state.finish(nil)
        }
    }
}
