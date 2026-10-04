import Foundation

/// Keeps cancellation attached to the queued request that owns the stream.
/// Session cancellation alone can be cleared when a queued C call starts.
enum QueuedGenerationStream {
    typealias Continuation = AsyncThrowingStream<GenerationEvent, Error>.Continuation

    static func make(
        queue: DispatchQueue,
        cancelActive: @escaping @Sendable () -> Void,
        operation: @escaping @Sendable (Continuation, @escaping @Sendable () -> Void) throws -> Void
    ) -> AsyncThrowingStream<GenerationEvent, Error> {
        AsyncThrowingStream { continuation in
            let request = Request(cancelActive: cancelActive)
            continuation.onTermination = { reason in
                if case .cancelled = reason { request.cancel() }
            }
            queue.async {
                guard request.begin() else {
                    continuation.finish(throwing: CancellationError())
                    return
                }
                defer { request.finish() }
                do {
                    try operation(continuation) { request.didReceiveEvent() }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
    }

    /// All state and active cancellation calls share the lock, so a late
    /// termination cannot cancel the next request after this worker exits.
    private final class Request: @unchecked Sendable {
        private enum Phase { case queued, running, finished }
        private let lock = NSLock()
        private var phase = Phase.queued
        private var cancelled = false
        private var receivedEvent = false
        private let cancelActive: @Sendable () -> Void

        init(cancelActive: @escaping @Sendable () -> Void) {
            self.cancelActive = cancelActive
        }

        func begin() -> Bool {
            lock.lock()
            defer { lock.unlock() }
            guard !cancelled else { return false }
            phase = .running
            return true
        }

        func cancel() {
            lock.lock()
            defer { lock.unlock() }
            guard phase != .finished else { return }
            cancelled = true
            if phase == .running { cancelActive() }
        }

        func didReceiveEvent() {
            lock.lock()
            defer { lock.unlock() }
            guard !receivedEvent else { return }
            receivedEvent = true
            // Rust arms cancellation inside ts_generate. Replay a cancel
            // racing that setup at its first event, after the flag is armed.
            if cancelled && phase == .running { cancelActive() }
        }

        func finish() {
            lock.lock()
            defer { lock.unlock() }
            phase = .finished
        }
    }
}
