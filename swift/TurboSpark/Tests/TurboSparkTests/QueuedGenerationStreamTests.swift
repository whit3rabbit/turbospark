import XCTest
@testable import TurboSpark

final class QueuedGenerationStreamTests: XCTestCase {
    private final class Recorder: @unchecked Sendable {
        private let lock = NSLock()
        private var entries: [String] = []

        func record(_ entry: String) {
            lock.lock()
            defer { lock.unlock() }
            entries.append(entry)
        }

        var values: [String] {
            lock.lock()
            defer { lock.unlock() }
            return entries
        }
    }

    private func drain(_ queue: DispatchQueue) async {
        await withCheckedContinuation { continuation in
            queue.async { continuation.resume() }
        }
    }

    func testCancelledQueuedGenerationDoesNotRunOrCancelTheNextGeneration() async throws {
        let queue = DispatchQueue(label: "test.queued-generation")
        let recorder = Recorder()
        queue.suspend()
        var suspended = true
        defer { if suspended { queue.resume() } }
        let stream = QueuedGenerationStream.make(
            queue: queue, cancelActive: { recorder.record("cancel") }
        ) { continuation, _ in
            recorder.record("cancelled generation ran")
            continuation.yield(.content("stale title"))
        }
        let consumer = Task {
            for try await _ in stream {}
        }
        consumer.cancel()
        _ = try? await consumer.value

        queue.resume()
        suspended = false
        await drain(queue)
        XCTAssertEqual(recorder.values, [], "Cancelled queued work must never enter generation.")

        let foreground = QueuedGenerationStream.make(
            queue: queue, cancelActive: { recorder.record("foreground cancelled") }
        ) { continuation, didReceiveEvent in
            recorder.record("foreground")
            didReceiveEvent()
            continuation.yield(.content("reply"))
        }
        var reply = ""
        for try await event in foreground {
            if case .content(let text) = event { reply += text }
        }
        XCTAssertEqual(reply, "reply")
        XCTAssertEqual(recorder.values, ["foreground"])
    }

    func testCancellingAnActiveGenerationStopsItAndReplaysCancellationAfterArming() async {
        let queue = DispatchQueue(label: "test.active-generation")
        let recorder = Recorder()
        let started = expectation(description: "Generation began before its first event")
        let release = DispatchSemaphore(value: 0)
        let stream = QueuedGenerationStream.make(
            queue: queue,
            cancelActive: {
                recorder.record("cancel")
                release.signal()
            }
        ) { _, didReceiveEvent in
            started.fulfill()
            guard release.wait(timeout: .now() + 2) == .success else {
                recorder.record("active cancellation missing")
                return
            }
            // Models reset their cancellation flag during setup. The first
            // event must reapply a cancellation that raced that setup.
            didReceiveEvent()
            didReceiveEvent()
        }
        let consumer = Task {
            for try await _ in stream {}
        }
        await fulfillment(of: [started], timeout: 1)
        consumer.cancel()
        _ = try? await consumer.value
        await drain(queue)

        XCTAssertEqual(recorder.values, ["cancel", "cancel"])
    }
}
