import Foundation
import TurboSpark

/// In-memory state for one image request. The PNG is deliberately transient;
/// `saveImage()` moves it into the profile-scoped artifact directory.
public struct AppImageJob: Identifiable, Sendable, Equatable {
    public enum Status: String, Sendable, Equatable {
        case waiting
        case generating
        case completed
        case cancelled
        case failed
    }

    public let id: UUID
    public let chatID: UUID
    public let options: ImageGenerateOptions
    public var status: Status
    public var stage: String?
    public var completed: Int
    public var total: Int
    public var result: ImageGenerationResult?
    public var savedPath: String?

    public init(
        id: UUID = UUID(),
        chatID: UUID,
        options: ImageGenerateOptions,
        status: Status = .waiting,
        stage: String? = nil,
        completed: Int = 0,
        total: Int = 0,
        result: ImageGenerationResult? = nil,
        savedPath: String? = nil
    ) {
        self.id = id
        self.chatID = chatID
        self.options = options
        self.status = status
        self.stage = stage
        self.completed = completed
        self.total = total
        self.result = result
        self.savedPath = savedPath
    }
}

/// Process-wide FIFO gate for heavyweight image jobs. Text generation and
/// server work share the native FFI gate; this actor serializes app jobs and
/// prevents two chats from opening competing image backends at once.
actor ImageJobCoordinator {
    static let shared = ImageJobCoordinator()

    private struct Waiter {
        let id: UUID
        let continuation: CheckedContinuation<Bool, Never>
    }

    private var available = true
    private var waiters: [Waiter] = []

    func acquire() async -> Bool {
        guard !Task.isCancelled else { return false }
        if available {
            available = false
            return true
        }
        let id = UUID()
        return await withTaskCancellationHandler(operation: {
            await withCheckedContinuation { continuation in
                if Task.isCancelled {
                    continuation.resume(returning: false)
                } else {
                    waiters.append(Waiter(id: id, continuation: continuation))
                }
            }
        }, onCancel: {
            Task { await self.cancel(id: id) }
        })
    }

    private func cancel(id: UUID) {
        guard let index = waiters.firstIndex(where: { $0.id == id }) else { return }
        let waiter = waiters.remove(at: index)
        waiter.continuation.resume(returning: false)
    }

    func release() {
        if let next = waiters.first {
            waiters.removeFirst()
            next.continuation.resume(returning: true)
        } else {
            available = true
        }
    }
}
