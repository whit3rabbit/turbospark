import Foundation

/// The `/stats` session summary, computed from the chat row plus its usage
/// ledger. A VALUE rather than view code so it can be tested and so the
/// sheet stays a dumb renderer.
public struct AppChatStats: Equatable, Sendable {
    public var title: String
    public var createdAt: Date
    public var updatedAt: Date
    public var userTurns: Int
    public var assistantTurns: Int
    public var toolCalls: Int
    public var failedToolCalls: Int
    public var attachments: Int
    public var images: Int
    public var todoTotal: Int
    public var todoCompleted: Int
    public var compactedMessages: Int
    public var hasSummary: Bool
    /// Recorded engine usage; nil when nothing has been recorded (a chat
    /// from before the ledger existed, or a ghost).
    public var usage: AppChatUsageLedger?
    /// Model most recently used in this chat, when the app knows one.
    public var modelAlias: String?

    /// Computes the summary. `totalChars` is the only estimated number and
    /// is deliberately ABSENT from the result: a fake token total beside
    /// real ones would be the number everyone quotes.
    public init(chat: AppChat, modelAlias: String?) {
        title = chat.title
        createdAt = chat.createdAt
        updatedAt = chat.updatedAt
        userTurns = chat.messages.filter { $0.role == .user }.count
        assistantTurns = chat.messages.filter { $0.role == .assistant }.count
        toolCalls = chat.messages.reduce(0) { $0 + $1.toolCalls.count }
        var failures = 0
        for message in chat.messages {
            for result in message.toolResults where result.isError {
                failures += 1
            }
        }
        failedToolCalls = failures
        attachments = chat.messages.reduce(0) { $0 + $1.imagePaths.count } + chat.draftAttachments.count
        images = chat.messages.reduce(0) { $0 + $1.imagePaths.count }
        todoTotal = chat.todos.count
        todoCompleted = chat.todos.filter { $0.isCompleted }.count
        compactedMessages = chat.compactedMessageCount
        hasSummary = !(chat.contextSummary ?? "").isEmpty
        usage = chat.usage
        self.modelAlias = modelAlias
    }

    /// Wall-clock age of the conversation, rounded for display.
    public var durationText: String {
        let interval = max(0, updatedAt.timeIntervalSince(createdAt))
        let formatter = DateComponentsFormatter()
        formatter.allowedUnits = [.day, .hour, .minute]
        formatter.maximumUnitCount = 2
        formatter.unitsStyle = .abbreviated
        return formatter.string(from: interval) ?? "0m"
    }
}
