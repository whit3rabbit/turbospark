import Foundation

// The Rust side emits camelCase, so every type here decodes with no
// `CodingKeys` and the two definitions cannot drift over a spelling.

/// A single message in a conversation.
public struct ChatMessage: Codable, Sendable, Equatable {
    /// The sender role of the message.
    public enum Role: String, Codable, Sendable {
        case system, developer, user, assistant, tool
    }

    /// The role of the message sender.
    public var role: Role
    /// The message text content.
    public var content: String

    /// Creates a new chat message with the given role and content.
    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }

    /// Convenience factory for a system message.
    public static func system(_ content: String) -> ChatMessage {
        ChatMessage(role: .system, content: content)
    }

    /// Convenience factory for a developer instruction message.
    public static func developer(_ content: String) -> ChatMessage {
        ChatMessage(role: .developer, content: content)
    }

    /// Convenience factory for a user message.
    public static func user(_ content: String) -> ChatMessage {
        ChatMessage(role: .user, content: content)
    }

    /// Convenience factory for an assistant message.
    public static func assistant(_ content: String) -> ChatMessage {
        ChatMessage(role: .assistant, content: content)
    }

    /// Convenience factory for a tool message.
    public static func tool(_ content: String) -> ChatMessage {
        ChatMessage(role: .tool, content: content)
    }
}

/// The result of fitting a conversation into a context window budget.
public struct WindowFitOutcome: Codable, Sendable, Equatable {
    /// The messages retained after pruning older turns.
    public let retained: [ChatMessage]
    /// The token count of the rendered retained messages.
    public let measuredTokens: Int
    /// Number of older turns removed to fit the budget.
    public let removedTurnCount: Int
    /// Whether there is room remaining for generation within the budget.
    public let hasRoomForGeneration: Bool
}
