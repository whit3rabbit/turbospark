import Foundation

// The Rust side emits camelCase, so every type here decodes with no
// `CodingKeys` and the two definitions cannot drift over a spelling.

/// One image attached to a message.
///
/// A placeholder as far as the prompt is concerned: the template renders one
/// marker per image and the tower's rows are injected at that marker, so what
/// travels here is only where the pixels can be found.
public enum ChatImage: Codable, Sendable, Equatable {
    /// A file the engine process can read. The engine and the app share an
    /// address space, so this costs no copy of the bytes.
    case path(String)
    /// A bare base64 payload, or a full `data:<media>;base64,<data>` URL.
    case base64(String)
}

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
    /// Images on this message, in order.
    ///
    /// **Empty is the whole of the compatibility story.** With no images this
    /// type encodes `content` as a bare string, which is byte for byte what
    /// this binding sent before images existed; only a message that carries
    /// one switches to the ordered-parts shape. That is why `content` stayed
    /// a `String` rather than becoming an enum -- it is read in dozens of
    /// places that have nothing to do with vision.
    ///
    /// Gate on `SessionInfo.vision.active` before offering a way to fill
    /// this: an install can carry a tower and still refuse every image.
    public var images: [ChatImage]
    /// Tool calls an `assistant` message made. Pass back what a turn returned
    /// (`GenerationResult.toolCalls`) when replaying it, so the next render
    /// shows the model its own call. Empty on every other message.
    public var toolCalls: [GenerationToolCall]
    /// On a `tool` message: the id of the call it answers. Keep the ids the
    /// engine returned: the Gemma template resolves a tool turn's function
    /// name by matching them and renders "unknown" otherwise.
    public var toolCallId: String?
    /// On a `tool` message: the name of the function that ran.
    public var name: String?

    /// Creates a new chat message with the given role and content.
    public init(
        role: Role, content: String, images: [ChatImage] = [],
        toolCalls: [GenerationToolCall] = [], toolCallId: String? = nil, name: String? = nil
    ) {
        self.role = role
        self.content = content
        self.images = images
        self.toolCalls = toolCalls
        self.toolCallId = toolCallId
        self.name = name
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

    /// Convenience factory for an assistant message. Pass `toolCalls` to replay
    /// a turn that called tools (its `content` may be empty).
    public static func assistant(_ content: String, toolCalls: [GenerationToolCall] = []) -> ChatMessage {
        ChatMessage(role: .assistant, content: content, toolCalls: toolCalls)
    }

    /// Convenience factory for a tool message. Give `toolCallId` (and `name`)
    /// when it answers a call, so the template can pair them.
    public static func tool(_ content: String, toolCallId: String? = nil, name: String? = nil) -> ChatMessage {
        ChatMessage(role: .tool, content: content, toolCallId: toolCallId, name: name)
    }

    // MARK: - Wire shape

    private enum CodingKeys: String, CodingKey { case role, content, toolCalls, toolCallId, name }

    private enum PartKind: String, Codable { case text, image }

    /// One part as Rust serializes it (crates/ffi/src/wire.rs WirePart). The
    /// Rust side has no skip_serializing_if, so an image part always carries
    /// both `path` and `base64` with the unused one as JSON null. A
    /// `[String: String]` decode throws on that null, so every field is
    /// optional here. Encoding stays on the dictionary form above.
    private struct WirePart: Decodable {
        var type: String?
        var text: String?
        var path: String?
        var base64: String?
    }

    /// **IMAGES ARE PREPENDED, NOT APPENDED, and that is matched to the
    /// reference rather than chosen.** `apply_chat_template(processor,
    /// config, question, num_images=1)` builds `[image, text]`, so the marker
    /// run comes first and the question follows. Appending would move every
    /// mRoPE position past the image and produce a different prompt for the
    /// same request.
    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(role, forKey: .role)
        // Omitted when empty, so a message without tools is byte for byte what
        // this binding sent before tools existed.
        if !toolCalls.isEmpty { try container.encode(toolCalls, forKey: .toolCalls) }
        try container.encodeIfPresent(toolCallId, forKey: .toolCallId)
        try container.encodeIfPresent(name, forKey: .name)
        guard !images.isEmpty else {
            // The pre-vision shape, byte for byte.
            try container.encode(content, forKey: .content)
            return
        }
        var parts: [[String: String]] = images.map { image in
            switch image {
            case .path(let p): return ["type": PartKind.image.rawValue, "path": p]
            case .base64(let b): return ["type": PartKind.image.rawValue, "base64": b]
            }
        }
        if !content.isEmpty {
            parts.append(["type": PartKind.text.rawValue, "text": content])
        }
        try container.encode(parts, forKey: .content)
    }

    /// Accepts either shape, so a value that made a round trip through the
    /// engine comes back equal to what went in.
    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        role = try container.decode(Role.self, forKey: .role)
        toolCalls = try container.decodeIfPresent([GenerationToolCall].self, forKey: .toolCalls) ?? []
        toolCallId = try container.decodeIfPresent(String.self, forKey: .toolCallId)
        name = try container.decodeIfPresent(String.self, forKey: .name)
        if let text = try? container.decode(String.self, forKey: .content) {
            content = text
            images = []
            return
        }
        let parts = try container.decodeIfPresent([WirePart].self, forKey: .content) ?? []
        var text = ""
        var decoded: [ChatImage] = []
        for part in parts {
            switch part.type {
            case PartKind.image.rawValue:
                if let p = part.path {
                    decoded.append(.path(p))
                } else if let b = part.base64 {
                    decoded.append(.base64(b))
                }
            default:
                text += part.text ?? ""
            }
        }
        content = text
        images = decoded
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
