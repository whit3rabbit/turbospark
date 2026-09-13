import Foundation

/// A reusable, app-wide system prompt. The selected row supplies the default
/// for chats without their own prompt and for the in-app server.
public struct AppSystemPrompt: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var name: String
    public var instructions: String

    public init(id: UUID = UUID(), name: String, instructions: String) {
        self.id = id
        self.name = name
        self.instructions = instructions
    }

    /// One malformed row must not discard a user's other saved prompts.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = container.decodeLenient(UUID.self, forKey: .id, fallback: UUID())
        name = container.decodeLenient(String.self, forKey: .name, fallback: "System Prompt")
        instructions = container.decodeLenient(String.self, forKey: .instructions, fallback: "")
    }

    /// Compact starter prompts for local models. They are normal stored rows,
    /// so a user may edit or remove them like any other prompt.
    public static let builtIns: [AppSystemPrompt] = [
        AppSystemPrompt(
            id: UUID(uuidString: "4AAB30AF-7E13-41CB-89E2-0FB4AA40A2D4")!,
            name: "TurboSpark Agent",
            instructions: "You are TurboSpark, a desktop coding agent on macOS. Use tools to inspect and change the project when asked. Keep edits focused, verify relevant work, and do not commit unless asked. Be concise."),
        AppSystemPrompt(
            id: UUID(uuidString: "C532F6E5-E4F4-4AC6-9545-5ECF0D3BB0DB")!,
            name: "Compact Agent",
            instructions: "Help with the task. Use tools for project work, inspect before editing, and verify changes. Be concise."),
        AppSystemPrompt(
            id: UUID(uuidString: "427B5BE6-BED8-427A-BFC2-4C61C6B38E33")!,
            name: "Code Reviewer",
            instructions: "Review code for correctness, regressions, and missing tests. Inspect evidence before conclusions. Do not change files unless asked.")
    ]
}
