import Foundation

// MARK: - Skill Tool (OpenCode / Claude / Antigravity Compatible)

/// Input parameters for loading a specialized workflow skill.
public struct SkillInput: Codable, Sendable, Equatable {
    /// Name of the skill to load from the project or global skill catalog.
    public var name: String
    /// Optional parameters/arguments to substitute in the skill template.
    public var arguments: [String: String]?

    public init(name: String, arguments: [String: String]? = nil) {
        self.name = name
        self.arguments = arguments
    }
}

/// Output returned when a skill is loaded and injected into prompt context.
public struct SkillOutput: Codable, Sendable, Equatable {
    /// Name of the activated skill.
    public var name: String
    /// Source directory where the skill definition was located.
    public var directory: String
    /// Scope of the skill (User or Project).
    public var scope: String
    /// Textual instructions, guides, and workflow rules of the skill.
    public var output: String

    public init(name: String, directory: String = "", scope: String = "User Scope", output: String) {
        self.name = name
        self.directory = directory
        self.scope = scope
        self.output = output
    }
}

// MARK: - OpenAI Tool Definition for Skill

/// OpenAI tool definition schemas for the skill loading subsystem.
public enum SkillToolDefinitions {
    public static let skill = OpenAITool.function(
        name: "skill",
        description: "Load a specialized skill and inject its workflow instructions, rules, and reference scripts into the conversation.",
        parameters: .object(
            properties: [
                "name": .string(description: "The name of the skill to load from project or user skills."),
                "arguments": .object(properties: [:], description: "Optional dictionary of named argument values to substitute in the skill.")
            ],
            required: ["name"]
        )
    )

    public static let all: [OpenAITool] = [skill]
}
