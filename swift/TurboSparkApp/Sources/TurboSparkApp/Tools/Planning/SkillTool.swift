import Foundation

// MARK: - Skill Tool (OpenCode Compatible)

/// Input parameters for loading a specialized workflow skill.
public struct SkillInput: Codable, Sendable, Equatable {
    /// Name of the skill to load from the project or global skill catalog.
    public var name: String

    public init(name: String) {
        self.name = name
    }
}

/// Output returned when a skill is loaded and injected into prompt context.
public struct SkillOutput: Codable, Sendable, Equatable {
    /// Name of the activated skill.
    public var name: String
    /// Source directory where the skill definition was located.
    public var directory: String
    /// Textual instructions, guides, and workflow rules of the skill.
    public var output: String

    public init(name: String, directory: String = "", output: String) {
        self.name = name
        self.directory = directory
        self.output = output
    }
}

// MARK: - OpenAI Tool Definition for Skill

/// OpenAI tool definition schemas for the skill loading subsystem.
public enum SkillToolDefinitions {
    public static let skill = OpenAITool.function(
        name: "skill",
        description: "Load a specialized skill and inject its workflow instructions and resources into the current conversation.",
        parameters: .object(
            properties: [
                "name": .string(description: "The name of the skill to load from available skills.")
            ],
            required: ["name"]
        )
    )

    public static let all: [OpenAITool] = [skill]
}
