import Foundation

// MARK: - Skill Tool (OpenCode Compatible)

public struct SkillInput: Codable, Sendable, Equatable {
    public var name: String

    public init(name: String) {
        self.name = name
    }
}

public struct SkillOutput: Codable, Sendable, Equatable {
    public var name: String
    public var directory: String
    public var output: String

    public init(name: String, directory: String = "", output: String) {
        self.name = name
        self.directory = directory
        self.output = output
    }
}

// MARK: - OpenAI Tool Definition for Skill

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
