import Foundation

/// Execution type for a user-defined custom tool.
public enum CustomToolExecutionType: String, Codable, CaseIterable, Sendable {
    case command = "command"
    case script = "script"
    case http = "http"
}

/// Execution configuration for a custom tool.
public struct CustomToolExecution: Codable, Sendable, Equatable {
    public var type: CustomToolExecutionType
    public var command: String?
    public var scriptContent: String?
    public var scriptInterpreter: String?
    public var httpURL: String?
    public var httpMethod: String?
    public var httpHeaders: [String: String]?
    public var arguments: [String]?
    public var environment: [String: String]?
    public var timeoutSeconds: Double?

    public init(
        type: CustomToolExecutionType = .command,
        command: String? = nil,
        scriptContent: String? = nil,
        scriptInterpreter: String? = "/bin/zsh",
        httpURL: String? = nil,
        httpMethod: String? = "POST",
        httpHeaders: [String: String]? = nil,
        arguments: [String]? = nil,
        environment: [String: String]? = nil,
        timeoutSeconds: Double? = 30.0
    ) {
        self.type = type
        self.command = command
        self.scriptContent = scriptContent
        self.scriptInterpreter = scriptInterpreter
        self.httpURL = httpURL
        self.httpMethod = httpMethod
        self.httpHeaders = httpHeaders
        self.arguments = arguments
        self.environment = environment
        self.timeoutSeconds = timeoutSeconds
    }
}

/// A user-defined custom tool model that can be advertised to the LLM and executed.
public struct CustomToolDefinition: Identifiable, Codable, Sendable, Equatable {
    public var id: UUID
    public var name: String
    public var displayName: String?
    public var toolDescription: String
    public var category: AppToolCategory
    public var scope: SkillScope
    public var parameters: JSONSchema
    public var execution: CustomToolExecution
    public var isEnabled: Bool
    public var sourcePath: String?

    public init(
        id: UUID = UUID(),
        name: String,
        displayName: String? = nil,
        toolDescription: String,
        category: AppToolCategory = .terminal,
        scope: SkillScope = .userGlobal,
        parameters: JSONSchema = .emptyObject(),
        execution: CustomToolExecution,
        isEnabled: Bool = true,
        sourcePath: String? = nil
    ) {
        self.id = id
        self.name = name
        self.displayName = displayName
        self.toolDescription = toolDescription
        self.category = category
        self.scope = scope
        self.parameters = parameters
        self.execution = execution
        self.isEnabled = isEnabled
        self.sourcePath = sourcePath
    }

    /// Names become file names (`<name>.json`, `custom_<name>_<id>.sh`), so
    /// only a short ASCII word with no path separators or dots is allowed.
    public static func isSafeName(_ name: String) -> Bool {
        guard (1...64).contains(name.utf8.count) else { return false }
        return name.utf8.allSatisfy {
            ($0 >= 0x30 && $0 <= 0x39) || ($0 >= 0x41 && $0 <= 0x5A)
                || ($0 >= 0x61 && $0 <= 0x7A) || $0 == 0x5F || $0 == 0x2D
        }
    }

    /// The category permissions and risk are decided on.
    ///
    /// **THE DECLARED CATEGORY IS A REQUEST, NOT A GRANT.** A definition file
    /// is written by whoever wrote the repo, and a `command` or `script` tool
    /// runs a shell whatever its file says, so it is never classified below
    /// `terminal`; an `http` tool is never below `web`. Declaring `fileRead`
    /// used to make a shell tool run silently under the default preset.
    public var effectiveCategory: AppToolCategory {
        switch execution.type {
        case .command, .script:
            return .terminal
        case .http:
            return category == .terminal ? .terminal : .web
        }
    }

    /// Converts definition into an OpenAI-compatible function calling specification.
    public var openAITool: OpenAITool {
        OpenAITool.function(
            name: name,
            description: toolDescription,
            parameters: parameters
        )
    }
}
