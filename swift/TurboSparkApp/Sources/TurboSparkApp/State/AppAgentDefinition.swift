import Foundation

/// Scope determining where an agent definition originates.
public enum AppAgentScope: String, Codable, CaseIterable, Sendable {
    case builtIn
    case userGlobal
    case project

    public var label: String {
        switch self {
        case .builtIn: return "Built-in"
        case .userGlobal: return "User"
        case .project: return "Project"
        }
    }

    public var isProjectScope: Bool {
        self == .project
    }
}

/// Provenance of an agent definition file across agent tooling ecosystems.
public enum AgentSourceAgent: String, Codable, CaseIterable, Sendable {
    case turboSpark = "turbospark"
    case claude = "claude"
    case openCode = "opencode"
    case pi = "pi"
    case antigravity = "antigravity"
    case cursor = "cursor"
    case custom = "custom"

    public var displayName: String {
        switch self {
        case .turboSpark: return "TurboSpark"
        case .claude: return "Claude Code"
        case .openCode: return "OpenCode"
        case .pi: return "Pi"
        case .antigravity: return "Antigravity"
        case .cursor: return "Cursor"
        case .custom: return "Custom"
        }
    }
}

/// A structured agent definition for specialized autonomous tasks.
public struct AppAgentDefinition: Identifiable, Codable, Equatable, Sendable {
    /// Unique identifier for the agent definition.
    public var id: UUID
    /// Unique identifier name of the agent (e.g. "explore", "plan", "reviewer").
    public var name: String
    /// Human-friendly display title.
    public var displayName: String
    /// Concise description of the agent and when to invoke it.
    public var agentDescription: String
    /// Specialized system prompt instructions for this agent.
    public var systemPrompt: String
    /// Whitelist of allowed tool names (if nil, all default tools except disallowed are available).
    public var tools: [String]?
    /// Blacklist of explicitly disallowed tool names.
    public var disallowedTools: [String]?
    /// Optional model identifier override.
    public var model: String?
    /// Maximum autonomous turn count for subagent execution.
    ///
    /// Clamped into `1...maxTurnsCeiling` by `init` (state#95), so the file
    /// on disk is a request rather than a setting: `max_turns: 100000` in an
    /// agent file out of a cloned repository is a model-driven loop with no
    /// user in it, each turn free to run tools.
    public var maxTurns: Int

    /// The most turns any agent file may ask for (state#95).
    ///
    /// 50 rather than a smaller number because the ceiling is there to bound
    /// an unattended loop, not to second-guess a legitimate long task -- and
    /// a project agent that shadows a built-in is additionally held to THAT
    /// built-in's own value, which is 5 for every one shipped.
    public static let maxTurnsCeiling = 50
    /// Originating tooling ecosystem.
    public var sourceAgent: AgentSourceAgent
    /// Scope of the agent definition.
    public var scope: AppAgentScope
    /// Path on disk if loaded from a file.
    public var filePath: String?
    /// Whether the agent is active and available for invocation.
    public var isEnabled: Bool

    public init(
        id: UUID = UUID(),
        name: String,
        displayName: String? = nil,
        agentDescription: String,
        systemPrompt: String,
        tools: [String]? = nil,
        disallowedTools: [String]? = nil,
        model: String? = nil,
        maxTurns: Int = 5,
        sourceAgent: AgentSourceAgent = .turboSpark,
        scope: AppAgentScope = .builtIn,
        filePath: String? = nil,
        isEnabled: Bool = true
    ) {
        self.id = id
        self.name = name
        self.displayName = displayName ?? name.capitalized
        self.agentDescription = agentDescription
        self.systemPrompt = systemPrompt
        self.tools = tools
        self.disallowedTools = disallowedTools
        self.model = model
        self.maxTurns = min(max(1, maxTurns), Self.maxTurnsCeiling)
        self.sourceAgent = sourceAgent
        self.scope = scope
        self.filePath = filePath
        self.isEnabled = isEnabled
    }

    /// Normalizes tool names and synonyms to canonical identifiers.
    public static func canonicalToolName(_ name: String) -> String {
        let lower = name.lowercased().replacingOccurrences(of: "_", with: "")
        switch lower {
        case "readfile", "fileread", "read", "viewfile", "cat":
            return "read_file"
        case "writefile", "filewrite", "write", "savefile":
            return "write_file"
        case "editfile", "fileedit", "edit":
            return "edit_file"
        case "applypatch", "patch":
            return "apply_patch"
        case "searchcode", "grep", "search", "searchfiles":
            return "search_code"
        case "listdirectory", "listdir", "ls", "glob":
            return "list_directory"
        case "runcommand", "bash", "shell", "exec", "terminal":
            return "run_command"
        case "webfetch", "fetchurl", "readurlcontent":
            return "webfetch"
        case "agent", "subagent", "task":
            return "agent"
        default:
            return name.lowercased()
        }
    }

    /// Resolves whether a tool name is permitted under this agent definition.
    public func isToolAllowed(_ toolName: String) -> Bool {
        let canonical = Self.canonicalToolName(toolName)
        let lower = toolName.lowercased()
        let stripped = lower.replacingOccurrences(of: "_", with: "")

        if let disallowed = disallowedTools {
            let disallowedCanonicals = Set(disallowed.map { Self.canonicalToolName($0) })
            let lowerDisallowed = Set(disallowed.map { $0.lowercased() })
            let strippedDisallowed = Set(disallowed.map { $0.lowercased().replacingOccurrences(of: "_", with: "") })
            if disallowedCanonicals.contains(canonical) || lowerDisallowed.contains(lower) || strippedDisallowed.contains(stripped) {
                return false
            }
        }

        if let allowed = tools {
            let allowedCanonicals = Set(allowed.map { Self.canonicalToolName($0) })
            let lowerAllowed = Set(allowed.map { $0.lowercased() })
            let strippedAllowed = Set(allowed.map { $0.lowercased().replacingOccurrences(of: "_", with: "") })
            return allowedCanonicals.contains(canonical) || lowerAllowed.contains(lower) || strippedAllowed.contains(stripped)
        }

        return true
    }
}
