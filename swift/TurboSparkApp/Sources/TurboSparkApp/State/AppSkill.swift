import Foundation

/// Scope of an agent skill definition.
public enum SkillScope: Codable, Equatable, Hashable, Sendable {
    /// User-global skill available across all projects (~/.turbospark/skills/ or user agent dirs).
    case userGlobal
    /// Project-local skill scoped to a specific project directory.
    case projectLocal(projectPath: String)
    /// Bundled default skill packaged with TurboSpark.
    case bundled
    /// Contributed by an installed plugin; `pluginID` is the
    /// `<name>@<origin>` id whose enable state gates the whole set.
    case plugin(pluginID: String)

    public var isProjectScope: Bool {
        if case .projectLocal = self {
            return true
        }
        return false
    }

    public var label: String {
        switch self {
        case .userGlobal:
            return "User Scope"
        case .projectLocal:
            return "Project Scope"
        case .bundled:
            return "Bundled"
        case .plugin:
            return "Plugin"
        }
    }

    public var badgeIcon: String {
        switch self {
        case .userGlobal:
            return "person.fill"
        case .projectLocal:
            return "folder.fill"
        case .bundled:
            return "cube.box.fill"
        case .plugin:
            return "puzzlepiece.extension.fill"
        }
    }
}

/// Known agent harness where a skill was created or imported from.
public enum SkillSourceAgent: String, Codable, CaseIterable, Identifiable, Sendable {
    case turboSpark = "turbospark"
    case claude = "claude"
    case cursor = "cursor"
    case gemini = "gemini"
    case antigravity = "antigravity"
    case openCode = "opencode"
    case pi = "pi"
    case copilot = "copilot"
    case windsurf = "windsurf"
    case kilo = "kilo"
    case crush = "crush"
    case cline = "cline"
    case forge = "forge"
    case qwen = "qwen"
    case custom = "custom"

    public var id: String { rawValue }

    public var displayName: String {
        switch self {
        case .turboSpark: return "TurboSpark"
        case .claude: return "Claude Code"
        case .cursor: return "Cursor"
        case .gemini: return "Gemini CLI"
        case .antigravity: return "Google Antigravity"
        case .openCode: return "OpenCode"
        case .pi: return "Pi Agent"
        case .copilot: return "GitHub Copilot"
        case .windsurf: return "Windsurf"
        case .kilo: return "Kilo Code"
        case .crush: return "Charm Crush"
        case .cline: return "Cline"
        case .forge: return "Forge"
        case .qwen: return "Qwen Code"
        case .custom: return "Custom"
        }
    }
}

/// Skill execution context mode.
public enum SkillExecutionContext: String, Codable, CaseIterable, Sendable {
    case inline
    case fork
}

/// Shell type for skill command execution.
public enum SkillShellType: String, Codable, CaseIterable, Sendable {
    case bash
    case powershell
}

/// Named argument definition for a skill.
public struct SkillArgument: Codable, Equatable, Hashable, Sendable {
    public var name: String
    public var placeholder: String?
    public var defaultValue: String?
    public var description: String?

    public init(
        name: String,
        placeholder: String? = nil,
        defaultValue: String? = nil,
        description: String? = nil
    ) {
        self.name = name
        self.placeholder = placeholder
        self.defaultValue = defaultValue
        self.description = description
    }
}

/// Permission pattern matching allowed tools (e.g. "Bash(git:*)" or "read_file").
public struct SkillPermissionPattern: Codable, Equatable, Hashable, Sendable {
    public var tool: String
    public var subPattern: String?

    public init(tool: String, subPattern: String? = nil) {
        self.tool = tool
        self.subPattern = subPattern
    }

    public static func parse(_ raw: String) -> SkillPermissionPattern {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        if let openParen = trimmed.firstIndex(of: "("), trimmed.hasSuffix(")") {
            let toolName = String(trimmed[..<openParen]).trimmingCharacters(in: .whitespaces)
            let subStart = trimmed.index(after: openParen)
            let subEnd = trimmed.index(before: trimmed.endIndex)
            let sub = String(trimmed[subStart..<subEnd]).trimmingCharacters(in: .whitespaces)
            return SkillPermissionPattern(tool: toolName, subPattern: sub)
        }
        return SkillPermissionPattern(tool: trimmed, subPattern: nil)
    }

    public func matches(tool: String, subTool: String? = nil) -> Bool {
        guard self.tool.lowercased() == tool.lowercased() else { return false }
        guard let pattern = subPattern else { return true }
        guard let sub = subTool else { return false }
        if pattern == "*" { return true }
        if pattern.hasSuffix("*") {
            let prefix = String(pattern.dropLast())
            return sub.lowercased().hasPrefix(prefix.lowercased())
        }
        return pattern.lowercased() == sub.lowercased()
    }
}

/// Skill manifest parsed from YAML frontmatter.
public struct SkillManifest: Codable, Equatable, Hashable, Sendable {
    public var name: String?
    public var description: String?
    public var allowedTools: [String]
    public var argumentHint: String?
    public var arguments: [SkillArgument]
    public var userInvocable: Bool
    public var disableModelInvocation: Bool?
    public var model: String?
    public var context: SkillExecutionContext
    public var agent: String?
    public var paths: [String]
    public var shell: SkillShellType

    public init(
        name: String? = nil,
        description: String? = nil,
        allowedTools: [String] = [],
        argumentHint: String? = nil,
        arguments: [SkillArgument] = [],
        userInvocable: Bool = true,
        disableModelInvocation: Bool? = nil,
        model: String? = nil,
        context: SkillExecutionContext = .inline,
        agent: String? = nil,
        paths: [String] = [],
        shell: SkillShellType = .bash
    ) {
        self.name = name
        self.description = description
        self.allowedTools = allowedTools
        self.argumentHint = argumentHint
        self.arguments = arguments
        self.userInvocable = userInvocable
        self.disableModelInvocation = disableModelInvocation
        self.model = model
        self.context = context
        self.agent = agent
        self.paths = paths
        self.shell = shell
    }

    public func displayName(fallback: String) -> String {
        if let name, !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return name.trimmingCharacters(in: .whitespacesAndNewlines)
        }
        return fallback
    }

    public var parsedPermissionPatterns: [SkillPermissionPattern] {
        allowedTools.map { SkillPermissionPattern.parse($0) }
    }

    public func isToolAllowed(tool: String, subTool: String? = nil) -> Bool {
        if allowedTools.isEmpty { return true }
        return parsedPermissionPatterns.contains { $0.matches(tool: tool, subTool: subTool) }
    }
}

/// A fully resolved skill with its metadata and instructional markdown body.
public struct AppSkill: Identifiable, Codable, Equatable, Sendable {
    public var id = UUID()
    public var manifest: SkillManifest
    public var content: String
    public var sourceURL: URL
    public var skillDirectoryURL: URL?
    public var scope: SkillScope
    public var agentOrigin: SkillSourceAgent
    public var isEnabled: Bool
    public var referenceFiles: [String]
    public var createdAt: Date
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        manifest: SkillManifest,
        content: String,
        sourceURL: URL,
        skillDirectoryURL: URL? = nil,
        scope: SkillScope = .userGlobal,
        agentOrigin: SkillSourceAgent = .turboSpark,
        isEnabled: Bool = true,
        referenceFiles: [String] = [],
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.manifest = manifest
        self.content = content
        self.sourceURL = sourceURL
        self.skillDirectoryURL = skillDirectoryURL
        self.scope = scope
        self.agentOrigin = agentOrigin
        self.isEnabled = isEnabled
        self.referenceFiles = referenceFiles
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    public var name: String {
        let fallback = sourceURL.deletingPathExtension().lastPathComponent
        return manifest.displayName(fallback: fallback == "SKILL" ? (skillDirectoryURL?.lastPathComponent ?? "skill") : fallback)
    }

    public var skillDescription: String {
        manifest.description ?? "Specialized skill workflow instructions."
    }

    public var isDirectoryBased: Bool {
        sourceURL.lastPathComponent.uppercased() == "SKILL.MD" && skillDirectoryURL != nil
    }
}
