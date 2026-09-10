import Foundation

/// High-level permission mode governing tool execution and risk gating (Unsloth Studio parity).
public enum AppPermissionMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Always ask: prompts before executing any mutating, terminal, network, or external tool.
    case ask
    /// Smart auto-approval: runs safe & low-risk development commands silently, prompts on high-risk operations.
    case auto
    /// Classifier-driven approval (`swift/docs/SWIFT_AGENT_MODE.md`): the
    /// local model judges each call the static ladder would have asked about.
    /// Safe calls run, risky ones are refused with a reason, and anything the
    /// classifier cannot decide asks anyway.
    case agentAuto
    /// Permissive: executes all tools without prompting within sandbox bounds.
    case permissive
    /// Full access: unrestricted, no approval prompts and the code sandbox is disabled.
    case fullAccess
    /// Strict read-only: denies mutating actions, file writes, terminal executions, and crons.
    case readOnly

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .ask: return "Ask for approval"
        case .auto: return "Approve for me"
        case .agentAuto: return "Agent (classifier)"
        case .permissive: return "Run automatically"
        case .fullAccess: return "Full access"
        case .readOnly: return "Strict Read-Only"
        }
    }

    public var shortLabel: String {
        switch self {
        case .ask: return "Ask for approval"
        case .auto: return "Approve for me"
        case .agentAuto: return "Agent"
        case .permissive: return "Run automatically"
        case .fullAccess: return "Full access"
        case .readOnly: return "Read Only"
        }
    }

    public var descriptionText: String {
        switch self {
        case .ask:
            return "Always ask before tool calls edit files or use the internet"
        case .auto:
            return "Run tool calls, but ask before high-risk actions like credential access, privilege escalation, or destructive commands"
        case .agentAuto:
            return "A local classifier reviews each tool call: safe ones run, risky ones are blocked with a reason, and uncertain ones ask you"
        case .permissive:
            return "Run tool calls without approval prompts inside the sandbox"
        case .fullAccess:
            return "Unrestricted: no approval prompts and the code sandbox is disabled"
        case .readOnly:
            return "Allows only inspection, search, and reading. Prohibits all file modifications, terminal executions, destructive MCP calls, and background automations."
        }
    }

    public var systemImage: String {
        switch self {
        case .ask: return "hand.raised.fill"
        case .auto: return "shield.lefthalf.filled"
        case .agentAuto: return "brain.head.profile"
        case .permissive: return "play.circle.fill"
        case .fullAccess: return "exclamationmark.triangle.fill"
        case .readOnly: return "lock.shield.fill"
        }
    }
}

/// Permission level for tool actions performed by an agent.
public enum AppToolPermission: String, Codable, CaseIterable, Identifiable, Sendable {
    case allow
    case ask
    case deny

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .allow: return "Always Allow"
        case .ask: return "Ask Before Execution"
        case .deny: return "Deny"
        }
    }

    public var systemImage: String {
        switch self {
        case .allow: return "checkmark.circle.fill"
        case .ask: return "questionmark.circle.fill"
        case .deny: return "nosign"
        }
    }
}

/// Granular permission matrix for a project workspace.
public struct AppProjectPermissions: Codable, Equatable, Sendable {
    /// High-level permission mode (defaults to .auto).
    public var mode: AppPermissionMode
    /// Permission for reading files, listing directories, and searching code.
    public var fileRead: AppToolPermission
    /// Permission for writing, editing, or deleting files.
    public var fileWrite: AppToolPermission
    /// Permission for running shell / terminal commands.
    public var terminal: AppToolPermission
    /// Permission for web requests and documentation fetching.
    public var web: AppToolPermission
    /// Permission for external MCP tools.
    public var mcp: AppToolPermission
    /// Permission for background workflow and cron automation.
    public var automation: AppToolPermission
    /// Persisted MCP ALLOW rules in `mcp__server` / `mcp__server__tool`
    /// syntax (`swift/docs/SWIFT_TOOLS.md`). A matching call skips the ask prompt
    /// but never the high-risk gate.
    public var mcpAllowRules: [String]
    /// Persisted MCP DENY rules in the same syntax. A matching call is
    /// refused outright, ahead of every other gate including session
    /// approvals, and its tools are stripped from the advertised list.
    public var mcpDenyRules: [String]

    public init(
        mode: AppPermissionMode = .auto,
        fileRead: AppToolPermission = .allow,
        fileWrite: AppToolPermission = .ask,
        terminal: AppToolPermission = .ask,
        web: AppToolPermission = .allow,
        mcp: AppToolPermission = .ask,
        automation: AppToolPermission = .ask,
        mcpAllowRules: [String] = [],
        mcpDenyRules: [String] = []
    ) {
        self.mode = mode
        self.fileRead = fileRead
        self.fileWrite = fileWrite
        self.terminal = terminal
        self.web = web
        self.mcp = mcp
        self.automation = automation
        self.mcpAllowRules = mcpAllowRules
        self.mcpDenyRules = mcpDenyRules
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.mode = try container.decodeIfPresent(AppPermissionMode.self, forKey: .mode) ?? .auto
        self.fileRead = try container.decodeIfPresent(AppToolPermission.self, forKey: .fileRead) ?? .allow
        self.fileWrite = try container.decodeIfPresent(AppToolPermission.self, forKey: .fileWrite) ?? .ask
        self.terminal = try container.decodeIfPresent(AppToolPermission.self, forKey: .terminal) ?? .ask
        self.web = try container.decodeIfPresent(AppToolPermission.self, forKey: .web) ?? .allow
        self.mcp = try container.decodeIfPresent(AppToolPermission.self, forKey: .mcp) ?? .ask
        self.automation = try container.decodeIfPresent(AppToolPermission.self, forKey: .automation) ?? .ask
        self.mcpAllowRules = try container.decodeIfPresent([String].self, forKey: .mcpAllowRules) ?? []
        self.mcpDenyRules = try container.decodeIfPresent([String].self, forKey: .mcpDenyRules) ?? []
    }

    /// Auto configuration: smart risk-gated execution (Unsloth Studio default).
    public static var auto: AppProjectPermissions {
        AppProjectPermissions(
            mode: .auto,
            fileRead: .allow,
            fileWrite: .allow,
            terminal: .allow,
            web: .allow,
            mcp: .allow,
            automation: .allow
        )
    }

    /// **The preset a project gets when nobody chose one.**
    ///
    /// Named rather than spelled out at each site because it was spelled out
    /// at each site and they disagreed: `AppProject.init`, the decode
    /// fallback and `AppModel.createProject` said `.standard` while the new
    /// project sheet said `.auto` (`terminal: .allow`, `fileWrite: .allow`,
    /// `mcp: .allow`). The sheet is the only one of the four a user actually
    /// goes through, so the permissive spelling was the shipped default and
    /// the conservative one was decoration.
    ///
    /// Every default now reads THIS. A fifth site spelling its own preset is
    /// the bug this constant exists to make visible.
    public static var newProjectDefault: AppProjectPermissions { .standard }

    /// Safe standard configuration requiring confirmation for state-mutating actions.
    public static var standard: AppProjectPermissions {
        AppProjectPermissions(
            mode: .auto,
            fileRead: .allow,
            fileWrite: .ask,
            terminal: .ask,
            web: .allow,
            mcp: .ask,
            automation: .ask
        )
    }

    /// Agent configuration: the same matrix as `.standard`, where the
    /// `.ask` categories are exactly what the classifier judges
    /// (`swift/docs/SWIFT_AGENT_MODE.md`). A category the user moves to
    /// `.allow` runs without the classifier; `.deny` refuses outright.
    public static var agent: AppProjectPermissions {
        AppProjectPermissions(
            mode: .agentAuto,
            fileRead: .allow,
            fileWrite: .ask,
            terminal: .ask,
            web: .allow,
            mcp: .ask,
            automation: .ask
        )
    }

    /// Permissive configuration allowing all actions without asking.
    public static var permissive: AppProjectPermissions {
        AppProjectPermissions(
            mode: .permissive,
            fileRead: .allow,
            fileWrite: .allow,
            terminal: .allow,
            web: .allow,
            mcp: .allow,
            automation: .allow
        )
    }

    /// Explicit ask configuration requiring user approval for all mutating actions.
    public static var alwaysAsk: AppProjectPermissions {
        AppProjectPermissions(
            mode: .ask,
            fileRead: .allow,
            fileWrite: .ask,
            terminal: .ask,
            web: .ask,
            mcp: .ask,
            automation: .ask
        )
    }

    /// Read-only configuration that denies file writes and terminal executions.
    public static var readOnly: AppProjectPermissions {
        AppProjectPermissions(
            mode: .readOnly,
            fileRead: .allow,
            fileWrite: .deny,
            terminal: .deny,
            web: .allow,
            mcp: .deny,
            automation: .deny
        )
    }

    /// Full access configuration: unrestricted tool actions without approval prompts.
    public static var fullAccess: AppProjectPermissions {
        AppProjectPermissions(
            mode: .fullAccess,
            fileRead: .allow,
            fileWrite: .allow,
            terminal: .allow,
            web: .allow,
            mcp: .allow,
            automation: .allow
        )
    }

    /// Resolves canonical permissions configuration for a given permission mode.
    public static func preset(for mode: AppPermissionMode) -> AppProjectPermissions {
        switch mode {
        case .ask: return .alwaysAsk
        case .auto: return .standard
        case .agentAuto: return .agent
        case .permissive: return .permissive
        case .fullAccess: return .fullAccess
        case .readOnly: return .readOnly
        }
    }
}

/// Agent specialization profile governing behavior and prompt engineering.
public enum AppAgentType: String, Codable, CaseIterable, Identifiable, Sendable {
    case general
    case coder
    case researcher
    case autonomous
    case custom

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .general: return "General Assistant"
        case .coder: return "Coding Agent"
        case .researcher: return "Document Analyst"
        case .autonomous: return "Autonomous Planner"
        case .custom: return "Custom Agent"
        }
    }

    public var descriptionText: String {
        switch self {
        case .general:
            return "Versatile conversational assistant with optional tool usage when requested."
        case .coder:
            return "Software engineer optimized for codebase navigation, editing, terminal commands, and debugging."
        case .researcher:
            return "Deep analytical agent focused on reading documents, code search, and contextual synthesis."
        case .autonomous:
            return "Self-directed agent that iterates over multi-step tool workflows to accomplish complex objectives."
        case .custom:
            return "User-configured agent with customized system instructions and tool capabilities."
        }
    }

    public var systemImage: String {
        switch self {
        case .general: return "bubble.left.and.bubble.right"
        case .coder: return "chevron.left.forwardslash.chevron.right"
        case .researcher: return "doc.text.magnifyingglass"
        case .autonomous: return "arrow.triangle.2.circlepath"
        case .custom: return "person.crop.circle.badge.gearshape"
        }
    }

    public var defaultSystemPrompt: String {
        switch self {
        case .general:
            return "You are a helpful and concise AI assistant. You can answer questions, provide explanations, and use tools when helpful."
        case .coder:
            return "You are an expert software engineer. Analyze the project structure, inspect files, write clean modular code, execute necessary commands, and verify your changes carefully."
        case .researcher:
            return "You are a thorough research and document analysis assistant. Carefully inspect project files, extract pertinent details, and synthesize accurate, well-structured summaries."
        case .autonomous:
            return "You are an autonomous problem solver. Plan your strategy step-by-step, invoke tools to gather information and make changes, check outcomes, and deliver a comprehensive final report."
        case .custom:
            return "You are a specialized AI assistant tailored for this project."
        }
    }
}

/// A project workspace managing a codebase, custom rules, permissions, and agent profile.
public struct AppProject: Identifiable, Codable, Equatable, Sendable {
    /// Unique identifier for the project.
    public var id = UUID()
    /// User-visible project name.
    public var name: String
    /// Absolute filesystem path to the project root directory.
    public var rootDirectoryPath: String?
    /// Selected agent behavior profile.
    public var agentType: AppAgentType
    /// Preference for resolving instructions (AGENTS.md vs CLAUDE.md).
    public var rulePreference: AppRulePreference
    /// User-supplied custom system prompt or project rules.
    public var customInstructions: String
    /// Tool execution permissions for this project.
    public var permissions: AppProjectPermissions
    /// Maximum autonomous tool-execution iterations per turn.
    public var maxAutonomousSteps: Int

    /// The range the picker offers, which is the range this field may hold
    /// (state#95).
    ///
    /// `projects_archive.json` is a plain file a user or a future release can
    /// write, and this one decoded whatever it found: a hand-edited 100000
    /// is an agent loop that runs tools until the model stops proposing them.
    /// Same rule as `clampedSetting` on the engine settings (state#35), and
    /// the same reason -- the clamp belongs where the value ENTERS, not at
    /// the one call site that happens to read it.
    public static let autonomousStepRange = 1...15

    static func clampedSteps(_ value: Int) -> Int {
        min(max(value, autonomousStepRange.lowerBound), autonomousStepRange.upperBound)
    }
    /// Project-specific MCP external servers.
    public var mcpServers: [McpServerConfig]
    /// Project-specific Forge Guardrails override (nil = auto/model default).
    public var forgeGuardrailsEnabled: Bool?
    /// Carry a bounded execution state between agent steps instead of the full
    /// transcript (docs/SKILL_STATE.md). Opt-in: off leaves the loop unchanged.
    public var skillStateEnabled: Bool
    /// Project-scope plugin enable state, keyed `<plugin>@<origin>`
    /// (`swift/docs/SWIFT_PLUGINS.md`). Overrides the user setting: a project may
    /// turn off a plugin it does not trust without turning it off everywhere.
    public var enabledPlugins: [String: Bool]
    public var enabledSkills: [String: Bool] = [:]
    public var marketplaces = ProjectMarketplaces()
    public var localPluginPaths: [String] = []
    public var enabledMcpServers: [String: Bool] = [:]
    /// MCP servers from the project's own config files the user has
    /// APPROVED. A name here (or covered by `approveAllProjectMcpServers`)
    /// imports without re-prompting when the config re-declares it.
    public var approvedMcpJsonServers: [String]
    /// MCP servers from the project's config files the user has REJECTED.
    /// A rejected name is never auto-imported and never re-prompted; the
    /// project MCP sheet remains the place to change one's mind.
    public var rejectedMcpJsonServers: [String]
    /// Approve every current and FUTURE server declared by this project's
    /// config files without prompting (the reference implementation's
    /// `enableAllProjectMcpServers`).
    public var approveAllProjectMcpServers: Bool
    /// Timestamp when the project was created.
    public var createdAt: Date
    /// Timestamp when the project was last updated.
    public var updatedAt: Date

    public init(
        id: UUID = UUID(),
        name: String,
        rootDirectoryPath: String? = nil,
        agentType: AppAgentType = .coder,
        rulePreference: AppRulePreference = .agentsFirst,
        customInstructions: String = "",
        permissions: AppProjectPermissions = .newProjectDefault,
        maxAutonomousSteps: Int = 5,
        mcpServers: [McpServerConfig] = [],
        forgeGuardrailsEnabled: Bool? = nil,
        skillStateEnabled: Bool = false,
        enabledPlugins: [String: Bool] = [:],
        approvedMcpJsonServers: [String] = [],
        rejectedMcpJsonServers: [String] = [],
        approveAllProjectMcpServers: Bool = false,
        createdAt: Date = Date(),
        updatedAt: Date = Date()
    ) {
        self.id = id
        self.name = name
        self.rootDirectoryPath = rootDirectoryPath
        self.agentType = agentType
        self.rulePreference = rulePreference
        self.customInstructions = customInstructions
        self.permissions = permissions
        self.maxAutonomousSteps = Self.clampedSteps(maxAutonomousSteps)
        self.mcpServers = mcpServers
        self.forgeGuardrailsEnabled = forgeGuardrailsEnabled
        self.skillStateEnabled = skillStateEnabled
        self.enabledPlugins = enabledPlugins
        self.approvedMcpJsonServers = approvedMcpJsonServers
        self.rejectedMcpJsonServers = rejectedMcpJsonServers
        self.approveAllProjectMcpServers = approveAllProjectMcpServers
        self.createdAt = createdAt
        self.updatedAt = updatedAt
    }

    enum CodingKeys: String, CodingKey {
        case id, name, rootDirectoryPath, agentType, rulePreference, customInstructions
        case permissions, maxAutonomousSteps, mcpServers, forgeGuardrailsEnabled
        case skillStateEnabled, enabledPlugins, enabledSkills, enabledMcpServers, marketplaces, localPluginPaths, createdAt, updatedAt
        case approvedMcpJsonServers, rejectedMcpJsonServers, approveAllProjectMcpServers
    }

    public init(from decoder: Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.id = try container.decodeIfPresent(UUID.self, forKey: .id) ?? UUID()
        // Tolerant throughout (state#45). A strict `name` throws for the
        // whole PROJECTS archive on one row, and `decodeIfPresent` of an ENUM
        // throws on an unknown raw value -- absence is the only thing it
        // tolerates, and a value a newer release wrote is not absence.
        self.name = try container.decodeIfPresent(String.self, forKey: .name) ?? "Untitled Project"
        self.rootDirectoryPath = try container.decodeIfPresent(String.self, forKey: .rootDirectoryPath)
        self.agentType = container.decodeTolerant(
            AppAgentType.self, forKey: .agentType, fallback: .coder)
        self.rulePreference = container.decodeTolerant(
            AppRulePreference.self, forKey: .rulePreference, fallback: .agentsFirst)
        self.customInstructions = try container.decodeIfPresent(String.self, forKey: .customInstructions) ?? ""
        self.permissions = try container.decodeIfPresent(AppProjectPermissions.self, forKey: .permissions) ?? .newProjectDefault
        self.maxAutonomousSteps = Self.clampedSteps(
            try container.decodeIfPresent(Int.self, forKey: .maxAutonomousSteps) ?? 5)
        self.mcpServers = try container.decodeLossyArray(McpServerConfig.self, forKey: .mcpServers)
        self.forgeGuardrailsEnabled = try container.decodeIfPresent(Bool.self, forKey: .forgeGuardrailsEnabled)
        self.skillStateEnabled = try container.decodeIfPresent(Bool.self, forKey: .skillStateEnabled) ?? false
        self.enabledPlugins = try container.decodeIfPresent([String: Bool].self, forKey: .enabledPlugins) ?? [:]
        self.localPluginPaths = try container.decodeIfPresent([String].self, forKey: .localPluginPaths) ?? []
        self.marketplaces = try container.decodeIfPresent(ProjectMarketplaces.self, forKey: .marketplaces) ?? ProjectMarketplaces()
        self.enabledSkills = try container.decodeIfPresent([String: Bool].self, forKey: .enabledSkills) ?? [:]
        self.enabledMcpServers = try container.decodeIfPresent([String: Bool].self, forKey: .enabledMcpServers) ?? [:]
        self.approvedMcpJsonServers = try container.decodeIfPresent([String].self, forKey: .approvedMcpJsonServers) ?? []
        self.rejectedMcpJsonServers = try container.decodeIfPresent([String].self, forKey: .rejectedMcpJsonServers) ?? []
        self.approveAllProjectMcpServers = try container.decodeIfPresent(Bool.self, forKey: .approveAllProjectMcpServers) ?? false
        self.createdAt = try container.decodeIfPresent(Date.self, forKey: .createdAt) ?? Date()
        self.updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? Date()
    }

    /// Resolved URL to the root directory if configured.
    public var rootDirectoryURL: URL? {
        guard let rootDirectoryPath, !rootDirectoryPath.isEmpty else { return nil }
        return URL(fileURLWithPath: rootDirectoryPath, isDirectory: true)
    }

    /// Whether a valid root directory is assigned.
    public var hasRootDirectory: Bool {
        guard let path = rootDirectoryPath, !path.isEmpty else { return false }
        var isDir: ObjCBool = false
        return FileManager.default.fileExists(atPath: path, isDirectory: &isDir) && isDir.boolValue
    }
}

/// Archive container for persisting all application projects and the active selection.
public struct AppProjectArchive: Codable, Sendable {
    /// Identifier of the active project, or nil for all chats.
    public var selectedProjectID: UUID?
    /// All saved projects.
    public var projects: [AppProject]

    public init(selectedProjectID: UUID? = nil, projects: [AppProject] = []) {
        self.selectedProjectID = selectedProjectID
        self.projects = projects
    }

    public static func empty() -> AppProjectArchive {
        AppProjectArchive(selectedProjectID: nil, projects: [])
    }
}

/// Filesystem storage utilities for saving and loading project archives.
public enum AppProjectFileStore {
    private static var storageDirectory: URL {
        AppStorageRoot.directory
    }

    private static var archiveFileURL: URL {
        storageDirectory.appendingPathComponent("projects_archive.json")
    }

    /// Loads the saved project archive from disk or returns an empty default.
    public static func load() -> AppProjectArchive {
        AppJSONStore.load(AppProjectArchive.self, from: archiveFileURL, label: "project archive")
            ?? AppProjectArchive.empty()
    }

    /// Persists the project archive to disk atomically.
    public static func save(_ archive: AppProjectArchive) {
        AppJSONStore.save(archive, to: archiveFileURL, label: "Project archive")
    }
}
