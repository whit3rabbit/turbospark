import Foundation

/// Central registry for managing built-in and discovered agents.
public final class AgentManager: @unchecked Sendable {
    public static let shared = AgentManager()

    private let fileManager = FileManager.default
    private static let disabledAgentsDefaultsKey = "TurboSpark.disabledAgentNames"

    public init() {}

    // MARK: - Enabled / Disabled Persistence

    private var disabledAgentNames: Set<String> {
        get { Set(UserDefaults.standard.stringArray(forKey: Self.disabledAgentsDefaultsKey) ?? []) }
        set { UserDefaults.standard.set(Array(newValue), forKey: Self.disabledAgentsDefaultsKey) }
    }

    public func isAgentDisabled(name: String) -> Bool {
        disabledAgentNames.contains(name.lowercased())
    }

    public func setAgentEnabled(_ enabled: Bool, name: String) {
        var names = disabledAgentNames
        let key = name.lowercased()
        if enabled {
            names.remove(key)
        } else {
            names.insert(key)
        }
        disabledAgentNames = names
    }

    // MARK: - Built-in Agents

    public static let explorePrompt = """
    You are a fast, read-only file search and codebase exploration specialist.
    Your role is EXCLUSIVELY to search, read, and analyze code.
    You are strictly prohibited from creating or modifying files.
    Use glob, read_file, search_code, and read-only shell commands to explore the codebase.
    Report your findings clearly, concisely, and directly.
    """

    public static let planPrompt = """
    You are a software architecture and implementation planning specialist.
    Analyze requirements, inspect existing codebase design and patterns, and construct detailed,
    step-by-step implementation plans, identifying potential edge cases, verification steps, and trade-offs.
    """

    public static let generalPurposePrompt = """
    You are a versatile, autonomous coding assistant capable of codebase exploration,
    file editing, command execution, and problem solving.
    """

    public static let reviewerPrompt = """
    You are a code review and security verification specialist.
    Analyze code changes, identify bugs, edge cases, performance bottlenecks, and security hazards.
    Provide constructive feedback and specific recommendations.
    """

    public var builtInAgents: [AppAgentDefinition] {
        [
            AppAgentDefinition(
                name: "explore",
                displayName: "Codebase Explorer",
                agentDescription: "Fast read-only agent specialized for exploring codebases and answering questions without modifying files.",
                systemPrompt: Self.explorePrompt,
                disallowedTools: [
                    "write_file", "save_file", "filewrite", "write",
                    "edit_file", "fileedit", "edit",
                    "apply_patch", "applypatch",
                    "agent", "subagent"
                ],
                maxTurns: 5,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "explore")
            ),
            AppAgentDefinition(
                name: "plan",
                displayName: "Architect & Planner",
                agentDescription: "Specialized planning agent for designing architectures, researching requirements, and structuring implementation steps.",
                systemPrompt: Self.planPrompt,
                disallowedTools: [
                    "write_file", "save_file", "filewrite", "write",
                    "edit_file", "fileedit", "edit",
                    "apply_patch", "applypatch"
                ],
                maxTurns: 5,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "plan")
            ),
            AppAgentDefinition(
                name: "general-purpose",
                displayName: "General Purpose",
                agentDescription: "Full autonomous assistant capable of reading, searching, editing files, and running commands.",
                systemPrompt: Self.generalPurposePrompt,
                maxTurns: 6,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "general-purpose")
            ),
            AppAgentDefinition(
                name: "reviewer",
                displayName: "Code Reviewer",
                agentDescription: "Code review and security audit specialist for verifying correctness and code quality.",
                systemPrompt: Self.reviewerPrompt,
                disallowedTools: [
                    "write_file", "save_file", "filewrite", "write",
                    "edit_file", "fileedit", "edit",
                    "apply_patch", "applypatch"
                ],
                maxTurns: 4,
                sourceAgent: .turboSpark,
                scope: .builtIn,
                isEnabled: !isAgentDisabled(name: "reviewer")
            )
        ]
    }

    // MARK: - Standard Directories

    public var defaultUserAgentsDirectory: URL {
        let home = fileManager.homeDirectoryForCurrentUser
        let dir = home.appendingPathComponent(".turbospark", isDirectory: true)
            .appendingPathComponent("agents", isDirectory: true)
        try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    public var knownUserAgentRoots: [(agent: AgentSourceAgent, relativePath: String)] {
        [
            (.turboSpark, ".turbospark/agents"),
            (.claude, ".claude/agents"),
            (.openCode, ".config/opencode/agents"),
            (.pi, ".pi/agent/agents"),
            (.antigravity, ".gemini/antigravity/agents"),
            (.antigravity, ".agents/agents"),
            (.cursor, ".cursor/agents")
        ]
    }

    public var knownProjectAgentSubdirectories: [(agent: AgentSourceAgent, relativePath: String)] {
        [
            (.turboSpark, ".turbospark/agents"),
            (.claude, ".claude/agents"),
            (.openCode, ".opencode/agents"),
            (.pi, ".pi/agents"),
            (.antigravity, ".agents/agents"),
            (.cursor, ".cursor/agents")
        ]
    }

    // MARK: - Discovery

    public func discoverUserAgents(includeExternalAgents: Bool = true) -> [AppAgentDefinition] {
        var agents: [AppAgentDefinition] = []
        var seenNames = Set<String>()

        let primaryURL = defaultUserAgentsDirectory
        let primaryAgents = scanDirectory(primaryURL, scope: .userGlobal, defaultAgent: .turboSpark)
        for agent in primaryAgents {
            let key = agent.name.lowercased()
            if !seenNames.contains(key) {
                seenNames.insert(key)
                agents.append(agent)
            }
        }

        if includeExternalAgents {
            let home = fileManager.homeDirectoryForCurrentUser
            for (sourceAgent, relPath) in knownUserAgentRoots where sourceAgent != .turboSpark {
                let dirURL = home.appendingPathComponent(relPath, isDirectory: true)
                let extAgents = scanDirectory(dirURL, scope: .userGlobal, defaultAgent: sourceAgent)
                for agent in extAgents {
                    let key = agent.name.lowercased()
                    if !seenNames.contains(key) {
                        seenNames.insert(key)
                        agents.append(agent)
                    }
                }
            }
        }

        return agents
    }

    public func discoverProjectAgents(projectURL: URL) -> [AppAgentDefinition] {
        var agents: [AppAgentDefinition] = []
        var seenNames = Set<String>()

        for (sourceAgent, relPath) in knownProjectAgentSubdirectories {
            let dirURL = projectURL.appendingPathComponent(relPath, isDirectory: true)
            let found = scanDirectory(dirURL, scope: .project, defaultAgent: sourceAgent)
            for agent in found {
                let key = agent.name.lowercased()
                if !seenNames.contains(key) {
                    seenNames.insert(key)
                    agents.append(agent)
                }
            }
        }

        return agents
    }

    public func resolveEffectiveAgents(projectURL: URL?) -> [AppAgentDefinition] {
        var map: [String: AppAgentDefinition] = [:]

        // 1. Built-in agents
        for agent in builtInAgents {
            map[agent.name.lowercased()] = agent
        }

        // 2. User agents override built-in
        let userAgents = discoverUserAgents()
        for agent in userAgents {
            map[agent.name.lowercased()] = agent
        }

        // 3. Project agents override user
        if let projectURL {
            let projAgents = discoverProjectAgents(projectURL: projectURL)
            for agent in projAgents {
                map[agent.name.lowercased()] = agent
            }
        }

        return map.values.sorted { $0.name < $1.name }
    }

    public func findAgent(name: String, projectURL: URL? = nil) -> AppAgentDefinition? {
        let lower = name.lowercased()
        let all = resolveEffectiveAgents(projectURL: projectURL)
        return all.first { $0.name.lowercased() == lower }
    }

    // MARK: - Scanning Directory

    private func scanDirectory(_ dirURL: URL, scope: AppAgentScope, defaultAgent: AgentSourceAgent) -> [AppAgentDefinition] {
        guard fileManager.fileExists(atPath: dirURL.path) else { return [] }
        guard let enumerator = fileManager.enumerator(
            at: dirURL,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else { return [] }

        var results: [AppAgentDefinition] = []
        for case let fileURL as URL in enumerator {
            let ext = fileURL.pathExtension.lowercased()
            guard ext == "md" || ext == "json" else { continue }

            if var agent = try? AgentParser.parseFile(at: fileURL, scope: scope, sourceAgent: defaultAgent) {
                agent.isEnabled = !isAgentDisabled(name: agent.name)
                results.append(agent)
            }
        }
        return results
    }
}
