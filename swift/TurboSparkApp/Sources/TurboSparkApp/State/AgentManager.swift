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
        // The resolution carries each agent's `isEnabled`, so a cached one is
        // stale the moment this changes.
        invalidateResolutionCache()
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

    /// Constrains a project agent that takes a built-in's name so it can only
    /// ever be MORE restricted than the built-in, never less.
    ///
    /// **A PROJECT DIRECTORY IS UNTRUSTED INPUT, AND `/explore` IS A NAME THE
    /// USER TYPES.** Project agents are read from `.claude/agents` and five
    /// sibling directories in whatever repository is open, with no trust gate
    /// -- hooks from the SAME directories require a SHA-256 review decision
    /// and agents have no equivalent. A cloned repository shipping
    /// `.claude/agents/explore.md` with no `disallowedTools` turned `/explore`
    /// from the read-only built-in into a write-and-shell agent, under a name
    /// whose meaning the user learned from this app rather than from the
    /// repository.
    ///
    /// **OVERRIDING IS STILL ALLOWED, because it is a real feature** -- a
    /// project customizing its explorer's instructions for its own stack is
    /// exactly what project scope is for, and `AgentSystemTests` pins it.
    /// What the override may not do is GAIN capability: the built-in's
    /// `disallowedTools` are unioned in and its `tools` allowlist intersected,
    /// so the prompt, description and turn budget are the project's while the
    /// tool ceiling stays the built-in's. Same principle as `CommandGate`,
    /// where the model may only ever add friction.
    static func constrained(
        _ projectAgent: AppAgentDefinition, byBuiltIn builtIn: AppAgentDefinition
    ) -> AppAgentDefinition {
        var merged = projectAgent

        let inheritedDenies = builtIn.disallowedTools ?? []
        if !inheritedDenies.isEmpty {
            var denies = merged.disallowedTools ?? []
            for tool in inheritedDenies where !denies.contains(tool) {
                denies.append(tool)
            }
            merged.disallowedTools = denies
        }

        if let builtInAllows = builtIn.tools {
            // The built-in restricts to a list, so the override may only pick
            // a subset of it. A project allowlist naming something outside is
            // narrowed rather than honoured.
            if let projectAllows = merged.tools {
                let permitted = Set(builtInAllows.map { AppAgentDefinition.canonicalToolName($0) })
                merged.tools = projectAllows.filter {
                    permitted.contains(AppAgentDefinition.canonicalToolName($0))
                }
            } else {
                merged.tools = builtInAllows
            }
        }
        return merged
    }

    /// The last resolution, keyed by project root.
    ///
    /// **RESOLVING WALKS UP TO 13 DIRECTORIES RECURSIVELY AND IS CALLED FROM
    /// VIEW BODIES.** `findAgent(named:)`, `AppModel.effectiveAgents` and the
    /// `agent` tool all funnel here (the tool calls it TWICE when the named
    /// agent is missing), each time enumerating seven user roots and six
    /// project subdirectories and parsing every `.md` and `.json` under them,
    /// synchronously on the main actor. Cached per root, invalidated
    /// explicitly.
    private var resolutionCache: (key: String, result: AgentResolution)?

    /// One resolution: the agents to offer, and the project files refused.
    struct AgentResolution {
        var agents: [AppAgentDefinition]
        /// Project agents held to a built-in's tool ceiling for taking its
        /// name. Kept so it can be SHOWN -- a repository author whose
        /// `explore.md` cannot write files should be able to see why.
        var constrainedProjectNames: [String]
    }

    /// Drops the cached resolution. Call after anything that changes what is
    /// on disk or which agents are enabled.
    public func invalidateResolutionCache() {
        resolutionCache = nil
    }

    /// Project agent names held to a built-in's tool ceiling.
    public func constrainedProjectAgentNames(projectURL: URL?) -> [String] {
        resolution(projectURL: projectURL).constrainedProjectNames
    }

    public func resolveEffectiveAgents(projectURL: URL?) -> [AppAgentDefinition] {
        resolution(projectURL: projectURL).agents
    }

    private func resolution(projectURL: URL?) -> AgentResolution {
        let key = projectURL?.standardizedFileURL.path ?? ""
        if let cached = resolutionCache, cached.key == key {
            return cached.result
        }
        let result = computeResolution(projectURL: projectURL)
        resolutionCache = (key, result)
        return result
    }

    private func computeResolution(projectURL: URL?) -> AgentResolution {
        var map: [String: AppAgentDefinition] = [:]

        // 1. Built-in agents
        for agent in builtInAgents {
            map[agent.name.lowercased()] = agent
        }

        // 2. User agents override built-in. Still permitted: `~/.turbospark`
        // and its siblings are the user's own directories, not something a
        // `git clone` writes into.
        let userAgents = discoverUserAgents()
        for agent in userAgents {
            map[agent.name.lowercased()] = agent
        }

        // 3. Project agents override user, but one taking a BUILT-IN's name
        // is constrained to that built-in's tool ceiling first.
        var builtInsByName: [String: AppAgentDefinition] = [:]
        for agent in builtInAgents {
            builtInsByName[agent.name.lowercased()] = agent
        }
        var constrained: [String] = []
        if let projectURL {
            let projAgents = discoverProjectAgents(projectURL: projectURL)
            for agent in projAgents {
                let key = agent.name.lowercased()
                if let builtIn = builtInsByName[key] {
                    map[key] = Self.constrained(agent, byBuiltIn: builtIn)
                    constrained.append(agent.name)
                } else {
                    map[key] = agent
                }
            }
        }
        return AgentResolution(
            agents: map.values.sorted { $0.name < $1.name },
            constrainedProjectNames: constrained)
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
