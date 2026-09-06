import Foundation

/// Central registry for managing built-in and discovered agents.
public final class AgentManager: @unchecked Sendable {
    public static let shared = AgentManager()

    private let fileManager = FileManager.default

    public init() {}

    // MARK: - Enabled / Disabled Persistence

    private var disabledAgentNames: Set<String> {
        // Under `AppStorageRoot`, not `UserDefaults.standard` (state#57).
        get { DisabledItemStore.names(for: .agents) }
        set { DisabledItemStore.setNames(newValue, for: .agents) }
    }

    /// The persisted key for one agent.
    ///
    /// **SCOPE PLUS NAME, NOT NAME ALONE** (state#57), which is the rule
    /// `SkillManager` already followed and this half never did. A project
    /// agent taking a built-in's name is the supported way to override one
    /// (state#22 constrains what it may DO, not whether it may exist), so
    /// keying on the name alone disabled the built-in and the override
    /// together -- and back. Old name-only keys are still honoured on read,
    /// so nobody's existing preference is silently forgotten.
    private static func disabledKey(scope: AppAgentScope, name: String) -> String {
        "\(scope == .project ? "project" : "user"):\(name.lowercased())"
    }

    public func isAgentDisabled(name: String, scope: AppAgentScope = .userGlobal) -> Bool {
        let names = disabledAgentNames
        return names.contains(Self.disabledKey(scope: scope, name: name))
            || names.contains(name.lowercased())
    }

    public func setAgentEnabled(_ enabled: Bool, name: String, scope: AppAgentScope = .userGlobal) {
        var names = disabledAgentNames
        let key = Self.disabledKey(scope: scope, name: name)
        if enabled {
            names.remove(key)
            // The legacy name-only key too, or an agent disabled before this
            // landed can never be re-enabled.
            names.remove(name.lowercased())
        } else {
            names.insert(key)
        }
        disabledAgentNames = names
        // The resolution carries each agent's `isEnabled`, so a cached one is
        // stale the moment this changes.
        invalidateResolutionCache()
    }

    // MARK: - Standard Directories

    /// Standard user agents directory: `~/.turbospark/agents` for the Default
    /// profile, inside that profile's own folder for anyone else.
    public var defaultUserAgentsDirectory: URL {
        UserProfileStore.userScopeSubdirectory("agents")
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

        // Same isolation rule as skills: the cross-agent roots are shared
        // home-directory trees a non-default profile does not read.
        if includeExternalAgents, UserProfileStore.isDefault {
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
            let found = scanDirectory(
                dirURL, scope: .project, defaultAgent: sourceAgent, containedIn: projectURL)
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
    /// USER TYPES** (state#22). Project agents are read from `.claude/agents` and five
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

        // **THE TURN BUDGET IS A RESTRICTION LIKE ANY OTHER** (state#95).
        // state#22's rule is that a project agent taking a built-in's name
        // may only add friction, and `maxTurns` was not held to it: a
        // repository `explore.md` declaring 50 turns got 50 where the
        // built-in it shadows allows 5.
        merged.maxTurns = min(merged.maxTurns, builtIn.maxTurns)

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
    ///
    /// **GUARDED, BECAUSE THIS TYPE IS `@unchecked Sendable` AND THE CACHE IS
    /// READ OFF THE MAIN ACTOR** (state#69). See `SkillManager`'s twin: the
    /// writers are main-actor and `AppToolRegistry.execute` -- a
    /// `nonisolated async` function on the cooperative pool -- is not, so the
    /// `agent` tool's two `findAgent` calls race every settings-pane write.
    private let cacheLock = NSLock()
    private var resolutionCache: (key: String, result: AgentResolution)?

    /// One resolution: the agents to offer, and the project files refused.
    struct AgentResolution {
        var agents: [AppAgentDefinition]
        /// Project agents held to a built-in's tool ceiling for taking its
        /// name. Kept so it can be SHOWN -- a repository author whose
        /// `explore.md` cannot write files should be able to see why.
        var constrainedProjectNames: [String]
        /// USER agents a project agent is currently shadowing (state#105).
        ///
        /// **PRECEDENCE WITH NO DISCLOSURE IS INDISTINGUISHABLE FROM A BROKEN
        /// AGENT**, which is exactly the reasoning `SkillManager.shadowedUserSkillNames`
        /// was written down for; the agent side has the constrained list for
        /// the BUILT-IN case and had nothing for the user one. A cloned
        /// repository can take the name of an agent the user wrote, and the
        /// built-in ceiling does not apply to it -- there is no built-in to
        /// draw the ceiling from.
        var shadowedUserNames: [String]
    }

    /// Drops the cached resolution. Call after anything that changes what is
    /// on disk or which agents are enabled.
    public func invalidateResolutionCache() {
        cacheLock.lock()
        resolutionCache = nil
        cacheLock.unlock()
    }

    /// Project agent names held to a built-in's tool ceiling.
    public func constrainedProjectAgentNames(projectURL: URL?) -> [String] {
        resolution(projectURL: projectURL).constrainedProjectNames
    }

    /// User agent names a project agent is currently overriding (state#105).
    public func shadowedUserAgentNames(projectURL: URL?) -> [String] {
        resolution(projectURL: projectURL).shadowedUserNames
    }

    public func resolveEffectiveAgents(projectURL: URL?) -> [AppAgentDefinition] {
        resolution(projectURL: projectURL).agents
    }

    private func resolution(projectURL: URL?) -> AgentResolution {
        let key = projectURL?.standardizedFileURL.path ?? ""
        cacheLock.lock()
        let cached = resolutionCache
        cacheLock.unlock()
        if let cached, cached.key == key {
            return cached.result
        }
        // Computed outside the lock; see `SkillManager`'s twin for why.
        let result = computeResolution(projectURL: projectURL)
        cacheLock.lock()
        resolutionCache = (key, result)
        cacheLock.unlock()
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
        var shadowed: [String] = []
        let userNames = Set(userAgents.map { $0.name.lowercased() })
        if let projectURL {
            let projAgents = discoverProjectAgents(projectURL: projectURL)
            for agent in projAgents {
                let key = agent.name.lowercased()
                if userNames.contains(key), builtInsByName[key] == nil {
                    shadowed.append(agent.name)
                }
                if let builtIn = builtInsByName[key] {
                    map[key] = Self.constrained(agent, byBuiltIn: builtIn)
                    constrained.append(agent.name)
                } else {
                    map[key] = agent
                }
            }
        }

        // 4. Plugin agents, last. Their names are namespaced (`plugin:name`),
        // so they cannot collide with any of the above and there is no
        // precedence question -- the colon IS the namespace. An installed
        // plugin never overrides what the user or a project defined.
        let pluginAgents = PluginManager.shared.pluginAgents(projectURL: projectURL)
        for agent in pluginAgents {
            map[agent.name.lowercased()] = agent
        }

        return AgentResolution(
            agents: map.values.sorted { $0.name < $1.name },
            constrainedProjectNames: constrained,
            shadowedUserNames: shadowed.sorted())
    }

    public func findAgent(name: String, projectURL: URL? = nil) -> AppAgentDefinition? {
        let lower = name.lowercased()
        let all = resolveEffectiveAgents(projectURL: projectURL)
        return all.first { $0.name.lowercased() == lower }
    }

    // MARK: - Scanning Directory

    /// - Parameter containedIn: the project root a PROJECT-scoped scan must
    ///   keep its files inside (state#39). Nil for a user scope.
    private func scanDirectory(
        _ dirURL: URL, scope: AppAgentScope, defaultAgent: AgentSourceAgent,
        containedIn root: URL? = nil
    ) -> [AppAgentDefinition] {
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

            if var agent = try? AgentParser.parseFile(
                at: fileURL, scope: scope, sourceAgent: defaultAgent, containedIn: root) {
                agent.isEnabled = !isAgentDisabled(name: agent.name, scope: scope)
                results.append(agent)
            }
        }
        return results
    }
}
