import Foundation

/// What an enabled plugin contributes, resolved through the same parsers the
/// first-party surfaces use: `SkillParser` for skills and slash commands,
/// `AgentParser` for agents, `ProjectMcpDetector` for MCP servers. A plugin
/// contribution is therefore shaped exactly like its hand-written
/// counterpart, differing only in the `plugin:` namespace on its name.
extension PluginManager {
    // MARK: - Skills and slash commands

    /// Every skill and slash command contributed by the enabled plugins for
    /// a project, named `<plugin>:<skill>` (nested command directories add
    /// namespace segments, Claude Code's `plugin:ns:name`).
    public func pluginSkills(projectURL: URL?) -> [AppSkill] {
        var skills: [AppSkill] = []
        var seen = Set<String>()
        for plugin in enabledPlugins(projectURL: projectURL) {
            for skill in skillsContributed(by: plugin) {
                let key = skill.name.lowercased()
                guard !seen.contains(key) else { continue }
                seen.insert(key)
                skills.append(skill)
            }
        }
        return skills
    }

    func skillsContributed(by plugin: LoadedPlugin) -> [AppSkill] {
        let fm = FileManager.default
        let root = plugin.directoryURL
        let context = expansionContext(for: plugin)
        var results: [AppSkill] = []

        // 1. Skills: the manifest's `skills` paths, else the conventional
        //    `skills/` directory. Claude Code treats a directory containing
        //    SKILL.md as one skill named after the directory.
        let skillDirs = plugin.manifest.skillPaths.isEmpty ? ["skills"] : plugin.manifest.skillPaths
        for relative in skillDirs {
            let dirURL = root.appendingPathComponent(relative)
            guard fm.fileExists(atPath: dirURL.path) else { continue }
            for parsed in SkillManager.shared.scanDirectory(
                dirURL, scope: .plugin(pluginID: plugin.id), defaultAgent: .claude,
                containedIn: root) {
                results.append(namespacedSkill(parsed, plugin: plugin, context: context))
            }
        }

        // 2. Slash commands: the manifest's `commands` specs, else the
        //    conventional `commands/` directory walked recursively.
        results.append(contentsOf: commandSkills(for: plugin, context: context))
        return results
    }

    private func commandSkills(for plugin: LoadedPlugin, context: ExpansionContext) -> [AppSkill] {
        let fm = FileManager.default
        var results: [AppSkill] = []

        if plugin.manifest.commandSpecs.isEmpty {
            let commandsDir = plugin.directoryURL.appendingPathComponent("commands", isDirectory: true)
            guard fm.fileExists(atPath: commandsDir.path) else { return [] }
            for file in markdownFiles(under: commandsDir) {
                let relative = file.path.replacingOccurrences(of: commandsDir.path + "/", with: "")
                let segments = relative.split(separator: "/").map(String.init)
                let base = segments.last.map { ($0 as NSString).deletingPathExtension } ?? relative
                // Nested directories add namespace segments:
                // commands/review/lint.md -> plugin:review:lint.
                let name = ([plugin.namespace] + segments.dropLast().map { $0.lowercased() } + [base.lowercased()])
                    .joined(separator: ":")
                if let parsed = try? SkillParser.parseFile(
                    at: file, scope: .plugin(pluginID: plugin.id), agentOrigin: .claude,
                    containedIn: plugin.directoryURL) {
                    results.append(namespacedSkill(parsed, plugin: plugin, context: context, forcedName: name))
                }
            }
            return results
        }

        for spec in plugin.manifest.commandSpecs {
            let fallbackName = spec.name ?? URL(fileURLWithPath: spec.sourcePath ?? "command")
                .deletingPathExtension().lastPathComponent
            let name = "\(plugin.namespace):\(fallbackName.lowercased())"

            if let content = spec.inlineContent {
                var manifest = SkillManifest(
                    name: name,
                    description: spec.commandDescription,
                    argumentHint: spec.argumentHint)
                manifest.allowedTools = expandAllowedTools(
                    manifest.allowedTools, context: context)
                results.append(finishedSkill(
                    AppSkill(
                        manifest: manifest,
                        content: PluginVariableExpander.expand(
                            content, pluginRoot: context.root, pluginData: context.data,
                            optionValue: context.optionValue, sensitiveKeys: context.sensitiveKeys),
                        sourceURL: plugin.directoryURL
                            .appendingPathComponent("commands/\(fallbackName).md"),
                        skillDirectoryURL: nil,
                        scope: .plugin(pluginID: plugin.id),
                        agentOrigin: .claude)))
                continue
            }

            guard let sourcePath = spec.sourcePath else { continue }
            let resolved = plugin.directoryURL.appendingPathComponent(sourcePath)
            if fm.fileExists(atPath: resolved.path) && !resolved.hasDirectoryPath {
                if let parsed = try? SkillParser.parseFile(
                    at: resolved, scope: .plugin(pluginID: plugin.id), agentOrigin: .claude,
                    containedIn: plugin.directoryURL) {
                    results.append(namespacedSkill(parsed, plugin: plugin, context: context, forcedName: name))
                }
            }
        }
        return results
    }

    private func markdownFiles(under dir: URL) -> [URL] {
        guard let enumerator = FileManager.default.enumerator(
            at: dir,
            includingPropertiesForKeys: [.isRegularFileKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants])
        else { return [] }
        return enumerator
            .compactMap { $0 as? URL }
            .filter { $0.pathExtension.lowercased() == "md" }
            .sorted { $0.path < $1.path }
    }

    /// Applies the plugin namespace and the variable-expansion contract to a
    /// parsed skill.
    private func namespacedSkill(
        _ skill: AppSkill, plugin: LoadedPlugin, context: ExpansionContext,
        forcedName: String? = nil
    ) -> AppSkill {
        var updated = skill
        let base = (forcedName ?? skill.name).lowercased()
        let namespaced = base.hasPrefix("\(plugin.namespace):") ? base : "\(plugin.namespace):\(base)"
        updated.manifest.name = namespaced
        updated.manifest.allowedTools = expandAllowedTools(updated.manifest.allowedTools, context: context)
        updated.content = PluginVariableExpander.expand(
            updated.content, pluginRoot: context.root, pluginData: context.data,
            optionValue: context.optionValue, sensitiveKeys: context.sensitiveKeys)
        return finishedSkill(updated)
    }

    private func expandAllowedTools(_ tools: [String], context: ExpansionContext) -> [String] {
        tools.map {
            PluginVariableExpander.expand(
                $0, pluginRoot: context.root, pluginData: context.data,
                optionValue: context.optionValue, sensitiveKeys: context.sensitiveKeys)
        }
    }

    /// The per-skill disable state is keyed on the NAMESPACED name, so it is
    /// applied here rather than inside the directory scan, which runs before
    /// the namespace exists.
    private func finishedSkill(_ skill: AppSkill) -> AppSkill {
        var updated = skill
        updated.isEnabled = skill.isEnabled
            && !SkillManager.shared.isSkillDisabled(scope: skill.scope, name: skill.name)
        return updated
    }

    // MARK: - Agents

    /// Subagent definitions from `agents/` (or the manifest's `agents`
    /// paths), named `<plugin>:<name>`.
    public func pluginAgents(projectURL: URL?) -> [AppAgentDefinition] {
        var agents: [AppAgentDefinition] = []
        var seen = Set<String>()
        for plugin in enabledPlugins(projectURL: projectURL) {
            for agent in agentsContributed(by: plugin) {
                let key = agent.name.lowercased()
                guard !seen.contains(key) else { continue }
                seen.insert(key)
                agents.append(agent)
            }
        }
        return agents
    }

    func agentsContributed(by plugin: LoadedPlugin) -> [AppAgentDefinition] {
        let fm = FileManager.default
        let root = plugin.directoryURL
        let agentDirs = plugin.manifest.agentPaths.isEmpty ? ["agents"] : plugin.manifest.agentPaths
        var results: [AppAgentDefinition] = []

        for relative in agentDirs {
            let dirURL = root.appendingPathComponent(relative)
            guard fm.fileExists(atPath: dirURL.path) else { continue }
            guard let enumerator = fm.enumerator(
                at: dirURL,
                includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles, .skipsPackageDescendants])
            else { continue }

            for case let fileURL as URL in enumerator {
                let ext = fileURL.pathExtension.lowercased()
                guard ext == "md" || ext == "json" else { continue }
                guard var agent = try? AgentParser.parseFile(
                    at: fileURL, scope: .plugin, sourceAgent: .custom,
                    containedIn: plugin.directoryURL)
                else { continue }
                // Claude Code's trust boundary, ported: a plugin agent's
                // `permissionMode`, `hooks` and `mcpServers` frontmatter are
                // IGNORED, because those escalate what a third-party file
                // may do. `AgentParser` never reads those keys, so the
                // boundary holds by construction rather than by scan; the
                // capability ceiling is `AppAgentDefinition.maxTurnsCeiling`
                // plus whatever `tools`/`disallowedTools` the file declares.
                agent.name = "\(plugin.namespace):\(agent.name.lowercased())"
                agent.isEnabled = !AgentManager.shared.isAgentDisabled(name: agent.name, scope: .plugin)
                results.append(agent)
            }
        }
        return results
    }

    // MARK: - MCP servers

    /// MCP servers contributed by enabled plugins, named
    /// `plugin:<plugin>:<server>` so they cannot collide with a user's own
    /// server names (the namespacing IS the collision rule, as in Claude
    /// Code).
    public func pluginMcpServers(projectURL: URL?) -> [McpServerConfig] {
        var servers: [McpServerConfig] = []
        var seen = Set<String>()
        for plugin in enabledPlugins(projectURL: projectURL) {
            for config in mcpServersContributed(by: plugin) {
                let key = config.name.lowercased()
                guard !seen.contains(key) else { continue }
                seen.insert(key)
                servers.append(config)
            }
        }
        return servers
    }

    func mcpServersContributed(by plugin: LoadedPlugin) -> [McpServerConfig] {
        let fm = FileManager.default
        let context = expansionContext(for: plugin)
        var raw: [McpServerConfig] = []

        // The conventional root .mcp.json, then any manifest-declared files
        // (which win on name collisions within the plugin, last wins).
        let rootMcpJSON = plugin.directoryURL.appendingPathComponent(".mcp.json")
        if fm.fileExists(atPath: rootMcpJSON.path),
            let rootMcpJSON = PathContainment.resolvedIfContained(
                rootMcpJSON, in: plugin.directoryURL) {
            raw.append(contentsOf: ProjectMcpDetector.parseConfigFile(
                at: rootMcpJSON, rootURL: plugin.directoryURL))
        }
        for relative in plugin.manifest.mcpServerFilePaths {
            let fileURL = plugin.directoryURL.appendingPathComponent(relative)
            guard fm.fileExists(atPath: fileURL.path),
                let fileURL = PathContainment.resolvedIfContained(
                    fileURL, in: plugin.directoryURL)
            else { continue }
            raw.append(contentsOf: ProjectMcpDetector.parseConfigFile(
                at: fileURL, rootURL: plugin.directoryURL))
        }
        if let inline = plugin.manifest.inlineMcpServersJSON,
            let dict = try? JSONSerialization.jsonObject(with: inline) as? [String: Any] {
            let wrapped = try? JSONSerialization.data(withJSONObject: ["mcpServers": dict])
            if let wrapped {
                raw.append(contentsOf: ProjectMcpDetector.parseConfigData(
                    wrapped, sourcePath: plugin.directoryURL.path,
                    sourceLabel: "plugin manifest", rootURL: plugin.directoryURL))
            }
        }

        return raw.map { config in
            var renamed = config
            renamed.name = "plugin:\(plugin.name):\(config.name)"
            renamed.isEnabled = true
            renamed.autoApprove = false
            renamed.sourcePath = plugin.directoryURL.path
            renamed.serverDescription = "Plugin: \(plugin.name)"
            return expandingPluginVariables(in: renamed, context: context)
        }
    }

    private func expandingPluginVariables(
        in config: McpServerConfig, context: ExpansionContext
    ) -> McpServerConfig {
        var updated = config
        func expand(_ input: String) -> String {
            PluginVariableExpander.expand(
                input, pluginRoot: context.root, pluginData: context.data,
                optionValue: context.optionValue, sensitiveKeys: context.sensitiveKeys,
                // A child process env is per-process and unreadable by other
                // users, the same reasoning the hook runner uses (state#60).
                preserveSensitive: true)
        }
        switch updated.transport {
        case .stdio(let command, let args, let env, let cwd, let passthrough):
            updated.transport = .stdio(
                command: expand(command),
                args: args.map(expand),
                env: env.mapValues(expand),
                cwd: cwd.map(expand),
                envPassthrough: passthrough)
        case .sse(let url, let headers):
            updated.transport = .sse(
                url: URL(string: expand(url.absoluteString)) ?? url,
                headers: headers.mapValues(expand))
        }
        return updated
    }

    // MARK: - Expansion context

    struct ExpansionContext {
        var root: String
        var data: String
        var optionValue: (String) -> String?
        var sensitiveKeys: Set<String>
    }

    func expansionContext(for plugin: LoadedPlugin) -> ExpansionContext {
        let sourceID = "plugin_\(plugin.name)"
        let stored = Self.hookOptionValues()[sourceID] ?? [:]
        let sensitive = Set(plugin.manifest.userConfig.filter(\.isSensitive).map(\.key))
        // Defaults from the manifest fill what the user has not set.
        let defaults = Dictionary(
            plugin.manifest.userConfig.compactMap { option in
                option.defaultValue.map { (option.key, $0) }
            },
            uniquingKeysWith: { first, _ in first })
        return ExpansionContext(
            root: plugin.directoryURL.path,
            data: dataDirectory(for: plugin).path,
            optionValue: { key in stored[key] ?? defaults[key] },
            sensitiveKeys: sensitive)
    }

    /// Reads the hooks options store directly. `AppHookStore` is a
    /// `@MainActor` singleton and this is reached from tool-execution paths;
    /// the file is plain JSON written by the same store.
    static func hookOptionValues() -> [String: [String: String]] {
        let url = AppStorageRoot.subdirectory("Hooks")
            .appendingPathComponent("hook_options_values.json")
        guard let data = try? Data(contentsOf: url),
            let dict = try? JSONDecoder().decode([String: [String: String]].self, from: data)
        else { return [:] }
        return dict
    }
}
