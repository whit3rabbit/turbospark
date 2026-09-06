import Foundation

extension AppHookStore {
    // MARK: - Reloading & Discovery

    /// Scans global directories, project root, and plugins for hooks.
    public func refresh(projectDirectory: String? = nil) {
        lastProjectDirectory = projectDirectory
        var loaded: [AppHookCommand] = []
        var diagnostics: [String] = []

        // 1. Load Custom in-app hooks
        loaded.append(contentsOf: loadCustomHooks())

        // 2. Discover Global User Config Hooks (~/.turbospark, ~/.claude)
        loaded.append(contentsOf: discoverGlobalHooks(diagnostics: &diagnostics))

        // 3. Discover Project Config Hooks
        if let projectDirectory, !projectDirectory.isEmpty {
            loaded.append(contentsOf: discoverProjectHooks(at: projectDirectory, diagnostics: &diagnostics))
        }

        // 4. Discover Plugin Hooks
        loaded.append(contentsOf: discoverPluginHooks(
            projectDirectory: projectDirectory, diagnostics: &diagnostics))

        self.hooks = loaded
        self.didRefreshAtLeastOnce = true
        self.discoveryDiagnostics = diagnostics
        recomputeSourceGroups(projectDirectory: projectDirectory)
    }

    func recomputeSourceGroups(projectDirectory: String?) {
        var groups: [String: AppHookSourceGroup] = [:]

        // User Config Group
        groups["user_config"] = AppHookSourceGroup(
            id: "user_config",
            title: "User config",
            subtitle: "All projects (~/.turbospark or ~/.claude)",
            sourceType: .userConfig
        )

        // Project Config Group (if project active)
        if let projectDirectory, !projectDirectory.isEmpty {
            let projectName = URL(fileURLWithPath: projectDirectory).lastPathComponent
            groups["project_config"] = AppHookSourceGroup(
                id: "project_config",
                title: "Project config (\(projectName))",
                subtitle: "Configured in \(projectName) (.turbospark or .claude)",
                sourceType: .projectConfig
            )
            groups["local_config"] = AppHookSourceGroup(
                id: "local_config",
                title: "Local settings (\(projectName))",
                subtitle: "settings.local.json in \(projectName), not shared",
                sourceType: .localConfig
            )
        }

        // Custom in-app hooks group
        groups["custom"] = AppHookSourceGroup(
            id: "custom",
            title: "Custom Hooks",
            subtitle: "In-app created automation and guardrail hooks",
            sourceType: .custom
        )

        // Add hooks into their respective groups
        for hook in hooks {
            let groupKey: String
            switch hook.sourceType {
            case .userConfig:
                groupKey = "user_config"
            case .localConfig:
                groupKey = groups["local_config"] != nil ? "local_config" : "user_config"
            case .projectConfig:
                groupKey = "project_config"
            case .custom:
                groupKey = "custom"
            case .plugin:
                groupKey = "plugin_\(hook.pluginName ?? "generic")"
                if groups[groupKey] == nil {
                    // Option specs come from the plugin's own `userConfig`
                    // manifest, not a guess: this used to return hardcoded
                    // demo specs keyed on the plugin's NAME, so a real
                    // option panel could not exist.
                    let specs = lastLoadedPlugins
                        .first { $0.name.lowercased() == (hook.pluginName ?? "").lowercased() }?
                        .manifest.userConfig
                        .map(\.optionSpec) ?? []
                    groups[groupKey] = AppHookSourceGroup(
                        id: groupKey,
                        title: hook.pluginName ?? "Plugin",
                        subtitle: "Plugin hooks and options",
                        sourceType: .plugin,
                        pluginName: hook.pluginName,
                        optionSpecs: specs
                    )
                }
            }

            if var group = groups[groupKey] {
                group.hooks.append(hook)
                if !isHookTrusted(hook) {
                    group.unreviewedCount += 1
                }
                groups[groupKey] = group
            }
        }

        // Sort groups: User Config, Project Config, Local Config (all
        // config-derived), then Custom, then Plugins.
        let sorted = groups.values.sorted { g1, g2 in
            let order1 = sortOrder(for: g1.sourceType)
            let order2 = sortOrder(for: g2.sourceType)
            if order1 == order2 {
                return g1.title < g2.title
            }
            return order1 < order2
        }

        self.sourceGroups = sorted
    }

    private func sortOrder(for type: AppHookSourceType) -> Int {
        switch type {
        case .userConfig: return 0
        case .projectConfig: return 1
        case .localConfig: return 2
        case .custom: return 3
        case .plugin: return 4
        }
    }

    // MARK: - Discovery Parsers

    private func discoverGlobalHooks(diagnostics: inout [String]) -> [AppHookCommand] {
        var results: [AppHookCommand] = []
        let home = fileManager.homeDirectoryForCurrentUser

        let candidates: [URL]
        if UserProfileStore.isDefault {
            candidates = [
                home.appendingPathComponent(".turbospark/hooks.json"),
                home.appendingPathComponent(".turbospark/settings.json"),
                home.appendingPathComponent(".claude/settings.json")
            ]
        } else {
            // A non-default profile owns its user-scope hook config inside
            // its own folder and reads no shared tree. Only the dedicated
            // hooks.json is scanned there: the profile's settings.json is
            // the app's own settings store, not a hook source.
            candidates = [AppStorageRoot.file("hooks.json")]
        }
        for fileURL in candidates where fileManager.fileExists(atPath: fileURL.path) {
            results.append(contentsOf: parseHooksFile(at: fileURL, sourceType: .userConfig, diagnostics: &diagnostics))
        }

        // Global `settings.local.json`: not part of Claude Code's own
        // precedence table (that file is project-scoped there), but this
        // app also honors a global one defensively -- same `.localConfig`
        // source as the project-scoped file below, since both are "local,
        // not shared" by the same rule.
        let localCandidates: [URL]
        if UserProfileStore.isDefault {
            localCandidates = [
                home.appendingPathComponent(".turbospark/settings.local.json"),
                home.appendingPathComponent(".claude/settings.local.json")
            ]
        } else {
            localCandidates = []
        }
        for fileURL in localCandidates where fileManager.fileExists(atPath: fileURL.path) {
            results.append(contentsOf: parseHooksFile(at: fileURL, sourceType: .localConfig, diagnostics: &diagnostics))
        }

        return results
    }

    private func discoverProjectHooks(at directoryPath: String, diagnostics: inout [String]) -> [AppHookCommand] {
        var results: [AppHookCommand] = []
        let projectURL = URL(fileURLWithPath: directoryPath, isDirectory: true)

        let candidates = [
            projectURL.appendingPathComponent(".turbospark/hooks.json"),
            projectURL.appendingPathComponent(".turbospark/settings.json"),
            projectURL.appendingPathComponent(".claude/settings.json"),
            projectURL.appendingPathComponent("hooks/hooks.json")
        ]
        for fileURL in candidates where fileManager.fileExists(atPath: fileURL.path) {
            results.append(contentsOf: parseHooksFile(at: fileURL, sourceType: .projectConfig, diagnostics: &diagnostics))
        }

        let localCandidates = [
            projectURL.appendingPathComponent(".turbospark/settings.local.json"),
            projectURL.appendingPathComponent(".claude/settings.local.json")
        ]
        for fileURL in localCandidates where fileManager.fileExists(atPath: fileURL.path) {
            results.append(contentsOf: parseHooksFile(at: fileURL, sourceType: .localConfig, diagnostics: &diagnostics))
        }

        return results
    }

    /// Enumerates ENABLED plugins through `PluginManager` and parses their
    /// hooks: the conventional `hooks/hooks.json` (or a bare `hooks.json`
    /// from the pre-plugin stub era), the manifest's declared `./file.json`
    /// paths, and the manifest's inline hooks schema.
    private func discoverPluginHooks(
        projectDirectory: String?, diagnostics: inout [String]
    ) -> [AppHookCommand] {
        var results: [AppHookCommand] = []
        let projectURL = projectDirectory.map { URL(fileURLWithPath: $0, isDirectory: true) }
        let plugins: [LoadedPlugin]
        if let pluginProvider {
            plugins = pluginProvider(projectDirectory)
        } else {
            plugins = PluginManager.shared.enabledPlugins(projectURL: projectURL)
        }
        lastLoadedPlugins = plugins

        for plugin in plugins {
            var candidates: [URL] = [
                plugin.directoryURL.appendingPathComponent("hooks/hooks.json"),
                plugin.directoryURL.appendingPathComponent("hooks.json")
            ]
            for relative in plugin.manifest.hookFilePaths {
                candidates.append(plugin.directoryURL.appendingPathComponent(relative))
            }
            var seenPaths = Set<String>()
            for candidate in candidates where fileManager.fileExists(atPath: candidate.path) {
                guard seenPaths.insert(candidate.standardizedFileURL.path).inserted else { continue }
                results.append(contentsOf: parseHooksFile(
                    at: candidate, sourceType: .plugin, pluginName: plugin.name,
                    diagnostics: &diagnostics))
            }
            if let inline = plugin.manifest.inlineHooksJSON {
                results.append(contentsOf: parseHooksData(
                    inline, sourceType: .plugin, sourcePath: plugin.directoryURL.path,
                    pluginName: plugin.name, diagnostics: &diagnostics))
            }
        }
        return results
    }

    private func parseHooksFile(
        at fileURL: URL,
        sourceType: AppHookSourceType,
        pluginName: String? = nil,
        diagnostics: inout [String]
    ) -> [AppHookCommand] {
        guard let data = try? Data(contentsOf: fileURL) else { return [] }
        return parseHooksData(
            data, sourceType: sourceType, sourcePath: fileURL.path,
            pluginName: pluginName, diagnostics: &diagnostics)
    }

    /// The data-level form, shared by config files and a plugin manifest's
    /// inline hooks schema (which has no file behind it).
    func parseHooksData(
        _ data: Data,
        sourceType: AppHookSourceType,
        sourcePath: String,
        pluginName: String?,
        diagnostics: inout [String]
    ) -> [AppHookCommand] {
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            diagnostics.append("Could not parse \(sourcePath) as JSON.")
            return []
        }

        var results: [AppHookCommand] = []
        let hooksContainer = (json["hooks"] as? [String: Any]) ?? json

        for (eventRaw, matchersRaw) in hooksContainer {
            guard let event = AppHookEvent(rawValue: eventRaw) else { continue }

            guard let matcherList = matchersRaw as? [[String: Any]] else { continue }
            for mObj in matcherList {
                let matcherPattern = mObj["matcher"] as? String
                guard let hookCommands = mObj["hooks"] as? [[String: Any]] else { continue }
                for hObj in hookCommands {
                    if let cmd = parseHookObject(hObj, event: event, matcher: matcherPattern, sourceType: sourceType, sourcePath: sourcePath, pluginName: pluginName) {
                        results.append(cmd)
                        // A prompt hook LOADS but is never evaluated; say so
                        // at discovery rather than letting it silently
                        // no-op at dispatch time.
                        if cmd.type == .prompt {
                            diagnostics.append("\"\(cmd.name)\" in \(sourcePath) is a prompt hook, which this client does not evaluate; it is skipped at dispatch.")
                        }
                    } else {
                        diagnostics.append("Dropped an unparseable \(event.rawValue) hook entry in \(sourcePath) (missing or empty command).")
                    }
                }
            }
        }
        return results
    }

    private func parseHookObject(
        _ dict: [String: Any],
        event: AppHookEvent,
        matcher: String?,
        sourceType: AppHookSourceType,
        sourcePath: String,
        pluginName: String?
    ) -> AppHookCommand? {        // Claude Code omits `type` for a command hook; default to `.command`
        // rather than dropping the entry, matching Gotcha 39's rule (a
        // missing optional key is a claim about what silence means, and here
        // it means "the common case").
        let hookType: AppHookType
        if let typeStr = dict["type"] as? String {
            // "agent" is Claude Code's LLM-evaluator sibling of "prompt".
            // Falling through to the `.command` default here would run the
            // agent's PROMPT TEXT as a shell command, so it maps to the
            // other unevaluated type instead.
            hookType = AppHookType(rawValue: typeStr) ?? (typeStr == "agent" ? .prompt : .command)
        } else {
            hookType = .command
        }

        let command = (dict["command"] as? String) ?? (dict["prompt"] as? String) ?? (dict["url"] as? String) ?? ""
        guard !command.isEmpty else { return nil }

        let ifCond = dict["if"] as? String
        let shellStr = dict["shell"] as? String ?? "zsh"
        let shell = AppHookShell(rawValue: shellStr) ?? .zsh
        let timeout = (dict["timeout"] as? Double) ?? 600.0
        let statusMsg = dict["statusMessage"] as? String
        let isAsync = (dict["async"] as? Bool) ?? false

        let fallbackName: String
        if let statusMsg, !statusMsg.isEmpty {
            fallbackName = statusMsg
        } else if let pluginName {
            fallbackName = "\(pluginName) \(event.rawValue) Hook"
        } else {
            fallbackName = "\(event.rawValue) Hook"
        }

        return AppHookCommand(
            name: fallbackName,
            event: event,
            type: hookType,
            command: command,
            ifCondition: ifCond,
            matcher: matcher,
            shell: shell,
            timeoutSeconds: timeout,
            statusMessage: statusMsg,
            isAsync: isAsync,
            isEnabled: true,
            sourceType: sourceType,
            sourcePath: sourcePath,
            pluginName: pluginName
        )
    }
}
