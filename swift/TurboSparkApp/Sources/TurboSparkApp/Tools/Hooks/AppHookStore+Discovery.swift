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
        loaded.append(contentsOf: discoverPluginHooks(diagnostics: &diagnostics))

        self.hooks = loaded
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
                    groups[groupKey] = AppHookSourceGroup(
                        id: groupKey,
                        title: hook.pluginName ?? "Plugin",
                        subtitle: "Plugin hooks and options",
                        sourceType: .plugin,
                        pluginName: hook.pluginName,
                        optionSpecs: defaultOptionSpecs(for: hook.pluginName)
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

        // Sort groups: User Config first, Project Config second, Custom third, then Plugins
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
        case .custom: return 2
        case .localConfig: return 3
        case .plugin: return 4
        }
    }

    // MARK: - Discovery Parsers

    private func discoverGlobalHooks(diagnostics: inout [String]) -> [AppHookCommand] {
        var results: [AppHookCommand] = []
        let home = fileManager.homeDirectoryForCurrentUser

        let candidates = [
            home.appendingPathComponent(".turbospark/hooks.json"),
            home.appendingPathComponent(".turbospark/settings.json"),
            home.appendingPathComponent(".claude/settings.json")
        ]
        for fileURL in candidates where fileManager.fileExists(atPath: fileURL.path) {
            results.append(contentsOf: parseHooksFile(at: fileURL, sourceType: .userConfig, diagnostics: &diagnostics))
        }

        // Global `settings.local.json`: not part of Claude Code's own
        // precedence table (that file is project-scoped there), but this
        // app also honors a global one defensively -- same `.localConfig`
        // source as the project-scoped file below, since both are "local,
        // not shared" by the same rule.
        let localCandidates = [
            home.appendingPathComponent(".turbospark/settings.local.json"),
            home.appendingPathComponent(".claude/settings.local.json")
        ]
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

    private func discoverPluginHooks(diagnostics: inout [String]) -> [AppHookCommand] {
        var results: [AppHookCommand] = []
        let home = fileManager.homeDirectoryForCurrentUser

        let pluginDirs = [
            home.appendingPathComponent(".turbospark/plugins"),
            home.appendingPathComponent(".claude/plugins")
        ]

        for dir in pluginDirs where fileManager.fileExists(atPath: dir.path) {
            if let contents = try? fileManager.contentsOfDirectory(atPath: dir.path) {
                for pluginName in contents where !pluginName.hasPrefix(".") {
                    let pluginPath = dir.appendingPathComponent(pluginName)
                    let hookCandidates = [
                        pluginPath.appendingPathComponent("hooks/hooks.json"),
                        pluginPath.appendingPathComponent("hooks.json")
                    ]
                    for candidate in hookCandidates where fileManager.fileExists(atPath: candidate.path) {
                        results.append(contentsOf: parseHooksFile(at: candidate, sourceType: .plugin, pluginName: pluginName, diagnostics: &diagnostics))
                    }
                }
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
        guard let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            diagnostics.append("Could not parse \(fileURL.path) as JSON.")
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
                    if let cmd = parseHookObject(hObj, event: event, matcher: matcherPattern, sourceType: sourceType, sourcePath: fileURL.path, pluginName: pluginName) {
                        results.append(cmd)
                    } else {
                        diagnostics.append("Dropped an unparseable \(event.rawValue) hook entry in \(fileURL.path) (missing or empty command).")
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
    ) -> AppHookCommand? {
        // Claude Code omits `type` for a command hook; default to `.command`
        // rather than dropping the entry, matching Gotcha 39's rule (a
        // missing optional key is a claim about what silence means, and here
        // it means "the common case").
        let hookType: AppHookType
        if let typeStr = dict["type"] as? String {
            hookType = AppHookType(rawValue: typeStr) ?? .command
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

    private func defaultOptionSpecs(for pluginName: String?) -> [AppHookOptionSpec] {
        guard let pluginName else { return [] }
        if pluginName.contains("forge") {
            return [
                AppHookOptionSpec(key: "api_key", type: .string, title: "API Key", description: "Forge authentication token", isSensitive: true),
                AppHookOptionSpec(key: "strict_mode", type: .boolean, title: "Strict Mode", description: "Enforce strict lint checks", defaultValue: "true")
            ]
        } else if pluginName.contains("secret") {
            return [
                AppHookOptionSpec(key: "scan_depth", type: .number, title: "Scan Depth", description: "Maximum directory depth to scan", defaultValue: "5"),
                AppHookOptionSpec(key: "mask_findings", type: .boolean, title: "Mask Findings", description: "Redact found secret tokens in logs", defaultValue: "true")
            ]
        }
        return []
    }
}
