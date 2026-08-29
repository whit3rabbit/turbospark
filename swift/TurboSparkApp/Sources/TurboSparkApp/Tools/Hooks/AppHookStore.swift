import Foundation
import Combine

/// Manages persistence, discovery, trusted command hashes, and options for lifecycle hooks.
@MainActor
public final class AppHookStore: ObservableObject {
    public static let shared = AppHookStore()

    @Published public private(set) var hooks: [AppHookCommand] = []
    @Published public private(set) var trustedHashes: Set<String> = []
    @Published public private(set) var optionValues: [String: [String: String]] = [:] // [SourceID: [OptionKey: OptionValue]]
    @Published public private(set) var sourceGroups: [AppHookSourceGroup] = []

    private let fileManager = FileManager.default

    private var storageDirectory: URL {
        let appSupport = fileManager.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = appSupport.appendingPathComponent("TurboSpark/Hooks", isDirectory: true)
        try? fileManager.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private var trustedHashesFileURL: URL {
        storageDirectory.appendingPathComponent("trusted_hook_hashes.json")
    }

    private var customHooksFileURL: URL {
        storageDirectory.appendingPathComponent("custom_hooks.json")
    }

    private var optionsValuesFileURL: URL {
        storageDirectory.appendingPathComponent("hook_options_values.json")
    }

    public init() {
        loadTrustedHashes()
        loadOptionValues()
    }

    // MARK: - Trust & Review Checks

    /// Returns true if the hook's command and parameters have been explicitly reviewed and trusted by the user.
    public func isHookTrusted(_ hook: AppHookCommand) -> Bool {
        // Custom in-app created hooks are trusted by default
        if hook.sourceType == .custom {
            return true
        }
        return trustedHashes.contains(hook.contentHash)
    }

    /// Marks a hook as trusted by storing its SHA-256 content hash.
    public func trustHook(_ hook: AppHookCommand) {
        trustedHashes.insert(hook.contentHash)
        saveTrustedHashes()
        recomputeSourceGroups(projectDirectory: nil)
    }

    /// Revokes trust for a hook.
    public func untrustHook(_ hook: AppHookCommand) {
        trustedHashes.remove(hook.contentHash)
        saveTrustedHashes()
        recomputeSourceGroups(projectDirectory: nil)
    }

    /// Trusts all currently unreviewed hooks in a source group.
    public func trustAllInGroup(_ groupID: String) {
        if let group = sourceGroups.first(where: { $0.id == groupID }) {
            for hook in group.hooks {
                trustedHashes.insert(hook.contentHash)
            }
            saveTrustedHashes()
            recomputeSourceGroups(projectDirectory: nil)
        }
    }

    // MARK: - Toggle & Option Updates

    public func toggleHookEnabled(id: UUID) {
        if let index = hooks.firstIndex(where: { $0.id == id }) {
            hooks[index].isEnabled.toggle()
            if hooks[index].sourceType == .custom {
                saveCustomHooks()
            }
            recomputeSourceGroups(projectDirectory: nil)
        }
    }

    public func updateOptionValue(sourceID: String, key: String, value: String) {
        var current = optionValues[sourceID] ?? [:]
        current[key] = value
        optionValues[sourceID] = current
        saveOptionValues()
    }

    public func getOptionValue(sourceID: String, key: String, defaultVal: String? = nil) -> String {
        optionValues[sourceID]?[key] ?? defaultVal ?? ""
    }

    // MARK: - Custom Hook Creation & Deletion

    public func addCustomHook(_ hook: AppHookCommand) {
        var newHook = hook
        newHook.sourceType = .custom
        trustedHashes.insert(newHook.contentHash)
        hooks.append(newHook)
        saveCustomHooks()
        saveTrustedHashes()
        recomputeSourceGroups(projectDirectory: nil)
    }

    public func updateCustomHook(_ hook: AppHookCommand) {
        if let index = hooks.firstIndex(where: { $0.id == hook.id }) {
            hooks[index] = hook
            saveCustomHooks()
            recomputeSourceGroups(projectDirectory: nil)
        }
    }

    public func deleteCustomHook(id: UUID) {
        hooks.removeAll { $0.id == id }
        saveCustomHooks()
        recomputeSourceGroups(projectDirectory: nil)
    }

    // MARK: - Reloading & Discovery

    /// Scans global directories, project root, and plugins for hooks.
    public func refresh(projectDirectory: String? = nil) {
        var loaded: [AppHookCommand] = []

        // 1. Load Custom in-app hooks
        loaded.append(contentsOf: loadCustomHooks())

        // 2. Discover Global User Config Hooks (~/.turbospark, ~/.claude)
        loaded.append(contentsOf: discoverGlobalHooks())

        // 3. Discover Project Config Hooks
        if let projectDirectory, !projectDirectory.isEmpty {
            loaded.append(contentsOf: discoverProjectHooks(at: projectDirectory))
        }

        // 4. Discover Plugin Hooks
        loaded.append(contentsOf: discoverPluginHooks())

        self.hooks = loaded
        recomputeSourceGroups(projectDirectory: projectDirectory)
    }

    private func recomputeSourceGroups(projectDirectory: String?) {
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
            case .userConfig, .localConfig:
                groupKey = "user_config"
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

    private func discoverGlobalHooks() -> [AppHookCommand] {
        var results: [AppHookCommand] = []
        let home = fileManager.homeDirectoryForCurrentUser

        let candidates = [
            home.appendingPathComponent(".turbospark/hooks.json"),
            home.appendingPathComponent(".turbospark/settings.json"),
            home.appendingPathComponent(".claude/settings.json")
        ]

        for fileURL in candidates where fileManager.fileExists(atPath: fileURL.path) {
            results.append(contentsOf: parseHooksFile(at: fileURL, sourceType: .userConfig))
        }
        return results
    }

    private func discoverProjectHooks(at directoryPath: String) -> [AppHookCommand] {
        var results: [AppHookCommand] = []
        let projectURL = URL(fileURLWithPath: directoryPath, isDirectory: true)

        let candidates = [
            projectURL.appendingPathComponent(".turbospark/hooks.json"),
            projectURL.appendingPathComponent(".turbospark/settings.json"),
            projectURL.appendingPathComponent(".claude/settings.json"),
            projectURL.appendingPathComponent("hooks/hooks.json")
        ]

        for fileURL in candidates where fileManager.fileExists(atPath: fileURL.path) {
            results.append(contentsOf: parseHooksFile(at: fileURL, sourceType: .projectConfig))
        }
        return results
    }

    private func discoverPluginHooks() -> [AppHookCommand] {
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
                        results.append(contentsOf: parseHooksFile(at: candidate, sourceType: .plugin, pluginName: pluginName))
                    }
                }
            }
        }
        return results
    }

    private func parseHooksFile(at fileURL: URL, sourceType: AppHookSourceType, pluginName: String? = nil) -> [AppHookCommand] {
        guard let data = try? Data(contentsOf: fileURL),
              let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }

        var results: [AppHookCommand] = []
        let hooksContainer = (json["hooks"] as? [String: Any]) ?? json

        for (eventRaw, matchersRaw) in hooksContainer {
            guard let event = AppHookEvent(rawValue: eventRaw) else { continue }

            if let matcherList = matchersRaw as? [[String: Any]] {
                for mObj in matcherList {
                    let matcherPattern = mObj["matcher"] as? String
                    if let hookCommands = mObj["hooks"] as? [[String: Any]] {
                        for hObj in hookCommands {
                            if let cmd = parseHookObject(hObj, event: event, matcher: matcherPattern, sourceType: sourceType, sourcePath: fileURL.path, pluginName: pluginName) {
                                results.append(cmd)
                            }
                        }
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
        guard let typeStr = dict["type"] as? String,
              let hookType = AppHookType(rawValue: typeStr) else {
            return nil
        }

        let command = (dict["command"] as? String) ?? (dict["prompt"] as? String) ?? (dict["url"] as? String) ?? ""
        guard !command.isEmpty else { return nil }

        let ifCond = dict["if"] as? String
        let shellStr = dict["shell"] as? String ?? "zsh"
        let shell = AppHookShell(rawValue: shellStr) ?? .zsh
        let timeout = (dict["timeout"] as? Double) ?? 30.0
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

    // MARK: - Persistence IO

    private func loadTrustedHashes() {
        if let data = try? Data(contentsOf: trustedHashesFileURL),
           let list = try? JSONDecoder().decode([String].self, from: data) {
            self.trustedHashes = Set(list)
        } else {
            self.trustedHashes = []
        }
    }

    private func saveTrustedHashes() {
        let list = Array(trustedHashes).sorted()
        if let data = try? JSONEncoder().encode(list) {
            try? data.write(to: trustedHashesFileURL, options: .atomic)
        }
    }

    private func loadOptionValues() {
        if let data = try? Data(contentsOf: optionsValuesFileURL),
           let dict = try? JSONDecoder().decode([String: [String: String]].self, from: data) {
            self.optionValues = dict
        } else {
            self.optionValues = [:]
        }
    }

    private func saveOptionValues() {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(optionValues) {
            try? data.write(to: optionsValuesFileURL, options: .atomic)
        }
    }

    private func loadCustomHooks() -> [AppHookCommand] {
        guard let data = try? Data(contentsOf: customHooksFileURL),
              let list = try? JSONDecoder().decode([AppHookCommand].self, from: data) else {
            return []
        }
        return list
    }

    private func saveCustomHooks() {
        let customOnly = hooks.filter { $0.sourceType == .custom }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(customOnly) {
            try? data.write(to: customHooksFileURL, options: .atomic)
        }
    }
}
