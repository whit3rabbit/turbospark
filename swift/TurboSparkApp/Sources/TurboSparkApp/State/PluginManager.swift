import Foundation

/// Central registry of installed plugins: discovery across the TurboSpark
/// and Claude Code roots, the enable cascade, and the accessors the
/// contribution surfaces (skills, agents, hooks, MCP) read through.
///
/// Precedence on a lowercased name collision, first match wins:
/// `~/.turbospark/plugins` flat dirs, then locally-registered folders, then
/// the versioned install cache, then Claude Code's own installs (read-only).
/// That is Claude Code's "session beats marketplace beats builtin" order
/// adapted to this app's roots, and the shadowed entries are recorded as
/// diagnostics rather than vanishing.
///
/// The resolution cache is guarded by an NSLock for the same reason
/// `SkillManager`'s is (state#69): `AppToolRegistry.execute` reads it off
/// the cooperative pool while every writer is main-actor.
public final class PluginManager: @unchecked Sendable {
    public static let shared = PluginManager()

    /// The writable root: flat plugin dirs, the versioned install cache, the
    /// per-plugin data dirs and the install ledger all live under here.
    public let turboSparkRoot: URL
    /// Claude Code's own plugin root, read-only. Read, never written --
    /// exactly like `~/.claude/skills` and `~/.claude/agents`.
    public let claudeRoot: URL

    private let lock = NSLock()
    private var resolutionCache: (key: String, result: PluginResolution)?

    public struct PluginResolution: Sendable {
        /// All discovered plugins, precedence-ordered. Includes DISABLED
        /// ones; `enabledPlugins(projectURL:)` filters.
        public var plugins: [LoadedPlugin]
        /// One line per plugin that failed to load, named. A corrupt
        /// manifest fails its plugin alone; siblings load.
        public var loadErrors: [String]
    }

    // Enable-state providers, injectable for tests. Defaults read the same
    // stores the app persists through, so there is no second copy of the
    // enable state anywhere.
    private let userEnableProvider: () -> [String: Bool]
    private let projectEnableProvider: (URL) -> [String: Bool]
    private let claudeEnableProvider: () -> [String: Bool]


    public init(
        turboSparkRoot: URL? = nil,
        claudeRoot: URL? = nil,
        userEnableProvider: (() -> [String: Bool])? = nil,
        projectEnableProvider: ((URL) -> [String: Bool])? = nil,
        claudeEnableProvider: (() -> [String: Bool])? = nil
    ) {
        let home = FileManager.default.homeDirectoryForCurrentUser
        // Profile-aware roots: the Default profile keeps both shared trees,
        // any other profile holds its plugins in its own folder and reads no
        // cross-harness root at all.
        let profileStoreDir = UserProfileStore.storeDirectory()
        self.turboSparkRoot =
            turboSparkRoot
            ?? (UserProfileStore.isDefault
                ? home.appendingPathComponent(".turbospark/plugins", isDirectory: true)
                : UserProfileStore.userScopeSubdirectory("plugins"))
        self.claudeRoot =
            claudeRoot
            ?? (UserProfileStore.isDefault
                ? home.appendingPathComponent(".claude/plugins", isDirectory: true)
                // Read-only and nonexistent inside the profile's folder, so
                // nothing shared is discovered. The `?? home` is unreachable
                // while `isDefault` is false; it only satisfies the Optional.
                : (profileStoreDir ?? home)
                    .appendingPathComponent(".claude/plugins", isDirectory: true))
        self.userEnableProvider = userEnableProvider ?? { MacAppSettingsFileStore.load().enabledPlugins }
        self.projectEnableProvider = projectEnableProvider ?? { projectURL in
            let path = projectURL.standardizedFileURL.path
            let project = AppProjectFileStore.load().projects.first {
                $0.rootDirectoryPath != nil
                    && URL(fileURLWithPath: $0.rootDirectoryPath!).standardizedFileURL.path == path
            }
            return project?.enabledPlugins ?? [:]
        }
        self.claudeEnableProvider = claudeEnableProvider ?? {
            Self.claudeEnabledPlugins(settingsURL: home.appendingPathComponent(".claude/settings.json"))
        }
    }

    // MARK: - Resolution

    public func invalidateResolutionCache() {
        lock.lock()
        resolutionCache = nil
        lock.unlock()
    }

    public func resolve(projectURL: URL?) -> PluginResolution {
        let key = projectURL?.standardizedFileURL.path ?? ""
        lock.lock()
        let cached = resolutionCache
        lock.unlock()
        if let cached, cached.key == key {
            return cached.result
        }
        let result = computeResolution(projectKey: key)
        lock.lock()
        resolutionCache = (key, result)
        lock.unlock()
        return result
    }

    /// The enabled plugins for a project, precedence-ordered.
    public func enabledPlugins(projectURL: URL?) -> [LoadedPlugin] {
        let resolution = resolve(projectURL: projectURL)
        return resolution.plugins.filter { isResolvedEnabled($0, projectURL: projectURL, userMap: nil) }
    }

    public func findPlugin(
        named name: String, projectURL: URL?
    ) -> LoadedPlugin? {
        resolve(projectURL: projectURL).plugins.first { $0.name.lowercased() == name.lowercased() }
    }

    /// Whether one resolved plugin is enabled, under the cascade: project
    /// overrides user overrides Claude Code's own setting; absent means
    /// enabled, because an installed plugin that nothing disabled should
    /// run.
    public func isEnabled(pluginID: String, projectURL: URL?) -> Bool {
        isResolvedEnabledID(pluginID: pluginID, projectURL: projectURL)
    }

    private func isResolvedEnabled(
        _ plugin: LoadedPlugin, projectURL: URL?, userMap: [String: Bool]?
    ) -> Bool {
        if let projectURL {
            let projectMap = projectEnableProvider(projectURL)
            if let value = projectMap[plugin.id] { return value }
        }
        let userMap = userMap ?? userEnableProvider()
        if let value = userMap[plugin.id] { return value }
        if plugin.origin == .claudeInterop {
            let claudeMap = claudeEnableProvider()
            if let value = claudeMap[plugin.id] { return value }
        }
        return true
    }

    private func isResolvedEnabledID(pluginID: String, projectURL: URL?) -> Bool {
        guard let plugin = resolve(projectURL: projectURL).plugins.first(where: { $0.id == pluginID }) else {
            return false
        }
        return isResolvedEnabled(plugin, projectURL: projectURL, userMap: nil)
    }

    // MARK: - Discovery

    private func computeResolution(projectKey: String) -> PluginResolution {
        var plugins: [LoadedPlugin] = []
        var errors: [String] = []
        var shadowed: Set<String> = []

        func admit(_ candidate: LoadedPlugin) {
            let key = candidate.name.lowercased()
            if plugins.contains(where: { $0.name.lowercased() == key }) {
                shadowed.insert(candidate.name)
                return
            }
            plugins.append(candidate)
        }

        // 1. TurboSpark flat dirs. The plugin hook discovery that predates
        //    this manager scanned exactly here, so anything it found, this
        //    finds.
        for url in flatPluginDirectories(in: turboSparkRoot) {
            loadPlugin(at: url, origin: .turboSpark, marketplaceName: nil, admit: admit, errors: &errors)
        }

        // 2. Locally-registered folders (the --plugin-dir analog).
        let projectFolders = AppProjectFileStore.load().projects.first {
            $0.rootDirectoryURL?.standardizedFileURL.path == projectKey
        }?.localPluginPaths ?? []
        for url in localPluginDirectories() + projectFolders.map({ URL(fileURLWithPath: $0) }) {
            loadPlugin(at: url, origin: .local, marketplaceName: nil, admit: admit, errors: &errors)
        }

        // 3. The versioned install caches, ours then Claude Code's.
        let ourLedger = ledgerInstallPaths(root: turboSparkRoot)
        for (marketplace, _, version, url) in scopedCacheDirectories(projectKey: projectKey) {
            loadPlugin(
                at: url, origin: .marketplace, marketplaceName: marketplace,
                fallbackVersion: ourLedger[url.standardizedFileURL.path]?.version ?? version,
                admit: admit, errors: &errors)
        }

        let claudeLedger = ledgerInstallPaths(root: claudeRoot)
        for url in flatPluginDirectories(in: claudeRoot) {
            loadPlugin(
                at: url, origin: .claudeInterop, marketplaceName: nil,
                extraDiagnostics: [Self.readOnlyInteropNote],
                admit: admit, errors: &errors)
        }
        for (marketplace, _, version, url) in cachePluginDirectories(in: claudeRoot) {
            loadPlugin(
                at: url, origin: .claudeInterop, marketplaceName: marketplace,
                fallbackVersion: claudeLedger[url.standardizedFileURL.path]?.version ?? version,
                extraDiagnostics: [Self.readOnlyInteropNote],
                admit: admit, errors: &errors)
        }

        var shadowedNotes: [String] = []
        for name in shadowed.sorted() {
            shadowedNotes.append("A second plugin named '\(name)' was found and skipped (first match wins).")
        }
        return PluginResolution(plugins: plugins, loadErrors: errors + shadowedNotes)
    }

    /// Claude Code's plugins are another application's state; this app reads
    /// them and does not write there. Said on every interop row so a user
    /// who toggles one here knows where the enable state lives.
    static let readOnlyInteropNote =
        "Installed by Claude Code; managed read-only here. Enable state: TurboSpark first, then Claude Code's own setting."

    private func loadPlugin(
        at url: URL,
        origin: PluginOriginKind,
        marketplaceName: String?,
        fallbackVersion: String? = nil,
        extraDiagnostics: [String] = [],
        admit: (LoadedPlugin) -> Void,
        errors: inout [String]
    ) {
        let dirName = url.lastPathComponent
        let manifestURL = url.appendingPathComponent(PluginManifestParser.manifestRelativePath)

        let manifest: PluginManifest
        if FileManager.default.fileExists(atPath: manifestURL.path) {
            do {
                let data = try Data(contentsOf: manifestURL)
                manifest = try PluginManifestParser.parseManifest(
                    data: data, fallbackName: dirName, sourceDescription: manifestURL.path)
            } catch let error as PluginLoadError {
                errors.append(error.errorDescription ?? error.reason)
                return
            } catch {
                errors.append("Plugin '\(dirName)': could not read \(manifestURL.path): \(error.localizedDescription)")
                return
            }
        } else if Self.hasContributionSignature(url) {
            // No manifest is fine when convention directories carry the
            // plugin; a synthesized one names the source (Claude Code's
            // rule).
            manifest = PluginManifestParser.synthesizedManifest(
                name: dirName, sourceDescription: url.path)
        } else {
            // Neither manifest nor conventions: not a plugin directory.
            return
        }

        let loaded = LoadedPlugin(
            name: manifest.name,
            manifest: manifest,
            directoryURL: url,
            origin: origin,
            marketplaceName: marketplaceName,
            version: manifest.version ?? fallbackVersion ?? "unknown",
            diagnostics: extraDiagnostics)
        admit(loaded)
    }

    /// The convention directories that make an unmanifested directory a
    /// plugin. `.claude-plugin` alone is NOT a signature: an empty marker
    /// dir would otherwise synthesize a nameless plugin.
    static func hasContributionSignature(_ url: URL) -> Bool {
        let fm = FileManager.default
        let signatures = ["commands", "agents", "skills", "hooks"]
        for name in signatures where fm.fileExists(atPath: url.appendingPathComponent(name).path) {
            return true
        }
        return fm.fileExists(atPath: url.appendingPathComponent(".mcp.json").path)
    }

    private func flatPluginDirectories(in root: URL) -> [URL] {
        let fm = FileManager.default
        guard fm.fileExists(atPath: root.path),
            let entries = try? fm.contentsOfDirectory(
                at: root, includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles])
        else { return [] }
        return entries
            .filter { $0.hasDirectoryPath && !$0.lastPathComponent.hasPrefix(".") }
            .filter { $0.lastPathComponent != "cache" && $0.lastPathComponent != "marketplaces" && $0.lastPathComponent != "data" }
            .sorted { $0.lastPathComponent < $1.lastPathComponent }
    }

    /// Walks `<root>/cache/<marketplace>/<plugin>/<version>/`, keeping the
    /// newest version per `<marketplace, plugin>` pair (updates leave the old
    /// version dir in place until uninstall).
    private func cachePluginDirectories(in root: URL) -> [(marketplace: String, plugin: String, version: String, url: URL)] {
        let fm = FileManager.default
        let cacheRoot = root.appendingPathComponent("cache", isDirectory: true)
        guard fm.fileExists(atPath: cacheRoot.path),
            let marketplaces = try? fm.contentsOfDirectory(at: cacheRoot, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles])
        else { return [] }

        var results: [(String, String, String, URL)] = []
        for marketplaceDir in marketplaces where marketplaceDir.hasDirectoryPath {
            guard let pluginDirs = try? fm.contentsOfDirectory(at: marketplaceDir, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { continue }
            for pluginDir in pluginDirs where pluginDir.hasDirectoryPath {
                guard let versionDirs = try? fm.contentsOfDirectory(at: pluginDir, includingPropertiesForKeys: [.isDirectoryKey], options: [.skipsHiddenFiles]) else { continue }
                let versions = versionDirs
                    .filter { $0.hasDirectoryPath && !$0.lastPathComponent.hasPrefix(".") }
                    .sorted { $0.lastPathComponent.localizedStandardCompare($1.lastPathComponent) == .orderedAscending }
                if let newest = versions.last {
                    results.append((
                        marketplaceDir.lastPathComponent,
                        pluginDir.lastPathComponent,
                        newest.lastPathComponent,
                        newest
                    ))
                }
            }
        }
        return results
    }

    private func scopedCacheDirectories(projectKey: String) -> [(String, String, String, URL)] {
        let ledger = PluginLedgerStore(root: turboSparkRoot).load()
        var result: [(String, String, String, URL)] = []
        for id in ledger.plugins.keys.sorted() {
            guard let records = ledger.plugins[id] else { continue }
            let matching = records.filter {
                $0.scope == "user" || ($0.scope == "project" && !($0.projectPath ?? "").isEmpty
                    && $0.projectPath.map { URL(fileURLWithPath: $0).standardizedFileURL.path } == projectKey)
            }
            // A project-specific version takes precedence over an inherited user version.
            guard let record = matching.first(where: { $0.scope == "project" }) ?? matching.first else { continue }
            let parts = id.split(separator: "@", maxSplits: 1).map(String.init)
            guard parts.count == 2 else { continue }
            result.append((parts[1], parts[0], record.version, URL(fileURLWithPath: record.installPath)))
        }
        for entry in cachePluginDirectories(in: turboSparkRoot) {
            let id = "\(entry.plugin)@\(entry.marketplace)"
            let trackedFamily = ledger.plugins.keys.contains { key in
                let parts = key.split(separator: "@", maxSplits: 1).map(String.init)
                return parts.count == 2 && PluginMarketplaceManager.sanitizedComponent(parts[0]) == entry.plugin
                    && PluginMarketplaceManager.sanitizedComponent(parts[1]) == entry.marketplace
            }
            if ledger.plugins[id] == nil && !trackedFamily {
                result.append((entry.marketplace, entry.plugin, entry.version, entry.url))
            }
        }
        return result
    }

    private func localPluginDirectories() -> [URL] {
        PluginLedgerStore(root: turboSparkRoot).loadLocalPluginPaths().map {
            URL(fileURLWithPath: $0, isDirectory: true)
        }
    }

    private func ledgerInstallPaths(root: URL) -> [String: (id: String, version: String)] {
        let ledger = PluginLedgerStore(root: root).load()
        var byPath: [String: (String, String)] = [:]
        for (id, records) in ledger.plugins {
            for record in records {
                byPath[URL(fileURLWithPath: record.installPath).standardizedFileURL.path] = (id, record.version)
            }
        }
        return byPath
    }

    /// MARK: - Claude Code enable state

    /// Reads `enabledPlugins` out of Claude Code's own settings.json, so an
    /// interop plugin the user disabled THERE stays disabled here. Array
    /// values are the version-constraint form, which counts as enabled.
    static func claudeEnabledPlugins(settingsURL: URL) -> [String: Bool] {
        guard let data = try? Data(contentsOf: settingsURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
            let enabled = json["enabledPlugins"] as? [String: Any]
        else { return [:] }
        var result: [String: Bool] = [:]
        for (key, value) in enabled {
            if let flag = value as? Bool {
                result[key] = flag
            } else if let constraints = value as? [Any] {
                result[key] = !constraints.isEmpty
            }
        }
        return result
    }

    // MARK: - Per-plugin paths

    /// The persistent per-plugin data directory, lazily created and deleted
    /// on last-scope uninstall. Sanitized like Claude Code's cache paths:
    /// every component outside `[A-Za-z0-9_-]` becomes `-`, so an id can
    /// never climb out with `..`.
    public func dataDirectory(for plugin: LoadedPlugin) -> URL {
        let raw = "\(plugin.name)-\(plugin.originKey)"
        let sanitized = String(raw.map { character in
            character.isLetter || character.isNumber || character == "-" || character == "_"
                ? character : "-"
        })
        let dir = turboSparkRoot
            .appendingPathComponent("data", isDirectory: true)
            .appendingPathComponent(sanitized, isDirectory: true)
        try? FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }
}
