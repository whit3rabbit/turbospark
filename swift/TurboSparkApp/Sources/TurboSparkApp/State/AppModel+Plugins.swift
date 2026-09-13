import Foundation

extension AppModel {
    // MARK: - Loading

    /// Re-resolves installed plugins from disk. Call after anything that
    /// changes what is on disk or which plugins are enabled -- the resolution
    /// cache lives in `PluginManager` and every contribution surface reads
    /// through it.
    public func reloadPlugins() {
        PluginManager.shared.invalidateResolutionCache()
        let resolution = PluginManager.shared.resolve(
            projectURL: selectedProject?.rootDirectoryURL)
        installedPlugins = resolution.plugins
        pluginLoadDiagnostics = resolution.loadErrors
    }

    // MARK: - Enable / disable

    /// Toggles a plugin at USER scope and persists it. Project-scope
    /// overrides are set per project in the project archive and win over
    /// this; see `swift/docs/SWIFT_PLUGINS.md` for the cascade.
    public func setPluginEnabled(_ plugin: LoadedPlugin, _ enabled: Bool) {
        pluginEnableState[plugin.id] = enabled
        persistSettings()
        pluginStateChanged()
        showToast(
            "\(plugin.name) \(enabled ? "enabled" : "disabled").",
            style: .info)
    }

    /// Whether the toggle should show on/off for a plugin under the current
    /// selection: the resolved cascade, not the raw stored value.
    public func isPluginEnabled(_ plugin: LoadedPlugin) -> Bool {
        PluginManager.shared.isEnabled(
            pluginID: plugin.id, projectURL: selectedProject?.rootDirectoryURL)
    }

    /// Every path a plugin's contributions reach a turn through has to see
    /// the same state at the same time, so one change invalidates all of
    /// them together.
    func pluginStateChanged() {
        PluginManager.shared.invalidateResolutionCache()
        SkillManager.shared.invalidateResolutionCache()
        AgentManager.shared.invalidateResolutionCache()
        reloadPlugins()
        reloadSkills()
        reloadAgents()
        AppHookStore.shared.refresh(projectDirectory: selectedProject?.rootDirectoryPath)
        refreshMcpToolCatalog(for: selectedProject)
    }

    // MARK: - Local folder plugins

    /// Registers a folder as a plugin (the `--plugin-dir` analog). Refuses a
    /// directory with neither a manifest nor a contribution directory,
    /// because a registered non-plugin would otherwise appear enabled while
    /// contributing nothing at all.
    public func addLocalPluginFolder(at path: String, scope: PluginInstallScope) {
        let url = URL(fileURLWithPath: path, isDirectory: true)
        let fm = FileManager.default
        guard fm.fileExists(atPath: url.path) else {
            showToast("Folder not found: \(path)", style: .error)
            return
        }
        let hasManifest = fm.fileExists(
            atPath: url.appendingPathComponent(PluginManifestParser.manifestRelativePath).path)
        guard hasManifest || PluginManager.hasContributionSignature(url) else {
            showToast(
                "That folder is not a plugin: it has no \(PluginManifestParser.manifestRelativePath) and no commands/, agents/, skills/, hooks/ or .mcp.json.",
                style: .error)
            return
        }
        switch scope {
        case .user: PluginLedgerStore(root: nil).addLocalPluginPath(url.standardizedFileURL.path)
        case .project(let captured):
            guard var project = projects.first(where: { $0.id == captured.id }) else { return }
            if !project.localPluginPaths.contains(url.standardizedFileURL.path) {
                project.localPluginPaths.append(url.standardizedFileURL.path)
                updateProject(project)
            }
        }
        pluginStateChanged()
        showToast("Registered plugin folder \(url.lastPathComponent).", style: .info)
    }

    public func removeLocalPluginFolder(_ plugin: LoadedPlugin, scope: PluginInstallScope) {
        switch scope {
        case .user:
            PluginLedgerStore(root: nil).removeLocalPluginPath(plugin.directoryURL.standardizedFileURL.path)
        case .project(let captured):
            guard var project = projects.first(where: { $0.id == captured.id }) else { return }
            project.localPluginPaths.removeAll { $0 == plugin.directoryURL.standardizedFileURL.path }
            updateProject(project)
        }
        pluginStateChanged()
        showToast("Removed plugin folder \(plugin.name).", style: .info)
    }

    // MARK: - Marketplace install / uninstall

    /// Installs a marketplace entry and leaves the plugin enabled (Claude
    /// Code's behavior: install turns it on; the hooks inside still pass the
    /// trust gate before anything runs).
    @discardableResult
    public func installPlugin(
        entry: PluginManifestParser.MarketplaceEntry,
        marketplaceName: String,
        checkoutDirectory: URL?,
        marketplaceSource: MarketplaceSource,
        scope: PluginInstallScope
    ) async -> PluginMarketplaceManager.InstallOutcome? {
        do {
            let outcome = try await PluginMarketplaceManager.shared.install(
                entry: entry,
                marketplaceName: marketplaceName,
                checkoutDirectory: checkoutDirectory,
                marketplaceSource: marketplaceSource,
                scope: scope.ledgerValue,
                projectRootURL: scope.projectRootURL)
            // Default ON: an explicit `false` from a previous install would
            // otherwise survive a reinstall and read as a broken install.
            setPluginPreference(id: outcome.pluginID, enabled: true, scope: scope)
            pluginStateChanged()
            showToast(
                "Installed \(outcome.pluginName) \(outcome.version).",
                style: .info)
            return outcome
        } catch {
            showToast("Install failed: \(error.localizedDescription)", style: .error)
            return nil
        }
    }

    public func uninstallPlugin(_ plugin: LoadedPlugin, scope: PluginInstallScope) {
        if plugin.origin == .local { removeLocalPluginFolder(plugin, scope: scope); return }
        _ = uninstallPluginID(plugin.id, scope: scope)
    }

    // MARK: - Project-scope override

    /// Sets a plugin's enable state for the SELECTED project only. The
    /// project map wins over the user one, which is what makes "off in this
    /// repo" expressible at all.
    public func setPluginEnabledForSelectedProject(_ plugin: LoadedPlugin, _ enabled: Bool) {
        guard var project = selectedProject else {
            showToast("Select a project to set a per-project plugin override.", style: .warning)
            return
        }
        project.enabledPlugins[plugin.id] = enabled
        updateProject(project)
        pluginStateChanged()
        showToast("\(plugin.name) \(enabled ? "enabled" : "disabled") for \(project.name).", style: .info)
    }
}

/// Where an install or uninstall applies. `ledgerValue` is the string the
/// v2 ledger records; Claude Code's scopes this app does not have (managed,
/// flag) do not exist here.
public enum PluginInstallScope: Sendable, Equatable {
    case user
    case project(AppProject)

    public var ledgerValue: String {
        if case .project = self { return "project" }
        return "user"
    }

    public var projectRootURL: URL? {
        if case .project(let project) = self {
            return project.rootDirectoryURL
        }
        return nil
    }

    public var label: String {
        if case .project(let project) = self {
            return "Project: \(project.name)"
        }
        return "User (all chats)"
    }
}
