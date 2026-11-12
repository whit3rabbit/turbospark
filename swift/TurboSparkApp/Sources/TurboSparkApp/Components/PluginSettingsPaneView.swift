import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Settings pane for installed plugins: the same surfaces Claude Code's
/// `/plugins` panel manages, adapted to this app's settings style. Docs:
/// swift/docs/SWIFT_PLUGINS.md.
@MainActor
public struct PluginSettingsPaneView: View {
    @ObservedObject var model: AppModel

    @State private var projectID: UUID?
    @State private var searchQuery = ""
    @State private var selectedPluginID: String? = nil
    @State private var isShowingMarketplace = false
    @State private var pluginToRemove: LoadedPlugin? = nil
    @State private var isConfirmingRemove = false

    public init(model: AppModel, projectID: UUID? = nil) {
        self.model = model
        self._projectID = State(initialValue: projectID)
    }

    private var scopedProject: AppProject? { model.projects.first { $0.id == projectID } }
    private var plugins: [LoadedPlugin] {
        // Reading the published revision makes a management edit refresh this resolution.
        _ = model.installedPlugins
        return PluginManager.shared.resolve(projectURL: scopedProject?.rootDirectoryURL).plugins
    }

    private var filteredPlugins: [LoadedPlugin] {
        let query = searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !query.isEmpty else { return plugins }
        return plugins.filter {
            $0.name.lowercased().contains(query) ||
            $0.id.lowercased().contains(query) ||
            $0.pluginDescription.lowercased().contains(query)
        }
    }

    private var selectedPlugin: LoadedPlugin? {
        if let selectedPluginID {
            return plugins.first { $0.id == selectedPluginID }
        }
        return filteredPlugins.first
    }

    public var body: some View {
        VStack(spacing: 0) {
            header

            if plugins.isEmpty {
                emptyState
            } else if filteredPlugins.isEmpty {
                ExtensionNoMatchesState(query: searchQuery)
            } else {
                ExtensionSplitView {
                    pluginList
                } detail: {
                    if let plugin = selectedPlugin {
                        PluginDetailView(
                            model: model,
                            plugin: plugin,
                            scopedProject: scopedProject,
                            onUninstall: {
                                pluginToRemove = plugin
                                isConfirmingRemove = true
                            })
                    } else {
                        Text("Select a plugin to inspect its contributions.", bundle: .module)
                            .themedFont(.base)
                            .foregroundStyle(.appSecondary)
                    }
                }
            }
        }
        .sheet(isPresented: $isShowingMarketplace) {
            PluginMarketplaceSheet(model: model, projectID: projectID)
        }
        .confirmationDialog(
            Text("Remove Plugin", bundle: .module),
            isPresented: $isConfirmingRemove,
            presenting: pluginToRemove
        ) { plugin in
            ForEach(model.pluginInstallationScopes(plugin.id), id: \.label) { scope in
                Button(role: .destructive) {
                    model.uninstallPlugin(plugin, scope: scope)
                } label: {
                    Text("Remove from \(Self.scopeName(scope))", bundle: .module)
                }
            }
            if plugin.origin == .local {
                Button(role: .destructive) { model.removeLocalPluginFolder(plugin, scope: scopedProject.map(PluginInstallScope.project) ?? .user) } label: { Text("Unregister folder", bundle: .module) }
            }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: { plugin in
            Text("Removes only the selected installation. Other projects keep their installation. Shared files remain until the last installation is removed.", bundle: .module)
        }
    }

    /// `PluginInstallScope.label` is fixed English built in the model layer
    /// (and is the dialog's row identity), so the shown name is composed here
    /// from catalog keys.
    static func scopeName(_ scope: PluginInstallScope) -> String {
        switch scope {
        case .user:
            return AppStrings.localized("User (all chats)")
        case .project(let project):
            return AppStrings.localized("Project: \(project.name)")
        }
    }

    // MARK: - Header

    /// `[scope][search] ... [refresh][Import][Browse Marketplace]`, the same
    /// row as Skills and MCP Servers. The marketplace is this pane's one
    /// primary action (plugins are installed, not written here); a local
    /// folder is the secondary source, in the Import menu.
    private var header: some View {
        ExtensionPaneHeader {
            ExtensionPaneIntro(text: Text("Bundles of commands, skills, agents, hooks and MCP servers, installed from a marketplace or a local folder.", bundle: .module))
        } filters: {
            ExtensionScopePicker(model: model, projectID: $projectID)
            AppSearchField(text: $searchQuery, prompt: Text("Search plugins...", bundle: .module))
        } actions: {
            ExtensionRefreshButton(label: Text("Refresh", bundle: .module), action: refresh)
            ExtensionImportMenu { addLocalFolderItem }
            browseMarketplaceButton
        } compactActions: {
            ExtensionMoreMenu {
                Button(action: refresh) {
                    Label { Text("Refresh", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                }
                Divider()
                addLocalFolderItem
            }
            browseMarketplaceButton
        }
    }

    private var addLocalFolderItem: some View {
        Button {
            addLocalFolder()
        } label: {
            Label { Text("Add Local Folder...", bundle: .module) } icon: { Image(systemName: "folder.badge.plus") }
        }
    }

    private var browseMarketplaceButton: some View {
        Button {
            isShowingMarketplace = true
        } label: {
            Label { Text("Browse Marketplace", bundle: .module) } icon: { Image(systemName: "cart") }
        }
        .appAction(.primary)
    }

    /// Re-scans both plugin roots and the hook sources plugins contribute.
    private func refresh() {
        model.reloadPlugins()
        AppHookStore.shared.refresh(
            projectDirectory: model.selectedProject?.rootDirectoryPath)
    }

    private func addLocalFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = AppStrings.localized("Choose a folder containing .claude-plugin/plugin.json or plugin directories")
        if panel.runModal() == .OK, let url = panel.url {
            model.addLocalPluginFolder(at: url.path, scope: scopedProject.map(PluginInstallScope.project) ?? .user)
        }
    }

    // MARK: - List

    private var pluginList: some View {
        VStack(spacing: 0) {
            if !model.pluginLoadDiagnostics.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(model.pluginLoadDiagnostics, id: \.self) { line in
                        Label(line, systemImage: "exclamationmark.triangle")
                            .themedFont(.small)
                            .foregroundStyle(AppAttention.color)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                Rectangle().fill(.appBorder).frame(height: 1)
            }

            // One read of the enable stores per render, not per row.
            let enabledIDs = PluginManager.shared.enabledPluginIDs(projectURL: scopedProject?.rootDirectoryURL)
            List(filteredPlugins, selection: $selectedPluginID) { plugin in
                PluginListRow(
                    plugin: plugin,
                    isEnabled: enabledIDs.contains(plugin.id),
                    toggle: { model.setPluginPreference(id: plugin.id, enabled: $0, scope: scopedProject.map(PluginInstallScope.project) ?? .user) })
                    .tag(plugin.id)
            }
            .listStyle(.sidebar)
            .scrollContentBackground(.hidden)
        }
    }

    // MARK: - Empty state

    /// Secondary actions only: Browse Marketplace is the header's primary.
    private var emptyState: some View {
        AppEmptyState(
            systemImage: "puzzlepiece.extension",
            title: Text("No plugins installed.", bundle: .module),
            message: Text("Browse a marketplace, or add a local folder containing .claude-plugin/plugin.json.", bundle: .module)
        ) {
            Button { addLocalFolder() } label: { Text("Add Local Folder...", bundle: .module) }
                .appAction(.secondary)
        }
    }
}

/// One row of the installed list.
private struct PluginListRow: View {
    let plugin: LoadedPlugin
    let isEnabled: Bool
    let toggle: (Bool) -> Void

    var body: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(plugin.name)
                    .themedFont(.base, weight: .medium)
                Text(verbatim: "\(plugin.originKey) - \(plugin.version)")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            // Named by the plugin so VoiceOver does not announce an unnamed
            // switch in every row.
            Toggle(isOn: Binding(get: { isEnabled }, set: toggle)) {
                Text(plugin.name)
            }
            .labelsHidden()
            .toggleStyle(.switch)
        }
        .padding(.vertical, 2)
    }
}

/// What a plugin contributes, counted the way the loaders count: manifest
/// declarations falling back to the convention directories.
struct PluginContributionCounts {
    let commands: Int
    let skills: Int
    let agents: Int
    let hooks: Int
    let mcpServers: Int

    init(plugin: LoadedPlugin) {
        let fm = FileManager.default
        let dir = plugin.directoryURL

        func directoryCount(_ name: String, extensions: Set<String>, recursive: Bool) -> Int {
            let url = dir.appendingPathComponent(name)
            guard let enumerator = fm.enumerator(
                at: url, includingPropertiesForKeys: [.isRegularFileKey],
                options: [.skipsHiddenFiles]) else { return 0 }
            return enumerator
                .compactMap { $0 as? URL }
                .filter { extensions.contains($0.pathExtension.lowercased()) }
                .count
        }

        if plugin.manifest.commandSpecs.isEmpty {
            commands = directoryCount("commands", extensions: ["md"], recursive: true)
        } else {
            commands = plugin.manifest.commandSpecs.count
        }

        if plugin.manifest.skillPaths.isEmpty {
            let skillsDir = dir.appendingPathComponent("skills")
            let entries = (try? fm.contentsOfDirectory(atPath: skillsDir.path)) ?? []
            skills = entries.filter { !$0.hasPrefix(".") }.count
        } else {
            skills = plugin.manifest.skillPaths.count
        }

        if plugin.manifest.agentPaths.isEmpty {
            agents = directoryCount("agents", extensions: ["md", "json"], recursive: true)
        } else {
            agents = plugin.manifest.agentPaths.count
        }

        var hookCount = 0
        var hookSources: [URL] = [
            dir.appendingPathComponent("hooks/hooks.json"),
            dir.appendingPathComponent("hooks.json")
        ]
        hookSources.append(contentsOf: plugin.manifest.hookFilePaths.map { dir.appendingPathComponent($0) })
        for url in hookSources where fm.fileExists(atPath: url.path) {
            if let data = try? Data(contentsOf: url),
                let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                let container = (json["hooks"] as? [String: Any]) ?? json
                for (_, matchers) in container {
                    if let matcherList = matchers as? [[String: Any]] {
                        for matcher in matcherList {
                            if let hooks = matcher["hooks"] as? [[String: Any]] {
                                hookCount += hooks.count
                            }
                        }
                    }
                }
            }
        }
        if plugin.manifest.inlineHooksJSON != nil {
            hookCount += 1
        }
        hooks = hookCount

        var serverCount = 0
        if fm.fileExists(atPath: dir.appendingPathComponent(".mcp.json").path) {
            serverCount += 1
        }
        serverCount += plugin.manifest.mcpServerFilePaths.count
        if let inline = plugin.manifest.inlineMcpServersJSON,
            let dict = try? JSONSerialization.jsonObject(with: inline) as? [String: Any] {
            serverCount += dict.count
        }
        mcpServers = serverCount
    }
}
