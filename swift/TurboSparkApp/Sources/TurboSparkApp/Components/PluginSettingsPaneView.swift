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
    @ObservedObject private var hookStore = AppHookStore.shared
    @Environment(\.appTheme) private var theme

    @State private var selectedPluginID: String? = nil
    @State private var isShowingMarketplace = false
    @State private var pluginToRemove: LoadedPlugin? = nil
    @State private var isConfirmingRemove = false

    public init(model: AppModel) {
        self.model = model
    }

    private var selectedPlugin: LoadedPlugin? {
        if let selectedPluginID {
            return model.installedPlugins.first { $0.id == selectedPluginID }
        }
        return model.installedPlugins.first
    }

    public var body: some View {
        VStack(spacing: 0) {
            headerControlBar
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(Color(nsColor: .windowBackgroundColor))

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            if model.installedPlugins.isEmpty {
                emptyState
            } else {
                HStack(spacing: 0) {
                    pluginList
                        .frame(width: 300)
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

                    Rectangle()
                        .fill(TurboSparkTheme.hairlineColor)
                        .frame(width: 1)

                    if let plugin = selectedPlugin {
                        pluginDetail(plugin)
                    } else {
                        Text("Select a plugin to inspect its contributions.", bundle: .module)
                            .themedFont(.base)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
            }
        }
        .sheet(isPresented: $isShowingMarketplace) {
            PluginMarketplaceSheet(model: model)
        }
        .confirmationDialog(
            "Remove Plugin",
            isPresented: $isConfirmingRemove,
            presenting: pluginToRemove
        ) { plugin in
            Button("Remove '\(plugin.name)'", role: .destructive) {
                model.uninstallPlugin(plugin, scope: .user)
            }
            Button("Cancel", role: .cancel) {}
        } message: { plugin in
            Text("Uninstalls the plugin from user scope and removes its install cache. Hooks it contributed stop immediately.", bundle: .module)
        }
    }

    // MARK: - Header

    private var headerControlBar: some View {
        HStack(spacing: 10) {
            Button {
                isShowingMarketplace = true
            } label: {
                Label("Browse Marketplace", systemImage: "cart")
            }
            Button {
                addLocalFolder()
            } label: {
                Label("Add Local Folder", systemImage: "folder.badge.plus")
            }
            Spacer()
            Button {
                model.reloadPlugins()
                AppHookStore.shared.refresh(
                    projectDirectory: model.selectedProject?.rootDirectoryPath)
            } label: {
                Label("Refresh", systemImage: "arrow.clockwise")
            }
            .help("Re-scan both plugin roots")
        }
        .font(theme.ui(.base))
    }

    private func addLocalFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.message = "Choose a folder containing .claude-plugin/plugin.json or plugin directories"
        if panel.runModal() == .OK, let url = panel.url {
            model.addLocalPluginFolder(at: url.path)
        }
    }

    // MARK: - List

    private var pluginList: some View {
        VStack(spacing: 0) {
            if !model.pluginLoadDiagnostics.isEmpty {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(model.pluginLoadDiagnostics, id: \.self) { line in
                        Label(line, systemImage: "exclamationmark.triangle")
                            .font(theme.ui(.small))
                            .foregroundStyle(.orange)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                Rectangle().fill(TurboSparkTheme.hairlineColor).frame(height: 1)
            }

            List(model.installedPlugins, selection: $selectedPluginID) { plugin in
                PluginListRow(
                    plugin: plugin,
                    isEnabled: model.isPluginEnabled(plugin),
                    toggle: { model.setPluginEnabled(plugin, $0) })
                    .tag(plugin.id)
            }
            .listStyle(.sidebar)
        }
    }

    // MARK: - Detail

    private func pluginDetail(_ plugin: LoadedPlugin) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .firstTextBaseline) {
                    Text(plugin.name)
                        .font(theme.ui(.title3, weight: .semibold))
                    Text(plugin.version)
                        .font(theme.ui(.small))
                        .foregroundStyle(.secondary)
                    Spacer()
                    originBadge(plugin)
                }

                if !plugin.pluginDescription.isEmpty {
                    Text(plugin.pluginDescription)
                        .font(theme.ui(.base))
                        .foregroundStyle(.secondary)
                }

                LabeledRow(label: "ID", value: plugin.id)
                LabeledRow(label: "Location", value: plugin.directoryURL.path)

                contributionSummary(plugin)

                if plugin.origin == .claudeInterop {
                    Label(
                        "Installed by Claude Code and managed read-only here. Removing it only clears TurboSpark's enable record.",
                        systemImage: "info.circle")
                        .font(theme.ui(.small))
                        .foregroundStyle(.secondary)
                }

                if !plugin.manifest.unsupportedNotes.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Not applied by this client", bundle: .module)
                            .font(theme.ui(.base, weight: .semibold))
                        ForEach(plugin.manifest.unsupportedNotes, id: \.self) { note in
                            Label(note, systemImage: "minus.circle")
                                .font(theme.ui(.small))
                                .foregroundStyle(.secondary)
                        }
                    }
                }

                optionsEditor(plugin)

                if model.selectedProject != nil {
                    projectOverrideSection(plugin)
                }

                HStack {
                    Spacer()
                    if plugin.origin == .marketplace || plugin.origin == .local {
                        Button(role: .destructive) {
                            pluginToRemove = plugin
                            isConfirmingRemove = true
                        } label: {
                            Label("Uninstall", systemImage: "trash")
                        }
                    }
                }
            }
            .padding(20)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func originBadge(_ plugin: LoadedPlugin) -> some View {
        let (text, color): (String, Color) = {
            switch plugin.origin {
            case .turboSpark: return ("TurboSpark", .blue)
            case .marketplace: return ("Marketplace: \(plugin.marketplaceName ?? "?")", .green)
            case .local: return ("Local folder", .orange)
            case .claudeInterop: return ("Claude Code", .purple)
            }
        }()
        return Text(text)
            .themedFont(points: 10, weight: .medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }

    private func contributionSummary(_ plugin: LoadedPlugin) -> some View {
        let counts = PluginContributionCounts(plugin: plugin)
        return VStack(alignment: .leading, spacing: 6) {
            Text("Contributions", bundle: .module)
                .font(theme.ui(.base, weight: .semibold))
            HStack(spacing: 14) {
                ContributionCount(label: "Commands", count: counts.commands)
                ContributionCount(label: "Skills", count: counts.skills)
                ContributionCount(label: "Agents", count: counts.agents)
                ContributionCount(label: "Hooks", count: counts.hooks)
                ContributionCount(label: "MCP servers", count: counts.mcpServers)
            }
            Text("Counts reflect what this client loads from the plugin as installed. Hooks always pass the SHA-256 trust review before they run.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.secondary)
        }
    }

    private func optionsEditor(_ plugin: LoadedPlugin) -> some View {
        let specs = plugin.manifest.userConfig.map(\.optionSpec)
        let sourceID = "plugin_\(plugin.name)"
        return Group {
            if !specs.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Options", bundle: .module)
                        .font(theme.ui(.base, weight: .semibold))
                    ForEach(specs) { spec in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(spec.title).font(theme.ui(.base))
                                if !spec.description.isEmpty {
                                    Text(spec.description)
                                        .font(theme.ui(.small))
                                        .foregroundStyle(.secondary)
                                }
                            }
                            Spacer()
                            optionField(spec: spec, sourceID: sourceID)
                        }
                    }
                    Text("Values reach the plugin's hooks and MCP servers as CLAUDE_PLUGIN_OPTION_<KEY> environment variables and ${user_config.KEY} substitutions. Sensitive values are masked and never substituted into visible content.", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.secondary)
                }
            }
        }
    }

    @ViewBuilder
    private func optionField(spec: AppHookOptionSpec, sourceID: String) -> some View {
        let binding = Binding(
            get: { hookStore.getOptionValue(sourceID: sourceID, key: spec.key, defaultVal: spec.defaultValue ?? "") },
            set: { hookStore.updateOptionValue(sourceID: sourceID, key: spec.key, value: $0) })
        switch spec.type {
        case .boolean:
            Toggle("", isOn: Binding(
                get: { (hookStore.getOptionValue(sourceID: sourceID, key: spec.key, defaultVal: spec.defaultValue ?? "false")) == "true" },
                set: { hookStore.updateOptionValue(sourceID: sourceID, key: spec.key, value: $0 ? "true" : "false") }))
                .labelsHidden()
        default:
            Group {
                // The row already draws `spec.title`; on macOS the field's own
                // title is a visible label too, so it rendered twice.
                if spec.isSensitive {
                    SecureField(spec.title, text: binding)
                        .labelsHidden()
                } else {
                    TextField(spec.title, text: binding)
                        .labelsHidden()
                        .textFieldStyle(.roundedBorder)
                        .frame(maxWidth: 240)
                }
            }
            .frame(maxWidth: 240)
        }
    }

    private func projectOverrideSection(_ plugin: LoadedPlugin) -> some View {
        let project = model.selectedProject!
        let resolved = project.enabledPlugins[plugin.id].map { $0 ? "on" : "off" }
        return VStack(alignment: .leading, spacing: 6) {
            Text("Project override", bundle: .module)
                .font(theme.ui(.base, weight: .semibold))
            HStack {
                Text("For \(project.name):", bundle: .module)
                    .font(theme.ui(.base))
                Spacer()
                Picker("", selection: Binding(
                    get: { resolved ?? "inherit" },
                    set: { newValue in
                        switch newValue {
                        case "on": model.setPluginEnabledForSelectedProject(plugin, true)
                        case "off": model.setPluginEnabledForSelectedProject(plugin, false)
                        default:
                            var updated = project
                            updated.enabledPlugins.removeValue(forKey: plugin.id)
                            model.updateProject(updated)
                            model.reloadPlugins()
                        }
                    })) {
                    Text("Inherit user setting", bundle: .module).tag("inherit")
                    Text("Enabled", bundle: .module).tag("on")
                    Text("Disabled", bundle: .module).tag("off")
                }
                .labelsHidden()
                .frame(width: 200)
            }
        }
    }

    // MARK: - Empty state

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "puzzlepiece.extension")
                .themedFont(points: 36)
                .foregroundStyle(.tertiary)
            Text("No plugins installed.", bundle: .module)
                .themedFont(.base)
                .foregroundStyle(.secondary)
            Text("Browse a marketplace, or add a local folder containing .claude-plugin/plugin.json.", bundle: .module)
                .themedFont(.base)
                .foregroundStyle(.tertiary)
                .multilineTextAlignment(.center)
            HStack {
                Button("Browse Marketplace") { isShowingMarketplace = true }
                Button("Add Local Folder") { addLocalFolder() }
            }
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
                Text("\(plugin.originKey) - \(plugin.version)", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Toggle("", isOn: Binding(get: { isEnabled }, set: toggle))
                .labelsHidden()
                .toggleStyle(.switch)
        }
        .padding(.vertical, 2)
    }
}

private struct ContributionCount: View {
    let label: String
    let count: Int

    var body: some View {
        VStack(spacing: 2) {
            Text("\(count)", bundle: .module)
                .themedFont(.title3, weight: .semibold)
            Text(label)
                .themedFont(.tiny)
                .foregroundStyle(.secondary)
        }
        .accessibilityElement(children: .combine)
        .accessibilityLabel("\(count) \(label)")
    }
}

private struct LabeledRow: View {
    let label: String
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            Text(label)
                .themedFont(.base, weight: .medium)
                .frame(width: 80, alignment: .leading)
            Text(value)
                .themedFont(.small)
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
            Spacer()
        }
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
