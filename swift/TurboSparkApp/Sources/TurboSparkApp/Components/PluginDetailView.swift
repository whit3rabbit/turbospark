import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The detail column of the Plugins pane: identity, origin, contributions,
/// options, the project override and Uninstall.
///
/// A dedicated subview so the pane's state stays private (Gotcha 15); the
/// pane owns the removal confirmation and receives Uninstall as a closure.
@MainActor
struct PluginDetailView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var hookStore = AppHookStore.shared
    @Environment(\.appTheme) private var theme
    let plugin: LoadedPlugin
    let scopedProject: AppProject?
    let onUninstall: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                HStack(alignment: .firstTextBaseline) {
                    Text(plugin.name)
                        .font(theme.ui(.title3, weight: .semibold))
                        .accessibilityAddTraits(.isHeader)
                    Text(plugin.version)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                    Spacer()
                    originBadge
                }

                if !plugin.pluginDescription.isEmpty {
                    Text(plugin.pluginDescription)
                        .font(theme.ui(.base))
                        .foregroundStyle(.appSecondary)
                }

                PluginLabeledRow(label: Text("ID", bundle: .module), value: plugin.id)
                HStack(alignment: .firstTextBaseline) {
                    Text("Location", bundle: .module)
                        .themedFont(.base, weight: .medium)
                        .frame(minWidth: 80, alignment: .leading)
                    SettingsPathText(plugin.directoryURL.path, size: .small)
                    Spacer()
                }

                contributionSummary

                if plugin.origin == .claudeInterop {
                    Label {
                        Text("Installed by Claude Code and managed read-only here. Removing it only clears TurboSpark's enable record.", bundle: .module)
                    } icon: {
                        Image(systemName: "info.circle")
                    }
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                }

                if !plugin.manifest.unsupportedNotes.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Not applied by this client", bundle: .module)
                            .font(theme.ui(.base, weight: .semibold))
                            .accessibilityAddTraits(.isHeader)
                        ForEach(plugin.manifest.unsupportedNotes, id: \.self) { note in
                            Label(note, systemImage: "minus.circle")
                                .font(theme.ui(.small))
                                .foregroundStyle(.appSecondary)
                        }
                    }
                }

                optionsEditor

                if let project = scopedProject {
                    projectOverrideSection(project)
                }

                HStack {
                    Spacer()
                    if plugin.origin == .marketplace || plugin.origin == .local {
                        Button(role: .destructive, action: onUninstall) {
                            Label { Text("Uninstall", bundle: .module) } icon: { Image(systemName: "trash") }
                        }
                        .appAction(.destructive, size: .small)
                    }
                }
            }
            .padding(SettingsLayout.detailInset)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    /// Product names stay verbatim; the two descriptive origins are catalog
    /// keys.
    private var originBadge: some View {
        let style: AnyShapeStyle
        let text: Text
        switch plugin.origin {
        case .turboSpark:
            text = Text(verbatim: "TurboSpark")
            style = AnyShapeStyle(.appAccent)
        case .marketplace:
            text = Text("Marketplace: \(plugin.marketplaceName ?? "?")", bundle: .module)
            style = AnyShapeStyle(.appAccent)
        case .local:
            text = Text("Local folder", bundle: .module)
            style = AnyShapeStyle(AppAttention.color)
        case .claudeInterop:
            text = Text(verbatim: "Claude Code")
            style = AnyShapeStyle(.appSecondary)
        }
        return text
            .themedFont(.tiny, weight: .medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(style.opacity(0.15)))
            .foregroundStyle(style)
    }

    private var contributionSummary: some View {
        let counts = PluginContributionCounts(plugin: plugin)
        return VStack(alignment: .leading, spacing: 6) {
            Text("Contributions", bundle: .module)
                .font(theme.ui(.base, weight: .semibold))
                .accessibilityAddTraits(.isHeader)
            HStack(spacing: 14) {
                PluginContributionCount(label: Text("Commands", bundle: .module), count: counts.commands)
                PluginContributionCount(label: Text("Skills", bundle: .module), count: counts.skills)
                PluginContributionCount(label: Text("Agents", bundle: .module), count: counts.agents)
                PluginContributionCount(label: Text("Hooks", bundle: .module), count: counts.hooks)
                PluginContributionCount(label: Text("MCP servers", bundle: .module), count: counts.mcpServers)
            }
            Text("Counts reflect what this client loads from the plugin as installed. Hooks always pass the SHA-256 trust review before they run.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
        }
    }

    @ViewBuilder
    private var optionsEditor: some View {
        let specs = plugin.manifest.userConfig.map(\.optionSpec)
        let sourceID = "plugin_\(plugin.name)"
        if !specs.isEmpty {
            VStack(alignment: .leading, spacing: 8) {
                Text("Options", bundle: .module)
                .settingsControl("Options", pane: .plugins, timing: .nextTurn)
                    .font(theme.ui(.base, weight: .semibold))
                    .accessibilityAddTraits(.isHeader)
                ForEach(specs) { spec in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(spec.title).font(theme.ui(.base))
                            if !spec.description.isEmpty {
                                Text(spec.description)
                                    .font(theme.ui(.small))
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                        Spacer()
                        optionField(spec: spec, sourceID: sourceID)
                    }
                }
                Text("Values reach the plugin's hooks and MCP servers as CLAUDE_PLUGIN_OPTION_<KEY> environment variables and ${user_config.KEY} substitutions. Sensitive values are masked and never substituted into visible content.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
    }

    @ViewBuilder
    private func optionField(spec: AppHookOptionSpec, sourceID: String) -> some View {
        let binding = Binding(
            get: { hookStore.getOptionValue(sourceID: sourceID, key: spec.key, defaultVal: spec.defaultValue ?? "") },
            set: {
                hookStore.updateOptionValue(
                    sourceID: sourceID, key: spec.key, value: $0,
                    isSensitive: spec.isSensitive)
            })
        switch spec.type {
        case .boolean:
            // Named by the option: the row draws the title beside it, and an
            // empty label left VoiceOver announcing an unnamed switch.
            Toggle(isOn: Binding(
                get: { (hookStore.getOptionValue(sourceID: sourceID, key: spec.key, defaultVal: spec.defaultValue ?? "false")) == "true" },
                set: {
                    hookStore.updateOptionValue(
                        sourceID: sourceID, key: spec.key, value: $0 ? "true" : "false",
                        isSensitive: spec.isSensitive)
                })) {
                Text(spec.title)
            }
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

    private func projectOverrideSection(_ project: AppProject) -> some View {
        let resolved = project.enabledPlugins[plugin.id].map { $0 ? "on" : "off" }
        return VStack(alignment: .leading, spacing: 6) {
            Text("Project override", bundle: .module)
                    .settingsControl("Project override", pane: .plugins, timing: .nextTurn)
                .font(theme.ui(.base, weight: .semibold))
                .accessibilityAddTraits(.isHeader)
            HStack {
                Text("For \(project.name):", bundle: .module)
                    .font(theme.ui(.base))
                Spacer()
                Picker(selection: Binding(
                    get: { resolved ?? "inherit" },
                    set: { newValue in
                        switch newValue {
                        case "on": model.setPluginPreference(id: plugin.id, enabled: true, scope: .project(project))
                        case "off": model.setPluginPreference(id: plugin.id, enabled: false, scope: .project(project))
                        default:
                            model.setPluginPreference(id: plugin.id, enabled: nil, scope: .project(project))
                        }
                    })) {
                    Text("Inherit user setting", bundle: .module)
                    .settingsControl("Inherit user setting", pane: .plugins, timing: .nextTurn).tag("inherit")
                    Text("Enabled", bundle: .module).tag("on")
                    Text("Disabled", bundle: .module).tag("off")
                } label: {
                    Text("Project override", bundle: .module)
                }
                .labelsHidden()
                .fixedSize()
            }
        }
    }
}

private struct PluginContributionCount: View {
    let label: Text
    let count: Int

    var body: some View {
        // Combined, the children read as "3 Commands"; no separate label.
        VStack(spacing: 2) {
            Text(verbatim: "\(count)")
                .themedFont(.title3, weight: .semibold)
            label
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
        .accessibilityElement(children: .combine)
    }
}

private struct PluginLabeledRow: View {
    let label: Text
    let value: String

    var body: some View {
        HStack(alignment: .firstTextBaseline) {
            label
                .themedFont(.base, weight: .medium)
                .frame(minWidth: 80, alignment: .leading)
            Text(verbatim: value)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .textSelection(.enabled)
            Spacer()
        }
    }
}
