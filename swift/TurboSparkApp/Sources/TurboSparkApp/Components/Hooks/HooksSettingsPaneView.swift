import SwiftUI

/// Main Codex-style Hooks and Lifecycle Management Settings Pane.
///
/// Laid out like the other extension panes (`ExtensionPaneHeader`): purpose
/// line, `[search] ... [refresh][Import][Add Hook]`, then the configuration
/// sources. The shell already titles the pane, so the inner "Hooks" heading
/// that repeated it is gone; its settings-search anchor moved to the purpose
/// line.
@MainActor
public struct HooksSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var hookStore: AppHookStore = .shared

    @State private var searchText = ""
    @State private var selectedGroupID: String? = nil
    @State private var showingAddHookSheet = false
    @State private var showingPluginMarketplace = false
    @State private var activeDetailGroup: AppHookSourceGroup? = nil
    @State private var showDiagnosticsDetail = false

    public init(model: AppModel) {
        self.model = model
    }

    private var totalUnreviewedCount: Int {
        hookStore.sourceGroups.reduce(0) { $0 + $1.unreviewedCount }
    }

    private var filteredGroups: [AppHookSourceGroup] {
        if searchText.trimmingCharacters(in: .whitespaces).isEmpty {
            return hookStore.sourceGroups
        }
        let q = searchText.lowercased()
        return hookStore.sourceGroups.filter { group in
            group.title.lowercased().contains(q) ||
            group.subtitle.lowercased().contains(q) ||
            group.hooks.contains(where: { $0.name.lowercased().contains(q) || $0.command.lowercased().contains(q) })
        }
    }

    public var body: some View {
        VStack(spacing: 0) {
            header

            VStack(alignment: .leading, spacing: 14) {
                if totalUnreviewedCount > 0 {
                    reviewStatus
                }
                if !hookStore.discoveryDiagnostics.isEmpty {
                    discoveryDiagnosticsBanner
                }

                if filteredGroups.isEmpty {
                    emptyState
                } else {
                    groupList
                }
            }
            .padding(ExtensionPaneLayout.contentInsets)
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
        }
        .onAppear(perform: refresh)
        .sheet(item: $activeDetailGroup) { group in
            HookSourceDetailSheet(groupID: group.id)
        }
        .sheet(isPresented: $showingAddHookSheet) {
            HookEditorSheet(
                onSave: { newHook in
                    hookStore.addCustomHook(newHook)
                    showingAddHookSheet = false
                },
                onDismiss: {
                    showingAddHookSheet = false
                }
            )
        }
        .sheet(isPresented: $showingPluginMarketplace, onDismiss: refresh) {
            PluginMarketplaceSheet(model: model)
        }
    }

    private func refresh() {
        hookStore.refresh(projectDirectory: model.selectedProject?.rootDirectoryPath)
    }

    // MARK: - Header

    /// Hooks arrive from config files and from plugins; the only thing to
    /// import is a plugin, so the Import menu leads to the plugin
    /// marketplace rather than a hooks-specific source that does not exist.
    private var header: some View {
        ExtensionPaneHeader {
            ExtensionPaneIntro(
                text: Text("Manage lifecycle hooks from config and enabled plugins.", bundle: .module),
                documentation: URL(string: "https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview#hooks"),
                documentationHint: Text("Opens the Claude Code hooks documentation in your browser.", bundle: .module))
                .settingsControl("Hooks", pane: .hooks, timing: .nextTurn)
        } filters: {
            AppSearchField(text: $searchText, prompt: Text("Search hooks, commands, plugins...", bundle: .module))
        } actions: {
            ExtensionRefreshButton(label: Text("Refresh", bundle: .module), action: refresh)
            ExtensionImportMenu { pluginMarketplaceItem }
            addHookButton
        } compactActions: {
            ExtensionMoreMenu {
                Button(action: refresh) {
                    Label { Text("Refresh", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                }
                Divider()
                pluginMarketplaceItem
            }
            addHookButton
        }
    }

    private var pluginMarketplaceItem: some View {
        Button {
            showingPluginMarketplace = true
        } label: {
            Label { Text("Browse Plugin Marketplace", bundle: .module) } icon: { Image(systemName: "puzzlepiece.extension") }
        }
    }

    private var addHookButton: some View {
        Button {
            showingAddHookSheet = true
        } label: {
            Label { Text("Add Hook", bundle: .module) } icon: { Image(systemName: "plus") }
        }
        .appAction(.primary)
    }

    /// Trust state, not an action: it used to sit among the header buttons.
    private var reviewStatus: some View {
        HStack(spacing: 4) {
            Image(systemName: "exclamationmark.triangle.fill")
                .themedFont(.tiny)
                .foregroundStyle(.orange)
                .accessibilityHidden(true)
            Text("Needs review: \(totalUnreviewedCount)", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.orange)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 5)
        .background(Color.orange.opacity(0.12))
        .clipShape(Capsule())
        .overlay(
            Capsule()
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
        .accessibilityElement(children: .combine)
    }

    // MARK: - Groups

    private var groupList: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                let configGroups = filteredGroups.filter { $0.sourceType == .userConfig || $0.sourceType == .projectConfig || $0.sourceType == .localConfig }
                if !configGroups.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("From Config", bundle: .module)
                            .settingsControl("From Config", pane: .hooks, timing: .nextTurn)
                            .themedFont(.small, weight: .semibold)
                            .foregroundStyle(.appSecondary)
                            .accessibilityAddTraits(.isHeader)

                        ForEach(configGroups) { group in
                            sourceGroupCard(group: group)
                        }
                    }
                }

                let pluginGroups = filteredGroups.filter { $0.sourceType == .plugin }
                if !pluginGroups.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Plugins", bundle: .module)
                            .settingsControl("Plugins", pane: .hooks, timing: .nextTurn)
                            .themedFont(.small, weight: .semibold)
                            .foregroundStyle(.appSecondary)
                            .accessibilityAddTraits(.isHeader)

                        ForEach(pluginGroups) { group in
                            sourceGroupCard(group: group)
                        }
                    }
                }

                let customGroups = filteredGroups.filter { $0.sourceType == .custom }
                if !customGroups.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Custom & App Hooks", bundle: .module)
                            .settingsControl("Custom & App Hooks", pane: .hooks, timing: .nextTurn)
                            .themedFont(.small, weight: .semibold)
                            .foregroundStyle(.appSecondary)
                            .accessibilityAddTraits(.isHeader)

                        ForEach(customGroups) { group in
                            sourceGroupCard(group: group)
                        }
                    }
                }
            }
            .padding(.vertical, 4)
        }
    }

    // MARK: - Discovery Diagnostics Banner

    private var discoveryDiagnosticsBanner: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                showDiagnosticsDetail.toggle()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .themedFont(.small)
                        .foregroundStyle(.orange)
                    Text("Issues found while scanning hook config files: \(hookStore.discoveryDiagnostics.count)", bundle: .module)
                        .themedFont(.small, weight: .medium)
                        .foregroundStyle(.orange)
                    Spacer()
                    Image(systemName: showDiagnosticsDetail ? "chevron.up" : "chevron.down")
                        .themedFont(.tiny, weight: .semibold)
                        .foregroundStyle(.orange)
                }
            }
            .buttonStyle(.plain)
            .appPointerCursor()

            if showDiagnosticsDetail {
                VStack(alignment: .leading, spacing: 4) {
                    ForEach(Array(hookStore.discoveryDiagnostics.enumerated()), id: \.offset) { _, message in
                        Text(message)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                            .textSelection(.enabled)
                    }
                }
            }
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 10)
        .background(Color.orange.opacity(0.12))
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(Color.orange.opacity(0.3), lineWidth: 1)
        )
    }

    // MARK: - Source Group Card

    private func sourceGroupCard(group: AppHookSourceGroup) -> some View {
        Button {
            activeDetailGroup = group
        } label: {
            HStack(spacing: 12) {
                // Icon
                Image(systemName: groupIcon(for: group.sourceType))
                    .themedFont(.base)
                    .foregroundStyle(.appSecondary)
                    .frame(width: 28, height: 28)
                    .background(.appPage)
                    .clipShape(RoundedRectangle(cornerRadius: 6))

                // Title and hook count
                VStack(alignment: .leading, spacing: 2) {
                    Text(group.title)
                        .themedFont(.small, weight: .semibold)
                        .foregroundStyle(.appText)

                    Text("Hooks: \(group.hooks.count)", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }

                Spacer()

                // Unreviewed badge if applicable
                if group.unreviewedCount > 0 {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.circle.fill")
                            .themedFont(.tiny)
                            .foregroundStyle(.orange)
                        Text("Needs review: \(group.unreviewedCount)", bundle: .module)
                            .themedFont(.tiny, weight: .medium)
                            .foregroundStyle(.orange)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.12))
                    .clipShape(Capsule())
                }

                Image(systemName: "chevron.right")
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appSecondary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(.appSurface.opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(.appBorder.opacity(0.35), lineWidth: 1)
            )
        }
        .buttonStyle(.plain)
        .appPointerCursor()
    }

    private func groupIcon(for type: AppHookSourceType) -> String {
        switch type {
        case .userConfig: return "gearshape"
        case .projectConfig: return "folder"
        case .localConfig: return "slider.horizontal.2.square"
        case .plugin: return "shippingbox"
        case .custom: return "link.badge.plus"
        }
    }

    /// Two states, two messages: a fresh install with no hooks used to read
    /// `No hooks found matching ''`. Add Hook is the header's primary, so
    /// the empty state offers the other source, plugins.
    @ViewBuilder
    private var emptyState: some View {
        if searchText.isEmpty {
            AppEmptyState(
                systemImage: "link.badge.plus",
                title: Text("No hooks configured", bundle: .module),
                message: Text("Hooks run a command at a point in the assistant's work, such as before a tool runs. Add one here, or install a plugin that brings its own.", bundle: .module)
            ) {
                Button {
                    showingPluginMarketplace = true
                } label: {
                    Text("Browse Plugin Marketplace", bundle: .module)
                }
                .appAction(.secondary)
            }
        } else {
            ExtensionNoMatchesState(query: searchText)
        }
    }
}
