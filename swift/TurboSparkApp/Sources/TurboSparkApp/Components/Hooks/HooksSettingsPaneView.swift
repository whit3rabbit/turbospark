import SwiftUI

/// Main Codex-style Hooks and Lifecycle Management Settings Pane.
public struct HooksSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var hookStore: AppHookStore = .shared

    @State private var searchText = ""
    @State private var selectedGroupID: String? = nil
    @State private var showingAddHookSheet = false
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
        VStack(alignment: .leading, spacing: 14) {
            headerBar
            if !hookStore.discoveryDiagnostics.isEmpty {
                discoveryDiagnosticsBanner
            }
            searchBar
            Divider()

            if filteredGroups.isEmpty {
                emptyState
            } else {
                ScrollView {
                    VStack(alignment: .leading, spacing: 18) {
                        // Config groups section
                        let configGroups = filteredGroups.filter { $0.sourceType == .userConfig || $0.sourceType == .projectConfig || $0.sourceType == .localConfig }
                        if !configGroups.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("From Config", bundle: .module)
                    .settingsControl("From Config", pane: .hooks, timing: .nextTurn)
                                    .themedFont(.small, weight: .semibold)
                                    .foregroundStyle(.appSecondary)

                                ForEach(configGroups) { group in
                                    sourceGroupCard(group: group)
                                }
                            }
                        }

                        // Plugins section
                        let pluginGroups = filteredGroups.filter { $0.sourceType == .plugin }
                        if !pluginGroups.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Plugins", bundle: .module)
                    .settingsControl("Plugins", pane: .hooks, timing: .nextTurn)
                                    .themedFont(.small, weight: .semibold)
                                    .foregroundStyle(.appSecondary)

                                ForEach(pluginGroups) { group in
                                    sourceGroupCard(group: group)
                                }
                            }
                        }

                        // Custom hooks section
                        let customGroups = filteredGroups.filter { $0.sourceType == .custom }
                        if !customGroups.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Custom & App Hooks", bundle: .module)
                    .settingsControl("Custom & App Hooks", pane: .hooks, timing: .nextTurn)
                                    .themedFont(.small, weight: .semibold)
                                    .foregroundStyle(.appSecondary)

                                ForEach(customGroups) { group in
                                    sourceGroupCard(group: group)
                                }
                            }
                        }
                    }
                    .padding(.vertical, 4)
                }
            }
        }
        .padding(16)
        .onAppear {
            hookStore.refresh(projectDirectory: model.selectedProject?.rootDirectoryPath)
        }
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
    }

    // MARK: - Header Bar

    private var headerBar: some View {
        HStack(alignment: .center) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Hooks", bundle: .module)
                    .settingsControl("Hooks", pane: .hooks, timing: .nextTurn)
                    .themedFont(.title2, weight: .bold)
                HStack(spacing: 4) {
                    Text("Manage lifecycle hooks from config and enabled plugins.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    Button("Learn more") {
                        if let url = URL(string: "https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview#hooks") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                    .themedFont(.small)
                    .buttonStyle(.link)
                    .help("Open Hooks Documentation: https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview#hooks")
                    .accessibilityLabel("Learn more about hooks")
                    .accessibilityHint("Opens Claude Code hooks documentation in web browser")
                    .accessibilityAddTraits(.isLink)
                }
            }

            Spacer()

            if totalUnreviewedCount > 0 {
                HStack(spacing: 4) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .themedFont(.tiny)
                        .foregroundStyle(.orange)
                    Text("\(totalUnreviewedCount) needs review", bundle: .module)
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
            }

            Button {
                hookStore.refresh(projectDirectory: model.selectedProject?.rootDirectoryPath)
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .controlSize(.regular)
            .help("Refresh and re-scan hooks from disk")

            Button {
                showingAddHookSheet = true
            } label: {
                Label("Add Hook", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.regular)
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
                    Text("\(hookStore.discoveryDiagnostics.count) discovery issue\(hookStore.discoveryDiagnostics.count == 1 ? "" : "s") found while scanning hook config files", bundle: .module)
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

    // MARK: - Search Bar

    private var searchBar: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.appSecondary)
                .themedFont(.small)
            TextField("Search hooks, commands, plugins...", text: $searchText)
                .textFieldStyle(.plain)
                .themedFont(.small)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(.appBorder.opacity(0.3), lineWidth: 1)
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

                    Text("\(group.hooks.count) hook\(group.hooks.count == 1 ? "" : "s")", bundle: .module)
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
                        Text("\(group.unreviewedCount) needs review", bundle: .module)
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

    private var emptyState: some View {
        VStack(spacing: 12) {
            Image(systemName: "link.badge.plus")
                .themedFont(.display)
                .foregroundStyle(.appSecondary)
            // Two states, two sentences: a fresh install with no hooks used
            // to read `No hooks found matching ''`.
            if searchText.isEmpty {
                Text("No hooks configured", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                Button {
                    showingAddHookSheet = true
                } label: {
                    Label("Add Hook", systemImage: "plus")
                }
                .buttonStyle(.bordered)
            } else {
                Text("No hooks found matching '\(searchText)'", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}
