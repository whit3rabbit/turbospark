import SwiftUI

/// Main Codex-style Hooks and Lifecycle Management Settings Pane.
public struct HooksSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var hookStore: AppHookStore = .shared

    @State private var searchText = ""
    @State private var selectedGroupID: String? = nil
    @State private var showingAddHookSheet = false
    @State private var activeDetailGroup: AppHookSourceGroup? = nil

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
                                Text("From Config")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)

                                ForEach(configGroups) { group in
                                    sourceGroupCard(group: group)
                                }
                            }
                        }

                        // Plugins section
                        let pluginGroups = filteredGroups.filter { $0.sourceType == .plugin }
                        if !pluginGroups.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Plugins")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)

                                ForEach(pluginGroups) { group in
                                    sourceGroupCard(group: group)
                                }
                            }
                        }

                        // Custom hooks section
                        let customGroups = filteredGroups.filter { $0.sourceType == .custom }
                        if !customGroups.isEmpty {
                            VStack(alignment: .leading, spacing: 8) {
                                Text("Custom & App Hooks")
                                    .font(.caption.weight(.semibold))
                                    .foregroundStyle(.secondary)

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
                Text("Hooks")
                    .font(.title2.weight(.bold))
                HStack(spacing: 4) {
                    Text("Manage lifecycle hooks from config and enabled plugins.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Button("Learn more") {
                        if let url = URL(string: "https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/overview#hooks") {
                            NSWorkspace.shared.open(url)
                        }
                    }
                    .font(.caption)
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
                        .font(.caption2)
                        .foregroundStyle(.orange)
                    Text("\(totalUnreviewedCount) needs review")
                        .font(.caption.weight(.semibold))
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

    // MARK: - Search Bar

    private var searchBar: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .foregroundStyle(.secondary)
                .font(.caption)
            TextField("Search hooks, commands, plugins...", text: $searchText)
                .textFieldStyle(.plain)
                .font(.caption)
            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)
            }
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 6)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 6))
        .overlay(
            RoundedRectangle(cornerRadius: 6)
                .stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1)
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
                    .font(.system(size: 16))
                    .foregroundStyle(.secondary)
                    .frame(width: 28, height: 28)
                    .background(Color(nsColor: .windowBackgroundColor))
                    .clipShape(RoundedRectangle(cornerRadius: 6))

                // Title and hook count
                VStack(alignment: .leading, spacing: 2) {
                    Text(group.title)
                        .font(.subheadline.weight(.semibold))
                        .foregroundStyle(.primary)

                    Text("\(group.hooks.count) hook\(group.hooks.count == 1 ? "" : "s")")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                // Unreviewed badge if applicable
                if group.unreviewedCount > 0 {
                    HStack(spacing: 4) {
                        Image(systemName: "exclamationmark.circle.fill")
                            .font(.caption2)
                            .foregroundStyle(.orange)
                        Text("\(group.unreviewedCount) needs review")
                            .font(.caption2.weight(.medium))
                            .foregroundStyle(.orange)
                    }
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.12))
                    .clipShape(Capsule())
                }

                Image(systemName: "chevron.right")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 14)
            .padding(.vertical, 12)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
            .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
            .overlay(
                RoundedRectangle(cornerRadius: 8, style: .continuous)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.35), lineWidth: 1)
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
                .font(.system(size: 40))
                .foregroundStyle(.secondary)
            Text("No hooks found matching '\(searchText)'")
                .font(.subheadline)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}
