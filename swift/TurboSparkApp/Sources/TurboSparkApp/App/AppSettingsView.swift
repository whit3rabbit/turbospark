import SwiftUI
import TurboSpark

public struct AppSettingsView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var appearanceManager = AppearanceManager.shared

    @State private var selectedTab: SettingsTab = .appearance
    @State private var searchText: String = ""

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        HStack(spacing: 0) {
            settingsSidebar
                .frame(width: 220)
                .background(Color(nsColor: .windowBackgroundColor).opacity(0.85))

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(width: 1)

            VStack(spacing: 0) {
                // Top header for right pane
                HStack {
                    Text(selectedTab.title)
                        .font(theme.ui(.title3, weight: .semibold))
                    Spacer()
                }
                .padding(.horizontal, 24)
                .padding(.vertical, 16)
                .background(Color(nsColor: .windowBackgroundColor))

                Rectangle()
                    .fill(TurboSparkTheme.hairlineColor)
                    .frame(height: 1)

                tabContent
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .font(theme.ui(.base))
        .frame(minWidth: 780, minHeight: 580)
        .onReceive(NotificationCenter.default.publisher(for: .openSettingsTab)) { notification in
            if let tab = notification.object as? SettingsTab {
                selectedTab = tab
            }
        }
    }

    // MARK: - Sidebar
    private var settingsSidebar: some View {
        VStack(spacing: 12) {
            // Search field
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                    .font(theme.ui(.small))
                    .accessibilityHidden(true)
                TextField("Search settings...", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(theme.ui(.small))
                    .accessibilityLabel("Search settings")
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(Color(nsColor: .separatorColor).opacity(0.3), lineWidth: 1)
            )
            .padding(.horizontal, 12)
            .padding(.top, 14)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    sidebarCategory(
                        title: "Personal",
                        tabs: SettingsTab.allCases.filter { $0.category == "Personal" }
                    )
                    sidebarCategory(
                        title: "Engine & Coding",
                        tabs: SettingsTab.allCases.filter { $0.category == "Engine & Coding" }
                    )
                }
                .padding(.horizontal, 8)
                .padding(.bottom, 12)
            }
        }
    }

    private func sidebarCategory(title: String, tabs: [SettingsTab]) -> some View {
        let filtered = tabs.filter { tab in
            searchText.isEmpty ||
            tab.title.localizedCaseInsensitiveContains(searchText) ||
            tab.keywords.contains { $0.localizedCaseInsensitiveContains(searchText) }
        }

        return VStack(alignment: .leading, spacing: 2) {
            if !filtered.isEmpty {
                Text(title)
                    .font(theme.ui(.tiny, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .accessibilityAddTraits(.isHeader)

                ForEach(filtered) { tab in
                    let isSelected = selectedTab == tab
                    Button {
                        selectedTab = tab
                    } label: {
                        HStack(spacing: 8) {
                            Image(systemName: tab.systemImage)
                                .font(theme.ui(points: 13, weight: .medium))
                                .frame(width: 18)
                                .foregroundStyle(isSelected ? appearanceManager.activeAccentColor(isDark: false) : .secondary)
                                .accessibilityHidden(true)

                            Text(tab.title)
                                .font(theme.ui(.base))
                                .foregroundStyle(isSelected ? .primary : .secondary)

                            Spacer()
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            isSelected ? Color.accentColor.opacity(0.12) : Color.clear,
                            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
                        )
                    }
                    .buttonStyle(.plain)
                    .help(tab.title)
                    .accessibilityLabel(tab.title)
                    .accessibilityHint("Switches settings view to \(tab.title)")
                    .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
                    .appPointerCursor()
                }
            }
        }
    }

    // MARK: - Tab Content
    @ViewBuilder
    private var tabContent: some View {
        switch selectedTab {
        case .appearance:
            AppearanceSettingsPaneView()
        case .general:
            GeneralSettingsPaneView(model: model)
        case .profiles:
            ProfilesSettingsPaneView(model: model)
        case .shortcuts:
            KeyboardShortcutsSettingsPaneView()
        case .permissions:
            PermissionsSettingsPaneView(model: model)
        case .models:
            ModelsSettingsPaneView(model: model)
        case .engine:
            EngineSettingsPaneView(model: model)
        case .safety:
            SafetySettingsPaneView(model: model)
        case .mcp:
            McpSettingsPaneView(model: model)
        case .skills:
            SkillsSettingsPaneView(model: model)
        case .memory:
            MemorySettingsPaneView(model: model)
        case .agents:
            AgentsSettingsPaneView(model: model)
        case .plugins:
            PluginSettingsPaneView(model: model)
        case .automation:
            CronJobsSettingsPaneView(model: model)
        case .hooks:
            HooksSettingsPaneView(model: model)
        }
    }
}
