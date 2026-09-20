import AppKit
import SwiftUI
import TurboSpark

public struct AppSettingsView: View {
    /// The Settings scene sizes its window to this view and the window is not
    /// user-resizable, so the ideal size in `body` is the size the settings
    /// window actually opens at. It is derived from the visible frame (same
    /// approach as `TurboSparkApp.defaultWindowSize`) so a larger display gets
    /// a roomier pane while a small display clamps to the minimum.
    static var defaultWindowSize: CGSize {
        guard let visible = NSScreen.main?.visibleFrame else {
            return CGSize(width: 1040, height: 720)
        }
        return CGSize(
            width: min(max(940, visible.width * 0.7), 1440),
            height: min(max(660, visible.height * 0.8), 940)
        )
    }

    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var appearanceManager = AppearanceManager.shared

    @State private var selectedTab: SettingsTab = .appearance
    @State private var highlightedControl: String?
    @State private var searchText: String = ""

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        HStack(spacing: 0) {
            settingsSidebar
                .frame(width: 240)
                .background(.appPage.opacity(0.85))

            Rectangle()
                .fill(.appBorder)
                .frame(width: 1)

            VStack(spacing: 0) {
                // Top header for right pane
                HStack {
                    Text(selectedTab.title)
                        .font(theme.ui(.title3, weight: .semibold))
                    Spacer()
                }
                .padding(.horizontal, 28)
                .padding(.vertical, 16)
                .background(.appPage)

                Rectangle()
                    .fill(.appBorder)
                    .frame(height: 1)

                ScrollViewReader { proxy in
                    tabContent
                        .environment(\.settingsTarget, highlightedControl)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                        .task(id: highlightedControl) {
                            guard let id = highlightedControl else { return }
                            // The target pane must mount before its scroll view knows the anchor.
                            try? await Task.sleep(for: .milliseconds(150))
                            guard !Task.isCancelled else { return }
                            proxy.scrollTo(id, anchor: .center)
                        }
                }
            }
            // A soft edge on the content's leading side so the sidebar seam
            // reads in both light and dark themes. The bare hairline alone
            // disappears against the two near-identical page backgrounds, and
            // a shadow cast by the divider would be covered by this column.
            .overlay(alignment: .leading) {
                LinearGradient(
                    colors: [
                        theme.foreground.opacity(theme.isDark ? 0.22 : 0.08),
                        .clear
                    ],
                    startPoint: .leading,
                    endPoint: .trailing
                )
                .frame(width: 6)
                .allowsHitTesting(false)
            }
        }
        .font(theme.ui(.base))
        .frame(
            minWidth: 940,
            idealWidth: Self.defaultWindowSize.width,
            minHeight: 660,
            idealHeight: Self.defaultWindowSize.height
        )
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
                    .foregroundStyle(.appSecondary)
                    .font(theme.ui(.small))
                    .accessibilityHidden(true)
                TextField("Search settings...", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(theme.ui(.small))
                    .accessibilityLabel("Search settings")
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(.appBorder.opacity(0.3), lineWidth: 1)
            )
            .padding(.horizontal, 12)
            .padding(.top, 14)

            ScrollView {
                VStack(alignment: .leading, spacing: 14) {
                    if !searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        controlSearchResults
                    }
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
                    .foregroundStyle(.appSecondary)
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
                                .font(theme.ui(.callout, weight: .medium))
                                .frame(width: 18)
                                .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                                .accessibilityHidden(true)

                            Text(tab.title)
                                .font(theme.ui(.base))
                                .foregroundStyle(isSelected ? theme.foreground : theme.secondaryText)

                            Spacer()
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 6)
                        .background(
                            isSelected ? theme.selection : Color.clear,
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

    private var controlSearchResults: some View {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        let results = SettingsControlCatalog.entries.filter {
            $0.title.localizedCaseInsensitiveContains(query)
                || NSLocalizedString($0.title, bundle: .module, comment: "").localizedCaseInsensitiveContains(query)
        }
        return VStack(alignment: .leading, spacing: 8) {
            if results.isEmpty {
                Text("No matching controls", bundle: .module).themedFont(.small)
            }
            ForEach(results) { result in
                Button {
                    selectedTab = result.pane
                    highlightedControl = result.navigationID
                    searchText = ""
                } label: {
                    VStack(alignment: .leading, spacing: 3) {
                        Text(LocalizedStringKey(result.title), bundle: .module).themedFont(.small, weight: .medium)
                        Text(result.pane.title).themedFont(.tiny).foregroundStyle(.appSecondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .help(Text(LocalizedStringKey(result.timing.rawValue), bundle: .module))
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
            McpManagementView(model: model)
        case .skills:
            SkillsSettingsPaneView(model: model)
        case .memory:
            MemorySettingsPaneView(model: model)
        case .soul:
            SoulSettingsPaneView(model: model)
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
