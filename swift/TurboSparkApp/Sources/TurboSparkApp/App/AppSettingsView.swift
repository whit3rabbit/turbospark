import AppKit
import SwiftUI
import TurboSpark

/// Spacing shared by the settings window and every pane, so pane content
/// starts on the same edge as the header title instead of each pane picking
/// its own 16, 20 or 24pt inset.
enum SettingsLayout {
    /// Leading and trailing inset of the header title and of pane content.
    static let horizontalInset: CGFloat = 36
    /// Around a scrolling, non-Form pane's content.
    static let contentInsets = EdgeInsets(
        top: 28, leading: horizontalInset, bottom: 36, trailing: horizontalInset)
    /// Around a `.grouped` Form. The Form adds its own section margin inside
    /// its scroll view, so it takes a smaller outer inset than a ScrollView
    /// pane. 16pt is what these panes already used.
    static let formInsets = EdgeInsets(top: 16, leading: 16, bottom: 16, trailing: 16)
    /// Vertical padding of the full-width header bar on list/detail panes
    /// (Skills, Plugins, Agents).
    static let headerBarVerticalPadding: CGFloat = 14
    /// Inset of a list/detail pane's detail column.
    static let detailInset: CGFloat = 28
}

public struct AppSettingsView: View {
    /// The Settings scene ignores `idealWidth`/`idealHeight` and opens at the
    /// content's minimum, so a min/ideal frame left every display at the
    /// 940x660 floor. A fixed frame is the size the window actually opens at
    /// (and it is not user-resizable). Derived from the visible frame, same
    /// approach as `TurboSparkApp.defaultWindowSize`: roughly three quarters
    /// of a large display, clamped to a floor that still fits a 1280x800 one.
    static var defaultWindowSize: CGSize {
        windowSize(forVisibleFrame: NSScreen.main?.visibleFrame)
    }

    /// Separate from `NSScreen` so the clamps are testable without a display.
    static func windowSize(forVisibleFrame visible: CGRect?) -> CGSize {
        guard let visible else {
            return CGSize(width: 1180, height: 800)
        }
        return CGSize(
            width: min(max(940, visible.width * 0.76), 1480),
            height: min(max(660, visible.height * 0.86), 1000)
        )
    }

    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var appearanceManager = AppearanceManager.shared

    @State private var selectedTab: SettingsTab = .appearance
    @State private var highlightedControl: String?
    @State private var searchText: String = ""
    /// Captured once: `body` re-evaluates on every model publish, and
    /// re-reading `NSScreen.main` there would resize the open window whenever
    /// the key window moved to another display.
    @State private var windowSize = AppSettingsView.defaultWindowSize

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        HStack(spacing: 0) {
            settingsSidebar
                .frame(width: 252)
                .background(.appPage.opacity(0.85))

            Rectangle()
                .fill(.appBorder)
                .frame(width: 1)

            VStack(spacing: 0) {
                // Top header for right pane
                HStack {
                    Text(LocalizedStringKey(selectedTab.title), bundle: .module)
                        .font(theme.ui(.title3, weight: .semibold))
                    Spacer()
                }
                .padding(.horizontal, SettingsLayout.horizontalInset)
                .padding(.vertical, 20)
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
        .frame(width: windowSize.width, height: windowSize.height)
        .onReceive(NotificationCenter.default.publisher(for: .openSettingsTab)) { notification in
            if let tab = notification.object as? SettingsTab {
                selectedTab = tab
            }
        }
        .onAppear { consumeRequestedTab() }
        .onChange(of: model.requestedSettingsTab) { _, _ in consumeRequestedTab() }
    }

    private func consumeRequestedTab() {
        guard let tab = model.requestedSettingsTab else { return }
        selectedTab = tab
        model.requestedSettingsTab = nil
    }

    // MARK: - Sidebar
    private var settingsSidebar: some View {
        VStack(spacing: 16) {
            // Search field
            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.appSecondary)
                    .font(theme.ui(.small))
                    .accessibilityHidden(true)
                TextField("Search settings...", text: $searchText)
                    .textFieldStyle(.plain)
                    .font(theme.ui(.small))
                    .accessibilityLabel("Search settings")
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 8))
            .overlay(
                RoundedRectangle(cornerRadius: 8)
                    .stroke(.appBorder.opacity(0.3), lineWidth: 1)
            )
            .padding(.horizontal, 16)
            .padding(.top, 20)

            ScrollView {
                VStack(alignment: .leading, spacing: 22) {
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
                .padding(.horizontal, 12)
                .padding(.bottom, 16)
            }
        }
    }

    private func sidebarCategory(title: String, tabs: [SettingsTab]) -> some View {
        let filtered = tabs.filter { tab in
            searchText.isEmpty ||
            tab.title.localizedCaseInsensitiveContains(searchText) ||
            // The title the user actually sees, in the active language.
            NSLocalizedString(tab.title, bundle: .module, comment: "")
                .localizedCaseInsensitiveContains(searchText) ||
            tab.keywords.contains { $0.localizedCaseInsensitiveContains(searchText) }
        }

        return VStack(alignment: .leading, spacing: 3) {
            if !filtered.isEmpty {
                Text(title)
                    .font(theme.ui(.tiny, weight: .semibold))
                    .foregroundStyle(.appSecondary)
                    .padding(.horizontal, 12)
                    .padding(.bottom, 6)
                    .accessibilityAddTraits(.isHeader)

                ForEach(filtered) { tab in
                    let isSelected = selectedTab == tab
                    Button {
                        selectedTab = tab
                    } label: {
                        HStack(spacing: 10) {
                            Image(systemName: tab.systemImage)
                                .font(theme.ui(.callout, weight: .medium))
                                .frame(width: 20)
                                .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                                .accessibilityHidden(true)

                            Text(LocalizedStringKey(tab.title), bundle: .module)
                                .font(theme.ui(.base))
                                .foregroundStyle(isSelected ? theme.foreground : theme.secondaryText)

                            Spacer()
                        }
                        .padding(.horizontal, 12)
                        .padding(.vertical, 8)
                        // The whole row, not just the glyphs, takes the click.
                        .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
                        .background(
                            isSelected ? theme.selection : Color.clear,
                            in: RoundedRectangle(cornerRadius: 8, style: .continuous)
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
        case .browser:
            BrowserSettingsPaneView(model: model)
        }
    }
}
