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
    /// content's minimum, so a min/ideal frame alone left every display at the
    /// floor. `SettingsWindowFrame` applies this size to the AppKit window the
    /// first time it opens; the content itself only declares `minimumSize`, so
    /// the user can resize the window to recover a crowded pane. Derived from
    /// the visible frame, same approach as `TurboSparkApp.defaultWindowSize`:
    /// roughly three quarters of a large display, clamped to a floor that
    /// still fits a 1280x800 one.
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

    /// The smallest window the panes still lay out in: the 252pt sidebar plus
    /// a content column the adaptive headers fold into two rows within.
    static let minimumSize = CGSize(width: 820, height: 560)

    /// Settings search matches the title the user actually SEES, which
    /// follows the in-app language, as well as the English source and the
    /// tab's keywords. The query is trimmed: an untrimmed lone space hid
    /// every tab whose title has no space (Appearance, General) while the
    /// control results, which already trimmed, showed nothing.
    static func matches(_ title: String, keywords: [String] = [], query rawQuery: String) -> Bool {
        let query = rawQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        return query.isEmpty
            || title.localizedCaseInsensitiveContains(query)
            || AppStrings.localized(String.LocalizationValue(title)).localizedCaseInsensitiveContains(query)
            || keywords.contains { $0.localizedCaseInsensitiveContains(query) }
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
                .frame(minWidth: 220, idealWidth: 252, maxWidth: 280)
                .background(.appPage)

            Rectangle()
                .fill(.appBorder)
                .frame(width: 1)

            VStack(spacing: 0) {
                // Top header for right pane
                HStack {
                    Text(LocalizedStringKey(selectedTab.title), bundle: .module)
                        .font(theme.ui(.title3, weight: .semibold))
                        .accessibilityAddTraits(.isHeader)
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
        .frame(
            minWidth: Self.minimumSize.width, maxWidth: .infinity,
            minHeight: Self.minimumSize.height, maxHeight: .infinity)
        .background(SettingsWindowFrame(minimumSize: Self.minimumSize, defaultSize: windowSize))
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
            AppSearchField(
                text: $searchText,
                prompt: Text("Search settings...", bundle: .module),
                minWidth: 0, idealWidth: 220, maxWidth: .infinity)
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
            Self.matches(tab.title, keywords: tab.keywords, query: searchText)
        }

        return VStack(alignment: .leading, spacing: 3) {
            if !filtered.isEmpty {
                Text(LocalizedStringKey(title), bundle: .module)
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
                    .help(Text(LocalizedStringKey(tab.title), bundle: .module))
                    .accessibilityLabel(Text(LocalizedStringKey(tab.title), bundle: .module))
                    .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
                    .appPointerCursor()
                }
            }
        }
    }

    private var controlSearchResults: some View {
        let query = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        let results = SettingsControlCatalog.entries.filter {
            Self.matches($0.title, query: query)
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
                        Text(LocalizedStringKey(result.pane.title), bundle: .module)
                            .themedFont(.tiny).foregroundStyle(.appSecondary)
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
        case .tools:
            ToolsSettingsPaneView(model: model)
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
        case .computerUse:
            ComputerUseSettingsPaneView(model: model)
        case .network:
            NetworkSettingsPaneView(model: model)
        }
    }
}
