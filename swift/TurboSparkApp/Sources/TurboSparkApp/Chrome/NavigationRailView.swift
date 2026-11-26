import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The icon rail: top-level navigation, on screen in every section and state.
///
/// The section's panel (`AppSidebarView`) sits beside it and can be hidden;
/// this cannot, so navigation is never stranded. While a panel is shown the
/// rail takes the panel's background and draws the hairline between them, so
/// the pair reads as one surface rather than two bands (Gotcha 17).
///
/// Redesign notes: selection is now a single indicator bar that SLIDES
/// between rows (`matchedGeometryEffect`) instead of a tinted rounded rect
/// that appears and disappears per button. One moving mark reads as "you are
/// here"; five independently fading rects read as five separate states. The
/// tinted background is kept underneath it, and hover gets its own quieter
/// fill so an unselected icon still answers the pointer.
@MainActor
struct NavigationRailView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    /// A panel is on screen beside the rail (drives the shared surface).
    let isPanelVisible: Bool
    /// The user's panel preference. A section whose panel lists destinations
    /// (Audio, Images, Installed, API) is a plain button when selecting it
    /// will show that panel, and a menu of the same destinations when it will
    /// not.
    let isPanelPreferred: Bool

    private var itemSize: CGFloat { min(42, 34 * theme.sidebarScale) }
    @State private var hoveredSection: AppModel.AppNavigationSection? = nil
    @State private var isSettingsHovered = false
    /// The live chord map, so rail tooltips show what is bound now.
    @ObservedObject private var keybinds = KeybindOverrideStore.shared
    @ObservedObject private var layoutStore = SidebarLayoutStore.shared
    @Namespace private var selectionNamespace

    var body: some View {
        VStack(spacing: 4) {
            ScrollView {
                VStack(spacing: 4) {
                    let groups = layoutStore.layout.groups(
                        includesChat: true, keeping: model.activeSection.navigationRow)
                    ForEach(Array(groups.enumerated()), id: \.offset) { index, sections in
                        if index > 0 {
                            SidebarDivider().padding(.horizontal, 12).padding(.vertical, 4)
                        }
                        ForEach(sections) { section in
                            Group {
                                if Self.hasDestinationMenu(section) && !isPanelPreferred {
                                    railMenu(section)
                                } else {
                                    railButton(section)
                                }
                            }
                            .modifier(CustomizableRow(section: section, store: layoutStore))
                        }
                    }
                }
            }
            .scrollIndicators(.hidden)
            Spacer(minLength: 0)
            ProfileMenuButton(model: model, itemSize: itemSize)
            settingsButton
        }
        // Starts below the top bar so the first icon clears the traffic
        // lights, which a `.hiddenTitleBar` window draws over this column.
        .padding(.top, AppChromeLayout.topBarHeight + 6)
        .padding(.bottom, 8)
        .frame(width: AppChromeLayout.navigationRailWidth)
        .frame(maxHeight: .infinity)
        .background(
            isPanelVisible
                ? TurboSparkTheme.sidebarBackgroundColor(
                    isDark: theme.isDark, reduceTransparency: theme.reduceTransparency)
                : TurboSparkTheme.railBackgroundColor(
                    isDark: theme.isDark, reduceTransparency: theme.reduceTransparency))
        .overlay(alignment: .trailing) {
            // Overlay, not layout width: `leftChromeWidth` counts the rail
            // and the panel only.
            if isPanelVisible {
                Rectangle()
                    .fill(.appBorder)
                    .frame(width: 0.5)
                    .accessibilityHidden(true)
            }
        }
        .tsAnimation(TSMotion.select, value: model.activeSection)
    }

    /// Every panel except Chat's lists the section's destinations, so with
    /// the panel hidden those would be unreachable without this menu. Chat's
    /// panel is its conversation list, which the chat pane does not need.
    /// Discover has the Models panel but no rail icon of its own: the Models
    /// icon covers it.
    nonisolated static func hasDestinationMenu(_ section: AppModel.AppNavigationSection) -> Bool {
        section.hasSidebarPanel && section != .chat && section != .modelHub
    }

    private func railMenu(_ section: AppModel.AppNavigationSection) -> some View {
        let isSelected = model.activeSection.navigationRow == section
        let isHovered = hoveredSection == section
        return Menu {
            switch section {
            case .audio:
                AudioDestinationMenu(model: model)
            case .images:
                SidebarModeMenu<ImagesWorkspaceMode> { mode in
                    model.imagesMode = mode
                    model.activeSection = .images
                }
            case .modelManager:
                SidebarModeMenu<InstalledModelKind> { kind in
                    model.installedModelKind = kind
                    model.activeSection = .modelManager
                }
            case .server:
                SidebarModeMenu<APIWorkspaceTab> { tab in
                    model.apiWorkspaceTab = tab
                    model.activeSection = .server
                }
            default:
                // Unreachable: `hasDestinationMenu` admits only the cases above.
                EmptyView()
            }
        } label: {
            Image(systemName: isSelected ? section.selectedSystemImage : section.systemImage)
                .font(theme.ui(.callout, weight: .medium))
                .foregroundStyle(isSelected ? theme.accentText : theme.secondaryText)
                .frame(width: itemSize, height: itemSize)
                .background(theme.accent.opacity(isSelected ? 0.14 : 0),
                            in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        }
        // `.button` + `.plain`: `.borderlessButton` flattens the label and
        // drops the tint.
        .menuStyle(.button).buttonStyle(.plain)
        .menuIndicator(.hidden).fixedSize()
        // Same hover answer as the plain button beside it: a menu-labeled
        // rail item used to be the one row that ignored the pointer.
        .background {
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .fill(theme.foreground.opacity(!isSelected && isHovered ? 0.07 : 0))
        }
        .onHover { hovering in
            TSMotion.animate(TSMotion.hover) {
                if hovering {
                    hoveredSection = section
                } else if hoveredSection == section {
                    hoveredSection = nil
                }
            }
        }
        .help(Text(LocalizedStringKey(section.title), bundle: .module)
              + (keybinds.chord(for: .navigation(section)).map { Text(verbatim: " (\($0.display))") }
                 ?? Text(verbatim: "")))
        .accessibilityLabel(Text(LocalizedStringKey(section.title), bundle: .module))
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    private func railButton(_ section: AppModel.AppNavigationSection) -> some View {
        let isSelected = model.activeSection.navigationRow == section
        let isHovered = hoveredSection == section

        return Button {
            model.activeSection = section
        } label: {
            Image(systemName: isSelected ? section.selectedSystemImage : section.systemImage)
                .font(theme.ui(.callout, weight: .medium))
                .foregroundStyle(isSelected ? theme.accentText : theme.secondaryText)
                .frame(width: itemSize, height: itemSize)
                .background {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(
                            isSelected
                                ? theme.selection
                                : theme.foreground.opacity(isHovered ? 0.07 : 0))
                }
                // The one moving mark. Only the selected row hosts it, so
                // SwiftUI interpolates its frame from the old row to the new.
                .overlay(alignment: .leading) {
                    if isSelected {
                        Capsule()
                            .fill(theme.accent)
                            .frame(width: 2.5, height: itemSize * 0.55)
                            .offset(x: -7)
                            .matchedGeometryEffect(id: "railSelection", in: selectionNamespace)
                    }
                }
                // Symbol swap (outline to filled) crossfades instead of
                // popping. Cheap: one glyph, only on selection change.
                .contentTransition(.symbolEffect(.replace))
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.92))
        // The live chord from the keybind store, not a hardcoded default: a
        // rebinding must not leave the tooltip advertising a dead chord.
        .help(Text(LocalizedStringKey(section.title), bundle: .module)
              + (keybinds.chord(for: .navigation(section)).map { Text(verbatim: " (\($0.display))") }
                 ?? Text(verbatim: "")))
        .accessibilityLabel(Text(LocalizedStringKey(section.title), bundle: .module))
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .onHover { hovering in
            TSMotion.animate(TSMotion.hover) {
                if hovering {
                    hoveredSection = section
                } else if hoveredSection == section {
                    hoveredSection = nil
                }
            }
        }
    }

    private var settingsButton: some View {
        SettingsLink {
            Image(systemName: "gearshape")
                .font(theme.ui(.callout, weight: .medium))
                .foregroundStyle(.appSecondary)
                .frame(width: itemSize, height: itemSize)
                .background {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(theme.foreground.opacity(isSettingsHovered ? 0.07 : 0))
                }
                .rotationEffect(.degrees(isSettingsHovered ? 30 : 0))
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.92))
        .help(Text("Settings", bundle: .module) + Text(verbatim: " (Cmd+,)"))
        .accessibilityLabel(Text("Settings", bundle: .module))
        .onHover { hovering in
            TSMotion.animate(TSMotion.select) { isSettingsHovered = hovering }
        }
    }
}
