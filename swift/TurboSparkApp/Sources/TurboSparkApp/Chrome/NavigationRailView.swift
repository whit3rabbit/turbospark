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
    @Namespace private var selectionNamespace

    var body: some View {
        VStack(spacing: 4) {
            ScrollView {
                VStack(spacing: 4) {
                    ForEach(SidebarNavigationGroup.allCases) { group in
                        if group != .creation {
                            SidebarDivider().padding(.horizontal, 12).padding(.vertical, 4)
                        }
                        ForEach(group.sections) { section in
                            if Self.hasDestinationMenu(section) && !isPanelPreferred {
                                railMenu(section)
                            } else {
                                railButton(section)
                            }
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
    nonisolated static func hasDestinationMenu(_ section: AppModel.AppNavigationSection) -> Bool {
        section.hasSidebarPanel && section != .chat
    }

    private func railMenu(_ section: AppModel.AppNavigationSection) -> some View {
        let isSelected = model.activeSection == section
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
        .help(Text(LocalizedStringKey(section.title), bundle: .module))
        .accessibilityLabel(Text(LocalizedStringKey(section.title), bundle: .module))
        .accessibilityAddTraits(isSelected ? .isSelected : [])
    }

    private func railButton(_ section: AppModel.AppNavigationSection) -> some View {
        let isSelected = model.activeSection == section
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
        .help(Text(LocalizedStringKey(section.title), bundle: .module)
              + Text(verbatim: " (Cmd+\(section.shortcutKey))"))
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
