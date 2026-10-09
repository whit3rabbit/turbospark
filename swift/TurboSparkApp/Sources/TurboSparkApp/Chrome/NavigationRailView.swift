import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The icon rail: the sidebar's COLLAPSED presentation.
///
/// This used to be a permanent band of its own, sitting to the left of the
/// chat sidebar, and the reason was that sections had to stay reachable while
/// the sidebar was hidden. That constraint has not gone away -- it is met by
/// the COLUMN now rather than by having two of them. `AppSidebarView` shows
/// this view when collapsed and `SidebarSectionNavView` when expanded, so
/// navigation is on screen in both states and in every section.
///
/// Nothing here changed in the merge, which is the point: the collapsed
/// sidebar IS the rail that was already shipping.
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
                            if section == .audio {
                                Menu {
                                    AudioDestinationMenu(model: model)
                                } label: {
                                    Image(systemName: section.systemImage)
                                        .font(theme.ui(.callout, weight: .medium))
                                        .foregroundStyle(model.activeSection == .audio ? theme.accent : theme.secondaryText)
                                        .frame(width: itemSize, height: itemSize)
                                        .background(theme.accent.opacity(model.activeSection == .audio ? 0.14 : 0),
                                                    in: RoundedRectangle(cornerRadius: 8))
                                }
                                .menuStyle(.borderlessButton).menuIndicator(.hidden).fixedSize()
                                .help(Text("Audio", bundle: .module))
                                .accessibilityLabel(Text("Audio", bundle: .module))
                                .accessibilityAddTraits(model.activeSection == .audio ? .isSelected : [])
                            } else {
                                railButton(section)
                            }
                        }
                    }
                }
            }
            .scrollIndicators(.hidden)
            Spacer(minLength: 0)
            settingsButton
        }
        // Starts below the top bar so the first icon clears the traffic
        // lights, which a `.hiddenTitleBar` window draws over this column.
        .padding(.top, AppChromeLayout.topBarHeight + 6)
        .padding(.bottom, 8)
        .frame(width: AppChromeLayout.navigationRailWidth)
        .frame(maxHeight: .infinity)
        .background(TurboSparkTheme.railBackgroundColor(
            isDark: theme.isDark, reduceTransparency: theme.reduceTransparency))
        .animation(TSMotion.select, value: model.activeSection)
    }

    private func railButton(_ section: AppModel.AppNavigationSection) -> some View {
        let isSelected = model.activeSection == section
        let isHovered = hoveredSection == section

        return Button {
            model.activeSection = section
        } label: {
            Image(systemName: isSelected ? section.selectedSystemImage : section.systemImage)
                .font(theme.ui(.callout, weight: .medium))
                .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
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
            withAnimation(TSMotion.hover) {
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
            withAnimation(TSMotion.select) { isSettingsHovered = hovering }
        }
    }
}
