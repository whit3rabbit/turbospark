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

    @ScaledMetric private var itemSize: CGFloat = 34
    @State private var hoveredSection: AppModel.AppNavigationSection? = nil
    @State private var isSettingsHovered = false
    @Namespace private var selectionNamespace

    var body: some View {
        VStack(spacing: 4) {
            ForEach(AppModel.AppNavigationSection.allCases) { section in
                railButton(section)
            }
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
                .foregroundStyle(isSelected ? theme.accent : Color.secondary)
                .frame(width: itemSize, height: itemSize)
                .background {
                    RoundedRectangle(cornerRadius: 8, style: .continuous)
                        .fill(
                            isSelected
                                ? theme.accent.opacity(0.14)
                                : Color.primary.opacity(isHovered ? 0.07 : 0))
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
        .help("\(section.title) (⌘\(String(section.shortcutKey)))")
        .accessibilityLabel(section.title)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint("Switches the main pane to \(section.title)")
        .overlay(alignment: .leading) {
            if isHovered {
                RailTooltip(title: section.title, shortcut: "⌘\(String(section.shortcutKey))")
                    .offset(x: itemSize + 12)
                    .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .leading)))
                    .zIndex(100)
            }
        }
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
                        .fill(Color.primary.opacity(isSettingsHovered ? 0.07 : 0))
                }
                .rotationEffect(.degrees(isSettingsHovered ? 30 : 0))
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.92))
        .help("Settings (⌘,)")
        .accessibilityLabel("Settings")
        .accessibilityHint("Opens application settings window")
        .overlay(alignment: .leading) {
            if isSettingsHovered {
                RailTooltip(title: "Settings", shortcut: "⌘,")
                    .offset(x: itemSize + 12)
                    .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .leading)))
                    .zIndex(100)
            }
        }
        .onHover { hovering in
            withAnimation(TSMotion.select) { isSettingsHovered = hovering }
        }
    }
}

/// Floating hover tooltip badge for navigation rail icons.
private struct RailTooltip: View {
    @Environment(\.appTheme) private var theme
    let title: String
    let shortcut: String

    var body: some View {
        HStack(spacing: 6) {
            Text(title)
                .font(theme.ui(.tiny, weight: .medium))
                .foregroundStyle(Color.primary)
                .lineLimit(1)

            Text(shortcut)
                .font(theme.code(.small, weight: .semibold))
                .foregroundStyle(Color.secondary)
                .padding(.horizontal, 4)
                .padding(.vertical, 1.5)
                .background(
                    Color(nsColor: .separatorColor).opacity(0.3),
                    in: RoundedRectangle(cornerRadius: 3))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .background(
            Color(nsColor: .windowBackgroundColor)
                .shadow(.drop(color: Color.black.opacity(0.2), radius: 5, x: 0, y: 2)),
            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5)
        )
        .fixedSize()
        .allowsHitTesting(false)
    }
}
