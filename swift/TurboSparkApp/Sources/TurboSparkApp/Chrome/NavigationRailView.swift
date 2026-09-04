import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Always-visible icon rail carrying the top-level sections.
///
/// The rail is what makes the rest of the chrome collapsible: with sections
/// living here, the chat sidebar holds only conversations and can be hidden
/// without stranding navigation.
@MainActor
struct NavigationRailView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @ScaledMetric private var itemSize: CGFloat = 34
    @State private var hoveredSection: AppModel.AppNavigationSection? = nil
    @State private var isSettingsHovered = false

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
        .background(TurboSparkTheme.railBackgroundColor)
    }

    private func railButton(_ section: AppModel.AppNavigationSection) -> some View {
        let isSelected = model.activeSection == section
        return Button {
            model.activeSection = section
        } label: {
            Image(systemName: isSelected ? section.selectedSystemImage : section.systemImage)
                .font(theme.ui(points: 15, weight: .medium))
                .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                .frame(width: itemSize, height: itemSize)
                .background(
                    isSelected ? TurboSparkTheme.accentColor.opacity(0.14) : Color.clear,
                    in: RoundedRectangle(cornerRadius: 8, style: .continuous))
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .help("\(section.title) (⌘\(String(section.shortcutKey)))")
        .accessibilityLabel(section.title)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint("Switches the main pane to \(section.title)")
        .overlay(alignment: .leading) {
            if hoveredSection == section {
                RailTooltip(title: section.title, shortcut: "⌘\(String(section.shortcutKey))")
                    .offset(x: itemSize + 12)
                    .transition(.opacity.combined(with: .scale(scale: 0.95, anchor: .leading)))
                    .zIndex(100)
            }
        }
        .onHover { isHovered in
            withAnimation(.easeInOut(duration: 0.12)) {
                if isHovered {
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
                .font(theme.ui(points: 15, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: itemSize, height: itemSize)
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
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
        .onHover { isHovered in
            withAnimation(.easeInOut(duration: 0.12)) {
                isSettingsHovered = isHovered
            }
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
                .font(theme.ui(points: 11, weight: .medium))
                .foregroundStyle(Color.primary)
                .lineLimit(1)

            Text(shortcut)
                .font(theme.code(points: 9, weight: .semibold))
                .foregroundStyle(Color.secondary)
                .padding(.horizontal, 4)
                .padding(.vertical, 1.5)
                .background(Color(nsColor: .separatorColor).opacity(0.3), in: RoundedRectangle(cornerRadius: 3))
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
                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
        )
        .fixedSize()
        .allowsHitTesting(false)
    }
}
