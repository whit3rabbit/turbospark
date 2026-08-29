import SwiftUI

/// Always-visible icon rail carrying the top-level sections.
///
/// The rail is what makes the rest of the chrome collapsible: with sections
/// living here, the chat sidebar holds only conversations and can be hidden
/// without stranding navigation.
struct NavigationRailView: View {
    @ObservedObject var model: AppModel

    @ScaledMetric private var itemSize: CGFloat = 34

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
                .font(.system(size: 15, weight: .medium))
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
    }

    private var settingsButton: some View {
        SettingsLink {
            Image(systemName: "gearshape")
                .font(.system(size: 15, weight: .medium))
                .foregroundStyle(.secondary)
                .frame(width: itemSize, height: itemSize)
                .contentShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        }
        .buttonStyle(.plain)
        .help("Settings (⌘,)")
        .accessibilityLabel("Settings")
    }
}
