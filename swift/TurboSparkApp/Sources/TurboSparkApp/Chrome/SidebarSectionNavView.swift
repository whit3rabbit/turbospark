import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The top-level sections as LABELED rows, for the expanded sidebar.
///
/// This is the same navigation `NavigationRailView` draws as icons, in the
/// other of the sidebar's two presentations. The two are deliberately one
/// control in two shapes rather than two controls: both write
/// `model.activeSection`, both read it back for selection, and neither can be
/// on screen while the other is -- `AppSidebarView` picks exactly one.
///
/// **The titles are LOCALIZED here and are plain `String`s on the enum.**
/// `AppNavigationSection.title` predates this view and fed `.help` and
/// `.accessibilityLabel`, which are the non-localizing overloads, so nothing
/// needed a catalog key while the rail showed icons alone. A visible label
/// does, hence `Text(LocalizedStringKey(...), bundle: .module)` and the five
/// keys in `Localizable.xcstrings`. Keeping the enum free of SwiftUI is worth
/// the indirection: it lives in the state layer, which no view type reaches.
@MainActor
struct SidebarSectionNavView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @ScaledMetric private var rowHeight: CGFloat = 30
    @State private var hoveredSection: AppModel.AppNavigationSection?
    @Namespace private var selectionNamespace

    var body: some View {
        VStack(spacing: 2) {
            ForEach(AppModel.AppNavigationSection.allCases) { section in
                sectionRow(section)
            }
        }
        .padding(.horizontal, 8)
        .animation(TSMotion.select, value: model.activeSection)
    }

    private func sectionRow(_ section: AppModel.AppNavigationSection) -> some View {
        let isSelected = model.activeSection == section
        let isHovered = hoveredSection == section

        return Button {
            model.activeSection = section
        } label: {
            HStack(spacing: 9) {
                Image(systemName: isSelected ? section.selectedSystemImage : section.systemImage)
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(isSelected ? theme.accent : Color.secondary)
                    .frame(width: 18)
                    // Symbol swap (outline to filled) crossfades instead of
                    // popping, matching the rail's own treatment.
                    .contentTransition(.symbolEffect(.replace))

                Text(LocalizedStringKey(section.title), bundle: .module)
                    .font(theme.ui(.small, weight: isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? Color.primary : Color.secondary)
                    .lineLimit(1)

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .frame(height: rowHeight)
            .background {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(
                        isSelected
                            ? theme.accent.opacity(0.14)
                            : Color.primary.opacity(isHovered ? 0.07 : 0))
            }
            // The rail's one moving mark, kept at this width. Only the
            // selected row hosts it, so SwiftUI interpolates its frame from
            // the old row to the new instead of fading five separate rects.
            .overlay(alignment: .leading) {
                if isSelected {
                    Capsule()
                        .fill(theme.accent)
                        .frame(width: 2.5, height: rowHeight * 0.5)
                        .offset(x: -4)
                        .matchedGeometryEffect(id: "sidebarSelection", in: selectionNamespace)
                }
            }
            .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.98))
        // No hover tooltip: the label is already on screen, which is the
        // whole point of this presentation. The rail keeps its tooltip
        // because there the title is the only thing a tooltip can add.
        .help("\(section.title) (⌘\(String(section.shortcutKey)))")
        .accessibilityLabel(section.title)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint("Switches the main pane to \(section.title)")
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
}
