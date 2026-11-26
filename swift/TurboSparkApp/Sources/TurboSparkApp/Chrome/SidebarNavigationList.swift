import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Labeled workspace navigation for the unified expanded sidebar.
///
/// Every destination is a visible row, grouped (main, create, workspace) and
/// separated by hairlines: primary navigation does not hide in a pop-up
/// (HIG, Sidebars). Discover is a mode of Models, so it lights the Models row.
@MainActor
struct SidebarNavigationList: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    /// When embedded inside `ChatSidebarView`, the top `[Chat | Projects]`
    /// switcher already owns Chat, so omitting the redundant `Chat` row saves
    /// a row of vertical height.
    var includesChat = true
    @ObservedObject private var keybinds = KeybindOverrideStore.shared
    @State private var hoveredSection: AppModel.AppNavigationSection?
    @ObservedObject private var layoutStore = SidebarLayoutStore.shared

    /// Rows per group in the user's arrangement, with Chat left out when the
    /// switcher owns it. Empty groups are dropped so no stray divider is drawn,
    /// and the section being shown is always listed.
    private var groups: [[AppModel.AppNavigationSection]] {
        layoutStore.layout.groups(
            includesChat: includesChat, keeping: model.activeSection.navigationRow)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            ForEach(Array(groups.enumerated()), id: \.offset) { index, sections in
                if index > 0 { SidebarDivider().padding(.vertical, 4) }
                ForEach(sections) { section in
                    sectionRow(section)
                }
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 6)
        .tsAnimation(TSMotion.select, value: model.activeSection)
    }

    private func sectionRow(_ section: AppModel.AppNavigationSection) -> some View {
        let isSelected = model.activeSection.navigationRow == section
        let isHovered = hoveredSection == section

        return Button {
            TSMotion.animate(TSMotion.select) {
                model.activeSection = section
            }
        } label: {
            HStack(spacing: 9) {
                Image(systemName: isSelected ? section.selectedSystemImage : section.systemImage)
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(isSelected ? theme.accentText : theme.secondaryText)
                    .frame(width: 18 * theme.sidebarScale)
                    .accessibilityHidden(true)

                Text(LocalizedStringKey(section.title), bundle: .module)
                    .font(theme.ui(.small, weight: isSelected ? .semibold : .regular))
                    .foregroundStyle(isSelected ? theme.foreground : theme.secondaryText)
                    .lineLimit(1)
                    .truncationMode(.tail)

                Spacer(minLength: 0)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .frame(minHeight: theme.sidebarRowHeight - 2)
            .background {
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .fill(
                        isSelected
                            ? theme.selection
                            : theme.foreground.opacity(isHovered ? 0.07 : 0))
            }
            .contentShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.985))
        .help(
            Text(LocalizedStringKey(section.title), bundle: .module)
                + (keybinds.chord(for: .navigation(section)).map { Text(verbatim: " (\($0.display))") }
                    ?? Text(verbatim: "")))
        .accessibilityLabel(Text(LocalizedStringKey(section.title), bundle: .module))
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .modifier(CustomizableRow(section: section, store: layoutStore))
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
}

/// Pinned bottom footer for the expanded unified sidebar: user profile &
/// appearance menu on the leading side, Settings link on the trailing side.
@MainActor
struct SidebarFooterView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var isSettingsHovered = false

    private var itemSize: CGFloat { min(34, 28 * theme.sidebarScale) }

    var body: some View {
        VStack(spacing: 0) {
            SidebarDivider()
            HStack(spacing: 6) {
                ProfileMenuButton(model: model, itemSize: itemSize, showsLabel: true)
                Spacer(minLength: 4)
                SettingsLink {
                    Image(systemName: "gearshape")
                        .font(theme.ui(.small, weight: .medium))
                        .foregroundStyle(.appSecondary)
                        .frame(width: itemSize, height: itemSize)
                        .background {
                            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                                .fill(theme.foreground.opacity(isSettingsHovered ? 0.07 : 0))
                        }
                        .rotationEffect(.degrees(isSettingsHovered ? 30 : 0))
                        .contentShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                }
                .buttonStyle(TSPressScaleStyle(scale: 0.92))
                .help(Text("Settings", bundle: .module) + Text(verbatim: " (Cmd+,)"))
                .accessibilityLabel(Text("Settings", bundle: .module))
                .onHover { hovering in
                    TSMotion.animate(TSMotion.select) { isSettingsHovered = hovering }
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
        }
    }
}
