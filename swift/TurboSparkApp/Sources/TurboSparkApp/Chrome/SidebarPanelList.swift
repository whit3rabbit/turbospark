import SwiftUI

// The shared pieces of a section panel beside the icon rail (Gotcha 17).
// Audio, Images, Installed and API all list their pages or modes with these,
// so the row treatment, hover, selection trait and header cannot drift
// between panels.

/// A section sub-mode listed in that section's panel, and in the rail's
/// fallback menu while the panel is hidden.
protocol SidebarPanelMode: Hashable, Identifiable, CaseIterable where AllCases == [Self] {
    /// The localized name, rendered through the bundled catalog.
    var titleText: Text { get }
    var symbol: String { get }
}

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// A panel's heading and its scrolling rows. The heading names the section
/// the rail selected; the rows are only what is in it, never the section list.
@MainActor
struct SidebarPanelList<Content: View>: View {
    @Environment(\.appTheme) private var theme
    let title: Text
    @ViewBuilder let content: () -> Content

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            title
                .font(theme.ui(.small, weight: .semibold))
                .foregroundStyle(.appText)
                .padding(.horizontal, 8)
                .accessibilityAddTraits(.isHeader)

            ScrollView {
                VStack(spacing: 2) { content() }
            }
            .scrollIndicators(.hidden)
        }
        .padding(.horizontal, 8)
        .padding(.top, 10)
        .frame(maxHeight: .infinity, alignment: .top)
    }
}

/// One destination row: glyph, title, selected fill. Titles wrap to two lines
/// rather than truncate, so long translations and Extra Large text still fit
/// the fixed panel width.
@MainActor
struct SidebarPanelRow: View {
    @Environment(\.appTheme) private var theme
    let title: Text
    let systemImage: String
    let isSelected: Bool
    let action: () -> Void
    @State private var isHovered = false

    var body: some View {
        Button(action: action) {
            HStack(spacing: 9) {
                Image(systemName: systemImage)
                    .frame(width: 18 * theme.sidebarScale)
                    .foregroundStyle(isSelected ? theme.accentText : theme.secondaryText)
                title
                    .foregroundStyle(isSelected ? theme.foreground : theme.secondaryText)
                    .lineLimit(2)
                    .fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 0)
            }
            .font(theme.ui(.small, weight: isSelected ? .semibold : nil))
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .frame(minHeight: theme.sidebarRowHeight)
            .background {
                RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                    .fill(
                        isSelected
                            ? theme.selection
                            : theme.foreground.opacity(isHovered ? 0.07 : 0))
            }
            .contentShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.98))
        .help(title)
        .accessibilityLabel(title)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .onHover { hovering in
            TSMotion.animate(TSMotion.hover) { isHovered = hovering }
        }
    }
}

/// A panel that lists one section's modes, plus optional content under them.
/// The rows form one group named by `groupLabel`, the VoiceOver name the
/// in-pane picker carried before it moved here.
@MainActor
struct SidebarModePanel<Mode: SidebarPanelMode, Footer: View>: View {
    let title: Text
    let groupLabel: Text
    @Binding var selection: Mode
    @ViewBuilder let footer: () -> Footer

    init(
        title: Text, groupLabel: Text, selection: Binding<Mode>,
        @ViewBuilder footer: @escaping () -> Footer
    ) {
        self.title = title
        self.groupLabel = groupLabel
        self._selection = selection
        self.footer = footer
    }

    var body: some View {
        SidebarPanelList(title: title) {
            VStack(spacing: 2) {
                ForEach(Mode.allCases) { mode in
                    SidebarPanelRow(
                        title: mode.titleText, systemImage: mode.symbol,
                        isSelected: selection == mode
                    ) { selection = mode }
                }
            }
            .accessibilityElement(children: .contain)
            .accessibilityLabel(groupLabel)
            footer()
        }
        .tsAnimation(TSMotion.select, value: selection)
    }
}

extension SidebarModePanel where Footer == EmptyView {
    init(title: Text, groupLabel: Text, selection: Binding<Mode>) {
        self.init(title: title, groupLabel: groupLabel, selection: selection) { EmptyView() }
    }
}

/// The rail's menu for a section whose panel is hidden: the same modes as the
/// panel, so hiding the panel never strands them (Gotcha 17, "navigation
/// never hides"). `AudioDestinationMenu` is the Audio equivalent.
@MainActor
struct SidebarModeMenu<Mode: SidebarPanelMode>: View {
    let select: (Mode) -> Void

    var body: some View {
        ForEach(Mode.allCases) { mode in
            Button { select(mode) } label: {
                Label { mode.titleText } icon: { Image(systemName: mode.symbol) }
            }
        }
    }
}
