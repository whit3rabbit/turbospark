import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The Audio section's panel beside the icon rail: every Audio page, one click
/// away, in the same order as `AudioDestinationMenu` (destinations, Library,
/// tools).
///
/// This replaced the Audio sub-rows that expanded under the Audio row of the
/// old labeled section list. The rail's Audio item is a plain button while
/// this panel is shown, and falls back to `AudioDestinationMenu` when the
/// panel is hidden, so the pages stay one step away either way.
@MainActor
struct AudioSidebarPanel: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @State private var hovered: AudioWorkspacePage?

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("Audio", bundle: .module)
                .font(theme.ui(.small, weight: .semibold))
                .foregroundStyle(.appText)
                .padding(.horizontal, 8)
                .accessibilityAddTraits(.isHeader)

            ScrollView {
                VStack(spacing: 2) {
                    ForEach(AudioWorkspacePage.destinations) { row($0) }
                    SidebarDivider().padding(.horizontal, 8).padding(.vertical, 6)
                    row(.library)
                    SidebarDivider().padding(.horizontal, 8).padding(.vertical, 6)
                    ForEach(AudioWorkspacePage.tools) { row($0) }
                }
            }
            .scrollIndicators(.hidden)
        }
        .padding(.horizontal, 8)
        .padding(.top, 10)
        .frame(maxHeight: .infinity, alignment: .top)
        .tsAnimation(TSMotion.select, value: controller.page)
    }

    private func row(_ page: AudioWorkspacePage) -> some View {
        let isSelected = controller.page == page
        return Button { model.openAudio(page) } label: {
            HStack(spacing: 9) {
                Image(systemName: page.symbol)
                    .frame(width: 18 * theme.sidebarScale)
                    .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                Text(LocalizedStringKey(page.title), bundle: .module)
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
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(
                        isSelected
                            ? theme.selection
                            : theme.foreground.opacity(hovered == page ? 0.07 : 0))
            }
            .contentShape(RoundedRectangle(cornerRadius: 7, style: .continuous))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.98))
        .help(Text(LocalizedStringKey(page.title), bundle: .module))
        .accessibilityLabel(Text(LocalizedStringKey(page.title), bundle: .module))
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .onHover { hovering in
            TSMotion.animate(TSMotion.hover) {
                if hovering {
                    hovered = page
                } else if hovered == page {
                    hovered = nil
                }
            }
        }
    }
}
