import SwiftUI

@MainActor
struct AudioSidebarDestinations: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @State private var hovered: AudioWorkspacePage?

    var body: some View {
        VStack(spacing: 2) {
            ForEach(AudioWorkspacePage.destinations) { page in
                let selected = controller.page == page
                Button { model.openAudio(page) } label: {
                    HStack(spacing: 9) {
                        Image(systemName: page.symbol).frame(width: 18 * theme.sidebarScale)
                        Text(LocalizedStringKey(page.title), bundle: .module)
                            .lineLimit(2)
                            .fixedSize(horizontal: false, vertical: true)
                        Spacer(minLength: 0)
                    }
                    .font(theme.ui(.small, weight: selected ? .semibold : nil))
                    .foregroundStyle(selected ? theme.accent : theme.secondaryText)
                    .padding(.horizontal, 8).padding(.vertical, 4)
                    .frame(minHeight: theme.sidebarRowHeight)
                    .background(selected ? theme.selection : theme.foreground.opacity(hovered == page ? 0.05 : 0),
                                in: RoundedRectangle(cornerRadius: 7))
                    .contentShape(Rectangle())
                }
                .buttonStyle(TSPressScaleStyle(scale: 0.98))
                .help(Text(LocalizedStringKey(page.title), bundle: .module))
                .accessibilityAddTraits(selected ? .isSelected : [])
                .onHover { hovered = $0 ? page : nil }
            }
        }
        .padding(.leading, 22)
        .padding(.bottom, 4)
    }
}

/// Shared by the compact rail, workspace header, and native View menu.
@MainActor
struct AudioDestinationMenu: View {
    @ObservedObject var model: AppModel

    var body: some View {
        ForEach(AudioWorkspacePage.destinations) { page in
            Button { model.openAudio(page) } label: {
                Label { Text(LocalizedStringKey(page.title), bundle: .module) }
                    icon: { Image(systemName: page.symbol) }
            }
        }
        Divider()
        Button { model.openAudio(.library) } label: {
            Label { Text("Audio library", bundle: .module) } icon: { Image(systemName: "books.vertical") }
        }
        ForEach(AudioWorkspacePage.tools) { page in
            Button { model.openAudio(page) } label: {
                Label { Text(LocalizedStringKey(page.title), bundle: .module) }
                    icon: { Image(systemName: page.symbol) }
            }
        }
    }
}
