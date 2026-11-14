import SwiftUI

/// OC-03: the Inbox rows of the chat sidebar. Sections reuse
/// `ChatSidebarChatRowView`, so the context menu (archive, delete behind the
/// existing confirmation) and the row's own busy guards are the ones the
/// Chats list has.
struct ChatSidebarInboxRows: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let layout: ChatInboxLayout
    @Binding var chatBeingRenamed: AppChat?
    @Binding var renameText: String
    @Binding var chatPendingDeletion: AppChat?
    @Binding var chatForSystemPrompt: AppChat?

    var body: some View {
        if layout.isEmpty {
            AppEmptyState(
                systemImage: "tray",
                title: layout.isSearching
                    ? Text("No matching chats", bundle: .module)
                    : Text("Nothing needs you", bundle: .module),
                message: layout.isSearching
                    ? Text("Try a different search term", bundle: .module)
                    : Text(
                        "Chats waiting on you, running, or with a new reply appear here.",
                        bundle: .module)
            ) {
                Button {
                    model.setChatSidebarMode(.chats)
                } label: {
                    Text("Show all chats", bundle: .module)
                }
                .appAction(.secondary, size: .small)
            }
            .padding(.vertical, 16)
        } else {
            ForEach(layout.groups) { group in
                sectionHeader(group)
                ForEach(group.chats) { chat in
                    ChatSidebarChatRowView(
                        model: model,
                        chat: chat,
                        chatBeingRenamed: $chatBeingRenamed,
                        renameText: $renameText,
                        chatPendingDeletion: $chatPendingDeletion,
                        chatForSystemPrompt: $chatForSystemPrompt)
                }
            }
        }
    }

    /// The count is a separate numeric Text (not interpolated into the
    /// localized title) so the title stays one catalog key; VoiceOver hears
    /// the title as a header and the count as its value.
    private func sectionHeader(_ group: ChatInboxLayout.Group) -> some View {
        HStack(spacing: 4) {
            Text(LocalizedStringKey(group.section.title), bundle: .module)
            Text(verbatim: group.chats.count.formatted())
        }
        .font(theme.ui(.tiny, weight: .semibold))
        .foregroundStyle(.appSecondary)
        .padding(.leading, 10)
        .padding(.top, 8)
        .textCase(nil)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text(LocalizedStringKey(group.section.title), bundle: .module))
        .accessibilityValue(Text(verbatim: group.chats.count.formatted()))
        .accessibilityAddTraits(.isHeader)
    }
}
