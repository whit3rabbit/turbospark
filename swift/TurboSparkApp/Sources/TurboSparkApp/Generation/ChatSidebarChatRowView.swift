import AppKit
import SwiftUI

/// Single chat row representation in the sidebar with hover actions and contextual actions.
struct ChatSidebarChatRowView: View {
    @ObservedObject var model: AppModel
    let chat: AppChat
    @Binding var chatBeingRenamed: AppChat?
    @Binding var renameText: String
    @Binding var chatPendingDeletion: AppChat?
    @State private var hoveredChatID: UUID?

    @ScaledMetric private var actionButtonSize: CGFloat = 26

    var body: some View {
        let isSelected = chat.id == model.selectedChatID
        let showsActions = hoveredChatID == chat.id || isSelected

        return HStack(spacing: 4) {
            Button {
                model.selectChat(id: chat.id)
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isSelected ? "bubble.left.fill" : "bubble.left")
                        .font(.caption)
                        .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(chat.title)
                            .font(.callout.weight(isSelected ? .semibold : .regular))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        if !chat.preview.isEmpty {
                            Text(chat.preview)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    Spacer(minLength: 0)
                }
                .padding(.leading, 9)
                .padding(.vertical, 8)
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .disabled(model.isRunning && !isSelected)
            .help("Switch to \(chat.title)")
            .accessibilityLabel(chat.title)
            .accessibilityValue(chat.preview.isEmpty
                                ? (isSelected ? "Selected" : "")
                                : "\(chat.preview)\(isSelected ? ", selected" : "")")
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)

            Menu {
                Button("Rename", systemImage: "pencil") {
                    renameText = chat.title
                    chatBeingRenamed = chat
                }
                Divider()
                Button("Delete", systemImage: "trash", role: .destructive) {
                    chatPendingDeletion = chat
                }
            } label: {
                Label("Chat actions", systemImage: "ellipsis")
                    .labelStyle(.iconOnly)
                    .frame(width: actionButtonSize, height: actionButtonSize)
                    .contentShape(Circle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Chat actions for \(chat.title)")
            // Always reachable from VoiceOver, even though sighted users
            // only see the button when hovering or when the row is selected.
            .accessibilityLabel("Chat actions for \(chat.title)")
            .opacity(showsActions ? 1 : 0)
            .disabled(model.isRunning)
            .padding(.trailing, 5)
        }
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: .rect(cornerRadius: 9))
        .contentShape(.rect(cornerRadius: 9))
        .onHover { isHovering in
            hoveredChatID = isHovering ? chat.id : nil
        }
        .contextMenu {
            Button("Rename") {
                renameText = chat.title
                chatBeingRenamed = chat
            }
            .disabled(model.isRunning)
            Button("Delete", role: .destructive) {
                chatPendingDeletion = chat
            }
            .disabled(model.isRunning)
        }
    }
}
