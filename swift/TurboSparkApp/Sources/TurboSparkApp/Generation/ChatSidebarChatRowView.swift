import AppKit
import SwiftUI

/// Single chat row in the sidebar, with hover actions and a context menu.
///
/// Redesign notes:
///
/// - Selection gains a leading accent bar on top of the tint. The tint alone
///   (`accentColor.opacity(0.12)`) disappears entirely when the accent is
///   near-white on a near-black window, which is exactly the shipped dark
///   default -- the selected chat was effectively unmarked.
/// - Hover now has its own fill. Previously hovering only revealed the "..."
///   button, so the row gave no feedback until the pointer reached the very
///   right edge of it.
/// - The "..." button fades and scales in rather than switching opacity, and
///   its animation is driven off one `isHovered` bool instead of the old
///   `hoveredChatID` (a per-row `@State` holding another row's id, which
///   could never be anything but this row or nil).
struct ChatSidebarChatRowView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let chat: AppChat
    @Binding var chatBeingRenamed: AppChat?
    @Binding var renameText: String
    @Binding var chatPendingDeletion: AppChat?
    @Binding var chatForSystemPrompt: AppChat?

    @State private var isHovered = false
    @ScaledMetric private var actionButtonSize: CGFloat = 26

    var body: some View {
        let isSelected = chat.id == model.selectedChatID
        let showsActions = isHovered || isSelected

        return HStack(spacing: 4) {
            selectButton(isSelected: isSelected)
            actionsMenu(showsActions: showsActions)
        }
        .background {
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .fill(
                    isSelected
                        ? theme.accent.opacity(0.13)
                        : Color.primary.opacity(isHovered ? 0.05 : 0))
        }
        // The "you are here" mark. Full-opacity accent on a 2.5pt bar
        // survives any accent/background pairing the theme allows.
        .overlay(alignment: .leading) {
            if isSelected {
                Capsule()
                    .fill(theme.accent)
                    .frame(width: 2.5)
                    .padding(.vertical, 9)
            }
        }
        .contentShape(.rect(cornerRadius: 9))
        .animation(TSMotion.hover, value: isHovered)
        .animation(TSMotion.select, value: isSelected)
        .onHover { hovering in
            isHovered = hovering
        }
        .contextMenu {
            Button(chat.isPinned ? "Unpin" : "Pin") {
                model.setChatPinned(id: chat.id, pinned: !chat.isPinned)
            }
            Button("Rename") {
                renameText = chat.title
                chatBeingRenamed = chat
            }
            .disabled(model.isRunning || chat.isGhost)
            Button(chat.systemPrompt == nil ? "Set System Prompt" : "Edit System Prompt") {
                chatForSystemPrompt = chat
            }
            .disabled(model.isRunning)
            Button("Duplicate") {
                _ = model.duplicateChat(id: chat.id)
            }
            .disabled(model.isRunning || chat.isGhost)
            Button(chat.isArchived ? "Unarchive" : "Archive") {
                model.setChatArchived(id: chat.id, archived: !chat.isArchived)
            }
            .disabled(chat.isGhost)
            Button("Delete", role: .destructive) {
                chatPendingDeletion = chat
            }
            .disabled(model.isRunning)
        }
    }

    private func selectButton(isSelected: Bool) -> some View {
        Button {
            withAnimation(TSMotion.select) {
                model.selectChat(id: chat.id)
            }
        } label: {
            HStack(spacing: 8) {
                rowIcon(isSelected: isSelected)

                VStack(alignment: .leading, spacing: 2) {
                    Text(chat.title)
                        .font(theme.ui(points: 12.5, weight: isSelected ? .semibold : .regular))
                        .foregroundStyle(.primary)
                        .lineLimit(1)
                    if !chat.preview.isEmpty {
                        Text(chat.preview)
                            .font(theme.ui(points: 11))
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                            .truncationMode(.tail)
                    }
                }
                Spacer(minLength: 0)
            }
            .padding(.leading, 9)
            .padding(.vertical, 8)
            .contentShape(.rect)
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.985))
        .disabled(model.isRunning && !isSelected)
        .help("Switch to \(chat.title)")
        .accessibilityLabel(chat.title)
        .accessibilityValue(chat.preview.isEmpty
                            ? (isSelected ? "Selected" : "")
                            : "\(chat.preview)\(isSelected ? ", selected" : "")")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }

    /// Ghost badge INSTEAD of the bubble icon, not beside it: the icon is
    /// what marks the row temporary, and doubling it would just spend
    /// sidebar width. A pinned chat reads the same way: the pin replaces
    /// the bubble, because the row has one icon slot and "this one is
    /// held" outranks "this is a chat".
    @ViewBuilder
    private func rowIcon(isSelected: Bool) -> some View {
        if chat.isGhost {
            Image(systemName: "ghost")
                .font(theme.ui(points: 11))
                .foregroundStyle(theme.accent)
                .accessibilityHidden(true)
        } else if chat.isPinned {
            Image(systemName: "pin.fill")
                .font(theme.ui(points: 10))
                .foregroundStyle(isSelected ? theme.accent : Color.secondary)
                .accessibilityHidden(true)
        } else if isSelected && model.isRunning {
            TaskProgressFlameIcon(size: 13)
        } else {
            Image(systemName: isSelected ? "bubble.left.fill" : "bubble.left")
                .font(theme.ui(points: 11))
                .foregroundStyle(isSelected ? theme.accent : Color.secondary)
                .contentTransition(.symbolEffect(.replace))
                .accessibilityHidden(true)
        }
    }

    private func actionsMenu(showsActions: Bool) -> some View {
        Menu {
            Button(chat.isPinned ? "Unpin" : "Pin", systemImage: chat.isPinned ? "pin.slash" : "pin") {
                model.setChatPinned(id: chat.id, pinned: !chat.isPinned)
            }
            // A ghost chat's title is a fixed "Temporary Chat"
            // (`AppModel+Submission.swift` skips auto-titling it for exactly
            // this reason); Rename is disabled here so that promise holds
            // for a manual override too.
            Button("Rename", systemImage: "pencil") {
                renameText = chat.title
                chatBeingRenamed = chat
            }
            .disabled(chat.isGhost)
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
        // Always reachable from VoiceOver, even though sighted users only
        // see the button when hovering or when the row is selected.
        .accessibilityLabel("Chat actions for \(chat.title)")
        .opacity(showsActions ? 1 : 0)
        .scaleEffect(showsActions ? 1 : 0.85)
        .disabled(model.isRunning)
        .padding(.trailing, 5)
    }
}
