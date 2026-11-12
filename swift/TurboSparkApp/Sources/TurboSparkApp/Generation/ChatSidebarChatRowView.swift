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
    private var actionButtonSize: CGFloat { theme.sidebarActionSize }

    var body: some View {
        let isSelected = chat.id == model.selectedChatID
        let showsActions = isHovered || isSelected

        return HStack(spacing: 4) {
            selectButton(isSelected: isSelected)
            actionsMenu(showsActions: showsActions)
        }
        .background {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .fill(
                    isSelected
                        ? theme.selection
                        : theme.foreground.opacity(isHovered ? 0.05 : 0))
                .overlay {
                    if isSelected {
                        RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                            .stroke(theme.accent.opacity(0.25), lineWidth: 1)
                    }
                }
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
        .contentShape(.rect(cornerRadius: AppMetrics.cardRadius))
        .tsAnimation(TSMotion.hover, value: isHovered)
        .tsAnimation(TSMotion.select, value: isSelected)
        .onHover { hovering in
            isHovered = hovering
        }
        .contextMenu {
            Button {
                model.setChatPinned(id: chat.id, pinned: !chat.isPinned)
            } label: { (chat.isPinned ? Text("Unpin", bundle: .module) : Text("Pin", bundle: .module)) }
            Button {
                renameText = chat.title
                chatBeingRenamed = chat
            } label: { Text("Rename", bundle: .module) }
            .disabled(model.isRunning || chat.isGhost)
            Button {
                chatForSystemPrompt = chat
            } label: { Text("Chat System Prompt", bundle: .module) }
            .disabled(model.isRunning)
            Button {
                _ = model.duplicateChat(id: chat.id)
            } label: { Text("Duplicate", bundle: .module) }
            .disabled(model.isRunning || chat.isGhost)
            Button {
                model.setChatArchived(id: chat.id, archived: !chat.isArchived)
            } label: { (chat.isArchived ? Text("Unarchive", bundle: .module) : Text("Archive", bundle: .module)) }
            .disabled(chat.isGhost)
            // The qwen-code "open in split view" action: the chat joins the
            // pane column without leaving the list. The selected chat is
            // refused by the model (a pane would mirror the main column),
            // so the item is disabled rather than a silent no-op.
            Button {
                model.toggleSplitPane(chatID: chat.id)
            } label: { (model.isSplitPane(chatID: chat.id) ? Text("Close Split Pane", bundle: .module) : Text("Open in Split View", bundle: .module)) }
            .disabled(chat.isGhost || model.isRunning || chat.id == model.selectedChatID)
            Button(role: .destructive) {
                chatPendingDeletion = chat
            } label: { Text("Delete", bundle: .module) }
            .disabled(model.isRunning)
        }
        .accessibilityAction(named: (chat.isPinned ? Text("Unpin", bundle: .module) : Text("Pin", bundle: .module))) {
            model.setChatPinned(id: chat.id, pinned: !chat.isPinned)
        }
        .accessibilityAction(named: Text("Rename", bundle: .module)) {
            if !model.isRunning && !chat.isGhost {
                renameText = chat.title
                chatBeingRenamed = chat
            }
        }
        .accessibilityAction(named: Text("Chat System Prompt", bundle: .module)) {
            if !model.isRunning {
                chatForSystemPrompt = chat
            }
        }
        .accessibilityAction(named: Text("Duplicate", bundle: .module)) {
            if !model.isRunning && !chat.isGhost {
                _ = model.duplicateChat(id: chat.id)
            }
        }
        .accessibilityAction(named: (chat.isArchived ? Text("Unarchive", bundle: .module) : Text("Archive", bundle: .module))) {
            if !chat.isGhost {
                model.setChatArchived(id: chat.id, archived: !chat.isArchived)
            }
        }
        .accessibilityAction(named: (model.isSplitPane(chatID: chat.id) ? Text("Close Split Pane", bundle: .module) : Text("Open in Split View", bundle: .module))) {
            if !chat.isGhost && !model.isRunning && chat.id != model.selectedChatID {
                model.toggleSplitPane(chatID: chat.id)
            }
        }
        .accessibilityAction(named: Text("Delete", bundle: .module)) {
            if !model.isRunning {
                chatPendingDeletion = chat
            }
        }
    }

    /// The chat's project accent, or nil for a projectless chat. Resolved
    /// off the chat's own projectID so a row never depends on which project
    /// is currently selected.
    private var projectAccent: Color? {
        guard chat.projectID != nil else { return nil }
        return ProjectAccentColor.forProject(id: chat.projectID).color
    }

    private func selectButton(isSelected: Bool) -> some View {
        Button {
            TSMotion.animate(TSMotion.select) {
                model.selectChat(id: chat.id)
            }
        } label: {
            HStack(spacing: 8) {
                rowIcon(isSelected: isSelected)

                VStack(alignment: .leading, spacing: 2) {
                    HStack(spacing: 4) {
                        Text(chat.title)
                            .font(theme.ui(.small, weight: isSelected ? .semibold : nil))
                            .foregroundStyle(.appText)
                            .lineLimit(1)
                        // The qwen-code workspace accent: one deterministic
                        // dot per project, so chats from different roots
                        // read apart without reading any words.
                        if let accent = projectAccent {
                            Circle()
                                .fill(accent)
                                .frame(width: 6, height: 6)
                                .help(Text("Projects", bundle: .module))
                                .accessibilityHidden(true)
                        }
                    }
                    if !chat.preview.isEmpty {
                        Text(chat.preview)
                            .font(theme.ui(.tiny))
                            .foregroundStyle(.appSecondary)
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
        .help(model.isRunning && !isSelected
            ? Text("Cannot switch chats while generation is running", bundle: .module)
            : Text("Switch to \(chat.title)", bundle: .module))
        .accessibilityLabel(chat.title)
        .accessibilityValue(Text(verbatim: chat.preview))
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
            GhostGlyph(size: 13, color: theme.accent)
                .accessibilityHidden(true)
        } else if chat.isPinned {
            Image(systemName: "pin.fill")
                .font(theme.ui(.tiny))
                .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                .accessibilityHidden(true)
        } else if isSelected && model.isRunning {
            TaskProgressFlameIcon(size: 13)
        } else {
            Image(systemName: isSelected ? "bubble.left.fill" : "bubble.left")
                .font(theme.ui(.tiny))
                .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                .contentTransition(.symbolEffect(.replace))
                .accessibilityHidden(true)
        }
    }

    private func actionsMenu(showsActions: Bool) -> some View {
        Menu {
            Button {
                model.setChatPinned(id: chat.id, pinned: !chat.isPinned)
            } label: {
                if chat.isPinned {
                    Label { Text("Unpin", bundle: .module) } icon: { Image(systemName: "pin.slash") }
                } else {
                    Label { Text("Pin", bundle: .module) } icon: { Image(systemName: "pin") }
                }
            }
            // A ghost chat's title is a fixed "Temporary Chat"
            // (`AppModel+Submission.swift` skips auto-titling it for exactly
            // this reason); Rename is disabled here so that promise holds
            // for a manual override too.
            Button {
                renameText = chat.title
                chatBeingRenamed = chat
            } label: {
                Label { Text("Rename", bundle: .module) } icon: { Image(systemName: "pencil") }
            }
            .disabled(chat.isGhost)
            Divider()
            Button(role: .destructive) {
                chatPendingDeletion = chat
            } label: {
                Label { Text("Delete", bundle: .module) } icon: { Image(systemName: "trash") }
            }
        } label: {
            Label { Text("Chat actions", bundle: .module) } icon: { Image(systemName: "ellipsis") }
                .labelStyle(.iconOnly)
                .frame(width: actionButtonSize, height: actionButtonSize)
                .contentShape(Circle())
        }
        .menuStyle(.borderlessButton)
        .menuIndicator(.hidden)
        .fixedSize()
        .help(Text("Chat actions for \(chat.title)", bundle: .module))
        // Always reachable from VoiceOver, even though sighted users only
        // see the button when hovering or when the row is selected.
        .accessibilityLabel(Text("Chat actions for \(chat.title)", bundle: .module))
        .opacity(showsActions ? 1 : 0)
        .scaleEffect(showsActions ? 1 : 0.85)
        .disabled(model.isRunning)
        .padding(.trailing, 5)
    }
}
