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
    /// Whether the running work pins THIS chat (ZP-23): its own turn, or work
    /// that is not a chat turn. Another chat's turn does not.
    private var isChatBusy: Bool { model.isChatBusyForRowActions(chat.id) }

    // Split out of `body` because Xcode 16.4 cannot type-check the whole
    // modifier chain in reasonable time. Explicit types keep each piece cheap.
    @ViewBuilder
    private func rowBackground(isSelected: Bool) -> some View {
        let hoverOpacity: Double = isHovered ? 0.05 : 0
        let fillColor: Color = isSelected ? theme.selection : theme.foreground.opacity(hoverOpacity)
        let shape = RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
        shape
            .fill(fillColor)
            .overlay {
                if isSelected {
                    shape.stroke(theme.accent.opacity(0.25), lineWidth: 1)
                }
            }
    }

    @ViewBuilder
    private func selectionBar(isSelected: Bool) -> some View {
        if isSelected {
            Capsule()
                .fill(theme.accent)
                .frame(width: 2.5)
                .padding(.vertical, 9)
        }
    }

    var body: some View {
        let isSelected = chat.id == model.selectedChatID
        let showsActions = isHovered || isSelected

        return HStack(spacing: 4) {
            selectButton(isSelected: isSelected)
            actionsMenu(showsActions: showsActions)
        }
        .background { rowBackground(isSelected: isSelected) }
        // The "you are here" mark. Full-opacity accent on a 2.5pt bar
        // survives any accent/background pairing the theme allows.
        .overlay(alignment: .leading) { selectionBar(isSelected: isSelected) }
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
            .disabled(isChatBusy || chat.isGhost)
            Button {
                chatForSystemPrompt = chat
            } label: { Text("Chat System Prompt", bundle: .module) }
            .disabled(isChatBusy)
            Button {
                _ = model.duplicateChat(id: chat.id)
            } label: { Text("Duplicate", bundle: .module) }
            .disabled(model.isRunning || chat.isGhost)
            Button {
                model.trajectoryPaneChatID = chat.id
            } label: { Text("View Trajectory", bundle: .module) }
            .disabled(chat.isGhost)
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
            // OC-04: the chat joins the session tab strip and becomes the
            // selected tab. The model refuses (with a toast) while the lane
            // verdict forbids the move.
            Button {
                model.openChatInSessionTab(id: chat.id)
            } label: { Text("Open in New Tab", bundle: .module) }
            .disabled(chat.isGhost || chat.isArchived)
            // OC-04: the same Recently Closed stack the Chat menu and the
            // strip's All Tabs menu show. Disabled while it is empty.
            Menu {
                ForEach(model.sessionTabs.closed.reversed(), id: \.chatID) { entry in
                    Button {
                        model.reopenClosedSessionTab(chatID: entry.chatID)
                    } label: {
                        Text(verbatim: model.sessionTabTitle(for: entry.chatID))
                    }
                }
            } label: { Text("Recently Closed", bundle: .module) }
            .disabled(model.sessionTabs.closed.isEmpty)
            Button(role: .destructive) {
                chatPendingDeletion = chat
            } label: { Text("Delete", bundle: .module) }
            .disabled(isChatBusy)
        }
        .accessibilityAction(named: (chat.isPinned ? Text("Unpin", bundle: .module) : Text("Pin", bundle: .module))) {
            model.setChatPinned(id: chat.id, pinned: !chat.isPinned)
        }
        .accessibilityAction(named: Text("Rename", bundle: .module)) {
            if !isChatBusy && !chat.isGhost {
                renameText = chat.title
                chatBeingRenamed = chat
            }
        }
        .accessibilityAction(named: Text("Chat System Prompt", bundle: .module)) {
            if !isChatBusy {
                chatForSystemPrompt = chat
            }
        }
        .accessibilityAction(named: Text("View Trajectory", bundle: .module)) {
            if !chat.isGhost { model.trajectoryPaneChatID = chat.id }
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
            if !isChatBusy {
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
                stateBadge
            }
            .padding(.leading, 9)
            .padding(.vertical, 8)
            .contentShape(.rect)
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.985))
        .disabled(!model.canSelectChat(id: chat.id))
        .help(!model.canSelectChat(id: chat.id)
            ? Text("Cannot switch chats while generation is running", bundle: .module)
            : Text("Switch to \(chat.title)", bundle: .module))
        .accessibilityLabel(chat.title)
        .accessibilityValue(stateAccessibilityValue)
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
        } else if model.chatTurnState(for: chat.id) == .generating {
            TaskProgressFlameIcon(size: 13)
        } else {
            Image(systemName: isSelected ? "bubble.left.fill" : "bubble.left")
                .font(theme.ui(.tiny))
                .foregroundStyle(isSelected ? theme.accent : theme.secondaryText)
                .contentTransition(.symbolEffect(.replace))
                .accessibilityHidden(true)
        }
    }

    private var slot: ChatRowSlot { model.chatRowSlot(for: chat) }

    /// The trailing half of the row's one status slot (ZP-02 on the ZP-23
    /// turn state). Pending pill, else unread dot, else the queued clock or
    /// the selected row's hand glyph. Each carries text (the pill) or help
    /// and a VoiceOver value (the dot, the glyphs), never color alone.
    /// Generating is the animated flame in the icon slot, so it has no badge.
    @ViewBuilder
    private var stateBadge: some View {
        switch slot {
        case .pending:
            HStack(spacing: 3) {
                Image(systemName: "hand.raised.fill")
                Text("Needs Approval", bundle: .module)
                    .lineLimit(1)
            }
            .font(theme.ui(.micro, weight: .semibold))
            .foregroundStyle(AppAttention.color)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .overlay(Capsule().stroke(AppAttention.color, lineWidth: 1))
            .fixedSize()
            .padding(.trailing, 4)
            .help(Text("Needs Approval", bundle: .module))
            .accessibilityHidden(true)
        case .unread:
            Circle()
                .fill(theme.accent)
                .frame(width: 8, height: 8)
                .padding(.trailing, 6)
                .help(Text("Unread", bundle: .module))
                .accessibilityHidden(true)
        case .running, .normal:
            switch model.chatTurnState(for: chat.id) {
            case .awaitingTool, .awaitingQuestion:
                // The selected row: the pill is for rows the user is not on.
                Image(systemName: "hand.raised.fill")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(AppAttention.color)
                    .help(Text("Needs Approval", bundle: .module))
                    .accessibilityHidden(true)
            case .queued:
                Image(systemName: "clock")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
                    .help(Text("Queued", bundle: .module))
                    .accessibilityHidden(true)
            case .generating, .idle:
                EmptyView()
            }
        }
    }

    private var stateAccessibilityValue: Text {
        // The same slot the eye reads, so VoiceOver and the dot agree.
        if slot == .unread {
            return chat.preview.isEmpty
                ? Text("Unread", bundle: .module)
                : Text("Unread: \(chat.preview)", bundle: .module)
        }
        switch model.chatTurnState(for: chat.id) {
        case .generating: return Text("Generating", bundle: .module)
        case .awaitingTool, .awaitingQuestion: return Text("Needs Approval", bundle: .module)
        case .queued: return Text("Queued", bundle: .module)
        case .idle: return Text(verbatim: chat.preview)
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
        .disabled(isChatBusy)
        .padding(.trailing, 5)
    }
}
