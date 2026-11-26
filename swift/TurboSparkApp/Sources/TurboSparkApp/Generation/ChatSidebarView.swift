import AppKit
import SwiftUI

// Isolated explicitly: only body is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The unified Chat & Projects sidebar panel.
///
/// Layout (inspired by Claude Desktop):
/// - Compact top bar: Chat/Projects switcher, collapsible Search icon, and
///   temporary-chat toggle.
/// - Expandable inline filter field (hidden until the Search icon is clicked).
/// - Primary "+ New chat" action button.
/// - Labeled workspace navigation (`Images`, `Audio`, `Installed`, `More`) so
///   every destination icon says what it does without a separate icon rail.
/// - Clean, high-density Recents list (with a compact Inbox toggle) in Chat
///   mode, or collapsible Project groups in Projects mode.
@MainActor
struct ChatSidebarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var searchText = ""
    @State private var isSearchVisible = false
    @FocusState private var isSearchFocused: Bool
    @State private var isRecentsExpanded = true
    @State private var chatBeingRenamed: AppChat?
    @State private var chatForSystemPrompt: AppChat?
    @State private var renameText = ""
    @State private var chatPendingDeletion: AppChat?
    @State private var showingProjectSettingsSheet = false
    @State private var projectBeingEdited: AppProject?
    @State private var projectForMcpSettings: AppProject?

    var body: some View {
        VStack(spacing: 0) {
            topActionRow
                .padding(.horizontal, 10)
                .padding(.top, 6)
                .padding(.bottom, 6)

            if isSearchVisible || !searchText.isEmpty {
                filterBar
                    .padding(.horizontal, 10)
                    .padding(.bottom, 6)
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            if model.interactionMode == .chat {
                newChatButton
                    .padding(.horizontal, 10)
                    .padding(.bottom, 4)
            }

            SidebarNavigationList(model: model, includesChat: false)

            SidebarDivider()

            contentView
        }
        .tsAnimation(TSMotion.hover, value: isSearchVisible)
        .sheet(isPresented: $showingProjectSettingsSheet) {
            ProjectSettingsSheet(
                model: model,
                editingProject: projectBeingEdited,
                onDismiss: {
                    showingProjectSettingsSheet = false
                    projectBeingEdited = nil
                }
            )
        }
        .sheet(item: $projectForMcpSettings) { proj in
            ProjectMcpSettingsSheet(
                model: model,
                projectID: proj.id,
                onDismiss: {
                    projectForMcpSettings = nil
                }
            )
        }
        .sheet(isPresented: Binding(
            get: { model.trajectoryPaneChatID != nil },
            set: { if !$0 { model.trajectoryPaneChatID = nil } })) {
            if let chatID = model.trajectoryPaneChatID {
                TrajectoryPaneView(
                    model: model, chatID: chatID,
                    onDismiss: { model.trajectoryPaneChatID = nil })
            }
        }
        .sheet(item: $chatForSystemPrompt) { chat in
            ChatSystemPromptSheet(
                model: model,
                chatID: chat.id,
                onDismiss: { chatForSystemPrompt = nil }
            )
        }
        .alert(
            Text("Rename", bundle: .module),
            isPresented: renameAlertPresented,
            presenting: chatBeingRenamed
        ) { chat in
            TextField(text: $renameText) { Text("Chat name", bundle: .module) }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
            Button {
                model.renameChat(id: chat.id, title: renameText)
            } label: { Text("Rename", bundle: .module) }
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        } message: { _ in
            Text("Choose a name that identifies this chat.", bundle: .module)
        }
        .alert(
            Text("Delete", bundle: .module),
            isPresented: deletionAlertPresented,
            presenting: chatPendingDeletion
        ) { chat in
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
            Button(role: .destructive) {
                model.deleteChat(id: chat.id)
            } label: { Text("Delete", bundle: .module) }
        } message: { chat in
            Text("\"\(chat.title)\" and its conversation history will be removed.", bundle: .module)
        }
    }

    // MARK: - Top Action Row

    private var topActionRow: some View {
        HStack(spacing: 4) {
            tabSelector
            searchToggleButton
            GhostChatButton(model: model)
        }
    }

    private var tabSelector: some View {
        HStack(spacing: 2) {
            ForEach(AppModel.AppInteractionMode.allCases) { mode in
                let isSelected = model.interactionMode == mode
                Button {
                    TSMotion.animate(TSMotion.select) {
                        model.setInteractionMode(mode)
                    }
                } label: {
                    HStack(spacing: 4) {
                        Image(systemName: mode.systemImage)
                            .font(theme.ui(.tiny, weight: .semibold))
                        Text(LocalizedStringKey(mode.title), bundle: .module)
                            .font(theme.ui(.tiny, weight: isSelected ? .semibold : .medium))
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
                    .frame(minHeight: theme.sidebarRowHeight - 6)
                    .foregroundStyle(isSelected ? theme.foreground : theme.secondaryText)
                    .background(
                        isSelected ? theme.selection : Color.clear,
                        in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius - 1, style: .continuous)
                    )
                    .contentShape(.rect(cornerRadius: AppMetrics.controlRadius - 1))
                }
                .buttonStyle(.plain)
                .help(Text(LocalizedStringKey(mode.title), bundle: .module))
                .accessibilityLabel(Text(LocalizedStringKey(mode.title), bundle: .module))
                .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
            }
        }
        .padding(2)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5)
        )
    }

    private var searchToggleButton: some View {
        let isActive = isSearchVisible || !searchText.isEmpty
        return AppIconButton(
            "magnifyingglass",
            label: Text("Search", bundle: .module)
        ) {
            TSMotion.animate(TSMotion.hover) {
                if isActive {
                    searchText = ""
                    isSearchVisible = false
                    isSearchFocused = false
                } else {
                    isSearchVisible = true
                    isSearchFocused = true
                }
            }
        }
        .buttonStyle(.plain)
        .font(theme.ui(.small, weight: .medium))
        .foregroundStyle(isActive ? theme.accentText : theme.secondaryText)
        .frame(width: theme.sidebarRowHeight - 2, height: theme.sidebarRowHeight - 2)
        .background {
            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                .fill(isActive ? theme.selection : Color.clear)
        }
        .contentShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .accessibilityAddTraits(isActive ? .isSelected : [])
    }

    // MARK: - Filter Bar (Hidden Behind Search Icon)

    private var filterBar: some View {
        AppSearchField(
            text: $searchText,
            prompt: model.interactionMode == .projects
                ? Text("Filter projects and tasks...", bundle: .module)
                : Text("Filter chats...", bundle: .module),
            minWidth: 100,
            idealWidth: 220,
            maxWidth: .infinity,
            focus: $isSearchFocused
        )
        .onExitCommand {
            TSMotion.animate(TSMotion.hover) {
                searchText = ""
                isSearchVisible = false
                isSearchFocused = false
            }
        }
    }

    // MARK: - Content View

    @ViewBuilder
    private var contentView: some View {
        switch model.interactionMode {
        case .projects:
            projectsContent
        case .chat:
            chatContent
        }
    }

    private var projectsContent: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Projects", bundle: .module)
                    .font(theme.ui(.tiny, weight: .semibold))
                    .foregroundStyle(.appSecondary)
                    .accessibilityAddTraits(.isHeader)

                Spacer()

                AppIconButton("plus", label: Text("Add Project", bundle: .module)) {
                    projectBeingEdited = nil
                    showingProjectSettingsSheet = true
                }
                .buttonStyle(.plain)
                .font(theme.ui(.tiny, weight: .semibold))
                .foregroundStyle(.appSecondary)
                .frame(width: theme.sidebarActionSize, height: theme.sidebarActionSize)
                .contentShape(Circle())
            }
            .padding(.horizontal, 14)
            .padding(.top, 8)
            .padding(.bottom, 2)

            ChatSidebarGroupedProjectsView(
                model: model,
                searchText: searchText,
                showingProjectSettingsSheet: $showingProjectSettingsSheet,
                projectBeingEdited: $projectBeingEdited,
                projectForMcpSettings: $projectForMcpSettings,
                chatBeingRenamed: $chatBeingRenamed,
                renameText: $renameText,
                chatPendingDeletion: $chatPendingDeletion,
                chatForSystemPrompt: $chatForSystemPrompt
            )
        }
    }

    private var chatContent: some View {
        VStack(spacing: 0) {
            if let activeProject = model.selectedProject {
                activeProjectScopeBar(activeProject)
                    .padding(.horizontal, 10)
                    .padding(.top, 8)
            }

            recentsHeaderRow
                .padding(.horizontal, 12)
                .padding(.top, 8)
                .padding(.bottom, 4)

            if isRecentsExpanded || model.chatSidebarMode == .inbox || !searchText.isEmpty {
                chatList
            } else {
                Spacer(minLength: 0)
            }
        }
    }

    /// Collapsible "Recents" heading on the leading side, and a compact quiet
    /// Inbox toggle on the trailing side (replacing the full-width Chats/Inbox
    /// pill bar).
    private var recentsHeaderRow: some View {
        let isInbox = model.chatSidebarMode == .inbox
        return HStack(spacing: 6) {
            Button {
                TSMotion.animate(TSMotion.select) {
                    if isInbox {
                        model.setChatSidebarMode(.chats)
                        isRecentsExpanded = true
                    } else {
                        isRecentsExpanded.toggle()
                    }
                }
            } label: {
                HStack(spacing: 4) {
                    Text("Recents", bundle: .module)
                        .font(theme.ui(.tiny, weight: .semibold))
                        .foregroundStyle(isInbox ? theme.secondaryText : theme.secondaryText)
                    Image(systemName: (!isInbox && isRecentsExpanded) ? "chevron.down" : "chevron.right")
                        .font(theme.ui(.micro, weight: .semibold))
                        .foregroundStyle(.appSecondary)
                        .accessibilityHidden(true)
                }
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .accessibilityAddTraits(.isHeader)

            Spacer(minLength: 4)

            Button {
                TSMotion.animate(TSMotion.select) {
                    model.setChatSidebarMode(isInbox ? .chats : .inbox)
                }
            } label: {
                HStack(spacing: 4) {
                    Image(systemName: isInbox ? "tray.fill" : "tray")
                        .font(theme.ui(.micro, weight: .medium))
                        .accessibilityHidden(true)
                    Text("Inbox", bundle: .module)
                        .font(theme.ui(.tiny, weight: isInbox ? .semibold : .medium))
                }
                .foregroundStyle(isInbox ? theme.accentText : theme.secondaryText)
                .padding(.horizontal, 6)
                .padding(.vertical, 2)
                .background(
                    isInbox ? theme.selection : Color.clear,
                    in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius - 1, style: .continuous)
                )
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .help(Text("Inbox", bundle: .module))
            .accessibilityLabel(Text("Inbox", bundle: .module))
            .accessibilityAddTraits(isInbox ? [.isButton, .isSelected] : .isButton)
        }
    }

    private func activeProjectScopeBar(_ project: AppProject) -> some View {
        HStack(spacing: 5) {
            Image(systemName: "folder.fill")
                .font(theme.ui(.tiny))
                .foregroundStyle(.appAccent)

            Text("Project: \(project.name)", bundle: .module)
                .font(theme.ui(.tiny, weight: .medium))
                .foregroundStyle(.appText)
                .lineLimit(1)

            Spacer()

            Button {
                model.selectProject(id: nil)
            } label: {
                HStack(spacing: 3) {
                    Text("All", bundle: .module)
                        .font(theme.ui(.tiny))
                    Image(systemName: "xmark")
                        .font(theme.ui(.micro))
                }
                .foregroundStyle(.appSecondary)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(.appSecondary.opacity(0.15), in: Capsule())
            }
            .buttonStyle(.plain)
            .help(Text("All Chats", bundle: .module))
            .accessibilityLabel(Text("All Chats", bundle: .module))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.appElevated, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
    }

    private var newChatButton: some View {
        Button {
            model.createChat()
        } label: {
            ViewThatFits(in: .horizontal) {
                HStack(spacing: 8) {
                    newChatLabel.fixedSize()
                    Spacer(minLength: 8)
                    Text(verbatim: "Cmd+N")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.appSecondary)
                        .fixedSize()
                        .accessibilityHidden(true)
                }
                newChatLabel
            }
        }
        .buttonStyle(SidebarActionButtonStyle())
        .disabled(!model.canCreateChat)
        .help(!model.canCreateChat
            ? Text("Cannot start a new chat while generation is running", bundle: .module)
            : (Text("New chat", bundle: .module) + Text(verbatim: " (Cmd+N)")))
    }

    private var newChatLabel: some View {
        Label {
            Text("New chat", bundle: .module)
                .font(theme.ui(.small, weight: .medium))
        } icon: {
            Image(systemName: "plus")
                .font(theme.ui(.small, weight: .medium))
                .accessibilityHidden(true)
        }
    }

    private var chatList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 1) {
                if model.chatSidebarMode == .inbox {
                    inboxRows
                } else {
                    chatsRows
                }
                archivedSection
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// OC-03: the Inbox grouping. Derived once per body, like the Chats
    /// layout, from the same scoped and sorted history.
    @ViewBuilder
    private var inboxRows: some View {
        let layout = ChatInboxLayout.make(
            chats: historyChats,
            searchText: searchText,
            selectedID: model.selectedChatID,
            retainedID: model.inboxRetainedChatID,
            turnState: { model.chatTurnState(for: $0) })
        ChatSidebarInboxRows(
            model: model,
            layout: layout,
            chatBeingRenamed: $chatBeingRenamed,
            renameText: $renameText,
            chatPendingDeletion: $chatPendingDeletion,
            chatForSystemPrompt: $chatForSystemPrompt)
    }

    @ViewBuilder
    private var chatsRows: some View {
        // Sorted, filtered and bucketed ONCE per body.
        let layout = ChatSidebarLayout.make(
            sortedHistory: historyChats, searchText: searchText)
        if layout.isEmpty {
            AppEmptyState(
                systemImage: "bubble.left.and.bubble.right",
                title: searchText.isEmpty ? Text("No chats yet", bundle: .module) : Text("No matching chats", bundle: .module),
                message: searchText.isEmpty ? Text("Start a conversation to see history here", bundle: .module) : Text("Try a different search term", bundle: .module)
            ) {
                if searchText.isEmpty {
                    Button {
                        model.createChat()
                    } label: {
                        Text("New chat", bundle: .module)
                    }
                    .appAction(.secondary, size: .small)
                    .disabled(model.isRunning)
                }
            }
            .padding(.vertical, 16)
        } else if !layout.isSearching {
            ForEach(layout.pinned) { chat in
                ChatSidebarChatRowView(
                    model: model,
                    chat: chat,
                    chatBeingRenamed: $chatBeingRenamed,
                    renameText: $renameText,
                    chatPendingDeletion: $chatPendingDeletion,
                    chatForSystemPrompt: $chatForSystemPrompt
                )
            }
            ForEach(layout.groups, id: \.bucket) { group in
                Text(LocalizedStringKey(group.bucket.label), bundle: .module)
                    .font(theme.ui(.micro, weight: .semibold))
                    .foregroundStyle(.appSecondary)
                    .padding(.leading, 10)
                    .padding(.top, 8)
                    .padding(.bottom, 2)
                    .textCase(nil)
                    .accessibilityAddTraits(.isHeader)
                ForEach(group.chats) { chat in
                    ChatSidebarChatRowView(
                        model: model,
                        chat: chat,
                        chatBeingRenamed: $chatBeingRenamed,
                        renameText: $renameText,
                        chatPendingDeletion: $chatPendingDeletion,
                        chatForSystemPrompt: $chatForSystemPrompt
                    )
                }
            }
        } else {
            ForEach(layout.matches) { chat in
                ChatSidebarChatRowView(
                    model: model,
                    chat: chat,
                    chatBeingRenamed: $chatBeingRenamed,
                    renameText: $renameText,
                    chatPendingDeletion: $chatPendingDeletion,
                    chatForSystemPrompt: $chatForSystemPrompt
                )
            }
        }
    }

    /// Archived chats live behind one disclosure, each row offering Restore
    /// and confirmed Delete.
    @ViewBuilder
    private var archivedSection: some View {
        let archived = model.archivedChats
        if !archived.isEmpty {
            DisclosureGroup {
                ForEach(archived) { chat in
                    HStack(spacing: 8) {
                        HStack(spacing: 8) {
                            Image(systemName: "archivebox")
                                .font(theme.ui(.tiny))
                                .foregroundStyle(.appSecondary)
                                .accessibilityHidden(true)
                            Text(chat.title)
                                .font(theme.ui(.small))
                                .lineLimit(1)
                                .truncationMode(.tail)
                        }
                        .accessibilityElement(children: .combine)

                        Spacer(minLength: 4)
                        Button {
                            model.setChatArchived(id: chat.id, archived: false)
                        } label: {
                            Text("Restore", bundle: .module)
                                .font(theme.ui(.tiny, weight: .medium))
                        }
                        .appAction(.quiet, size: .small)
                        .help(Text("Restore", bundle: .module))
                        // Asks first: this only stages the deletion alert.
                        AppIconButton("trash", label: Text("Delete", bundle: .module), role: .destructive) {
                            chatPendingDeletion = chat
                        }
                        .appAction(.quiet, size: .small)
                        .foregroundStyle(.appSecondary)
                    }
                    .padding(.leading, 12)
                    .padding(.vertical, 2)
                }
            } label: {
                Label {
                    Text("Archived", bundle: .module) + Text(verbatim: " (\(archived.count))")
                } icon: { Image(systemName: "archivebox") }
                    .font(theme.ui(.tiny, weight: .medium))
                    .foregroundStyle(.appSecondary)
            }
            .padding(.top, 10)
        }
    }

    private var historyChats: [AppChat] {
        // Vault-aware: a ghost chat's transcript lives encrypted off its row,
        // so the emptiness check goes through the model, which decrypts.
        sortedChats.filter { model.chatHasTranscript($0) }
    }

    private var sortedChats: [AppChat] {
        // Pinned first, recency inside each group -- `AppChat`'s one
        // ordering, shared with the projects view and prev/next navigation.
        AppChat.sortedForSidebar(model.filteredChats)
    }

    private var renameAlertPresented: Binding<Bool> {
        Binding(
            get: { chatBeingRenamed != nil },
            set: { if !$0 { chatBeingRenamed = nil } })
    }

    private var deletionAlertPresented: Binding<Bool> {
        Binding(
            get: { chatPendingDeletion != nil },
            set: { if !$0 { chatPendingDeletion = nil } })
    }
}
