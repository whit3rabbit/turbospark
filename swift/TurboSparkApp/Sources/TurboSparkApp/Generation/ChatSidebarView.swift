import AppKit
import SwiftUI

// Isolated explicitly: only body is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The CHAT LIST: tabs (Chat vs Projects), filter, and the rows themselves.
///
/// It is one child of `AppSidebarView` rather than the whole sidebar, and it
/// renders only while `activeSection == .chat`. The profile/settings footer
/// used to hang off the bottom of this view and moved up to `AppSidebarView`
/// when the two left columns merged: the footer has to be present in Files
/// and Server too, where there is no chat list to attach it to.
@MainActor
struct ChatSidebarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var searchText = ""
    @State private var chatBeingRenamed: AppChat?
    @State private var chatForSystemPrompt: AppChat?
    @State private var renameText = ""
    @State private var chatPendingDeletion: AppChat?
    @State private var showingProjectSettingsSheet = false
    @State private var projectBeingEdited: AppProject?
    @State private var projectForMcpSettings: AppProject?

    var body: some View {
        VStack(spacing: 0) {
            tabSelector
                .padding(.horizontal, 10)
                .padding(.top, 10)
                .padding(.bottom, 6)

            filterBar
                .padding(.horizontal, 10)
                .padding(.bottom, 8)

            SidebarDivider()

            contentView
        }
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

    // MARK: - Tab Selector

    private var tabSelector: some View {
        HStack(spacing: 4) {
            ForEach(AppModel.AppInteractionMode.allCases) { mode in
                let isSelected = model.interactionMode == mode
                Button {
                    withAnimation(.easeInOut(duration: 0.12)) {
                        model.setInteractionMode(mode)
                    }
                } label: {
                    HStack(spacing: 5) {
                        Image(systemName: mode.systemImage)
                            .font(theme.ui(.tiny, weight: .semibold))
                        Text(LocalizedStringKey(mode.title), bundle: .module)
                            .font(theme.ui(.tiny, weight: isSelected ? .semibold : .medium))
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 4)
                    .frame(minHeight: theme.sidebarRowHeight - 4)
                    .foregroundStyle(isSelected ? theme.foreground : theme.secondaryText)
                    .background(
                        isSelected ? theme.foreground.opacity(0.12) : Color.clear,
                        in: RoundedRectangle(cornerRadius: 6, style: .continuous)
                    )
                    .contentShape(.rect(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .help(Text(LocalizedStringKey(mode.title), bundle: .module))
                .accessibilityLabel(Text(LocalizedStringKey(mode.title), bundle: .module))
                .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
            }
        }
        .padding(2)
        .background(theme.foreground.opacity(0.04), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5)
        )
    }

    // MARK: - Filter Bar

    private var filterBar: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)

            TextField(text: $searchText) {
                Text(LocalizedStringKey(model.interactionMode == .projects
                     ? "Filter projects and tasks..." : "Filter chats..."), bundle: .module)
            }
            .textFieldStyle(.plain)
            .font(theme.ui(.tiny))

            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.plain)
                .help(Text("Clear filters", bundle: .module))
                .accessibilityLabel(Text("Clear filters", bundle: .module))
            }
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 5)
        .frame(minHeight: theme.sidebarRowHeight)
        .background(theme.surface, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5)
        )
        .accessibilityElement(children: .contain)
        .help(Text(LocalizedStringKey(model.interactionMode == .projects
              ? "Filter projects and tasks..." : "Filter chats..."), bundle: .module))
        .accessibilityLabel(Text(LocalizedStringKey(model.interactionMode == .projects
              ? "Filter projects and tasks..." : "Filter chats..."), bundle: .module))
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

                Button {
                    projectBeingEdited = nil
                    showingProjectSettingsSheet = true
                } label: {
                    Image(systemName: "plus")
                        .font(theme.ui(.tiny, weight: .semibold))
                        .foregroundStyle(.appSecondary)
                        .frame(width: theme.sidebarActionSize, height: theme.sidebarActionSize)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help(Text("Add Project", bundle: .module))
                .accessibilityLabel(Text("Add Project", bundle: .module))
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
            ScrollView {
                ChatSidebarProjectsSectionView(
                    model: model,
                    showingProjectSettingsSheet: $showingProjectSettingsSheet,
                    projectBeingEdited: $projectBeingEdited,
                    projectForMcpSettings: $projectForMcpSettings)
            }
            .frame(height: min(180, CGFloat(72 + model.projects.count * 48)) * theme.sidebarScale)
            SidebarDivider()
            newChatButton
                .padding(.horizontal, 10)
                .padding(.top, 8)
                .padding(.bottom, 6)

            if let activeProject = model.selectedProject {
                activeProjectScopeBar(activeProject)
                    .padding(.horizontal, 10)
                    .padding(.bottom, 6)
            }

            chatList
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
                .background(theme.foreground.opacity(0.06), in: Capsule())
            }
            .buttonStyle(.plain)
            .help(Text("All Chats", bundle: .module))
            .accessibilityLabel(Text("All Chats", bundle: .module))
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(theme.foreground.opacity(0.03), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
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
        .disabled(model.isRunning)
        .help(Text("New chat", bundle: .module) + Text(verbatim: " (Cmd+N)"))
    }

    private var newChatLabel: some View {
        Label {
            Text("New chat", bundle: .module)
                .font(theme.ui(.small, weight: .medium))
        } icon: {
            Image(systemName: "square.and.pencil")
                .font(theme.ui(.small))
                .accessibilityHidden(true)
        }
    }

    private var chatList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 3) {
                // Sorted, filtered and bucketed ONCE per body. This used to
                // re-run (with a sort and a vault-aware transcript check
                // each) for every mention below, and the body re-runs per
                // streamed token.
                let layout = ChatSidebarLayout.make(
                    sortedHistory: historyChats, searchText: searchText)
                if layout.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "bubble.left.and.bubble.right")
                            .font(theme.ui(.title2))
                            .foregroundStyle(.appSecondary)
                            .padding(.top, 18)
                            .padding(.bottom, 2)
                            .accessibilityHidden(true)
                        Text(LocalizedStringKey(searchText.isEmpty ? "No chats yet" : "No matching chats"), bundle: .module)
                            .font(theme.ui(.tiny, weight: .medium))
                            .foregroundStyle(.appSecondary)
                        Text(LocalizedStringKey(searchText.isEmpty ? "Start a conversation to see history here" : "Try a different search term"), bundle: .module)
                            .font(theme.ui(.tiny))
                            .foregroundStyle(.appSecondary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .padding(.horizontal, 12)
                    .accessibilityElement(children: .combine)
                } else if !layout.isSearching {
                    // The qwen-code session grouping: time-bucket headers
                    // over the recency-sorted list. Suppressed while a
                    // search filter is active -- the matches are what
                    // matters then, not when they happened.
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
                            .font(theme.ui(.tiny, weight: .semibold))
                            .foregroundStyle(.appSecondary)
                            .padding(.leading, 10)
                            .padding(.top, 8)
                            .textCase(nil)
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
                archivedSection
            }
            .padding(.horizontal, 8)
            .padding(.bottom, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    /// The qwen-code session-archive surface: archived chats leave the
    /// lists above and live behind one disclosure here, each row offering
    /// Restore and Delete.
    @ViewBuilder
    private var archivedSection: some View {
        // `archivedChats` is already in sidebar order.
        let archived = model.archivedChats
        if !archived.isEmpty {
            DisclosureGroup {
                ForEach(archived) { chat in
                    HStack(spacing: 8) {
                        Image(systemName: "archivebox")
                            .font(theme.ui(.tiny))
                            .foregroundStyle(.appSecondary)
                            .accessibilityHidden(true)
                        Text(chat.title)
                            .font(theme.ui(.small))
                            .lineLimit(1)
                            .truncationMode(.tail)
                        Spacer(minLength: 4)
                        Button {
                            model.setChatArchived(id: chat.id, archived: false)
                        } label: {
                            Text("Restore", bundle: .module)
                                .font(theme.ui(.tiny, weight: .medium))
                        }
                        .buttonStyle(.plain)
                        .help(Text("Restore", bundle: .module))
                        Button(role: .destructive) {
                            chatPendingDeletion = chat
                        } label: {
                            Image(systemName: "trash")
                                .font(theme.ui(.tiny))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.appSecondary)
                        .help(Text("Delete", bundle: .module))
                    }
                    .padding(.leading, 12)
                    .padding(.vertical, 2)
                    .accessibilityElement(children: .combine)
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
