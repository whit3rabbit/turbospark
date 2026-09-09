import AppKit
import SwiftUI

// Isolated explicitly: only body is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The adaptive sidebar changing depending on active tabs (Chat vs Projects) and active project.
@MainActor
struct ChatSidebarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @AppStorage(AppLanguage.storageKey)
    private var languageRawValue = AppLanguage.system.rawValue

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

            Divider()

            contentView

            ChatSidebarFooterView(
                chatCount: historyChats.count,
                languageRawValue: $languageRawValue
            )
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
            "Rename chat",
            isPresented: renameAlertPresented,
            presenting: chatBeingRenamed
        ) { chat in
            TextField("Chat name", text: $renameText)
            Button("Cancel", role: .cancel) {}
            Button("Rename") {
                model.renameChat(id: chat.id, title: renameText)
            }
            .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        } message: { _ in
            Text("Choose a name that identifies this chat.", bundle: .module)
        }
        .alert(
            "Delete chat?",
            isPresented: deletionAlertPresented,
            presenting: chatPendingDeletion
        ) { chat in
            Button("Cancel", role: .cancel) {}
            Button("Delete", role: .destructive) {
                model.deleteChat(id: chat.id)
            }
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
                            .font(theme.ui(points: 10, weight: .semibold))
                        Text(mode.title)
                            .font(theme.ui(points: 11.5, weight: isSelected ? .semibold : .medium))
                    }
                    .frame(maxWidth: .infinity)
                    .frame(height: 24)
                    .foregroundStyle(isSelected ? Color.primary : Color.secondary)
                    .background(
                        isSelected ? Color.primary.opacity(0.12) : Color.clear,
                        in: RoundedRectangle(cornerRadius: 6, style: .continuous)
                    )
                    .contentShape(.rect(cornerRadius: 6))
                }
                .buttonStyle(.plain)
                .accessibilityLabel("Switch to \(mode.title) tab")
                .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
            }
        }
        .padding(2)
        .background(Color.primary.opacity(0.04), in: RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
        )
    }

    // MARK: - Filter Bar

    private var filterBar: some View {
        HStack(spacing: 6) {
            Image(systemName: "magnifyingglass")
                .font(theme.ui(points: 11))
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)

            TextField(
                model.interactionMode == .projects ? "Filter projects and tasks..." : "Filter chats...",
                text: $searchText
            )
            .textFieldStyle(.plain)
            .font(theme.ui(points: 11))

            if !searchText.isEmpty {
                Button {
                    searchText = ""
                } label: {
                    Image(systemName: "xmark.circle.fill")
                        .font(theme.ui(points: 11))
                        .foregroundStyle(.tertiary)
                }
                .buttonStyle(.plain)
                .help("Clear filter")
                .accessibilityLabel("Clear filter")
            }
        }
        .padding(.horizontal, 8)
        .frame(height: 26)
        .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 6, style: .continuous))
        .overlay(
            RoundedRectangle(cornerRadius: 6, style: .continuous)
                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
        )
        .accessibilityElement(children: .contain)
        .accessibilityLabel(model.interactionMode == .projects ? "Filter projects and tasks" : "Filter chats")
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
                    .font(theme.ui(points: 11, weight: .semibold))
                    .foregroundStyle(.secondary)
                    .accessibilityAddTraits(.isHeader)

                Spacer()

                Button {
                    projectBeingEdited = nil
                    showingProjectSettingsSheet = true
                } label: {
                    Image(systemName: "plus")
                        .font(theme.ui(points: 11, weight: .semibold))
                        .foregroundStyle(.secondary)
                        .frame(width: 20, height: 20)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("Add codebase project")
                .accessibilityLabel("Add project")
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
                .font(theme.ui(points: 10))
                .foregroundStyle(TurboSparkTheme.accentColor)

            Text("Project: \(project.name)", bundle: .module)
                .font(theme.ui(points: 11, weight: .medium))
                .foregroundStyle(.primary)
                .lineLimit(1)

            Spacer()

            Button {
                model.selectProject(id: nil)
            } label: {
                HStack(spacing: 3) {
                    Text("All", bundle: .module)
                        .font(theme.ui(points: 10))
                    Image(systemName: "xmark")
                        .font(theme.ui(points: 9))
                }
                .foregroundStyle(.secondary)
                .padding(.horizontal, 5)
                .padding(.vertical, 2)
                .background(Color.primary.opacity(0.06), in: Capsule())
            }
            .buttonStyle(.plain)
            .help("Show chats from all projects")
            .accessibilityLabel("Clear project filter and show all chats")
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 6, style: .continuous))
    }

    private var newChatButton: some View {
        Button {
            model.createChat()
        } label: {
            HStack(spacing: 8) {
                Image(systemName: "square.and.pencil")
                    .font(theme.ui(points: 12))
                    .accessibilityHidden(true)
                Text("New chat", bundle: .module)
                    .font(theme.ui(points: 12, weight: .medium))
                Spacer()
                Text("Cmd+N", bundle: .module)
                    .font(theme.ui(points: 10))
                    .foregroundStyle(.tertiary)
                    .accessibilityHidden(true)
            }
            .padding(.horizontal, 10)
            .frame(maxWidth: .infinity, minHeight: 30)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background(
            TurboSparkTheme.surfaceColor,
            in: .rect(cornerRadius: 8))
        .overlay {
            RoundedRectangle(cornerRadius: 8)
                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 0.5)
        }
        .disabled(model.isRunning)
        .help("Create a new chat (Cmd+N)")
        .accessibilityHint("Starts a fresh conversation. Disabled while the model is generating.")
    }

    private var chatList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 3) {
                if filteredHistoryChats.isEmpty {
                    VStack(spacing: 6) {
                        Image(systemName: "bubble.left.and.bubble.right")
                            .font(theme.ui(points: 20))
                            .foregroundStyle(.tertiary)
                            .padding(.top, 18)
                            .padding(.bottom, 2)
                            .accessibilityHidden(true)
                        Text(searchText.isEmpty ? "No chats yet" : "No matching chats")
                            .font(theme.ui(points: 11, weight: .medium))
                            .foregroundStyle(.secondary)
                        Text(searchText.isEmpty ? "Start a conversation to see history here" : "Try a different search term")
                            .font(theme.ui(points: 10))
                            .foregroundStyle(.tertiary)
                            .multilineTextAlignment(.center)
                    }
                    .frame(maxWidth: .infinity)
                    .padding(.vertical, 16)
                    .padding(.horizontal, 12)
                    .accessibilityElement(children: .ignore)
                    .accessibilityLabel("No chats yet. Start a conversation to see history here.")
                } else if searchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    // The qwen-code session grouping: time-bucket headers
                    // over the recency-sorted list. Suppressed while a
                    // search filter is active -- the matches are what
                    // matters then, not when they happened.
                    let pinned = filteredHistoryChats.filter(\.isPinned)
                    let unpinned = filteredHistoryChats.filter { !$0.isPinned }
                    ForEach(pinned) { chat in
                        ChatSidebarChatRowView(
                            model: model,
                            chat: chat,
                            chatBeingRenamed: $chatBeingRenamed,
                            renameText: $renameText,
                            chatPendingDeletion: $chatPendingDeletion,
                            chatForSystemPrompt: $chatForSystemPrompt
                        )
                    }
                    ForEach(dateGroupedChats(approach: unpinned), id: \.bucket) { group in
                        Text(group.bucket.label)
                            .font(theme.ui(points: 10.5, weight: .semibold))
                            .foregroundStyle(.secondary)
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
                    ForEach(filteredHistoryChats) { chat in
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
        let archived = AppChat.sortedForSidebar(model.archivedChats)
        if !archived.isEmpty {
            DisclosureGroup {
                ForEach(archived) { chat in
                    HStack(spacing: 8) {
                        Image(systemName: "archivebox")
                            .font(theme.ui(points: 11))
                            .foregroundStyle(.tertiary)
                            .accessibilityHidden(true)
                        Text(chat.title)
                            .font(theme.ui(points: 12))
                            .lineLimit(1)
                            .truncationMode(.tail)
                        Spacer(minLength: 4)
                        Button {
                            model.setChatArchived(id: chat.id, archived: false)
                        } label: {
                            Text("Restore", bundle: .module)
                                .font(theme.ui(points: 10, weight: .medium))
                        }
                        .buttonStyle(.plain)
                        .help("Move this chat back into the chat list")
                        Button(role: .destructive) {
                            chatPendingDeletion = chat
                        } label: {
                            Image(systemName: "trash")
                                .font(theme.ui(points: 10))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.tertiary)
                        .help("Delete archived chat")
                    }
                    .padding(.leading, 12)
                    .padding(.vertical, 2)
                    .accessibilityElement(children: .combine)
                }
            } label: {
                Label(
                    archived.count == 1 ? "Archived (1 chat)" : "Archived (\(archived.count) chats)",
                    systemImage: "archivebox")
                    .font(theme.ui(points: 11, weight: .medium))
                    .foregroundStyle(.secondary)
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

    private var filteredHistoryChats: [AppChat] {
        let trimmed = searchText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return historyChats }
        return historyChats.filter {
            $0.title.localizedCaseInsensitiveContains(trimmed) ||
            $0.preview.localizedCaseInsensitiveContains(trimmed)
        }
    }

    private struct DateGroup {
        let bucket: ChatDateBucket
        let chats: [AppChat]
    }

    /// Buckets the recency-sorted chats, keeping the sort inside each
    /// bucket and dropping empty buckets.
    private func dateGroupedChats(approach chats: [AppChat]) -> [DateGroup] {
        ChatDateBucket.allCases.compactMap { bucket in
            let inBucket = chats.filter {
                ChatDateBucket.bucket(for: $0.updatedAt) == bucket
            }
            return inBucket.isEmpty ? nil : DateGroup(bucket: bucket, chats: inBucket)
        }
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
