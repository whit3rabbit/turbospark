import AppKit
import SwiftUI

// Isolated explicitly: only body is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Hierarchical view grouping projects and their child tasks matching the sidebar reference design.
@MainActor
struct ChatSidebarGroupedProjectsView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let searchText: String
    @Binding var showingProjectSettingsSheet: Bool
    @Binding var projectBeingEdited: AppProject?
    @Binding var projectForMcpSettings: AppProject?
    @Binding var chatBeingRenamed: AppChat?
    @Binding var renameText: String
    @Binding var chatPendingDeletion: AppChat?
    @Binding var chatForSystemPrompt: AppChat?

    @State private var expandedProjectIDs: Set<UUID> = []
    @State private var hoveredProjectID: UUID?
    @State private var hoveredChatID: UUID?

    @ScaledMetric private var actionButtonSize: CGFloat = 20
    private let defaultTaskLimit = 5

    var body: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
                if model.projects.isEmpty {
                    Button {
                        projectBeingEdited = nil
                        showingProjectSettingsSheet = true
                    } label: {
                        Label { Text("Add Project", bundle: .module) } icon: { Image(systemName: "folder.badge.plus") }
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                    }
                    .buttonStyle(.bordered)
                    .disabled(model.isRunning || model.submitting || model.pendingToolCall != nil)
                }
                if filteredProjects.isEmpty && unorganizedChats.isEmpty && !trimmedSearch.isEmpty {
                    emptyState
                } else {
                    ForEach(filteredProjects) { project in
                        projectSection(project)
                    }

                    if !unorganizedChats.isEmpty {
                        unorganizedSection
                    }
                }
            }
            .padding(.horizontal, 8)
            .padding(.top, 6)
            .padding(.bottom, 12)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Filtered Data

    private var trimmedSearch: String {
        searchText.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    private var filteredProjects: [AppProject] {
        if trimmedSearch.isEmpty {
            return model.projects
        }
        return model.projects.filter { project in
            if project.name.localizedCaseInsensitiveContains(trimmedSearch) {
                return true
            }
            // Check if any chat in this project matches the query. Ghost
            // and archived rows are excluded for the same reason `tasks`
            // excludes them: a match only counts if the chat is one the
            // list would actually show.
            return model.chats.contains { chat in
                chat.projectID == project.id && !chat.isGhost && !chat.isArchived &&
                    (chat.title.localizedCaseInsensitiveContains(trimmedSearch) ||
                     chat.preview.localizedCaseInsensitiveContains(trimmedSearch))
            }
        }
    }

    private func tasks(for projectID: UUID) -> [AppChat] {
        // Ghost and archived rows are NOT ordinary tasks: a ghost's row
        // must be reachable only through `enterGhostChat`, and archived
        // chats live behind the archive disclosure, not the project lists.
        // The Chat tab reads `filteredChats` for the same reason.
        let allTasks = AppChat.sortedForSidebar(
            model.chats.filter {
                $0.projectID == projectID && !$0.isGhost && !$0.isArchived
            })

        if trimmedSearch.isEmpty {
            return allTasks
        }

        // If project name matched, show all or matching tasks
        let project = model.projects.first { $0.id == projectID }
        if let project, project.name.localizedCaseInsensitiveContains(trimmedSearch) {
            return allTasks
        }

        return allTasks.filter {
            $0.title.localizedCaseInsensitiveContains(trimmedSearch) ||
            $0.preview.localizedCaseInsensitiveContains(trimmedSearch)
        }
    }

    private var unorganizedChats: [AppChat] {
        let chats = AppChat.sortedForSidebar(
            model.chats.filter { $0.projectID == nil && !$0.isGhost && !$0.isArchived })

        if trimmedSearch.isEmpty {
            return chats
        }

        return chats.filter {
            $0.title.localizedCaseInsensitiveContains(trimmedSearch) ||
            $0.preview.localizedCaseInsensitiveContains(trimmedSearch)
        }
    }

    // MARK: - Project Section

    private func projectSection(_ project: AppProject) -> some View {
        let isProjectSelected = model.selectedProjectID == project.id
        let projectTasks = tasks(for: project.id)
        let isExpanded = expandedProjectIDs.contains(project.id) || !trimmedSearch.isEmpty
        let visibleTasks = isExpanded ? projectTasks : Array(projectTasks.prefix(defaultTaskLimit))
        let hasMore = projectTasks.count > defaultTaskLimit && trimmedSearch.isEmpty

        return VStack(alignment: .leading, spacing: 2) {
            projectHeaderRow(project, isSelected: isProjectSelected)

            if projectTasks.isEmpty {
                Text("No tasks yet", bundle: .module)
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.tertiary)
                    .padding(.leading, 24)
                    .padding(.vertical, 4)
            } else {
                VStack(alignment: .leading, spacing: 2) {
                    ForEach(visibleTasks) { chat in
                        taskRow(chat, project: project)
                    }

                    if hasMore {
                        Button {
                            toggleExpansion(for: project.id)
                        } label: {
                            Text(isExpanded ? "Show less" : "Show more")
                                .font(theme.ui(.tiny, weight: .medium))
                                .foregroundStyle(.appSecondary)
                                .padding(.leading, 24)
                                .padding(.vertical, 4)
                                .contentShape(.rect)
                        }
                        .buttonStyle(.plain)
                        .accessibilityLabel(isExpanded ? "Show fewer tasks in \(project.name)" : "Show all \(projectTasks.count) tasks in \(project.name)")
                    }
                }
            }
        }
    }

    private func projectHeaderRow(_ project: AppProject, isSelected: Bool) -> some View {
        let isHovered = hoveredProjectID == project.id
        let showsActions = isHovered || isSelected

        return HStack(spacing: 6) {
            Button {
                model.selectProject(id: project.id)
            } label: {
                HStack(spacing: 7) {
                    Image(systemName: isSelected ? "folder.fill" : "folder")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)

                    Text(project.name)
                        .font(theme.ui(.small, weight: isSelected ? .semibold : .medium))
                        .foregroundStyle(.appText)
                        .lineLimit(1)

                    Spacer(minLength: 0)
                }
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .help("Switch to project \(project.name)")
            .accessibilityLabel("Project \(project.name)")
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)

            if showsActions {
                Button {
                    model.createChat(projectID: project.id)
                } label: {
                    Image(systemName: "plus")
                        .font(theme.ui(.tiny, weight: .medium))
                        .foregroundStyle(.appSecondary)
                        .frame(width: actionButtonSize, height: actionButtonSize)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("New task in \(project.name)")
                .accessibilityLabel("New task in \(project.name)")

                Menu {
                    Button {
                        model.createChat(projectID: project.id)
                    } label: {
                        Label { Text("New Task", bundle: .module) } icon: { Image(systemName: "plus") }
                    }
                    Divider()
                    Button {
                        projectBeingEdited = project
                        showingProjectSettingsSheet = true
                    } label: {
                        Label { Text("Project Settings", bundle: .module) } icon: { Image(systemName: "gearshape") }
                    }
                    Button {
                        projectForMcpSettings = project
                    } label: {
                        Label { Text("Manage MCP Servers...", bundle: .module) } icon: { Image(systemName: "server.rack") }
                    }
                    if let path = project.rootDirectoryPath {
                        Button {
                            let url = URL(fileURLWithPath: path)
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        } label: {
                            Label { Text("Reveal in Finder", bundle: .module) } icon: { Image(systemName: "folder") }
                        }
                    }
                    Divider()
                    Button(role: .destructive) {
                        model.deleteProject(id: project.id)
                    } label: {
                        Label { Text("Delete Project", bundle: .module) } icon: { Image(systemName: "trash") }
                    }
                } label: {
                    Label { Text("Project actions", bundle: .module) } icon: { Image(systemName: "ellipsis") }
                        .labelStyle(.iconOnly)
                        .frame(width: actionButtonSize, height: actionButtonSize)
                        .contentShape(Circle())
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help(Text("Project actions for \(project.name)", bundle: .module))
                .accessibilityLabel(Text("Project actions for \(project.name)", bundle: .module))
            }
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 4)
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.08) : Color.clear,
            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
        )
        .contentShape(.rect(cornerRadius: 6))
        .onHover { hovering in
            hoveredProjectID = hovering ? project.id : nil
        }
    }

    // MARK: - Task Row

    private func taskRow(_ chat: AppChat, project: AppProject?) -> some View {
        let isSelected = chat.id == model.selectedChatID
        let isHovered = hoveredChatID == chat.id
        let showsActions = isHovered || isSelected
        let isRecent = isRecentTask(chat.updatedAt)

        return HStack(spacing: 5) {
            Button {
                if let project {
                    if model.selectedProjectID != project.id {
                        model.selectProject(id: project.id)
                    }
                } else if model.selectedProjectID != nil {
                    // Same rule the search overlay's `open` applies: the
                    // Chat tab's list is project-scoped, so opening a chat
                    // OUTSIDE the selected project must move the project
                    // selection first or the chat opens invisibly.
                    model.selectProject(id: nil)
                }
                model.selectChat(id: chat.id)
            } label: {
                HStack(spacing: 6) {
                    taskIcon(chat: chat, isSelected: isSelected, isRecent: isRecent)

                    Text(chat.title)
                        .font(theme.ui(.tiny, weight: isSelected ? .medium : .regular))
                        .foregroundStyle(isSelected ? Color.primary : Color.primary.opacity(0.88))
                        .lineLimit(1)

                    Spacer(minLength: 4)

                    Text(MessageTimestampFormatter.compactRelativeString(for: chat.updatedAt))
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.tertiary)
                        .lineLimit(1)
                }
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .disabled(model.isRunning && !isSelected)
            .help("\(chat.title) - \(MessageTimestampFormatter.relativeString(for: chat.updatedAt))")
            .accessibilityLabel(chat.title)
            .accessibilityValue(isSelected ? "Selected" : "")
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)

            if showsActions {
                Menu {
                    Button {
                        renameText = chat.title
                        chatBeingRenamed = chat
                    } label: {
                        Label { Text("Rename", bundle: .module) } icon: { Image(systemName: "pencil") }
                    }
                    Divider()
                    Button(role: .destructive) {
                        chatPendingDeletion = chat
                    } label: {
                        Label { Text("Delete", bundle: .module) } icon: { Image(systemName: "trash") }
                    }
                } label: {
                    Label { Text("Task actions", bundle: .module) } icon: { Image(systemName: "ellipsis") }
                        .labelStyle(.iconOnly)
                        .frame(width: actionButtonSize, height: actionButtonSize)
                        .contentShape(Circle())
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("Task actions for \(chat.title)")
                .accessibilityLabel("Task actions for \(chat.title)")
                .disabled(model.isRunning)
            }
        }
        .padding(.leading, 18)
        .padding(.trailing, 6)
        .padding(.vertical, 4.5)
        .background(
            isSelected
                ? Color(nsColor: .selectedContentBackgroundColor).opacity(0.18)
                : (isHovered ? Color.primary.opacity(0.04) : Color.clear),
            in: RoundedRectangle(cornerRadius: 6, style: .continuous)
        )
        .contentShape(.rect(cornerRadius: 6))
        .onHover { hovering in
            hoveredChatID = hovering ? chat.id : nil
        }
        .contextMenu {
            Button {
                renameText = chat.title
                chatBeingRenamed = chat
            } label: { Text("Rename", bundle: .module) }
            .disabled(model.isRunning)
            Button(chat.systemPrompt == nil ? "Set System Prompt" : "Edit System Prompt") {
                chatForSystemPrompt = chat
            }
            .disabled(model.isRunning)
            Button(role: .destructive) {
                chatPendingDeletion = chat
            } label: { Text("Delete", bundle: .module) }
            .disabled(model.isRunning)
        }
    }

    @ViewBuilder
    private func taskIcon(chat: AppChat, isSelected: Bool, isRecent: Bool) -> some View {
        if isSelected && model.isRunning {
            TaskProgressFlameIcon(size: 11)
        } else if isRecent {
            Image(systemName: "sparkle")
                .font(theme.ui(.tiny, weight: .medium))
                .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                .accessibilityHidden(true)
        } else {
            Circle()
                .fill(Color.secondary.opacity(0.35))
                .frame(width: 4, height: 4)
                .padding(.horizontal, 3)
                .accessibilityHidden(true)
        }
    }

    private func isRecentTask(_ date: Date) -> Bool {
        Date().timeIntervalSince(date) < 86400 // Last 24 hours
    }

    private func toggleExpansion(for projectID: UUID) {
        if expandedProjectIDs.contains(projectID) {
            expandedProjectIDs.remove(projectID)
        } else {
            expandedProjectIDs.insert(projectID)
        }
    }

    // MARK: - Unorganized Section

    private var unorganizedSection: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 7) {
                Image(systemName: "tray")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
                Text("Other Tasks", bundle: .module)
                    .font(theme.ui(.small, weight: .medium))
                    .foregroundStyle(.appSecondary)
                Spacer()
            }
            .padding(.horizontal, 6)
            .padding(.vertical, 4)

            VStack(alignment: .leading, spacing: 2) {
                ForEach(unorganizedChats) { chat in
                    taskRow(chat, project: nil)
                }
            }
        }
    }

    // MARK: - Empty State

    private var emptyState: some View {
        VStack(spacing: 8) {
            Image(systemName: "folder.badge.plus")
                .font(theme.ui(.title2))
                .foregroundStyle(.tertiary)
                .padding(.top, 24)
            Text("No projects match filter", bundle: .module)
                .font(theme.ui(.tiny, weight: .medium))
                .foregroundStyle(.appSecondary)
            Button {
                projectBeingEdited = nil
                showingProjectSettingsSheet = true
            } label: { Text("Add Project", bundle: .module) }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
    }
}
