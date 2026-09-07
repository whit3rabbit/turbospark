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
                if filteredProjects.isEmpty && unorganizedChats.isEmpty {
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
            // Check if any chat in this project matches the query
            return model.chats.contains { chat in
                chat.projectID == project.id &&
                    (chat.title.localizedCaseInsensitiveContains(trimmedSearch) ||
                     chat.preview.localizedCaseInsensitiveContains(trimmedSearch))
            }
        }
    }

    private func tasks(for projectID: UUID) -> [AppChat] {
        let allTasks = model.chats
            .filter { $0.projectID == projectID }
            .sorted { $0.updatedAt > $1.updatedAt }

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
        let chats = model.chats
            .filter { $0.projectID == nil }
            .sorted { $0.updatedAt > $1.updatedAt }

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
                    .font(theme.ui(points: 11))
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
                                .font(theme.ui(points: 11, weight: .medium))
                                .foregroundStyle(.secondary)
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
                        .font(theme.ui(points: 11))
                        .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)

                    Text(project.name)
                        .font(theme.ui(points: 12, weight: isSelected ? .semibold : .medium))
                        .foregroundStyle(.primary)
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
                        .font(theme.ui(points: 10, weight: .medium))
                        .foregroundStyle(.secondary)
                        .frame(width: actionButtonSize, height: actionButtonSize)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("New task in \(project.name)")
                .accessibilityLabel("New task in \(project.name)")

                Menu {
                    Button("New Task", systemImage: "plus") {
                        model.createChat(projectID: project.id)
                    }
                    Divider()
                    Button("Project Settings", systemImage: "gearshape") {
                        projectBeingEdited = project
                        showingProjectSettingsSheet = true
                    }
                    Button("Manage MCP Servers...", systemImage: "server.rack") {
                        projectForMcpSettings = project
                    }
                    if let path = project.rootDirectoryPath {
                        Button("Reveal in Finder", systemImage: "folder") {
                            let url = URL(fileURLWithPath: path)
                            NSWorkspace.shared.activateFileViewerSelecting([url])
                        }
                    }
                    Divider()
                    Button("Delete Project", systemImage: "trash", role: .destructive) {
                        model.deleteProject(id: project.id)
                    }
                } label: {
                    Label("Project actions", systemImage: "ellipsis")
                        .labelStyle(.iconOnly)
                        .frame(width: actionButtonSize, height: actionButtonSize)
                        .contentShape(Circle())
                }
                .menuStyle(.borderlessButton)
                .menuIndicator(.hidden)
                .fixedSize()
                .help("Project actions for \(project.name)")
                .accessibilityLabel("Project actions for \(project.name)")
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
                }
                model.selectChat(id: chat.id)
            } label: {
                HStack(spacing: 6) {
                    taskIcon(chat: chat, isSelected: isSelected, isRecent: isRecent)

                    Text(chat.title)
                        .font(theme.ui(points: 11.5, weight: isSelected ? .medium : .regular))
                        .foregroundStyle(isSelected ? Color.primary : Color.primary.opacity(0.88))
                        .lineLimit(1)

                    Spacer(minLength: 4)

                    Text(MessageTimestampFormatter.compactRelativeString(for: chat.updatedAt))
                        .font(theme.ui(points: 10))
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
                    Button("Rename", systemImage: "pencil") {
                        renameText = chat.title
                        chatBeingRenamed = chat
                    }
                    Divider()
                    Button("Delete", systemImage: "trash", role: .destructive) {
                        chatPendingDeletion = chat
                    }
                } label: {
                    Label("Task actions", systemImage: "ellipsis")
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
            Button("Rename") {
                renameText = chat.title
                chatBeingRenamed = chat
            }
            .disabled(model.isRunning)
            Button(chat.systemPrompt == nil ? "Set System Prompt" : "Edit System Prompt") {
                chatForSystemPrompt = chat
            }
            .disabled(model.isRunning)
            Button("Delete", role: .destructive) {
                chatPendingDeletion = chat
            }
            .disabled(model.isRunning)
        }
    }

    @ViewBuilder
    private func taskIcon(chat: AppChat, isSelected: Bool, isRecent: Bool) -> some View {
        if isSelected && model.isRunning {
            TaskProgressFlameIcon(size: 11)
        } else if isRecent {
            Image(systemName: "sparkle")
                .font(theme.ui(points: 10, weight: .medium))
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
                    .font(theme.ui(points: 11))
                    .foregroundStyle(.secondary)
                Text("Other Tasks", bundle: .module)
                    .font(theme.ui(points: 12, weight: .medium))
                    .foregroundStyle(.secondary)
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
                .font(theme.ui(points: 22))
                .foregroundStyle(.tertiary)
                .padding(.top, 24)
            Text("No projects match filter", bundle: .module)
                .font(theme.ui(points: 11, weight: .medium))
                .foregroundStyle(.secondary)
            Button("Add Project") {
                projectBeingEdited = nil
                showingProjectSettingsSheet = true
            }
            .buttonStyle(.borderedProminent)
            .controlSize(.small)
            .padding(.top, 4)
        }
        .frame(maxWidth: .infinity)
        .padding(.vertical, 16)
    }
}
