import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Projects section in the chat sidebar showing all-chats selector and active projects.
@MainActor
struct ChatSidebarProjectsSectionView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @Binding var showingProjectSettingsSheet: Bool
    @Binding var projectBeingEdited: AppProject?
    @Binding var projectForMcpSettings: AppProject?
    @State private var hoveredProjectID: UUID?

    @ScaledMetric private var projectActionSize: CGFloat = 22

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
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
                    Label { Text("Add Project", bundle: .module) } icon: { Image(systemName: "plus") }
                        .labelStyle(.iconOnly)
                        .font(theme.ui(.tiny))
                        .foregroundStyle(.appSecondary)
                }
                .buttonStyle(.borderless)
                .help("Add codebase project")
                .accessibilityLabel("Add project")
                .accessibilityHint("Opens project settings to create a new project")
            }
            .padding(.horizontal, 10)
            .padding(.top, 8)
            .padding(.bottom, 2)

            allChatsRow

            ForEach(model.projects) { project in
                projectRow(project)
            }
        }
        .padding(.horizontal, 8)
        .padding(.bottom, 6)
    }

    private var allChatsRow: some View {
        let isSelected = model.selectedProjectID == nil
        return Button {
            model.selectProject(id: nil)
        } label: {
            HStack(spacing: 8) {
                Image(systemName: isSelected ? "folder.fill" : "folder")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                    .accessibilityHidden(true)
                Text("All Chats", bundle: .module)
                    .font(theme.ui(.small, weight: isSelected ? .semibold : .regular))
                    .foregroundStyle(.appText)
                Spacer()
                Text(verbatim: "\(model.chats.count)")
                    .font(theme.ui(.tiny))
                    .foregroundStyle(.appSecondary)
                    .accessibilityLabel("\(model.chats.count) chats")
            }
            .padding(.horizontal, 9)
            .padding(.vertical, 6)
            .contentShape(.rect)
        }
        .buttonStyle(.plain)
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: .rect(cornerRadius: 8)
        )
        .help("Show all conversations")
        .accessibilityLabel("All chats")
        .accessibilityValue("\(model.chats.count) total")
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint("Shows every chat regardless of project")
    }

    private func projectRow(_ project: AppProject) -> some View {
        let isSelected = model.selectedProjectID == project.id
        let showsActions = hoveredProjectID == project.id || isSelected

        return HStack(spacing: 4) {
            Button {
                model.chooseProjectForTask(id: project.id)
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isSelected ? "folder.fill.badge.gearshape" : "folder.badge.gearshape")
                        .font(theme.ui(.tiny))
                        .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(project.name)
                            .font(theme.ui(.small, weight: isSelected ? .semibold : .regular))
                            .foregroundStyle(.appText)
                            .lineLimit(1)
                        Text(project.agentType.label)
                            .font(theme.ui(.tiny))
                            .foregroundStyle(.appSecondary)
                    }
                    Spacer(minLength: 0)
                }
                .padding(.leading, 9)
                .padding(.vertical, 6)
                .contentShape(.rect)
            }
            .buttonStyle(.plain)
            .help("Switch to project \(project.name)")
            .accessibilityLabel("Project \(project.name)")
            .accessibilityValue("\(project.agentType.label)\(isSelected ? ", selected" : "")")
            .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)

            if showsActions {
                Button {
                    model.createChat(projectID: project.id)
                } label: {
                    Image(systemName: "plus")
                        .font(theme.ui(.tiny, weight: .semibold))
                        .foregroundStyle(.appSecondary)
                        .frame(width: projectActionSize, height: projectActionSize)
                        .contentShape(Circle())
                }
                .buttonStyle(.plain)
                .help("New chat in \(project.name)")
                .accessibilityLabel("New chat in \(project.name)")
            }

            Menu {
                Button {
                    model.createChat(projectID: project.id)
                } label: {
                    Label { Text("New Chat", bundle: .module) } icon: { Image(systemName: "plus") }
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
                    .frame(width: projectActionSize, height: projectActionSize)
                    .contentShape(Circle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help(Text("Project actions for \(project.name)", bundle: .module))
            // Always reachable from VoiceOver, even though sighted users
            // only see the button when hovering or when the row is selected.
            .accessibilityLabel(Text("Project actions for \(project.name)", bundle: .module))
            .opacity(showsActions ? 1 : 0)
            .padding(.trailing, 4)
        }
        .background(
            isSelected ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear,
            in: .rect(cornerRadius: 8)
        )
        .onHover { isHovering in
            hoveredProjectID = isHovering ? project.id : nil
        }
    }
}
