import AppKit
import SwiftUI

/// Projects section in the chat sidebar showing all-chats selector and active projects.
struct ChatSidebarProjectsSectionView: View {
    @ObservedObject var model: AppModel
    @Binding var showingProjectSettingsSheet: Bool
    @Binding var projectBeingEdited: AppProject?
    @Binding var projectForMcpSettings: AppProject?
    @State private var hoveredProjectID: UUID?

    @ScaledMetric private var projectActionSize: CGFloat = 22

    var body: some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack {
                Text("Projects")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Button {
                    projectBeingEdited = nil
                    showingProjectSettingsSheet = true
                } label: {
                    Label("Add Project", systemImage: "plus")
                        .labelStyle(.iconOnly)
                        .font(.caption)
                        .foregroundStyle(.secondary)
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
                    .font(.caption)
                    .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                    .accessibilityHidden(true)
                Text("All Chats")
                    .font(.callout.weight(isSelected ? .semibold : .regular))
                    .foregroundStyle(.primary)
                Spacer()
                Text("\(model.chats.count)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
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
                model.selectProject(id: project.id)
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: isSelected ? "folder.fill.badge.gearshape" : "folder.badge.gearshape")
                        .font(.caption)
                        .foregroundStyle(isSelected ? TurboSparkTheme.accentColor : Color.secondary)
                        .accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(project.name)
                            .font(.callout.weight(isSelected ? .semibold : .regular))
                            .foregroundStyle(.primary)
                            .lineLimit(1)
                        Text(project.agentType.label)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
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

            Menu {
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
                    .frame(width: projectActionSize, height: projectActionSize)
                    .contentShape(Circle())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help("Project actions for \(project.name)")
            // Always reachable from VoiceOver, even though sighted users
            // only see the button when hovering or when the row is selected.
            .accessibilityLabel("Project actions for \(project.name)")
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
