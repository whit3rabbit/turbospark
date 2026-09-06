import AppKit
import SwiftUI
import UniformTypeIdentifiers

/// Plus button and cascade submenu in the chat composer (Claude Code and Unsloth Studio parity).
///
/// Provides rapid access to attach files/photos, attach folders, insert slash commands,
/// manage and toggle MCP connectors, manage skills and plugins, switch projects, and toggle
/// intelligence modes (search, permissions).
struct PromptComposerPlusMenu: View {
    @ObservedObject var model: AppModel
    let iconButtonSize: CGFloat
    let isRunning: Bool
    let isExtracting: Bool
    let onAttachFiles: () -> Void
    let onAttachFolder: () -> Void
    let onNewProject: () -> Void
    let onAddMcpServer: () -> Void
    let onCreateSkill: () -> Void
    let onInsertPromptText: (String) -> Void

    @State private var isHovered: Bool = false

    var body: some View {
        if isExtracting {
            TaskProgressFlameIcon(size: 16)
                .frame(width: iconButtonSize, height: iconButtonSize)
                .help("Extracting documents...")
                .accessibilityLabel("Extracting documents")
        } else {
            Menu {
                filesSection
                slashCommandsSection
                connectorsSection
                pluginsSection
                projectsSection
                modesSection
            } label: {
                HStack(spacing: 3) {
                    Image(systemName: "plus")
                        .font(.system(size: 12, weight: .semibold))
                    Image(systemName: "chevron.down")
                        .font(.system(size: 7, weight: .bold))
                        .foregroundStyle(.secondary)
                }
                .frame(height: iconButtonSize)
                .padding(.horizontal, 7)
                .background(
                    Color.primary.opacity(isHovered ? 0.08 : 0.04),
                    in: Capsule()
                )
                .contentShape(Capsule())
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .disabled(isRunning)
            .onHover { isHovered = $0 }
            .help("Add files, folders, MCP servers, skills, and tools (Cmd+U)")
            .accessibilityLabel("Add options and tools")
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var filesSection: some View {
        Section {
            Button {
                onAttachFiles()
            } label: {
                Label("Add Files or Photos...", systemImage: "paperclip")
            }
            .keyboardShortcut("u", modifiers: .command)

            Button {
                onAttachFolder()
            } label: {
                Label("Add Folder...", systemImage: "folder.badge.plus")
            }

            Button {
                importGitOrUrlContext()
            } label: {
                Label("Import GitHub Issue or URL", systemImage: "link.badge.plus")
            }
        }
    }

    @ViewBuilder
    private var slashCommandsSection: some View {
        Section {
            Menu {
                Button {
                    onInsertPromptText("/explore ")
                } label: {
                    Label("/explore: Codebase search with subagent", systemImage: "magnifyingglass")
                }

                Button {
                    onInsertPromptText("/plan ")
                } label: {
                    Label("/plan: Multi-step implementation plan", systemImage: "list.bullet.clipboard")
                }

                Button {
                    onInsertPromptText("/agent ")
                } label: {
                    Label("/agent: Invoke specialized persona", systemImage: "person.crop.square")
                }

                let skills = model.effectiveSkills
                if !skills.isEmpty {
                    Divider()
                    ForEach(skills) { skill in
                        Button {
                            onInsertPromptText("/\(skill.name) ")
                        } label: {
                            Label("/\(skill.name): \(skill.skillDescription)", systemImage: "wand.and.stars")
                        }
                    }
                }
            } label: {
                Label("Slash Commands", systemImage: "command")
            }
        }
    }

    @ViewBuilder
    private var connectorsSection: some View {
        Section {
            Menu {
                let allMcpServers = model.globalMcpServers + (model.selectedProject?.mcpServers ?? [])
                if allMcpServers.isEmpty {
                    Text("No MCP Servers Configured")
                } else {
                    ForEach(allMcpServers) { server in
                        Button {
                            toggleMcpServer(server)
                        } label: {
                            Label(
                                server.name,
                                systemImage: server.isEnabled ? "checkmark.circle.fill" : "circle"
                            )
                        }
                    }
                }

                Divider()

                Button {
                    onAddMcpServer()
                } label: {
                    Label("Add MCP Server...", systemImage: "plus")
                }

                Button {
                    model.openSettings(tab: .mcp)
                } label: {
                    Label("Manage MCP Servers...", systemImage: "server.rack")
                }
            } label: {
                Label("Connectors (MCP)", systemImage: "point.3.filled.connected.trianglepath.dotted")
            }
        }
    }

    @ViewBuilder
    private var pluginsSection: some View {
        Section {
            Menu {
                let managedSkills = model.allManagedSkills
                if managedSkills.isEmpty {
                    Text("No Skills Installed")
                } else {
                    ForEach(managedSkills) { skill in
                        Menu {
                            Button {
                                onInsertPromptText("/\(skill.name) ")
                            } label: {
                                Label("Use in Chat (/\(skill.name))", systemImage: "arrow.right.circle")
                            }

                            Button {
                                model.toggleSkillEnabled(skill)
                            } label: {
                                Label(
                                    skill.isEnabled ? "Disable Skill" : "Enable Skill",
                                    systemImage: skill.isEnabled ? "checkmark" : "circle"
                                )
                            }
                        } label: {
                            Label(
                                skill.name,
                                systemImage: skill.isEnabled ? "wand.and.stars" : "wand.and.stars.inverse"
                            )
                        }
                    }
                }

                Divider()

                Button {
                    onCreateSkill()
                } label: {
                    Label("Create New Skill...", systemImage: "plus")
                }

                Button {
                    model.openSettings(tab: .skills)
                } label: {
                    Label("Manage Skills...", systemImage: "gearshape")
                }
            } label: {
                Label("Plugins and Skills", systemImage: "puzzlepiece.extension")
            }
        }
    }

    @ViewBuilder
    private var projectsSection: some View {
        Section {
            Menu {
                Button {
                    model.selectProject(id: nil)
                } label: {
                    Label(
                        "All Chats (No Project)",
                        systemImage: model.selectedProjectID == nil ? "checkmark" : ""
                    )
                }

                if !model.projects.isEmpty {
                    Divider()
                    ForEach(model.projects) { project in
                        Button {
                            model.selectProject(id: project.id)
                        } label: {
                            Label(
                                project.name,
                                systemImage: model.selectedProjectID == project.id ? "checkmark" : ""
                            )
                        }
                    }
                }

                Divider()

                Button {
                    onNewProject()
                } label: {
                    Label("New Project...", systemImage: "plus")
                }

                Button {
                    model.openSettings(tab: .permissions)
                } label: {
                    Label("Files and Permissions...", systemImage: "folder.badge.gearshape")
                }
            } label: {
                Label("Projects", systemImage: "folder")
            }
        }
    }

    @ViewBuilder
    private var modesSection: some View {
        Section {
            Button {
                model.webSearchEnabled.toggle()
            } label: {
                Label(
                    "Web Search",
                    systemImage: model.webSearchEnabled ? "checkmark.circle.fill" : "globe"
                )
            }

            Menu {
                let currentMode = model.selectedProject?.permissions.mode ?? model.activePermissionMode
                ForEach(AppPermissionMode.allCases) { mode in
                    Button {
                        setPermissionMode(mode)
                    } label: {
                        Label(
                            mode.label,
                            systemImage: currentMode == mode ? "checkmark" : ""
                        )
                    }
                }
            } label: {
                let currentMode = model.selectedProject?.permissions.mode ?? model.activePermissionMode
                Label("Tool Approval (\(currentMode.shortLabel))", systemImage: currentMode.systemImage)
            }
        }
    }

    // MARK: - Actions

    private func toggleMcpServer(_ server: McpServerConfig) {
        if model.globalMcpServers.contains(where: { $0.id == server.id }) {
            model.toggleGlobalMcpServer(id: server.id, isEnabled: !server.isEnabled)
        } else if let proj = model.selectedProject {
            model.toggleProjectMcpServer(projectID: proj.id, serverID: server.id, isEnabled: !server.isEnabled)
        }
    }

    private func setPermissionMode(_ mode: AppPermissionMode) {
        model.setEffectivePermissionMode(mode)
    }

    private func importGitOrUrlContext() {
        if let clip = NSPasteboard.general.string(forType: .string)?.trimmingCharacters(in: .whitespacesAndNewlines),
           clip.hasPrefix("http://") || clip.hasPrefix("https://") {
            onInsertPromptText("Analyze and resolve: \(clip)\n\n")
        } else {
            onInsertPromptText("Review and resolve GitHub issue: ")
        }
    }
}
