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
                .help(Text("Extracting documents...", bundle: .module))
                .accessibilityLabel(Text("Extracting documents", bundle: .module))
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
                        .themedFont(.small, weight: .semibold)
                    Image(systemName: "chevron.down")
                        .themedFont(.micro, weight: .bold)
                        .foregroundStyle(.appSecondary)
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
            .help(Text("Add files, folders, and tools (Cmd+U)", bundle: .module))
            .accessibilityLabel(Text("Add options and tools", bundle: .module))
        }
    }

    // MARK: - Sections

    @ViewBuilder
    private var filesSection: some View {
        Section {
            Button {
                onAttachFiles()
            } label: {
                Label { Text("Add Files or Photos...", bundle: .module) } icon: { Image(systemName: "paperclip") }
            }
            .keyboardShortcut("u", modifiers: .command)

            Button {
                onAttachFolder()
            } label: {
                Label {
                    Text("Add Folder…", bundle: .module)
                } icon: {
                    Image(systemName: "folder.badge.plus")
                }
            }

            Button {
                importGitOrUrlContext()
            } label: {
                Label { Text("Import GitHub Issue or URL", bundle: .module) } icon: { Image(systemName: "link.badge.plus") }
            }
        }
    }

    /// Built-in commands grouped by task (`SlashCommandMenuGroup`), with
    /// a search route first: inserting `/` opens the composer's own
    /// type-to-filter popup, which ranks every command and skill by name.
    @ViewBuilder
    private var slashCommandsSection: some View {
        Section {
            Menu {
                Button {
                    onInsertPromptText("/")
                } label: {
                    Label { Text("Search Commands...", bundle: .module) } icon: { Image(systemName: "magnifyingglass") }
                }

                Divider()

                ForEach(SlashCommandMenuGroup.grouped(BuiltInSlashCommand.all), id: \.group) { entry in
                    Menu {
                        ForEach(entry.commands) { command in
                            Button {
                                onInsertPromptText("/\(command.name) ")
                            } label: {
                                Label {
                                    Text(verbatim: "/\(command.name): ") + command.localizedSummary
                                } icon: {
                                    Image(systemName: command.iconName)
                                }
                            }
                        }
                    } label: {
                        Label { entry.group.title } icon: { Image(systemName: entry.group.systemImage) }
                    }
                }

                // The same two gates the submit-time slash handler applies:
                // a disabled or model-only skill listed here would be
                // refused with a toast the moment it was sent.
                let skills = model.effectiveSkills
                    .filter { $0.isEnabled && $0.manifest.userInvocable }
                if !skills.isEmpty {
                    Menu {
                        ForEach(skills) { skill in
                            Button {
                                onInsertPromptText("/\(skill.name) ")
                            } label: {
                                // A skill's name and description are the
                                // author's own text, shown as written.
                                Label {
                                    Text(verbatim: "/\(skill.name): \(skill.skillDescription)")
                                } icon: {
                                    Image(systemName: "wand.and.stars")
                                }
                            }
                        }
                    } label: {
                        Label { Text("Skills", bundle: .module) } icon: { Image(systemName: "wand.and.stars") }
                    }
                }
            } label: {
                Label { Text("Slash Commands", bundle: .module) } icon: { Image(systemName: "command") }
            }
        }
    }

    @ViewBuilder
    private var connectorsSection: some View {
        Section {
            Menu {
                let allMcpServers = AppToolCatalogMcp.resolvedServers(global: model.globalMcpServers, project: model.selectedProject)
                if allMcpServers.isEmpty {
                    Text("No MCP Servers Configured", bundle: .module)
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
                    Label { Text("Add MCP Server...", bundle: .module) } icon: { Image(systemName: "plus") }
                }

                Button {
                    model.openSettings(tab: .mcp)
                } label: {
                    Label { Text("Manage MCP Servers...", bundle: .module) } icon: { Image(systemName: "server.rack") }
                }
            } label: {
                Label { Text("Connectors (MCP)", bundle: .module) } icon: { Image(systemName: "point.3.filled.connected.trianglepath.dotted") }
            }
        }
    }

    @ViewBuilder
    private var pluginsSection: some View {
        Section {
            Menu {
                let managedSkills = model.allManagedSkills
                if managedSkills.isEmpty {
                    Text("No Skills Installed", bundle: .module)
                } else {
                    ForEach(managedSkills) { skill in
                        Menu {
                            Button {
                                onInsertPromptText("/\(skill.name) ")
                            } label: {
                                Label { Text("Use in Chat (/\(skill.name))", bundle: .module) } icon: { Image(systemName: "arrow.right.circle") }
                            }
                            // The submit handler refuses disabled and model-only skills.
                            .disabled(!skill.isEnabled || !skill.manifest.userInvocable)

                            Button {
                                model.toggleSkillEnabled(skill)
                            } label: {
                                Label {
                                    skill.isEnabled ? Text("Disable Skill", bundle: .module) : Text("Enable Skill", bundle: .module)
                                } icon: {
                                    Image(systemName: skill.isEnabled ? "checkmark" : "circle")
                                }
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
                    Label { Text("Create New Skill...", bundle: .module) } icon: { Image(systemName: "plus") }
                }

                Button {
                    model.openSettings(tab: .skills)
                } label: {
                    Label { Text("Manage Skills...", bundle: .module) } icon: { Image(systemName: "gearshape") }
                }
            } label: {
                Label { Text("Plugins and Skills", bundle: .module) } icon: { Image(systemName: "puzzlepiece.extension") }
            }
        }
    }

    @ViewBuilder
    private var projectsSection: some View {
        Section {
            Menu {
                Button {
                    model.chooseProjectForTask(id: nil)
                } label: {
                    Label { Text("All Chats (No Project)", bundle: .module) } icon: { Image(systemName: model.selectedProjectID == nil ? "checkmark" : "") }
                }

                if !model.projects.isEmpty {
                    Divider()
                    ForEach(model.projects) { project in
                        Button {
                            model.chooseProjectForTask(id: project.id)
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
                    Label { Text("New Project...", bundle: .module) } icon: { Image(systemName: "plus") }
                }

                Button {
                    model.openSettings(tab: .permissions)
                } label: {
                    Label { Text("Files and Permissions...", bundle: .module) } icon: { Image(systemName: "folder.badge.gearshape") }
                }
            } label: {
                Label { Text("Projects", bundle: .module) } icon: { Image(systemName: "folder") }
            }
        }
    }

    @ViewBuilder
    private var modesSection: some View {
        Section {
            if model.interactionMode == .chat {
                Toggle(isOn: Binding(
                    get: { model.chatToolsEnabled },
                    set: { newValue in
                        model.chatToolsEnabled = newValue
                        model.persistSettingsDebounced()
                    })) {
                    Label { Text("Tools in Chat", bundle: .module) } icon: { Image(systemName: "wrench.and.screwdriver") }
                }
            }

            // A Toggle so the menu shows the native checkmark and VoiceOver
            // reads the on/off state, rather than an icon swap.
            Toggle(isOn: $model.webSearchEnabled) {
                Label { Text("Web Search", bundle: .module) } icon: { Image(systemName: "globe") }
            }

            Menu {
                let currentMode = model.selectedProject?.permissions.mode ?? model.activePermissionMode
                ForEach(AppPermissionMode.allCases) { mode in
                    Button {
                        setPermissionMode(mode)
                    } label: {
                        Label {
                            approvalModeMenuTitle(mode)
                        } icon: {
                            Image(systemName: currentMode == mode ? "checkmark" : mode.systemImage)
                        }
                    }
                }
            } label: {
                let currentMode = model.selectedProject?.permissions.mode ?? model.activePermissionMode
                Label {
                    Text("Tool Approval (\(currentMode.approvalLabelText))", bundle: .module)
                } icon: {
                    Image(systemName: currentMode.systemImage)
                }
            }
        }
    }

    /// Each approval choice names what it ALLOWS, not only its short mode
    /// name: "Approve for me" or "Full access" alone does not say whether
    /// the sandbox stays on, and these rows change what tools may do to
    /// this machine without another prompt.
    private func approvalModeMenuTitle(_ mode: AppPermissionMode) -> Text {
        switch mode {
        case .ask:
            Text("Ask for approval: ask before tools edit files or use the internet", bundle: .module)
        case .auto:
            Text("Approve for me: run tools, ask before high-risk actions", bundle: .module)
        case .agentAuto:
            Text("Agent (classifier): a local classifier approves or blocks each call", bundle: .module)
        case .permissive:
            Text("Run automatically: run every tool without asking, sandbox on", bundle: .module)
        case .fullAccess:
            Text("Full access: run every tool without asking, sandbox off", bundle: .module)
        case .readOnly:
            Text("Strict Read-Only: only read and search, no edits or commands", bundle: .module)
        }
    }

    // MARK: - Actions

    private func toggleMcpServer(_ server: McpServerConfig) {
        if model.globalMcpServers.contains(where: { $0.id == server.id }) {
            if let project = model.selectedProject {
                model.setMcpOverride(serverID: server.id, enabled: !server.isEnabled, projectID: project.id)
            } else {
                model.toggleGlobalMcpServer(id: server.id, isEnabled: !server.isEnabled)
            }
        } else if let proj = model.selectedProject, proj.mcpServers.contains(where: { $0.id == server.id }) {
            model.toggleProjectMcpServer(projectID: proj.id, serverID: server.id, isEnabled: !server.isEnabled)
        } else {
            // Plugin contributions share their owner's enablement and trust decisions.
            model.openSettings(tab: .plugins)
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
