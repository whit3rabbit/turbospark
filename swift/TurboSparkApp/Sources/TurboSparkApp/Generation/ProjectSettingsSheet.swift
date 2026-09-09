import AppKit
import SwiftUI

/// Modal configuration sheet for editing project metadata, permissions, agent profiles, rules, and MCP servers.
struct ProjectSettingsSheet: View {
    @ObservedObject var model: AppModel
    let editingProject: AppProject?
    let onDismiss: () -> Void

    @State private var name: String = ""
    @State private var rootDirectoryPath: String = ""
    @State private var agentType: AppAgentType = .coder
    @State private var customInstructions: String = ""
    @State private var permissionMode: AppPermissionMode = .auto
    @State private var fileReadPermission: AppToolPermission = .allow
    @State private var fileWritePermission: AppToolPermission = .ask
    @State private var terminalPermission: AppToolPermission = .ask
    @State private var webPermission: AppToolPermission = .allow
    @State private var mcpPermission: AppToolPermission = .ask
    @State private var automationPermission: AppToolPermission = .ask
    @State private var maxAutonomousSteps: Double = 5
    @State private var skillStateEnabled: Bool = false
    @State private var rulePreference: AppRulePreference = .agentsFirst
    @State private var guardrailsOption: AppProjectGuardrailsOption = .auto
    @State private var rulesAutoDetectedMessage: String?
    @State private var showingMcpSheet = false

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    generalSection
                    agentSection
                    ProjectPermissionsSectionView(
                        permissionMode: $permissionMode,
                        fileReadPermission: $fileReadPermission,
                        fileWritePermission: $fileWritePermission,
                        terminalPermission: $terminalPermission,
                        webPermission: $webPermission,
                        mcpPermission: $mcpPermission,
                        automationPermission: $automationPermission,
                        guardrailsOption: $guardrailsOption
                    )
                    mcpSection
                    ProjectRulesSectionView(
                        rootDirectoryPath: rootDirectoryPath,
                        rulePreference: $rulePreference,
                        customInstructions: $customInstructions,
                        rulesAutoDetectedMessage: $rulesAutoDetectedMessage,
                        onAutoDetect: autoDetectRules
                    )
                }
                .padding(20)
            }
            Divider()
            footer
        }
        .frame(width: 580, height: 680)
        .onAppear(perform: populateInitialValues)
        .sheet(isPresented: $showingMcpSheet) {
            if let project = editingProject {
                ProjectMcpSettingsSheet(
                    model: model,
                    projectID: project.id,
                    onDismiss: { showingMcpSheet = false }
                )
            }
        }
    }

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(editingProject == nil ? "New Project" : "Project Settings")
                    .themedFont(.base, weight: .semibold)
                Text("Manage codebase root, agent profile, rules, and execution permissions.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") { onDismiss() }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var generalSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("General", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            VStack(alignment: .leading, spacing: 4) {
                Text("Project Name", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.secondary)
                TextField("e.g. My Rust Engine", text: $name)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Project name")
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Codebase Root Directory", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.secondary)
                HStack(spacing: 8) {
                    TextField("/path/to/codebase", text: $rootDirectoryPath)
                        .textFieldStyle(.roundedBorder)
                        .accessibilityLabel("Codebase root directory path")
                    Button("Choose...") {
                        selectFolder()
                    }
                    .buttonStyle(.bordered)
                    .help("Choose codebase root folder")
                    .accessibilityLabel("Choose codebase root folder")
                    .accessibilityHint("Opens folder picker to select root directory")
                }
            }
        }
    }

    private var agentSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Agent Specialization", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            Picker("Agent Profile", selection: $agentType) {
                ForEach(AppAgentType.allCases) { type in
                    Label(type.label, systemImage: type.systemImage).tag(type)
                }
            }
            .pickerStyle(.menu)
            .accessibilityLabel("Agent profile")

            Text(agentType.descriptionText)
                .themedFont(.small)
                .foregroundStyle(.secondary)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Autonomous Step Limit", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(maxAutonomousSteps)) steps", bundle: .module)
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                Slider(value: $maxAutonomousSteps, in: 1...15, step: 1)
                    .accessibilityLabel("Autonomous step limit")
                    .accessibilityValue("\(Int(maxAutonomousSteps)) steps")
            }

            VStack(alignment: .leading, spacing: 4) {
                Toggle("Bounded execution state", isOn: $skillStateEnabled)
                    .accessibilityHint(
                        "Carry a compact state between steps instead of the full transcript")
                Text(
                    skillStateEnabled
                        ? "Each step sees the task, a compact state and the newest tool result. "
                            + "Long runs stay within the context window and cost 3 to 5x fewer tokens."
                        : "Each step sees the whole transcript. Simple and exact, but long runs "
                            + "grow until they hit the context window."
                )
                .themedFont(.small)
                .foregroundStyle(.secondary)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var mcpSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("MCP External Servers & Detection", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    // Read the LIVE project, not the sheet-open snapshot:
                    // the count is wrong after the nested MCP sheet edits
                    // servers otherwise.
                    let liveCount = model.projects.first { $0.id == editingProject?.id }?
                        .mcpServers.count ?? editingProject?.mcpServers.count ?? 0
                    Text(liveCount > 0 ? "\(liveCount) server(s) configured for this project." : "Import .mcp.json or configure codebase MCP tools.")
                        .themedFont(.small)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                if editingProject != nil {
                    Button {
                        showingMcpSheet = true
                    } label: {
                        Label("Manage MCPs...", systemImage: "server.rack")
                    }
                    .buttonStyle(.bordered)
                    .help("Manage MCP servers for this project")
                } else {
                    Text("Save project to configure MCPs", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var footer: some View {
        HStack {
            if let editing = editingProject {
                Button("Delete Project", role: .destructive) {
                    model.deleteProject(id: editing.id)
                    onDismiss()
                }
                .buttonStyle(.plain)
                .foregroundStyle(.red)
            }
            Spacer()
            Button("Cancel") { onDismiss() }
                .buttonStyle(.plain)
            Button(editingProject == nil ? "Create Project" : "Save Changes") {
                save()
            }
            .buttonStyle(.borderedProminent)
            .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private func populateInitialValues() {
        if let editing = editingProject {
            name = editing.name
            rootDirectoryPath = editing.rootDirectoryPath ?? ""
            agentType = editing.agentType
            rulePreference = editing.rulePreference
            customInstructions = editing.customInstructions
            permissionMode = editing.permissions.mode
            fileReadPermission = editing.permissions.fileRead
            fileWritePermission = editing.permissions.fileWrite
            terminalPermission = editing.permissions.terminal
            webPermission = editing.permissions.web
            mcpPermission = editing.permissions.mcp
            automationPermission = editing.permissions.automation
            maxAutonomousSteps = Double(editing.maxAutonomousSteps)
            skillStateEnabled = editing.skillStateEnabled
            guardrailsOption = AppProjectGuardrailsOption.from(optionalBool: editing.forgeGuardrailsEnabled)
        } else {
            name = "New Project"
            agentType = .coder
            rulePreference = .agentsFirst
            guardrailsOption = .auto
            // `.standard`, matching `AppProject.init`, its decode fallback,
            // and `AppModel.createProject`. This branch used to seed
            // `.auto`, which is `terminal: .allow` / `fileWrite: .allow` /
            // `mcp: .allow` -- so every project made through this sheet ran
            // model-proposed shell commands with no prompt, while every
            // non-UI path defaulted to asking. Four call sites, one default:
            // a fifth that spells its own is the bug this comment exists to
            // stop. `.auto` is still one click away in the preset picker.
            let defaultPerms = AppProjectPermissions.newProjectDefault
            permissionMode = defaultPerms.mode
            fileReadPermission = defaultPerms.fileRead
            fileWritePermission = defaultPerms.fileWrite
            terminalPermission = defaultPerms.terminal
            webPermission = defaultPerms.web
            mcpPermission = defaultPerms.mcp
            automationPermission = defaultPerms.automation
        }
    }

    private func selectFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = "Select Codebase Root"
        if panel.runModal() == .OK, let url = panel.url {
            rootDirectoryPath = url.path
            if name == "New Project" || name.isEmpty {
                name = url.lastPathComponent
            }
            autoDetectRules()
        }
    }

    private func autoDetectRules() {
        guard !rootDirectoryPath.isEmpty else { return }
        if let result = model.detectProjectRulesDetails(directoryPath: rootDirectoryPath, preference: rulePreference) {
            customInstructions = result.content
            rulesAutoDetectedMessage = result.statusDescription
        } else {
            rulesAutoDetectedMessage = "No AGENTS.md / CLAUDE.md found in folder."
        }
    }

    private func save() {
        let perms = AppProjectPermissions(
            mode: permissionMode,
            fileRead: fileReadPermission,
            fileWrite: fileWritePermission,
            terminal: terminalPermission,
            web: webPermission,
            mcp: mcpPermission,
            automation: automationPermission
        )

        let trimmedPath = rootDirectoryPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let resolvedPath = trimmedPath.isEmpty ? nil : trimmedPath
        let guardrailsPref = guardrailsOption.asOptionalBool

        if let editing = editingProject {
            var updated = editing
            updated.name = name.trimmingCharacters(in: .whitespacesAndNewlines)
            updated.rootDirectoryPath = resolvedPath
            updated.agentType = agentType
            updated.rulePreference = rulePreference
            updated.customInstructions = customInstructions
            updated.permissions = perms
            updated.maxAutonomousSteps = Int(maxAutonomousSteps)
            updated.skillStateEnabled = skillStateEnabled
            updated.forgeGuardrailsEnabled = guardrailsPref
            // `editing` is the snapshot from sheet-open. The nested MCP
            // sheet edited the LIVE project (servers, allow/deny rules),
            // so carrying the snapshot's stale copies over `updateProject`
            // would silently revert every change made there. This sheet
            // owns none of those fields.
            if let live = model.projects.first(where: { $0.id == editing.id }) {
                updated.mcpServers = live.mcpServers
                updated.permissions.mcpAllowRules = live.permissions.mcpAllowRules
                updated.permissions.mcpDenyRules = live.permissions.mcpDenyRules
            }
            model.updateProject(updated)
        } else {
            model.createProject(
                name: name.trimmingCharacters(in: .whitespacesAndNewlines),
                rootDirectoryPath: resolvedPath,
                agentType: agentType,
                rulePreference: rulePreference,
                customInstructions: customInstructions,
                permissions: perms,
                maxAutonomousSteps: Int(maxAutonomousSteps),
                skillStateEnabled: skillStateEnabled,
                forgeGuardrailsEnabled: guardrailsPref
            )
        }
        onDismiss()
    }
}
