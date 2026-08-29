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
    @State private var rulePreference: AppRulePreference = .agentsFirst
    @State private var guardrailsOption: AppProjectGuardrailsOption = .auto
    @State private var rulesAutoDetectedMessage: String?
    @State private var showingMcpSheet = false

    enum AppProjectGuardrailsOption: String, CaseIterable, Identifiable {
        case auto = "auto"
        case enabled = "enabled"
        case disabled = "disabled"

        var id: String { rawValue }

        var asOptionalBool: Bool? {
            switch self {
            case .auto: return nil
            case .enabled: return true
            case .disabled: return false
            }
        }

        static func from(optionalBool: Bool?) -> AppProjectGuardrailsOption {
            guard let b = optionalBool else { return .auto }
            return b ? .enabled : .disabled
        }
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    generalSection
                    agentSection
                    permissionsSection
                    mcpSection
                    rulesSection
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
                    .font(.headline)
                Text("Manage codebase root, agent profile, rules, and execution permissions.")
                    .font(.caption)
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
            Text("General")
                .font(.subheadline.weight(.semibold))
                .accessibilityAddTraits(.isHeader)

            VStack(alignment: .leading, spacing: 4) {
                Text("Project Name")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("e.g. My Rust Engine", text: $name)
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel("Project name")
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Codebase Root Directory")
                    .font(.caption)
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
            Text("Agent Specialization")
                .font(.subheadline.weight(.semibold))
                .accessibilityAddTraits(.isHeader)

            Picker("Agent Profile", selection: $agentType) {
                ForEach(AppAgentType.allCases) { type in
                    Label(type.label, systemImage: type.systemImage).tag(type)
                }
            }
            .pickerStyle(.menu)
            .accessibilityLabel("Agent profile")

            Text(agentType.descriptionText)
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Autonomous Step Limit")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Text("\(Int(maxAutonomousSteps)) steps")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                Slider(value: $maxAutonomousSteps, in: 1...15, step: 1)
                    .accessibilityLabel("Autonomous step limit")
                    .accessibilityValue("\(Int(maxAutonomousSteps)) steps")
            }
        }
    }

    private var permissionsSection: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Tool Permissions & Risk Policy")
                    .font(.subheadline.weight(.semibold))
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Menu("Presets") {
                    Button("Auto (Recommended)") { applyPreset(.auto) }
                    Button("Always Ask") { applyPreset(.alwaysAsk) }
                    Button("Permissive (Allow All)") { applyPreset(.permissive) }
                    Button("Read Only") { applyPreset(.readOnly) }
                }
                .menuStyle(.borderlessButton)
                .font(.caption)
                .accessibilityLabel("Permission presets")
                .accessibilityHint("Applies a recommended set of tool permissions")
            }

            // Mode Selector
            VStack(alignment: .leading, spacing: 6) {
                Text("Execution Mode")
                    .font(.caption.weight(.medium))
                    .foregroundStyle(.secondary)

                Picker("Execution Mode", selection: $permissionMode) {
                    ForEach(AppPermissionMode.allCases) { mode in
                        Label(mode.shortLabel, systemImage: mode.systemImage).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityLabel("Permission execution mode")

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: permissionMode.systemImage)
                        .foregroundStyle(TurboSparkTheme.accentColor)
                        .font(.caption)
                    Text(permissionMode.descriptionText)
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 6))
            }

            // Granular Category Overrides
            VStack(spacing: 8) {
                permissionRow(title: "File Reading", desc: "List directories and inspect code files", selection: $fileReadPermission)
                permissionRow(title: "File Writing", desc: "Create, edit, or modify files", selection: $fileWritePermission)
                permissionRow(title: "Terminal Commands", desc: "Execute shell commands in project root", selection: $terminalPermission)
                permissionRow(title: "Web Requests", desc: "Fetch web documentation and search", selection: $webPermission)
                permissionRow(title: "MCP External Tools", desc: "Invoke external MCP server tools", selection: $mcpPermission)
                permissionRow(title: "Automation & Crons", desc: "Schedule background tasks and monitors", selection: $automationPermission)
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))

            // Forge Guardrails project option
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Forge Tool-Call Guardrails")
                        .font(.callout.weight(.medium))
                    Text("Repair malformed dialect calls and validate schemas")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Picker("", selection: $guardrailsOption) {
                    Text("Auto (Model Default)").tag(AppProjectGuardrailsOption.auto)
                    Text("Always Enabled").tag(AppProjectGuardrailsOption.enabled)
                    Text("Always Disabled").tag(AppProjectGuardrailsOption.disabled)
                }
                .pickerStyle(.menu)
                .frame(width: 175)
                .accessibilityLabel("Forge Guardrails project preference")
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func permissionRow(title: String, desc: String, selection: Binding<AppToolPermission>) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.callout.weight(.medium))
                Text(desc)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Picker("", selection: selection) {
                ForEach(AppToolPermission.allCases) { perm in
                    Text(perm.label).tag(perm)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 175)
            .accessibilityLabel("\(title) permission")
        }
    }

    private var mcpSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("MCP External Servers & Detection")
                        .font(.subheadline.weight(.semibold))
                        .accessibilityAddTraits(.isHeader)
                    let count = editingProject?.mcpServers.count ?? 0
                    Text(count > 0 ? "\(count) server(s) configured for this project." : "Import .mcp.json or configure codebase MCP tools.")
                        .font(.caption)
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
                    Text("Save project to configure MCPs")
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                }
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var rulesSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Project Rules & Instructions")
                    .font(.subheadline.weight(.semibold))
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                if !rootDirectoryPath.isEmpty {
                    Button("Detect CLAUDE.md / AGENTS.md") {
                        autoDetectRules()
                    }
                    .font(.caption)
                    .buttonStyle(.borderless)
                    .help("Detect project rules from AGENTS.md or CLAUDE.md")
                }
            }

            HStack {
                Text("Conflict Preference")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                Picker("Conflict Preference", selection: $rulePreference) {
                    ForEach(AppRulePreference.allCases) { pref in
                        Text(pref.label).tag(pref)
                    }
                }
                .pickerStyle(.menu)
                .frame(width: 180)
                .accessibilityLabel("Rules conflict preference")
            }

            if let rulesAutoDetectedMessage {
                Text(rulesAutoDetectedMessage)
                    .font(.caption)
                    .foregroundStyle(TurboSparkTheme.accentColor)
            }

            TextEditor(text: $customInstructions)
                .font(.callout.monospaced())
                .frame(height: 100)
                .padding(4)
                .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
                .overlay(RoundedRectangle(cornerRadius: 8).stroke(Color.secondary.opacity(0.2), lineWidth: 0.5))
                .accessibilityLabel("Project rules and instructions")
                .accessibilityHint("Free-form text sent to the model as project-specific guidance")
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
            guardrailsOption = AppProjectGuardrailsOption.from(optionalBool: editing.forgeGuardrailsEnabled)
        } else {
            name = "New Project"
            agentType = .coder
            rulePreference = .agentsFirst
            guardrailsOption = .auto
            applyPreset(.auto)
        }
    }

    private func applyPreset(_ permissions: AppProjectPermissions) {
        permissionMode = permissions.mode
        fileReadPermission = permissions.fileRead
        fileWritePermission = permissions.fileWrite
        terminalPermission = permissions.terminal
        webPermission = permissions.web
        mcpPermission = permissions.mcp
        automationPermission = permissions.automation
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
            updated.forgeGuardrailsEnabled = guardrailsPref
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
                forgeGuardrailsEnabled: guardrailsPref
            )
        }
        onDismiss()
    }
}
