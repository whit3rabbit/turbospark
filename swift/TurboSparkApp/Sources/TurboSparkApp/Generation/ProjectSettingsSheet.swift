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
    @State private var showingSkills = false
    @State private var showingPlugins = false
    @State private var showingAdvancedSettings = false
    @State private var syntextIndexEnabled: Bool = false
    @State private var isIndexing: Bool = false
    @State private var hasSyntextIndex: Bool = false
    @State private var indexStatsMessage: String? = nil

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 20) {
                    generalSection
                    DisclosureGroup(isExpanded: $showingAdvancedSettings) {
                        VStack(alignment: .leading, spacing: 20) {
                            agentSection
                            syntextIndexingSection
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
                            if editingProject != nil {
                                HStack {
                                    Button { showingSkills = true } label: { Text("Skills", bundle: .module) }
                                    Button { showingPlugins = true } label: { Text("Plugins", bundle: .module) }
                                }
                            }
                            ProjectRulesSectionView(
                                rootDirectoryPath: rootDirectoryPath,
                                rulePreference: $rulePreference,
                                customInstructions: $customInstructions,
                                rulesAutoDetectedMessage: $rulesAutoDetectedMessage,
                                onAutoDetect: autoDetectRules
                            )
                        }
                        .padding(.top, 12)
                    } label: {
                        Text("Settings", bundle: .module)
                            .themedFont(.small, weight: .medium)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(24)
            }
            Divider()
            footer
        }
        .frame(width: 580, height: showingAdvancedSettings ? 660 : 390)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 16))
        .overlay {
            RoundedRectangle(cornerRadius: 16)
                .strokeBorder(.appBorder, lineWidth: 1)
                .allowsHitTesting(false)
        }
        .sheet(isPresented: $showingSkills) {
            VStack {
                Button { showingSkills = false } label: { Text("Done", bundle: .module) }
                SkillsSettingsPaneView(model: model, projectID: editingProject?.id)
            }.frame(minWidth: 780, minHeight: 580)
        }
        .sheet(isPresented: $showingPlugins) {
            VStack {
                Button { showingPlugins = false } label: { Text("Done", bundle: .module) }
                PluginSettingsPaneView(model: model, projectID: editingProject?.id)
            }.frame(minWidth: 780, minHeight: 580)
        }
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
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button { onDismiss() } label: { Text("Close", bundle: .module) }
                .buttonStyle(.plain)
                .foregroundStyle(.appSecondary)
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
                    .foregroundStyle(.appSecondary)
                TextField("e.g. My Rust Engine", text: $name)
                    .textFieldStyle(.plain)
                    .themedFont(.base)
                    .padding(10)
                    .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
                    .overlay {
                        RoundedRectangle(cornerRadius: 8)
                            .strokeBorder(.appBorder, lineWidth: 1)
                            .allowsHitTesting(false)
                    }
                    .accessibilityLabel("Project name")
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Codebase Root Directory", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                HStack(spacing: 8) {
                    TextField("/path/to/codebase", text: $rootDirectoryPath)
                        .textFieldStyle(.plain)
                        .themedFont(.base)
                        .padding(10)
                        .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
                        .overlay {
                            RoundedRectangle(cornerRadius: 8)
                                .strokeBorder(.appBorder, lineWidth: 1)
                                .allowsHitTesting(false)
                        }
                        .accessibilityLabel("Codebase root directory path")
                    Button {
                        selectFolder()
                    } label: { Text("Choose...", bundle: .module) }
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

            Picker(selection: $agentType) {
                ForEach(AppAgentType.allCases) { type in
                    Label(type.label, systemImage: type.systemImage).tag(type)
                }
            } label: { Text("Agent Profile", bundle: .module) }
            .pickerStyle(.menu)
            .accessibilityLabel("Agent profile")

            Text(agentType.descriptionText)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Autonomous Step Limit", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    Spacer()
                    Text(verbatim: "\(Int(maxAutonomousSteps)) steps")
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }
                Slider(value: $maxAutonomousSteps, in: 1...15, step: 1)
                    .accessibilityLabel("Autonomous step limit")
                    .accessibilityValue("\(Int(maxAutonomousSteps)) steps")
            }

            VStack(alignment: .leading, spacing: 4) {
                Toggle(isOn: $skillStateEnabled) {
                Text("Bounded execution state", bundle: .module)
            }
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
                .foregroundStyle(.appSecondary)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
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
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                if editingProject != nil {
                    Button {
                        showingMcpSheet = true
                    } label: {
                        Label { Text("Manage MCPs...", bundle: .module) } icon: { Image(systemName: "server.rack") }
                    }
                    .buttonStyle(.bordered)
                    .help("Manage MCP servers for this project")
                } else {
                    Text("Save project to configure MCPs", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }
            .padding(12)
            .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var syntextIndexingSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 2) {
                Text("Code Search Index (Syntext)", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Text("Fast ripgrep-style indexed code search for this project.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            let trimmedPath = rootDirectoryPath.trimmingCharacters(in: .whitespacesAndNewlines)
            Toggle(isOn: $syntextIndexEnabled) {
                Text("Enable Syntext project indexing", bundle: .module)
            }
            .disabled(!model.syntextIndexingEnabled || trimmedPath.isEmpty)

            if !model.syntextIndexingEnabled {
                HStack(spacing: 8) {
                    Image(systemName: "info.circle")
                        .foregroundStyle(.appSecondary)
                    Text("Code indexing is disabled globally (Settings > General).", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
            } else if trimmedPath.isEmpty {
                Text("Set a codebase root directory above to enable indexing.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .padding(8)
            } else if syntextIndexEnabled || hasSyntextIndex {
                let rootURL = URL(fileURLWithPath: trimmedPath)
                HStack(spacing: 12) {
                    if isIndexing {
                        ProgressView()
                            .controlSize(.small)
                        Text("Indexing project codebase...", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    } else {
                        VStack(alignment: .leading, spacing: 2) {
                            if let msg = indexStatsMessage {
                                Text(msg)
                                    .themedFont(.small)
                                    .foregroundStyle(.appSecondary)
                            } else {
                                Text("Index ready for fast code search.", bundle: .module)
                                    .themedFont(.small)
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                        Spacer()
                        if syntextIndexEnabled {
                            Button(hasSyntextIndex ? "Reindex" : "Index Now") {
                                reindexProject(rootURL: rootURL)
                            }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .accessibilityLabel("Reindex project codebase")
                        }

                        if hasSyntextIndex {
                            Button(role: .destructive) {
                                deleteProjectIndex(rootURL: rootURL)
                            } label: { Text("Remove Index", bundle: .module) }
                            .buttonStyle(.bordered)
                            .controlSize(.small)
                            .accessibilityLabel("Remove project codebase index")
                        }
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(TurboSparkTheme.surfaceColor, in: RoundedRectangle(cornerRadius: 8))
            }
        }
    }

    private var footer: some View {
        HStack {
            if let editing = editingProject {
                Button(role: .destructive) {
                    model.deleteProject(id: editing.id)
                    onDismiss()
                } label: { Text("Delete Project", bundle: .module) }
                .buttonStyle(.plain)
                .foregroundStyle(.red)
            }
            Spacer()
            Button { onDismiss() } label: { Text("Cancel", bundle: .module) }
                .buttonStyle(.plain)
            Button(editingProject == nil ? "Create Project" : "Save Changes") {
                save()
            }
            .buttonStyle(.borderedProminent)
            .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                || model.isRunning || model.submitting || model.pendingToolCall != nil)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private func populateInitialValues() {
        if let editing = editingProject {
            showingAdvancedSettings = true
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
            syntextIndexEnabled = editing.syntextIndexEnabled
            refreshIndexStats()
        } else {
            name = ""
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
            syntextIndexEnabled = false
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
            refreshIndexStats()
        }
    }

    private func refreshIndexStats() {
        let trimmedPath = rootDirectoryPath.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedPath.isEmpty else {
            indexStatsMessage = nil
            hasSyntextIndex = false
            return
        }
        let rootURL = URL(fileURLWithPath: trimmedPath)
        Task {
            if await SyntextIndexManager.shared.isIndexed(for: rootURL) {
                if let stats = try? await SyntextIndexManager.shared.stats(for: rootURL) {
                    let mb = Double(stats.indexSizeBytes) / 1_048_576.0
                    await MainActor.run {
                        hasSyntextIndex = true
                        indexStatsMessage = String(format: "Indexed: %d documents (%.1f MB)", stats.totalDocuments, mb)
                    }
                    return
                }
            }
            await MainActor.run {
                hasSyntextIndex = false
                indexStatsMessage = "Not indexed yet."
            }
        }
    }

    private func reindexProject(rootURL: URL) {
        guard !isIndexing else { return }
        isIndexing = true
        Task {
            do {
                let stats = try await SyntextIndexManager.shared.buildIndex(for: rootURL)
                let mb = Double(stats.indexSizeBytes) / 1_048_576.0
                await MainActor.run {
                    isIndexing = false
                    hasSyntextIndex = true
                    indexStatsMessage = String(format: "Indexed: %d documents (%.1f MB)", stats.totalDocuments, mb)
                }
            } catch {
                await MainActor.run {
                    isIndexing = false
                    indexStatsMessage = "Indexing failed: \(error.localizedDescription)"
                }
            }
        }
    }

    private func deleteProjectIndex(rootURL: URL) {
        guard !isIndexing else { return }
        isIndexing = true
        Task {
            do {
                try await SyntextIndexManager.shared.deleteIndex(for: rootURL)
                await MainActor.run {
                    isIndexing = false
                    hasSyntextIndex = false
                    indexStatsMessage = "Not indexed yet."
                }
            } catch {
                await MainActor.run {
                    isIndexing = false
                    indexStatsMessage = "Failed to remove index: \(error.localizedDescription)"
                }
            }
        }
    }

    private func autoDetectRules() {
        guard !rootDirectoryPath.isEmpty else { return }
        if let result = model.detectLiveProjectInstructions(directoryPath: rootDirectoryPath, preference: rulePreference) {
            let files = result.detectedFiles.joined(separator: ", ")
            rulesAutoDetectedMessage = "Detected \(files). These files are applied live to every project turn."
        } else {
            rulesAutoDetectedMessage = "No AGENTS.md, CLAUDE.md, or CONTEXT.md found in folder."
        }
    }

    private func save() {
        guard !model.isRunning, !model.submitting, model.pendingToolCall == nil else { return }
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
            var updated = model.projects.first { $0.id == editing.id } ?? editing
            updated.name = name.trimmingCharacters(in: .whitespacesAndNewlines)
            updated.rootDirectoryPath = resolvedPath
            updated.agentType = agentType
            updated.rulePreference = rulePreference
            updated.customInstructions = customInstructions
            updated.permissions = perms
            updated.maxAutonomousSteps = Int(maxAutonomousSteps)
            updated.skillStateEnabled = skillStateEnabled
            updated.forgeGuardrailsEnabled = guardrailsPref
            updated.syntextIndexEnabled = syntextIndexEnabled
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
            model.setInteractionMode(.projects)
            model.createProject(
                name: name.trimmingCharacters(in: .whitespacesAndNewlines),
                rootDirectoryPath: resolvedPath,
                agentType: agentType,
                rulePreference: rulePreference,
                customInstructions: customInstructions,
                permissions: perms,
                maxAutonomousSteps: Int(maxAutonomousSteps),
                skillStateEnabled: skillStateEnabled,
                forgeGuardrailsEnabled: guardrailsPref,
                syntextIndexEnabled: syntextIndexEnabled
            )
        }
        onDismiss()
    }
}
