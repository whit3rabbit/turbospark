import AppKit
import SwiftUI

/// Modal configuration sheet for editing project metadata, permissions, agent profiles, rules, and MCP servers.
struct ProjectSettingsSheet: View {
    @Environment(\.appTheme) private var theme
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
    @State private var browserPermission: AppToolPermission = .ask
    @State private var computerUsePermission: AppToolPermission = .ask
    @State private var browserOriginAllowlist: [String] = []
    /// The origins as loaded, to tell an edit in this sheet from a grant an
    /// approval card made while it was open.
    @State private var loadedBrowserOriginAllowlist: [String] = []
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
    @State private var confirmingDelete = false

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
                                browserPermission: $browserPermission,
                                browserOriginAllowlist: $browserOriginAllowlist,
                                computerUsePermission: $computerUsePermission,
                                guardrailsOption: $guardrailsOption
                            )
                            mcpSection
                            if editingProject != nil {
                                HStack {
                                    Button { showingSkills = true } label: { Text("Skills", bundle: .module) }
                                        .appAction(.secondary)
                                    Button { showingPlugins = true } label: { Text("Plugins", bundle: .module) }
                                        .appAction(.secondary)
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
        .frame(minWidth: 540, idealWidth: 580, maxWidth: 640, minHeight: showingAdvancedSettings ? 500 : 360, idealHeight: showingAdvancedSettings ? 660 : 390)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .strokeBorder(.appBorder, lineWidth: 1)
                .allowsHitTesting(false)
        }
        .sheet(isPresented: $showingSkills) {
            VStack {
                Button { showingSkills = false } label: { Text("Done", bundle: .module) }
                    .keyboardShortcut(.defaultAction)
                SkillsSettingsPaneView(model: model, projectID: editingProject?.id)
            }.frame(minWidth: 780, minHeight: 580)
        }
        .sheet(isPresented: $showingPlugins) {
            VStack {
                Button { showingPlugins = false } label: { Text("Done", bundle: .module) }
                    .keyboardShortcut(.defaultAction)
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
                (editingProject == nil
                    ? Text("New Project", bundle: .module)
                    : Text("Project Settings", bundle: .module))
                    .themedFont(.base, weight: .semibold)
                Text("Manage codebase root, agent profile, rules, and execution permissions.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button { onDismiss() } label: { Text("Close", bundle: .module) }
                .appAction(.quiet)
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
                TextField(text: $name, prompt: Text("e.g. My Rust Engine", bundle: .module)) {
                    Text("Project name", bundle: .module)
                }
                    .textFieldStyle(.plain)
                    .themedFont(.base)
                    .padding(10)
                    .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                    .overlay {
                        RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                            .strokeBorder(.appBorder, lineWidth: 1)
                            .allowsHitTesting(false)
                    }
                    .accessibilityLabel(Text("Project name", bundle: .module))
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Codebase Root Directory", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                HStack(spacing: 8) {
                    // The prompt is a path example, not prose.
                    TextField(text: $rootDirectoryPath, prompt: Text(verbatim: "/path/to/codebase")) {
                        Text("Codebase root directory path", bundle: .module)
                    }
                        .textFieldStyle(.plain)
                        .themedFont(.base)
                        .padding(10)
                        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                                .strokeBorder(.appBorder, lineWidth: 1)
                                .allowsHitTesting(false)
                        }
                        .accessibilityLabel(Text("Codebase root directory path", bundle: .module))
                    Button {
                        selectFolder()
                    } label: { Text("Choose...", bundle: .module) }
                    .appAction(.secondary)
                    .help(Text("Choose codebase root folder", bundle: .module))
                    .accessibilityLabel(Text("Choose codebase root folder", bundle: .module))
                    .accessibilityHint(Text("Opens folder picker to select root directory", bundle: .module))
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
                    // Every agent label and description is a catalog key.
                    Label {
                        Text(LocalizedStringKey(type.label), bundle: .module)
                    } icon: {
                        Image(systemName: type.systemImage)
                    }
                    .tag(type)
                }
            } label: { Text("Agent Profile", bundle: .module) }
            .pickerStyle(.menu)
            .accessibilityLabel(Text("Agent profile", bundle: .module))

            Text(LocalizedStringKey(agentType.descriptionText), bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))

            VStack(alignment: .leading, spacing: 4) {
                HStack {
                    Text("Autonomous Step Limit", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    Spacer()
                    Text("Steps: \(Int(maxAutonomousSteps))", bundle: .module)
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                }
                Slider(value: $maxAutonomousSteps, in: 1...15, step: 1)
                    .accessibilityLabel(Text("Autonomous step limit", bundle: .module))
                    .accessibilityValue(Text("Steps: \(Int(maxAutonomousSteps))", bundle: .module))
            }

            VStack(alignment: .leading, spacing: 4) {
                Toggle(isOn: $skillStateEnabled) {
                Text("Bounded execution state", bundle: .module)
            }
                    .accessibilityHint(Text(
                        "Carry a compact state between steps instead of the full transcript", bundle: .module))
                (skillStateEnabled
                    ? Text(
                        "Each step sees the task, a compact state and the newest tool result. Long runs stay within the context window and cost 3 to 5x fewer tokens.",
                        bundle: .module)
                    : Text(
                        "Each step sees the whole transcript. Simple and exact, but long runs grow until they hit the context window.",
                        bundle: .module))
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
            .padding(10)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
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
                    (liveCount > 0
                        ? Text("Servers configured for this project: \(liveCount)", bundle: .module)
                        : Text("Import .mcp.json or configure codebase MCP tools.", bundle: .module))
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
                    .appAction(.secondary)
                    .help(Text("Manage MCP servers for this project", bundle: .module))
                } else {
                    Text("Save project to configure MCPs", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }
            .padding(12)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
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
                .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            } else if trimmedPath.isEmpty {
                Text("Set a codebase root directory above to enable indexing.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .padding(8)
            } else if syntextIndexEnabled || hasSyntextIndex {
                let rootURL = URL(fileURLWithPath: trimmedPath)
                HStack(spacing: 12) {
                    if isIndexing {
                        AppProgressBar(
                            value: nil,
                            label: Text("Indexing project codebase...", bundle: .module),
                            size: .compact
                        )
                        .frame(width: 80)
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
                            Button {
                                reindexProject(rootURL: rootURL)
                            } label: {
                                hasSyntextIndex ? Text("Reindex", bundle: .module) : Text("Index Now", bundle: .module)
                            }
                            .appAction(.secondary, size: .small)
                            .accessibilityLabel(Text("Reindex project codebase", bundle: .module))
                        }

                        if hasSyntextIndex {
                            Button {
                                deleteProjectIndex(rootURL: rootURL)
                            } label: { Text("Remove Index", bundle: .module) }
                            .appAction(.secondary, size: .small)
                            .accessibilityLabel(Text("Remove project codebase index", bundle: .module))
                        }
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            }
        }
    }

    private var footer: some View {
        HStack {
            if let editing = editingProject {
                // Asks first, like the sidebar's Delete Project: the
                // project's configuration has no way back.
                Button(role: .destructive) {
                    confirmingDelete = true
                } label: { Text("Delete Project", bundle: .module) }
                .appAction(.destructive)
                .confirmationDialog(
                    Text("Delete Project", bundle: .module),
                    isPresented: $confirmingDelete,
                    titleVisibility: .visible
                ) {
                    Button(role: .destructive) {
                        model.deleteProject(id: editing.id)
                        onDismiss()
                    } label: { Text("Delete Project", bundle: .module) }
                    Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
                } message: {
                    Text(verbatim: editing.name)
                }
            }
            Spacer()
            Button { onDismiss() } label: { Text("Cancel", bundle: .module) }
                .appAction(.secondary)
                .keyboardShortcut(.cancelAction)
            Button {
                save()
            } label: {
                editingProject == nil ? Text("Create Project", bundle: .module) : Text("Save Changes", bundle: .module)
            }
            .appAction(.primary)
            .keyboardShortcut(.defaultAction)
            .disabled(isSaveDisabled)
            .help(saveDisabledReason)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var isSaveDisabled: Bool {
        name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            || model.isRunning || model.submitting || model.pendingToolCall != nil
    }

    private var saveDisabledReason: Text {
        if name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return Text("Enter a project name to save", bundle: .module)
        }
        if model.isRunning || model.submitting || model.pendingToolCall != nil {
            return Text("Cannot save project while generation or tool execution is active", bundle: .module)
        }
        return editingProject == nil ? Text("Create Project", bundle: .module) : Text("Save Changes", bundle: .module)
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
            browserPermission = editing.permissions.browser
            computerUsePermission = editing.permissions.computerUse
            browserOriginAllowlist = editing.permissions.browserOriginAllowlist
            loadedBrowserOriginAllowlist = editing.permissions.browserOriginAllowlist
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
            browserPermission = defaultPerms.browser
            computerUsePermission = defaultPerms.computerUse
            browserOriginAllowlist = defaultPerms.browserOriginAllowlist
            syntextIndexEnabled = false
        }
    }

    private func selectFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = true
        panel.prompt = AppStrings.localized("Select Codebase Root")
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
                    let message = Self.indexedMessage(documents: Int(stats.totalDocuments), bytes: Int64(stats.indexSizeBytes))
                    await MainActor.run {
                        hasSyntextIndex = true
                        indexStatsMessage = message
                    }
                    return
                }
            }
            await MainActor.run {
                hasSyntextIndex = false
                indexStatsMessage = AppStrings.localized("Not indexed yet.")
            }
        }
    }

    private func reindexProject(rootURL: URL) {
        guard !isIndexing else { return }
        isIndexing = true
        Task {
            do {
                let stats = try await SyntextIndexManager.shared.buildIndex(for: rootURL)
                let message = Self.indexedMessage(documents: Int(stats.totalDocuments), bytes: Int64(stats.indexSizeBytes))
                await MainActor.run {
                    isIndexing = false
                    hasSyntextIndex = true
                    indexStatsMessage = message
                }
            } catch {
                await MainActor.run {
                    isIndexing = false
                    indexStatsMessage = AppStrings.localized("Indexing failed: \(error.localizedDescription)")
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
                    indexStatsMessage = AppStrings.localized("Not indexed yet.")
                }
            } catch {
                await MainActor.run {
                    isIndexing = false
                    indexStatsMessage = AppStrings.localized("Failed to remove index: \(error.localizedDescription)")
                }
            }
        }
    }

    private func autoDetectRules() {
        guard !rootDirectoryPath.isEmpty else { return }
        if let result = model.detectLiveProjectInstructions(directoryPath: rootDirectoryPath, preference: rulePreference) {
            let files = result.detectedFiles.joined(separator: ", ")
            rulesAutoDetectedMessage = AppStrings.localized(
                "Detected \(files). These files are applied live to every project turn.")
        } else {
            rulesAutoDetectedMessage = AppStrings.localized(
                "No AGENTS.md, CLAUDE.md, CONTEXT.md, or SOUL.md found in folder.")
        }
    }

    /// The index summary in the UI language. The size is formatted to one
    /// decimal first, so the catalog key carries a `%@` rather than a
    /// floating-point specifier.
    private static func indexedMessage(documents: Int, bytes: Int64) -> String {
        let megabytes = (Double(bytes) / 1_048_576.0).formatted(.number.precision(.fractionLength(1)))
        return AppStrings.localized("Indexed documents: \(documents) (\(megabytes) MB)")
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
            automation: automationPermission,
            browser: browserPermission,
            computerUse: computerUsePermission,
            browserOriginAllowlist: browserOriginAllowlist
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
                updated.permissions = Self.mergedPermissions(
                    edited: perms, live: live.permissions,
                    loadedOrigins: loadedBrowserOriginAllowlist)
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

    /// The sheet's edits laid over the live project's permissions.
    ///
    /// The sheet edits the mode, the category matrix and the origin list. The
    /// MCP rules and REPL roots are edited elsewhere (the MCP sheet, the
    /// approval cards) and always come from `live`. The origins come from
    /// `live` too unless the sheet changed them, so a grant an approval card
    /// made while the sheet was open survives a save that never touched them.
    static func mergedPermissions(
        edited: AppProjectPermissions, live: AppProjectPermissions, loadedOrigins: [String]
    ) -> AppProjectPermissions {
        var merged = edited
        merged.mcpAllowRules = live.mcpAllowRules
        merged.mcpDenyRules = live.mcpDenyRules
        merged.replFileAccessRoots = live.replFileAccessRoots
        if edited.browserOriginAllowlist == loadedOrigins {
            merged.browserOriginAllowlist = live.browserOriginAllowlist
        }
        return merged
    }
}
