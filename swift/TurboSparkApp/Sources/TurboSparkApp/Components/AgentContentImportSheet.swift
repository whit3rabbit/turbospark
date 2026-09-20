import AppKit
import SwiftUI

/// The unified Import wizard: scans other agent tools' home folders for
/// skills, agent definitions, and global MCP servers, and COPIES the
/// selected items into TurboSpark's own folders.
///
/// This is the only path through which other agents' content enters the
/// app when `MacAppSettings.autoLoadExternalAgentContent` is off (the
/// default): discovery never reads those folders, so nothing appears
/// uninvited. Nothing is pre-selected -- each row is an explicit choice,
/// which is the entire point of the wizard.
///
/// macOS 14 SDK note: this struct is `@MainActor` because `body` touches
/// `@MainActor`-isolated state (`AppModel`); the sheet chrome pattern is
/// shared with `McpImportSheet` and `SkillImportSheet`.
@MainActor
public struct AgentContentImportSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    public enum ImportCategory: String, CaseIterable, Identifiable, Sendable {
        case skills
        case agents
        case mcpServers

        public var id: String { rawValue }

        var localizedName: String {
            switch self {
            case .skills: return String(localized: "Skills", bundle: .module)
            case .agents: return String(localized: "Agents", bundle: .module)
            case .mcpServers: return String(localized: "MCP Servers", bundle: .module)
            }
        }
    }

    struct ImportCandidate: Identifiable {
        let id = UUID()
        let name: String
        let detail: String
        /// Path shown under the row, e.g. `~/.claude/skills/pdf` or the
        /// config file for an MCP entry.
        let sourceLocation: String
        let alreadyImported: Bool
        /// Skill directory or file to copy (Skills tab).
        let skillSource: URL?
        /// Agent definition file to copy (Agents tab).
        let agentSource: URL?
        /// Parsed server to add, disabled (MCP Servers tab). The command or
        /// URL is shown in `detail` BEFORE import, per the same rule as
        /// `McpImportSheet`.
        let mcpServer: McpServerConfig?
        var isSelected: Bool = false
    }

    struct CandidateGroup: Identifiable {
        let id: String
        let label: String
        var items: [ImportCandidate]
    }

    @State private var category: ImportCategory
    @State private var groups: [CandidateGroup] = []
    @State private var isLoading = false
    @State private var importToProjectScope: Bool = false
    private let capturedProjectID: UUID?

    public init(
        model: AppModel,
        initialCategory: ImportCategory = .skills,
        projectID: UUID? = nil
    ) {
        self.model = model
        self._category = State(initialValue: initialCategory)
        self.capturedProjectID = projectID ?? model.selectedProject?.id
        self._importToProjectScope = State(initialValue: projectID != nil)
    }

    private var project: AppProject? {
        model.projects.first { $0.id == capturedProjectID }
    }

    private var selectedCount: Int {
        groups.flatMap(\.items).filter(\.isSelected).count
    }

    public var body: some View {
        VStack(spacing: 0) {
            headerBar

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            controlsBar

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            content
        }
        .frame(minWidth: 680, minHeight: 520)
        .onAppear { scan() }
        .onChange(of: category) { _, _ in scan() }
        .onChange(of: importToProjectScope) { _, _ in scan() }
    }

    // MARK: - Chrome

    private var headerBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Import from Other Agents", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                Text(
                    "Copy skills, agents, and MCP servers from Claude, Codex, Cursor, and other agent tools into TurboSpark. Nothing is read from those folders until you import it here.",
                    bundle: .module
                )
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button {
                dismiss()
            } label: { Text("Close", bundle: .module) }
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.appPage)
    }

    private var controlsBar: some View {
        HStack {
            Picker(selection: $category) {
                ForEach(ImportCategory.allCases) { c in
                    Text(c.localizedName).tag(c)
                }
            } label: { Text("Category", bundle: .module) }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)

            Spacer()

            if category != .mcpServers {
                Text("Target Scope:", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                Picker(selection: $importToProjectScope) {
                    Text("User Scope (~/.turbospark)", bundle: .module).tag(false)
                    Text("Project Scope (.turbospark)", bundle: .module).tag(true)
                } label: { Text("Scope", bundle: .module) }
                .pickerStyle(.segmented)
                .frame(maxWidth: 320)
                .disabled(project == nil && !importToProjectScope)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(.appSurface.opacity(0.4))
    }

    // MARK: - Content

    @ViewBuilder
    private var content: some View {
        if isLoading {
            VStack(spacing: 12) {
                ProgressView()
                Text("Scanning other agents' folders...", bundle: .module)
                    .themedFont(.base)
                    .foregroundStyle(.appSecondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if groups.isEmpty {
            VStack(spacing: 16) {
                Image(systemName: "folder.badge.questionmark")
                    .themedFont(.display)
                    .foregroundStyle(.appSecondary)
                Text("No importable content found in other agent locations.", bundle: .module)
                    .themedFont(.base)
                    .foregroundStyle(.appSecondary)
                if category != .mcpServers {
                    Button {
                        selectCustomFolder()
                    } label: { Text("Choose Custom Folder...", bundle: .module) }
                        .buttonStyle(.bordered)
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(40)
        } else {
            VStack(spacing: 0) {
                candidateList
                Rectangle()
                    .fill(.appBorder)
                    .frame(height: 1)
                footerBar
            }
        }
    }

    private var candidateList: some View {
        List {
            ForEach($groups) { $group in
                Section {
                    ForEach($group.items) { $candidate in
                        candidateRow($candidate)
                    }
                } header: {
                    HStack {
                        Text(group.label)
                            .themedFont(.small, weight: .semibold)
                        Text(verbatim: "\(group.items.count)")
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)
                        Spacer()
                        Button {
                            for i in group.items.indices {
                                group.items[i].isSelected = !group.items[i].alreadyImported
                            }
                        } label: { Text("Select All", bundle: .module) }
                            .buttonStyle(.plain)
                            .themedFont(.tiny)
                    }
                }
            }
        }
    }

    private func candidateRow(_ candidate: Binding<ImportCandidate>) -> some View {
        HStack(spacing: 12) {
            Toggle("", isOn: candidate.isSelected)
                .labelsHidden()
                .disabled(candidate.wrappedValue.alreadyImported)
            VStack(alignment: .leading, spacing: 2) {
                HStack {
                    Text(candidate.wrappedValue.name)
                        .themedFont(.small, weight: .semibold)
                    if candidate.wrappedValue.alreadyImported {
                        Text("Already imported", bundle: .module)
                            .themedFont(.tiny)
                            .padding(.horizontal, 6)
                            .padding(.vertical, 1)
                            .background(Color.secondary.opacity(0.15))
                            .clipShape(Capsule())
                    }
                }
                Text(verbatim: candidate.wrappedValue.detail)
                    .themedCode(.tiny)
                    .foregroundStyle(.appSecondary)
                    .lineLimit(1)
                Text(verbatim: candidate.wrappedValue.sourceLocation)
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
            }
            Spacer()
        }
        .padding(.vertical, 2)
    }

    private var footerBar: some View {
        HStack {
            Button {
                for g in groups.indices {
                    for i in groups[g].items.indices {
                        groups[g].items[i].isSelected = false
                    }
                }
            } label: { Text("Deselect All", bundle: .module) }
                .buttonStyle(.plain)
                .themedFont(.small)

            if category == .mcpServers {
                Text(verbatim: "|").foregroundStyle(.tertiary)
                Text("Servers are added disabled.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
            }

            Spacer()

            Button(String(localized: "Import Selected (\(selectedCount))", bundle: .module)) {
                importSelected()
            }
            .buttonStyle(.borderedProminent)
            .tint(TurboSparkTheme.accentColor)
            .disabled(selectedCount == 0)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 10)
        .background(.appPage)
    }

    // MARK: - Scanning

    private func scan() {
        isLoading = true
        groups = []
        let category = self.category
        let projectURL = importToProjectScope ? project?.rootDirectoryURL : nil
        let allowSharedRoots = UserProfileStore.isDefault
        // Captured ON THE MAIN ACTOR: `model` is main-actor state and the
        // scan runs on a background queue.
        let existingMcpNames = model.globalMcpServers.map(\.name)

        DispatchQueue.global(qos: .userInitiated).async {
            let found: [CandidateGroup]
            switch category {
            case .skills:
                found = Self.scanSkills(
                    projectURL: projectURL, allowSharedRoots: allowSharedRoots)
            case .agents:
                found = Self.scanAgents(
                    projectURL: projectURL, allowSharedRoots: allowSharedRoots)
            case .mcpServers:
                found = Self.scanMcpServers(existingNames: existingMcpNames)
            }
            DispatchQueue.main.async {
                self.groups = found
                self.isLoading = false
            }
        }
    }

    /// Skills offered by other tools' user roots (user-scope targets) and
    /// project subdirectories (project-scope targets). `alreadyImported` is
    /// computed against the TARGET directory, so switching scope re-marks
    /// rows.
    nonisolated private static func scanSkills(
        projectURL: URL?, allowSharedRoots: Bool
    ) -> [CandidateGroup] {
        let manager = SkillManager.shared
        let home = FileManager.default.homeDirectoryForCurrentUser

        let targetNames: Set<String>
        if let projectURL {
            let targetDir = projectURL
                .appendingPathComponent(".turbospark", isDirectory: true)
                .appendingPathComponent("skills", isDirectory: true)
            targetNames = Set(manager.scanDirectory(
                targetDir, scope: .projectLocal(projectPath: projectURL.path),
                defaultAgent: .turboSpark
            ).map { $0.name.lowercased() })
        } else {
            targetNames = Set(manager.scanDirectory(
                manager.defaultUserSkillsDirectory, scope: .userGlobal,
                defaultAgent: .turboSpark
            ).map { $0.name.lowercased() })
        }

        var groups: [CandidateGroup] = []
        if allowSharedRoots {
            for (agent, relPath) in manager.knownUserAgentSkillRoots where agent != .turboSpark {
                let url = home.appendingPathComponent(relPath, isDirectory: true)
                guard FileManager.default.fileExists(atPath: url.path) else { continue }
                let items = manager.scanDirectory(url, scope: .userGlobal, defaultAgent: agent)
                    .map { skill in
                        ImportCandidate(
                            name: skill.name,
                            detail: skill.skillDescription,
                            sourceLocation: "~/\(relPath)/\(skill.sourceURL.lastPathComponent)",
                            alreadyImported: targetNames.contains(skill.name.lowercased()),
                            skillSource: skill.isDirectoryBased
                                ? (skill.skillDirectoryURL ?? skill.sourceURL)
                                : skill.sourceURL,
                            agentSource: nil,
                            mcpServer: nil)
                    }
                if !items.isEmpty {
                    groups.append(CandidateGroup(id: agent.rawValue, label: agent.displayName, items: items))
                }
            }
        }
        if let projectURL {
            for (agent, relPath) in manager.knownProjectSkillSubdirectories where agent != .turboSpark {
                let url = projectURL.appendingPathComponent(relPath, isDirectory: true)
                guard FileManager.default.fileExists(atPath: url.path) else { continue }
                let items = manager.scanDirectory(
                    url, scope: .projectLocal(projectPath: projectURL.path), defaultAgent: agent
                ).map { skill in
                    ImportCandidate(
                        name: skill.name,
                        detail: skill.skillDescription,
                        sourceLocation: "<project>/\(relPath)/\(skill.sourceURL.lastPathComponent)",
                        alreadyImported: targetNames.contains(skill.name.lowercased()),
                        skillSource: skill.isDirectoryBased
                            ? (skill.skillDirectoryURL ?? skill.sourceURL)
                            : skill.sourceURL,
                        agentSource: nil,
                        mcpServer: nil)
                }
                if !items.isEmpty {
                    groups.append(CandidateGroup(id: "project-\(agent.rawValue)", label: agent.displayName, items: items))
                }
            }
        }
        return groups
    }

    /// Agent definitions offered by other tools, same shape as the skills
    /// scan.
    nonisolated private static func scanAgents(
        projectURL: URL?, allowSharedRoots: Bool
    ) -> [CandidateGroup] {
        let manager = AgentManager.shared
        let home = FileManager.default.homeDirectoryForCurrentUser

        let targetNames: Set<String>
        if let projectURL {
            let targetDir = projectURL
                .appendingPathComponent(".turbospark", isDirectory: true)
                .appendingPathComponent("agents", isDirectory: true)
            targetNames = Set(manager.scanDirectory(
                targetDir, scope: .project, defaultAgent: .turboSpark
            ).map { $0.name.lowercased() })
        } else {
            targetNames = Set(manager.scanDirectory(
                manager.defaultUserAgentsDirectory, scope: .userGlobal,
                defaultAgent: .turboSpark
            ).map { $0.name.lowercased() })
        }

        var groups: [CandidateGroup] = []
        if allowSharedRoots {
            for (agent, relPath) in manager.knownUserAgentRoots where agent != .turboSpark {
                let url = home.appendingPathComponent(relPath, isDirectory: true)
                guard FileManager.default.fileExists(atPath: url.path) else { continue }
                let items = manager.scanDirectory(url, scope: .userGlobal, defaultAgent: agent)
                    .compactMap { agentDef -> ImportCandidate? in
                        guard let path = agentDef.filePath else { return nil }
                        return ImportCandidate(
                            name: agentDef.name,
                            detail: agentDef.agentDescription,
                            sourceLocation: "~/\(relPath)/\(URL(fileURLWithPath: path).lastPathComponent)",
                            alreadyImported: targetNames.contains(agentDef.name.lowercased()),
                            skillSource: nil,
                            agentSource: URL(fileURLWithPath: path),
                            mcpServer: nil)
                    }
                if !items.isEmpty {
                    groups.append(CandidateGroup(id: agent.rawValue, label: agent.displayName, items: items))
                }
            }
        }
        if let projectURL {
            for (agent, relPath) in manager.knownProjectAgentSubdirectories where agent != .turboSpark {
                let url = projectURL.appendingPathComponent(relPath, isDirectory: true)
                guard FileManager.default.fileExists(atPath: url.path) else { continue }
                let items = manager.scanDirectory(url, scope: .project, defaultAgent: agent)
                    .compactMap { agentDef -> ImportCandidate? in
                        guard let path = agentDef.filePath else { return nil }
                        return ImportCandidate(
                            name: agentDef.name,
                            detail: agentDef.agentDescription,
                            sourceLocation: "<project>/\(relPath)/\(URL(fileURLWithPath: path).lastPathComponent)",
                            alreadyImported: targetNames.contains(agentDef.name.lowercased()),
                            skillSource: nil,
                            agentSource: URL(fileURLWithPath: path),
                            mcpServer: nil)
                    }
                if !items.isEmpty {
                    groups.append(CandidateGroup(id: "project-\(agent.rawValue)", label: agent.displayName, items: items))
                }
            }
        }
        return groups
    }

    /// Global MCP servers from other tools' config files. Always imported
    /// into the GLOBAL list, disabled; name collisions against existing
    /// global servers are pre-marked.
    nonisolated private static func scanMcpServers(existingNames: [String]) -> [CandidateGroup] {
        ExternalAgentMcpReader.discoverSources().map { source in
            CandidateGroup(
                id: source.agent.rawValue,
                label: source.label,
                items: source.servers.map { server in
                    ImportCandidate(
                        name: server.name,
                        detail: server.commandSummary,
                        sourceLocation: source.configPath,
                        alreadyImported: McpServerConfig.nameIsTaken(
                            server.name, among: existingNames),
                        skillSource: nil,
                        agentSource: nil,
                        mcpServer: server)
                })
        }
    }

    // MARK: - Custom Folder

    /// The NSOpenPanel fallback carried over from the old skills import
    /// sheet: a folder or file the standard roots do not know about.
    private func selectCustomFolder() {
        guard !importToProjectScope || project?.rootDirectoryURL != nil else {
            model.showToast(
                String(localized: "The target project no longer exists.", bundle: .module),
                style: .error)
            return
        }
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Import"

        guard panel.runModal() == .OK, let selectedURL = panel.url else { return }
        switch category {
        case .skills:
            if importToProjectScope, let path = project?.rootDirectoryPath {
                model.importSkill(
                    from: selectedURL, targetScope: .projectLocal(projectPath: path))
            } else {
                model.importSkill(from: selectedURL, targetScope: .userGlobal)
            }
        case .agents:
            model.importAgent(
                from: selectedURL, targetScope: importToProjectScope ? .project : .userGlobal)
        case .mcpServers:
            break
        }
        dismiss()
    }

    // MARK: - Import

    private func importSelected() {
        guard !importToProjectScope || project?.rootDirectoryURL != nil else {
            model.showToast(
                String(localized: "The target project no longer exists.", bundle: .module),
                style: .error)
            return
        }
        let selected = groups.flatMap(\.items).filter(\.isSelected)
        guard !selected.isEmpty else { return }

        switch category {
        case .skills:
            let scope: SkillScope
            if importToProjectScope, let path = project?.rootDirectoryPath {
                scope = .projectLocal(projectPath: path)
            } else {
                scope = .userGlobal
            }
            for candidate in selected {
                guard let source = candidate.skillSource else { continue }
                model.importSkill(from: source, targetScope: scope)
            }
        case .agents:
            let scope: AppAgentScope = importToProjectScope ? .project : .userGlobal
            for candidate in selected {
                guard let source = candidate.agentSource else { continue }
                model.importAgent(from: source, targetScope: scope)
            }
        case .mcpServers:
            var imported = 0
            var skipped = 0
            for candidate in selected {
                guard var server = candidate.mcpServer else { continue }
                // Added DISABLED: enabling a server is an explicit decision
                // made in the MCP settings pane after the user has seen the
                // command or URL shown in this row.
                server.isEnabled = false
                server.autoApprove = false
                if McpServerConfig.nameIsTaken(
                    server.name, among: model.globalMcpServers.map(\.name))
                {
                    skipped += 1
                    continue
                }
                model.addGlobalMcpServer(server)
                imported += 1
            }
            if skipped > 0 {
                model.showToast(
                    String(
                        localized: "Added \(imported) servers; \(skipped) skipped (name already in use).",
                        bundle: .module),
                    style: .warning)
            } else {
                model.showToast(
                    String(localized: "Added \(imported) servers (disabled).", bundle: .module),
                    style: .success)
            }
        }
        dismiss()
    }
}
