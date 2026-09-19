import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Settings pane for discovering, creating, editing, and managing User- and Project-scoped skills.
@MainActor
public struct SkillsSettingsPaneView: View {
    @ObservedObject var model: AppModel

    public enum SkillScopeFilter: String, CaseIterable, Identifiable {
        case all = "All Skills"
        case user = "User Scope"
        case project = "Project Scope"

        public var id: String { rawValue }
    }

    @State private var scopeFilter: SkillScopeFilter = .all
    @State private var projectID: UUID?
    @State private var searchQuery: String = ""
    @State private var selectedSkillID: String? = nil
    @State private var isCreatingSkill: Bool = false
    @State private var isEditingSkill: Bool = false
    @State private var isImportingSkill: Bool = false
    @State private var skillToDelete: AppSkill? = nil
    @State private var isConfirmingDelete: Bool = false

    public init(model: AppModel, projectID: UUID? = nil) {
        self.model = model
        self._projectID = State(initialValue: projectID)
    }

    private var managedSkills: [AppSkill] {
        var result = model.userSkills
        if let root = scopedProject?.rootDirectoryURL {
            result += SkillManager.shared.discoverProjectSkills(projectRootURL: root)
        }
        result += PluginManager.shared.pluginSkills(projectURL: scopedProject?.rootDirectoryURL)
        return result.map { skill in
            var copy = skill
            copy.isEnabled = scopedProject?.enabledSkills[skill.name.lowercased()] ?? skill.isEnabled
            return copy
        }
    }
    private var scopedProject: AppProject? { model.projects.first { $0.id == projectID } }

    private var filteredSkills: [AppSkill] {
        let list: [AppSkill]
        switch scopeFilter {
        case .all:
            list = managedSkills
        case .user:
            list = managedSkills.filter { !$0.scope.isProjectScope }
        case .project:
            list = managedSkills.filter { $0.scope.isProjectScope }
        }

        if searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return list
        }
        let q = searchQuery.lowercased()
        return list.filter {
            $0.name.lowercased().contains(q) ||
            $0.skillDescription.lowercased().contains(q) ||
            $0.agentOrigin.displayName.lowercased().contains(q)
        }
    }

    private var selectedSkill: AppSkill? {
        if let id = selectedSkillID {
            return managedSkills.first { $0.sourceURL.path == id }
        }
        return filteredSkills.first
    }

    public var body: some View {
        VStack(spacing: 0) {
            ExtensionScopePicker(model: model, projectID: $projectID)
            // Header bar
            headerControlBar
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(.appPage)

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            if managedSkills.isEmpty {
                emptyStateView
            } else {
                HStack(spacing: 0) {
                    // Left list
                    skillsListView
                        .frame(width: 220)
                        .background(.appSurface.opacity(0.5))

                    Rectangle()
                        .fill(.appBorder)
                        .frame(width: 1)

                    // Right detail
                    if let current = selectedSkill {
                        skillDetailView(skill: current)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "wand.and.stars")
                                .themedFont(.display)
                                .foregroundStyle(.tertiary)
                            Text("Select a skill to inspect instructions and parameters.", bundle: .module)
                                .themedFont(.base)
                                .foregroundStyle(.appSecondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
            }
        }
        .sheet(isPresented: $isCreatingSkill) {
            SkillEditorSheet(model: model, skillToEdit: nil, defaultScope: projectID != nil ? .projectLocal(projectPath: scopedProject?.rootDirectoryPath ?? "") : .userGlobal)
        }
        .sheet(isPresented: $isEditingSkill) {
            if let skill = selectedSkill {
                SkillEditorSheet(model: model, skillToEdit: skill, defaultScope: skill.scope)
            }
        }
        .sheet(isPresented: $isImportingSkill) {
            SkillImportSheet(model: model, projectID: projectID)
        }
        .confirmationDialog(
            "Delete Skill",
            isPresented: $isConfirmingDelete,
            presenting: skillToDelete
        ) { skill in
            Button("Delete '\(skill.name)'", role: .destructive) {
                model.removeSkill(skill)
                if selectedSkillID == skill.sourceURL.path {
                    selectedSkillID = nil
                }
            }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: { skill in
            Text("Are you sure you want to delete '\(skill.name)' from disk? This action cannot be undone.", bundle: .module)
        }
    }

    // MARK: - Header Bar
    private var headerControlBar: some View {
        HStack(spacing: 12) {
            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                TextField("Search skills...", text: $searchQuery)
                    .textFieldStyle(.plain)
                    .themedFont(.base)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(.appSurface)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(.appBorder, lineWidth: 1)
            )

            Spacer()

            Button {
                model.reloadSkills()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .help("Refresh skills from disk")

            Button {
                isImportingSkill = true
            } label: {
                Label { Text("Import...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
            }
            .buttonStyle(.bordered)
            .help("Import skills from Claude, Cursor, Antigravity, or OpenCode")

            Button {
                isCreatingSkill = true
            } label: {
                Label { Text("New Skill", bundle: .module) } icon: { Image(systemName: "plus") }
            }
            .buttonStyle(.borderedProminent)
            .fixedSize()
        }
    }

    // MARK: - Skills List View
    private var skillsListView: some View {
        List(selection: $selectedSkillID) {
            ForEach(filteredSkills) { skill in
                skillRowView(skill: skill)
                    .tag(skill.sourceURL.path)
                    .padding(.vertical, 4)
            }
        }
        .listStyle(.sidebar)
        .scrollContentBackground(.hidden)
    }

    private func skillRowView(skill: AppSkill) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "wand.and.stars")
                    .foregroundStyle(skill.isEnabled ? TurboSparkTheme.accentColor : Color.secondary)
                    .themedFont(.small)

                Text(skill.name)
                    .themedFont(.base, weight: .semibold)
                    .foregroundStyle(skill.isEnabled ? Color.primary : Color.secondary)

                Spacer()

                // Scope badge
                scopeBadgeView(scope: skill.scope)

                // Shadowing disclosure (swift/docs/SWIFT_SKILLS.md 3B): a USER skill a
                // project skill currently overrides by name. Precedence
                // without the disclosure makes the user skill look broken --
                // it resolves to nothing while sitting right there, enabled.
                if isShadowedByProject(skill) {
                    shadowedBadgeView
                }
            }

            Text(skill.skillDescription)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .lineLimit(2)

            HStack(spacing: 8) {
                Label(skill.agentOrigin.displayName, systemImage: "cpu")
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)

                // **NO "N tools" CHIP** (state#48). `allowed-tools` is parsed
                // into `SkillManifest` and read by NOTHING:
                // `SkillManifest.isToolAllowed` has zero callers, and a skill
                // is inlined into a prompt rather than scoped, so there is no
                // window in which it could apply. The chip stated a
                // restriction that does not exist -- `swift/CLAUDE.md`
                // Gotcha 22's badge that cannot fail, in the direction that
                // matters, since a reader would take it for a permission
                // boundary. `disable-model-invocation` IS enforced now (in
                // the `skill` tool); scoping a skill's tools is a feature,
                // not a missing line, and is not claimed until it exists.

                if !skill.referenceFiles.isEmpty {
                    Text(verbatim: "\(skill.referenceFiles.count) files")
                        .themedFont(.tiny)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.top, 2)
        }
        .contentShape(Rectangle())
    }

    /// Whether a USER-scoped skill's name is currently overridden by a
    /// project skill (the comparison is the same case-insensitive key the
    /// precedence merge itself uses).
    private func isPluginSkill(_ skill: AppSkill) -> Bool {
        if case .plugin = skill.scope { return true }
        return false
    }

    private func isShadowedByProject(_ skill: AppSkill) -> Bool {
        guard skill.scope.isProjectScope == false else { return false }
        return SkillManager.shared.shadowedUserSkillNames(projectURL: scopedProject?.rootDirectoryURL).contains {
            $0.caseInsensitiveCompare(skill.name) == .orderedSame
        }
    }

    private var shadowedBadgeView: some View {
        HStack(spacing: 3) {
            Image(systemName: "arrow.2.squarepath")
                .themedFont(.micro)
            Text("Shadowed", bundle: .module)
                .themedFont(.micro, weight: .semibold)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Color.purple.opacity(0.15))
        .foregroundStyle(Color.purple)
        .clipShape(Capsule())
        .help("A project skill with this name takes precedence while that project is open.")
    }

    private func scopeBadgeView(scope: SkillScope) -> some View {
        HStack(spacing: 3) {
            Image(systemName: scope.badgeIcon)
                .themedFont(.micro)
            Text(scope.label)
                .themedFont(.micro, weight: .semibold)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(
            scope.isProjectScope
                ? Color.orange.opacity(0.15)
                : Color.blue.opacity(0.15)
        )
        .foregroundStyle(
            scope.isProjectScope ? Color.orange : Color.blue
        )
        .clipShape(Capsule())
    }

    // MARK: - Skill Detail View
    private func skillDetailView(skill: AppSkill) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                // Header card
                VStack(alignment: .leading, spacing: 8) {
                    VStack(alignment: .leading, spacing: 12) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(skill.name)
                                    .themedFont(.title2, weight: .bold)
                                scopeBadgeView(scope: skill.scope)
                            }
                            Text(skill.skillDescription)
                                .themedFont(.base)
                                .foregroundStyle(.appSecondary)
                        }

                        VStack(alignment: .leading, spacing: 8) {
                            if let project = scopedProject {
                                ExtensionOverridePicker(enabled: Binding(
                                    get: { project.enabledSkills[skill.name.lowercased()] },
                                    set: { model.setSkillOverride(name: skill.name, enabled: $0, projectID: project.id) }))
                            }
                            // The list greys a disabled skill and this pane
                            // had no way to re-enable it; the only toggle was
                            // in the composer's plus menu.
                            if scopedProject == nil && !isPluginSkill(skill) {
                            Toggle(isOn: Binding(
                                get: { skill.isEnabled },
                                set: { _ in model.toggleSkillEnabled(skill) }
                            )) {
                                Text("Enabled", bundle: .module)
                            }
            .settingsControl("Enabled", pane: .skills, timing: .nextTurn)
                            .toggleStyle(.switch)
                            .controlSize(.small)
                            } else {
                                Text(LocalizedStringKey(skill.isEnabled ? "Enabled" : "Disabled"), bundle: .module).themedFont(.small)
                            }
                            if case .plugin = skill.scope {
                                Button { model.openSettings(tab: .plugins) } label: { Text("Manage plugin", bundle: .module) }
                            }

                            if SkillManager.shared.ownsSkill(skill) {
                            Button {
                                isEditingSkill = true
                            } label: {
                                Label { Text("Edit", bundle: .module) } icon: { Image(systemName: "pencil") }
                            }
                            .buttonStyle(.bordered)

                            }
                            if !SkillManager.shared.ownsSkill(skill) {
                                Menu {
                                    Button { _ = model.importSkill(from: skill.skillDirectoryURL ?? skill.sourceURL, targetScope: .userGlobal) } label: { Text("Copy to User", bundle: .module) }
                                    if let path = scopedProject?.rootDirectoryPath {
                                        Button { _ = model.importSkill(from: skill.skillDirectoryURL ?? skill.sourceURL, targetScope: .projectLocal(projectPath: path)) } label: { Text("Copy to Project", bundle: .module) }
                                    }
                                } label: { Text("Copy to TurboSpark", bundle: .module)
                    .settingsControl("Copy to TurboSpark", pane: .skills, timing: .nextTurn) }
                                .help(Text("Skill actions", bundle: .module))
                            }
                            Button {
                                NSWorkspace.shared.activateFileViewerSelecting([skill.sourceURL])
                            } label: {
                                Label { Text("Reveal", bundle: .module) } icon: { Image(systemName: "folder") }
                            }
                            .buttonStyle(.bordered)

                            if SkillManager.shared.ownsSkill(skill) {
                            Button(role: .destructive) {
                                skillToDelete = skill
                                isConfirmingDelete = true
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.bordered)
                            .help(Text("Delete", bundle: .module))
                            .accessibilityLabel(Text("Delete", bundle: .module))
                            }
                        }
                    }

                    Divider()
                        .padding(.vertical, 4)

                    // Metadata badges
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: 12)], spacing: 8) {
                        metaItemView(title: "Agent Origin", value: skill.agentOrigin.displayName, icon: "terminal")
                        metaItemView(title: "Execution Context", value: skill.manifest.context.rawValue.capitalized, icon: "arrow.triangle.branch")
                        metaItemView(title: "Shell Type", value: skill.manifest.shell.rawValue, icon: "apple.terminal")
                        metaItemView(title: "Source Path", value: skill.sourceURL.lastPathComponent, icon: "doc.text")
                    }
                }
                .padding(16)
                .background(.appSurface)
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(.appBorder, lineWidth: 1)
                )

                // No "Allowed Tools & Permissions" section. `allowed-tools`
                // is enforced by nothing (see the row comment on state#48
                // above); this section rendered the same field with a green
                // shield, one scroll below the comment saying why not.

                // Path Triggers Section
                if !skill.manifest.paths.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Activation Path Triggers", bundle: .module)
                            .themedFont(.base, weight: .semibold)
                        FlowLayout(spacing: 6, lineSpacing: 6) {
                            ForEach(skill.manifest.paths, id: \.self) { pathPattern in
                                HStack(spacing: 4) {
                                    Image(systemName: "arrow.triangle.turn.up.right.diamond")
                                        .themedFont(.tiny)
                                    Text(pathPattern)
                                        .themedCode(.small)
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.purple.opacity(0.12))
                                .foregroundStyle(.purple)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }
                    }
                }

                // Reference Files Section
                if !skill.referenceFiles.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Directory Reference Files (\(skill.referenceFiles.count))", bundle: .module)
                            .themedFont(.base, weight: .semibold)
                        VStack(spacing: 4) {
                            ForEach(skill.referenceFiles, id: \.self) { fileName in
                                HStack(spacing: 6) {
                                    Image(systemName: "doc")
                                        .themedFont(.small)
                                        .foregroundStyle(.appSecondary)
                                    Text(fileName)
                                        .themedCode(.small)
                                    Spacer()
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(.appSurface.opacity(0.6))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }
                    }
                }

                // Instructions Markdown Content
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Instruction Body (SKILL.md)", bundle: .module)
                            .themedFont(.base, weight: .semibold)
                        Spacer()
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(skill.content, forType: .string)
                        } label: {
                            Label { Text("Copy", bundle: .module) } icon: { Image(systemName: "doc.on.doc") }
                                .themedFont(.small)
                        }
                        .buttonStyle(.borderless)
                    }

                    Text(skill.content)
                        .themedCode(.base)
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(.appElevated)
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(.appBorder, lineWidth: 1)
                        )
                }
            }
            .padding(20)
        }
    }

    private func metaItemView(title: String, value: String, icon: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
                Text(value)
                    .themedFont(.small, weight: .medium)
                    .lineLimit(1)
            }
        }
    }

    // MARK: - Empty State
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "wand.and.stars")
                .themedFont(.display)
                .foregroundStyle(.appSecondary)

            Text("No Skills Installed", bundle: .module)
                .themedFont(.title2, weight: .bold)

            Text("Skills allow you to package and inject specialized prompts, workflow instructions, reference scripts, and tool permissions into conversations.", bundle: .module)
                .themedFont(.base)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)

            HStack(spacing: 12) {
                Button {
                    isCreatingSkill = true
                } label: {
                    Label { Text("Create First Skill", bundle: .module) } icon: { Image(systemName: "plus") }
                }
                .buttonStyle(.borderedProminent)
                .fixedSize()

                Button {
                    isImportingSkill = true
                } label: {
                    Label { Text("Import from Agents...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                }
                .buttonStyle(.bordered)
            }
            .padding(.top, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}
