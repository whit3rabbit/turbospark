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

        /// The label to show for the filter. The raw value is its identity
        /// and is never displayed (it rendered English under every language).
        public var title: Text {
            switch self {
            case .all: return Text("All Skills", bundle: .module)
            case .user: return Text("User Scope", bundle: .module)
            case .project: return Text("Project Scope", bundle: .module)
            }
        }
    }

    @State private var scopeFilter: SkillScopeFilter = .all
    @State private var projectID: UUID?
    @State private var searchQuery: String = ""
    @State private var selectedSkillID: String? = nil
    @State private var isCreatingSkill: Bool = false
    @State private var isEditingSkill: Bool = false
    @State private var isImportingSkill: Bool = false
    @State private var isImportingFromAgents: Bool = false
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
            header

            if managedSkills.isEmpty {
                emptyStateView
            } else if filteredSkills.isEmpty {
                ExtensionNoMatchesState(query: searchQuery)
            } else {
                ExtensionSplitView {
                    skillsListView
                } detail: {
                    if let current = selectedSkill {
                        SkillDetailView(
                            model: model,
                            skill: current,
                            scopedProject: scopedProject,
                            onEdit: { isEditingSkill = true },
                            onDelete: {
                                skillToDelete = current
                                isConfirmingDelete = true
                            })
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "wand.and.stars")
                                .themedFont(.display)
                                .foregroundStyle(.tertiary)
                                .accessibilityHidden(true)
                            Text("Select a skill to inspect instructions and parameters.", bundle: .module)
                                .themedFont(.base)
                                .foregroundStyle(.appSecondary)
                        }
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
        .sheet(isPresented: $isImportingFromAgents) {
            AgentContentImportSheet(
                model: model, initialCategory: .skills, projectID: projectID)
        }
        .confirmationDialog(
            Text("Delete Skill", bundle: .module),
            isPresented: $isConfirmingDelete,
            presenting: skillToDelete
        ) { skill in
            Button(role: .destructive) {
                model.removeSkill(skill)
                if selectedSkillID == skill.sourceURL.path {
                    selectedSkillID = nil
                }
            } label: {
                Text("Delete '\(skill.name)'", bundle: .module)
            }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: { skill in
            Text("Are you sure you want to delete '\(skill.name)' from disk? This action cannot be undone.", bundle: .module)
        }
    }

    // MARK: - Header

    /// `[scope][search] ... [refresh][Import][New Skill]`, shared with the
    /// other extension panes. Both import sources sit in the Import menu:
    /// five controls in one row was what crowded this header.
    private var header: some View {
        ExtensionPaneHeader {
            ExtensionPaneIntro(text: Text("Reusable instructions and workflows the assistant can load into a conversation.", bundle: .module))
        } filters: {
            ExtensionScopePicker(model: model, projectID: $projectID)
            AppSearchField(text: $searchQuery, prompt: Text("Search skills...", bundle: .module))
        } actions: {
            ExtensionRefreshButton(label: Text("Refresh", bundle: .module)) { model.reloadSkills() }
            ExtensionImportMenu { importMenuItems }
            newSkillButton
        } compactActions: {
            ExtensionMoreMenu {
                Button { model.reloadSkills() } label: {
                    Label { Text("Refresh", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                }
                Divider()
                importMenuItems
            }
            newSkillButton
        }
    }

    @ViewBuilder
    private var importMenuItems: some View {
        Button {
            isImportingFromAgents = true
        } label: {
            Label { Text("From Other Agents...", bundle: .module) } icon: { Image(systemName: "person.2") }
        }
        .help(Text("Import skills, agents, or MCP servers from other agent tools", bundle: .module))

        Button {
            isImportingSkill = true
        } label: {
            Label { Text("Browse Marketplace", bundle: .module) } icon: { Image(systemName: "globe") }
        }
        .help(Text("Browse remote skill marketplaces", bundle: .module))
    }

    private var newSkillButton: some View {
        Button {
            isCreatingSkill = true
        } label: {
            Label { Text("New Skill", bundle: .module) } icon: { Image(systemName: "plus") }
        }
        .appAction(.primary)
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
                    .accessibilityHidden(true)

                Text(skill.name)
                    .themedFont(.base, weight: .semibold)
                    .foregroundStyle(skill.isEnabled ? Color.primary : Color.secondary)

                Spacer()

                SkillScopeBadge(scope: skill.scope)

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
                    Text("Files: \(skill.referenceFiles.count)", bundle: .module)
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
                .accessibilityHidden(true)
            Text("Shadowed", bundle: .module)
                .themedFont(.micro, weight: .semibold)
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(Color.purple.opacity(0.15))
        .foregroundStyle(Color.purple)
        .clipShape(Capsule())
        .help(Text("A project skill with this name takes precedence while that project is open.", bundle: .module))
    }

    // MARK: - Empty State

    /// Secondary actions only: New Skill is the header's primary action, and
    /// this offers the import routes the header folds into its menu.
    private var emptyStateView: some View {
        AppEmptyState(
            systemImage: "wand.and.stars",
            title: Text("No Skills Installed", bundle: .module),
            message: Text("Skills allow you to package and inject specialized prompts, workflow instructions, reference scripts, and tool permissions into conversations.", bundle: .module)
        ) {
            Button {
                isImportingFromAgents = true
            } label: {
                Label { Text("Import from Agents...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
            }
            .appAction(.secondary)
            Button {
                isImportingSkill = true
            } label: {
                Label { Text("Browse Marketplace", bundle: .module) } icon: { Image(systemName: "globe") }
            }
            .appAction(.secondary)
        }
    }
}
