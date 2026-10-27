import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Settings pane for discovering, inspecting, and managing built-in and custom Agents & Subagents.
@MainActor
public struct AgentsSettingsPaneView: View {
    @ObservedObject var model: AppModel

    public enum AgentScopeFilter: String, CaseIterable, Identifiable {
        case all = "All Agents"
        case builtIn = "Built-in"
        case user = "User Scope"
        case project = "Project Scope"

        public var id: String { rawValue }

        /// The label the filter shows. The raw value is its identity and is
        /// never displayed: `Text(filter.rawValue)` rendered English under
        /// every language.
        public var title: Text {
            switch self {
            case .all: return Text("All Agents", bundle: .module)
            case .builtIn: return Text("Built-in", bundle: .module)
            case .user: return Text("User Scope", bundle: .module)
            case .project: return Text("Project Scope", bundle: .module)
            }
        }
    }

    @State private var scopeFilter: AgentScopeFilter = .all
    @State private var searchQuery: String = ""
    @State private var selectedAgentID: UUID? = nil
    @State private var showingEditorSheet: Bool = false
    @State private var isImportingFromAgents: Bool = false
    @State private var agentToEdit: AppAgentDefinition? = nil
    @State private var agentToDelete: AppAgentDefinition? = nil

    public init(model: AppModel) {
        self.model = model
    }

    private var filteredAgents: [AppAgentDefinition] {
        let list: [AppAgentDefinition]
        switch scopeFilter {
        case .all:
            list = model.allManagedAgents
        case .builtIn:
            list = model.allManagedAgents.filter { $0.scope == .builtIn }
        case .user:
            list = model.allManagedAgents.filter { $0.scope == .userGlobal }
        case .project:
            list = model.allManagedAgents.filter { $0.scope == .project }
        }

        if searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return list
        }
        let q = searchQuery.lowercased()
        return list.filter {
            $0.name.lowercased().contains(q) ||
            $0.displayName.lowercased().contains(q) ||
            $0.agentDescription.lowercased().contains(q) ||
            $0.sourceAgent.displayName.lowercased().contains(q)
        }
    }

    private var selectedAgent: AppAgentDefinition? {
        if let id = selectedAgentID {
            return model.allManagedAgents.first { $0.id == id }
        }
        return filteredAgents.first
    }

    /// What the project's agent files are overriding (state#105).
    ///
    /// **PRECEDENCE WITH NO DISCLOSURE IS INDISTINGUISHABLE FROM A BROKEN
    /// AGENT.** A project agent taking a name deliberately shadows the user's
    /// own or a built-in; that is what precedence IS. But a user whose
    /// `explore` stopped writing files, or whose own `deploy` agent stopped
    /// behaving, reads it as their agent being broken rather than replaced --
    /// and a cloned repository can do either by name. `constrainedProjectAgentNames`
    /// existed for exactly this and had no view reading it, which is
    /// `swift/CLAUDE.md` Gotcha 36's tell: grep for the ACCESSOR, not for
    /// the feature.
    @ViewBuilder private var overrideDisclosureBanner: some View {
        let constrained = model.constrainedProjectAgentNames
        let shadowed = model.shadowedUserAgentNames
        if !constrained.isEmpty || !shadowed.isEmpty {
            VStack(alignment: .leading, spacing: 4) {
                if !shadowed.isEmpty {
                    let names = shadowed.joined(separator: ", ")
                    Label {
                        Text("This project overrides your own \(names).", bundle: .module)
                    } icon: {
                        Image(systemName: "arrow.triangle.branch")
                    }
                    .themedFont(.small)
                }
                if !constrained.isEmpty {
                    let names = constrained.joined(separator: ", ")
                    Label {
                        Text("\(names) took a built-in's name and is held to its tool limits.", bundle: .module)
                    } icon: {
                        Image(systemName: "lock.shield")
                    }
                    .themedFont(.small)
                }
            }
            .foregroundStyle(.appSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, SettingsLayout.horizontalInset)
            .padding(.vertical, 8)
            .background(.appSurface)
        }
    }

    @ViewBuilder private var projectInstructionsBanner: some View {
        if scopeFilter == .project, let project = model.selectedProject {
            HStack(spacing: 8) {
                Image(systemName: "doc.text")
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)
                Group {
                    if let root = project.rootDirectoryPath, !root.isEmpty {
                        if let detected = ProjectRuleDetector.detectRules(in: root, preference: project.rulePreference) {
                            Text(detected.statusDescription)
                        } else {
                            Text("No live AGENTS.md or CLAUDE.md detected in project root.", bundle: .module)
                        }
                    } else {
                        Text("No project folder attached. Set a folder in project settings to read AGENTS.md.", bundle: .module)
                    }
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                Spacer()
            }
            .padding(.horizontal, SettingsLayout.horizontalInset)
            .padding(.vertical, 6)
            .background(.appSurface)
        }
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            overrideDisclosureBanner
            projectInstructionsBanner

            if filteredAgents.isEmpty {
                if searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    emptyStateView
                } else {
                    ExtensionNoMatchesState(query: searchQuery)
                }
            } else {
                ExtensionSplitView {
                    agentsListView
                } detail: {
                    if let current = selectedAgent {
                        AgentDetailView(
                            model: model,
                            agent: current,
                            onEdit: {
                                agentToEdit = current
                                showingEditorSheet = true
                            },
                            onDelete: { agentToDelete = current })
                    } else {
                        VStack(spacing: 16) {
                            TSIdlingAgentsSparkView(size: 80)
                            Text("Select an agent to inspect system instructions and capabilities.", bundle: .module)
                                .themedFont(.base)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                }
            }
        }
        .onAppear {
            model.reloadAgents()
            if selectedAgentID == nil {
                selectedAgentID = filteredAgents.first?.id
            }
        }
        .sheet(isPresented: $showingEditorSheet) {
            AgentEditorSheet(model: model, agentToEdit: agentToEdit)
        }
        .sheet(isPresented: $isImportingFromAgents) {
            AgentContentImportSheet(
                model: model, initialCategory: .agents, projectID: nil)
        }
        .alert(
            Text("Delete Agent", bundle: .module),
            isPresented: Binding(
                get: { agentToDelete != nil },
                set: { if !$0 { agentToDelete = nil } }),
            presenting: agentToDelete
        ) { agent in
            Button(role: .destructive) {
                model.deleteAgent(agent)
                agentToDelete = nil
            } label: {
                Text("Delete", bundle: .module)
                    .settingsControl("Delete", pane: .agents, timing: .nextTurn)
            }
            Button {
                agentToDelete = nil
            } label: {
                Text("Cancel", bundle: .module)
            }
        } message: { agent in
            Text("Delete the agent file for '\(agent.name)'? This cannot be undone.", bundle: .module)
        }
    }

    // MARK: - Header

    /// `[scope][search] ... [reload][Import...][New Agent]`, shared with the
    /// other extension panes. The scope filter is a menu: a 320pt segmented
    /// control beside the search field and three actions overflowed the row.
    private var header: some View {
        ExtensionPaneHeader {
            ExtensionPaneIntro(text: Text("Subagents the assistant can hand a task to, each with its own instructions and tool limits.", bundle: .module))
        } filters: {
            Picker(selection: $scopeFilter) {
                ForEach(AgentScopeFilter.allCases) { filter in
                    filter.title.tag(filter)
                }
            } label: { Text("Scope", bundle: .module) }
            .settingsControl("Scope", pane: .agents, timing: .nextTurn)
            .pickerStyle(.menu)
            .fixedSize()
            AppSearchField(text: $searchQuery, prompt: Text("Search agents...", bundle: .module))
        } actions: {
            ExtensionRefreshButton(label: Text("Reload", bundle: .module)) { model.reloadAgents() }
            importButton
            newAgentButton
        } compactActions: {
            ExtensionMoreMenu {
                Button { model.reloadAgents() } label: {
                    Label { Text("Reload", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                }
                Button { isImportingFromAgents = true } label: {
                    Label { Text("Import...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                }
            }
            newAgentButton
        }
    }

    private var importButton: some View {
        Button {
            isImportingFromAgents = true
        } label: {
            Label { Text("Import...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
        }
        .appAction(.secondary)
        .help(Text("Import agents from other agent tools", bundle: .module))
    }

    private var newAgentButton: some View {
        Button {
            agentToEdit = nil
            showingEditorSheet = true
        } label: {
            Label { Text("New Agent", bundle: .module) } icon: { Image(systemName: "plus") }
        }
        .appAction(.primary)
        .help(Text("Create a new agent definition file", bundle: .module))
    }

    // MARK: - List View

    private var agentsListView: some View {
        List(selection: $selectedAgentID) {
            ForEach(filteredAgents) { agent in
                HStack(spacing: 10) {
                    // Named by the agent: an empty label left VoiceOver
                    // announcing a row of unnamed checkboxes.
                    Toggle(isOn: Binding(
                        get: { agent.isEnabled },
                        set: { _ in model.toggleAgentEnabled(agent) }
                    )) {
                        Text(agent.displayName)
                    }
                    .toggleStyle(.checkbox)
                    .labelsHidden()

                    VStack(alignment: .leading, spacing: 3) {
                        // Two lines and a tooltip: at the narrowest list
                        // width the badge left one line for "Codebase Ex...".
                        HStack(alignment: .firstTextBaseline, spacing: 6) {
                            Text(agent.displayName)
                                .themedFont(.base, weight: .semibold)
                                .lineLimit(2)
                                .fixedSize(horizontal: false, vertical: true)
                                .help(Text(verbatim: agent.displayName))
                            Spacer(minLength: 4)
                            AgentBadge(
                                text: Text(LocalizedStringKey(agent.scope.label), bundle: .module),
                                color: AgentBadge.scopeColor(agent.scope))
                        }

                        Text(agent.agentDescription)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(2)
                    }
                }
                .padding(.vertical, 4)
                .tag(agent.id)
            }
        }
        .listStyle(.sidebar)
        .scrollContentBackground(.hidden)
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        AppEmptyState(
            systemImage: "person.2.badge.gearshape",
            title: Text("No Agents Found", bundle: .module),
            message: Text("No agent definitions match the selected scope filter.", bundle: .module)
        ) {
            Button {
                isImportingFromAgents = true
            } label: {
                Label { Text("Import from Agents...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
            }
            .appAction(.secondary)
        }
    }
}
