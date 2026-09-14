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
    }

    @State private var scopeFilter: AgentScopeFilter = .all
    @State private var searchQuery: String = ""
    @State private var selectedAgentID: UUID? = nil
    @State private var showingEditorSheet: Bool = false
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
                    Label(
                        "This project overrides your own \(shadowed.joined(separator: ", ")).",
                        systemImage: "arrow.triangle.branch"
                    )
                    .themedFont(.small)
                }
                if !constrained.isEmpty {
                    Label(
                        "\(constrained.joined(separator: ", ")) took a built-in's name and is "
                            + "held to its tool limits.",
                        systemImage: "lock.shield"
                    )
                    .themedFont(.small)
                }
            }
            .foregroundStyle(.appSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.horizontal, 20)
            .padding(.vertical, 8)
            .background(.appSurface.opacity(0.5))
        }
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header bar
            headerControlBar
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(.appPage)

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            overrideDisclosureBanner

            if model.allManagedAgents.isEmpty {
                emptyStateView
            } else {
                HStack(spacing: 0) {
                    // Left list
                    agentsListView
                        .frame(width: 300)
                        .background(.appSurface.opacity(0.5))

                    Rectangle()
                        .fill(.appBorder)
                        .frame(width: 1)

                    // Right detail
                    if let current = selectedAgent {
                        agentDetailView(agent: current)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "person.2.badge.gearshape")
                                .themedFont(.display)
                                .foregroundStyle(.tertiary)
                            Text("Select an agent to inspect system instructions and capabilities.", bundle: .module)
                                .themedFont(.base)
                                .foregroundStyle(.appSecondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
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
            Text(verbatim: "Delete the agent file for '\(agent.name)'? This cannot be undone.")
        }
    }

    // MARK: - Header Bar

    private var headerControlBar: some View {
        HStack(spacing: 12) {
            Picker(selection: $scopeFilter) {
                ForEach(AgentScopeFilter.allCases) { filter in
                    Text(filter.rawValue).tag(filter)
                }
            } label: { Text("Scope", bundle: .module) }
            .settingsControl("Scope", pane: .agents, timing: .nextTurn)
            .pickerStyle(.segmented)
            .frame(width: 320)

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.appSecondary)
                TextField("Search agents...", text: $searchQuery)
                    .textFieldStyle(.plain)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(RoundedRectangle(cornerRadius: 6).fill(.appElevated))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(.appBorder, lineWidth: 1))

            Spacer()

            Button {
                model.reloadAgents()
            } label: {
                Label { Text("Reload", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
            }
            .buttonStyle(.bordered)
            .help("Rescan agents on disk")

            Button {
                agentToEdit = nil
                showingEditorSheet = true
            } label: {
                Label { Text("New Agent", bundle: .module) } icon: { Image(systemName: "plus") }
            }
            .buttonStyle(.borderedProminent)
            .help("Create a new agent definition file")
        }
    }

    // MARK: - List View

    private var agentsListView: some View {
        List(selection: $selectedAgentID) {
            ForEach(filteredAgents) { agent in
                HStack(spacing: 10) {
                    Toggle("", isOn: Binding(
                        get: { agent.isEnabled },
                        set: { _ in model.toggleAgentEnabled(agent) }
                    ))
                    .toggleStyle(.checkbox)
                    .labelsHidden()

                    VStack(alignment: .leading, spacing: 3) {
                        HStack(spacing: 6) {
                            Text(agent.displayName)
                                .themedFont(.base, weight: .semibold)
                                .lineLimit(1)
                            Spacer()
                            badgeView(text: agent.scope.label, color: scopeColor(agent.scope))
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
    }

    // MARK: - Detail View

    private func agentDetailView(agent: AppAgentDefinition) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                // Title and action header
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack(spacing: 8) {
                            Text(agent.displayName)
                                .themedFont(.title2, weight: .bold)

                            badgeView(text: agent.scope.label, color: scopeColor(agent.scope))
                            badgeView(text: agent.sourceAgent.displayName, color: .indigo)
                        }

                        Text(agent.agentDescription)
                            .themedFont(.base)
                            .foregroundStyle(.appSecondary)
                    }

                    Spacer()

                    // Edit and Delete only where the definition is a file
                    // this app owns: built-ins are code and plugin agents
                    // belong to their plugin, so neither has an editor.
                    if agent.scope == .userGlobal || agent.scope == .project {
                        HStack(spacing: 8) {
                            Button {
                                agentToEdit = agent
                                showingEditorSheet = true
                            } label: {
                                Label { Text("Edit", bundle: .module) } icon: { Image(systemName: "pencil") }
                            }
                            .buttonStyle(.bordered)

                            Button(role: .destructive) {
                                agentToDelete = agent
                            } label: {
                                Label { Text("Delete", bundle: .module) } icon: { Image(systemName: "trash") }
                            }
                            .buttonStyle(.bordered)
                        }
                    }

                    Toggle(isOn: Binding(
                        get: { agent.isEnabled },
                        set: { _ in model.toggleAgentEnabled(agent) }
                    )) {
                        Text("Enabled", bundle: .module)
                    }
            .settingsControl("Enabled", pane: .agents, timing: .nextTurn)
                    .toggleStyle(.switch)
                }

                Divider()

                // Configuration metadata
                VStack(alignment: .leading, spacing: 12) {
                    Text("Configuration", bundle: .module)
                        .themedFont(.base, weight: .semibold)

                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                        GridRow {
                            Text("Agent Identifier:", bundle: .module)
                                .foregroundStyle(.appSecondary)
                            Text(verbatim: "`\(agent.name)`")
                                .themedCode(.base)
                        }
                        GridRow {
                            Text("Max Turns:", bundle: .module)
                                .foregroundStyle(.appSecondary)
                            Text(verbatim: "\(agent.maxTurns)")
                        }
                        // No "Model Override" row: `agent.model` is parsed
                        // and read by nothing, a subagent runs on whatever is
                        // loaded, and the row sat between two that ARE
                        // enforced (`swift/docs/SWIFT_SETTINGS_AUDIT.md`).
                        if let disallowed = agent.disallowedTools, !disallowed.isEmpty {
                            GridRow {
                                Text("Disallowed Tools:", bundle: .module)
                                    .foregroundStyle(.appSecondary)
                                Text(disallowed.joined(separator: ", "))
                                    .themedFont(.small)
                                    .foregroundStyle(.red)
                            }
                        }
                        if let allowed = agent.tools, !allowed.isEmpty {
                            GridRow {
                                Text("Allowed Tools:", bundle: .module)
                                    .foregroundStyle(.appSecondary)
                                Text(allowed.joined(separator: ", "))
                                    .themedFont(.small)
                            }
                        }
                        if let path = agent.filePath {
                            GridRow {
                                Text("File Location:", bundle: .module)
                                    .foregroundStyle(.appSecondary)
                                Text(path)
                                    .themedFont(.small)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                        }
                    }
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 8).fill(.appSurface))

                // System Prompt / Instructions
                VStack(alignment: .leading, spacing: 8) {
                    Text("System Instructions", bundle: .module)
                        .themedFont(.base, weight: .semibold)

                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(agent.systemPrompt)
                            .themedCode(.small)
                            .textSelection(.enabled)
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .background(RoundedRectangle(cornerRadius: 8).fill(.appSurface))
                }

                // REPL Usage hint. Only for an ENABLED agent: the slash
                // command refuses a disabled one (state#93), so the hint
                // would advertise a command that fails.
                if agent.isEnabled {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("REPL Chat Invocation", bundle: .module)
                            .themedFont(.small, weight: .bold)
                        Text("You can invoke this subagent in chat or via slash command with clean context isolation:", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                        Text(verbatim: "`/\(agent.name) <task description>`")
                            .themedCode(.small)
                            .padding(8)
                            .background(RoundedRectangle(cornerRadius: 6).fill(.appElevated))
                    }
                }
            }
            .padding(24)
        }
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        VStack(spacing: 12) {
            Image(systemName: "person.2.badge.gearshape")
                .themedFont(.display)
                .foregroundStyle(.tertiary)
            Text("No Agents Found", bundle: .module)
                .themedFont(.title3, weight: .bold)
            Text("No agent definitions match the selected scope filter.", bundle: .module)
                .themedFont(.base)
                .foregroundStyle(.appSecondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Helpers

    private func badgeView(text: String, color: Color) -> some View {
        Text(text)
            .themedFont(.tiny, weight: .medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }

    private func scopeColor(_ scope: AppAgentScope) -> Color {
        switch scope {
        case .builtIn: return .blue
        case .userGlobal: return .purple
        case .project: return .green
        case .plugin: return .orange
        }
    }
}
