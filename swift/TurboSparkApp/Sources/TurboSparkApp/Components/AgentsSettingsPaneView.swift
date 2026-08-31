import AppKit
import SwiftUI

/// Settings pane for discovering, inspecting, and managing built-in and custom Agents & Subagents.
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

    public var body: some View {
        VStack(spacing: 0) {
            // Header bar
            headerControlBar
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(Color(nsColor: .windowBackgroundColor))

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            if model.allManagedAgents.isEmpty {
                emptyStateView
            } else {
                HStack(spacing: 0) {
                    // Left list
                    agentsListView
                        .frame(width: 300)
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

                    Rectangle()
                        .fill(TurboSparkTheme.hairlineColor)
                        .frame(width: 1)

                    // Right detail
                    if let current = selectedAgent {
                        agentDetailView(agent: current)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "person.2.badge.gearshape")
                                .font(.system(size: 36))
                                .foregroundStyle(.tertiary)
                            Text("Select an agent to inspect system instructions and capabilities.")
                                .font(.callout)
                                .foregroundStyle(.secondary)
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
    }

    // MARK: - Header Bar

    private var headerControlBar: some View {
        HStack(spacing: 12) {
            Picker("Scope", selection: $scopeFilter) {
                ForEach(AgentScopeFilter.allCases) { filter in
                    Text(filter.rawValue).tag(filter)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 320)

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("Search agents...", text: $searchQuery)
                    .textFieldStyle(.plain)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 5)
            .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .textBackgroundColor)))
            .overlay(RoundedRectangle(cornerRadius: 6).stroke(TurboSparkTheme.hairlineColor, lineWidth: 1))

            Spacer()

            Button {
                model.reloadAgents()
            } label: {
                Label("Reload", systemImage: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .help("Rescan agents on disk")
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
                                .font(.headline)
                                .lineLimit(1)
                            Spacer()
                            badgeView(text: agent.scope.label, color: scopeColor(agent.scope))
                        }

                        Text(agent.agentDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
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
                                .font(.title2.bold())

                            badgeView(text: agent.scope.label, color: scopeColor(agent.scope))
                            badgeView(text: agent.sourceAgent.displayName, color: .indigo)
                        }

                        Text(agent.agentDescription)
                            .font(.body)
                            .foregroundStyle(.secondary)
                    }

                    Spacer()

                    Toggle("Enabled", isOn: Binding(
                        get: { agent.isEnabled },
                        set: { _ in model.toggleAgentEnabled(agent) }
                    ))
                    .toggleStyle(.switch)
                }

                Divider()

                // Configuration metadata
                VStack(alignment: .leading, spacing: 12) {
                    Text("Configuration")
                        .font(.headline)

                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 8) {
                        GridRow {
                            Text("Agent Identifier:")
                                .foregroundStyle(.secondary)
                            Text("`\(agent.name)`")
                                .font(.system(.body, design: .monospaced))
                        }
                        GridRow {
                            Text("Max Turns:")
                                .foregroundStyle(.secondary)
                            Text("\(agent.maxTurns)")
                        }
                        if let model = agent.model {
                            GridRow {
                                Text("Model Override:")
                                    .foregroundStyle(.secondary)
                                Text(model)
                            }
                        }
                        if let disallowed = agent.disallowedTools, !disallowed.isEmpty {
                            GridRow {
                                Text("Disallowed Tools:")
                                    .foregroundStyle(.secondary)
                                Text(disallowed.joined(separator: ", "))
                                    .font(.caption)
                                    .foregroundStyle(.red)
                            }
                        }
                        if let allowed = agent.tools, !allowed.isEmpty {
                            GridRow {
                                Text("Allowed Tools:")
                                    .foregroundStyle(.secondary)
                                Text(allowed.joined(separator: ", "))
                                    .font(.caption)
                            }
                        }
                        if let path = agent.filePath {
                            GridRow {
                                Text("File Location:")
                                    .foregroundStyle(.secondary)
                                Text(path)
                                    .font(.caption)
                                    .lineLimit(1)
                                    .truncationMode(.middle)
                            }
                        }
                    }
                }
                .padding()
                .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))

                // System Prompt / Instructions
                VStack(alignment: .leading, spacing: 8) {
                    Text("System Instructions")
                        .font(.headline)

                    ScrollView(.horizontal, showsIndicators: false) {
                        Text(agent.systemPrompt)
                            .font(.system(.caption, design: .monospaced))
                            .textSelection(.enabled)
                            .padding(12)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .background(RoundedRectangle(cornerRadius: 8).fill(Color(nsColor: .controlBackgroundColor)))
                }

                // REPL Usage hint
                VStack(alignment: .leading, spacing: 6) {
                    Text("REPL Chat Invocation")
                        .font(.subheadline.bold())
                    Text("You can invoke this subagent in chat or via slash command with clean context isolation:")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text("`/\(agent.name) <task description>`")
                        .font(.system(.caption, design: .monospaced))
                        .padding(8)
                        .background(RoundedRectangle(cornerRadius: 6).fill(Color(nsColor: .textBackgroundColor)))
                }
            }
            .padding(24)
        }
    }

    // MARK: - Empty State

    private var emptyStateView: some View {
        VStack(spacing: 12) {
            Image(systemName: "person.2.badge.gearshape")
                .font(.system(size: 44))
                .foregroundStyle(.tertiary)
            Text("No Agents Found")
                .font(.title3.bold())
            Text("No agent definitions match the selected scope filter.")
                .font(.callout)
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    // MARK: - Helpers

    private func badgeView(text: String, color: Color) -> some View {
        Text(text)
            .font(.system(size: 10, weight: .medium))
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
        }
    }
}
