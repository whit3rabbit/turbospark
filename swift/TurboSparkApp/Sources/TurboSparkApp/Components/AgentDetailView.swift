import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The detail column of the Agents & Subagents pane.
///
/// A dedicated subview so the pane's state stays private (Gotcha 15). The
/// header stacks instead of sharing one row: title, badges, description,
/// Edit, Delete and the Enabled switch in a single `HStack` compressed the
/// description to a sliver in a narrow detail column.
@MainActor
struct AgentDetailView: View {
    @ObservedObject var model: AppModel
    let agent: AppAgentDefinition
    let onEdit: () -> Void
    let onDelete: () -> Void

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                Divider()
                configuration
                systemInstructions

                // REPL Usage hint. Only for an ENABLED agent: the slash
                // command refuses a disabled one (state#93), so the hint
                // would advertise a command that fails.
                if agent.isEnabled {
                    invocationHint
                }
            }
            .padding(SettingsLayout.detailInset)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(agent.displayName)
                .themedFont(.title2, weight: .bold)
                .fixedSize(horizontal: false, vertical: true)
                .accessibilityAddTraits(.isHeader)

            HStack(spacing: 6) {
                AgentBadge(text: Text(LocalizedStringKey(agent.scope.label), bundle: .module), color: AgentBadge.scopeColor(agent.scope))
                AgentBadge(text: Text(LocalizedStringKey(agent.sourceAgent.displayName), bundle: .module), color: .indigo)
            }

            Text(agent.agentDescription)
                .themedFont(.base)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: AppMetrics.actionSpacing) {
                Toggle(isOn: Binding(
                    get: { agent.isEnabled },
                    set: { _ in model.toggleAgentEnabled(agent) }
                )) {
                    Text("Enabled", bundle: .module)
                }
                .settingsControl("Enabled", pane: .agents, timing: .nextTurn)
                .toggleStyle(.switch)

                // Edit and Delete only where the definition is a file this
                // app owns: built-ins are code and plugin agents belong to
                // their plugin, so neither has an editor.
                if agent.scope == .userGlobal || agent.scope == .project {
                    if URL(fileURLWithPath: agent.filePath ?? "").pathExtension.lowercased() == "md" {
                        Button(action: onEdit) {
                            Label { Text("Edit", bundle: .module) } icon: { Image(systemName: "pencil") }
                        }
                        .appAction(.secondary, size: .small)
                    }

                    Button(role: .destructive, action: onDelete) {
                        Label { Text("Delete", bundle: .module) } icon: { Image(systemName: "trash") }
                    }
                    .appAction(.destructive, size: .small)
                }
            }
            .padding(.top, 4)
        }
    }

    private var configuration: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Configuration", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

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
                // No "Model Override" row: `agent.model` is parsed and read
                // by nothing, a subagent runs on whatever is loaded, and the
                // row sat between two that ARE enforced
                // (`swift/docs/SWIFT_SETTINGS_AUDIT.md`).
                if let disallowed = agent.disallowedTools, !disallowed.isEmpty {
                    GridRow {
                        Text("Disallowed Tools:", bundle: .module)
                            .foregroundStyle(.appSecondary)
                        Text(disallowed.joined(separator: ", "))
                            .themedFont(.small)
                            .foregroundStyle(AppAttention.color)
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
                        SettingsPathText(path, size: .small)
                    }
                }
            }
        }
        .appCardSurface(padding: 16)
    }

    private var systemInstructions: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("System Instructions", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            ScrollView(.horizontal, showsIndicators: false) {
                Text(agent.systemPrompt)
                    .themedCode(.small)
                    .textSelection(.enabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .appCardSurface(padding: 12)
        }
    }

    private var invocationHint: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text("REPL Chat Invocation", bundle: .module)
                .themedFont(.small, weight: .bold)
            Text("You can invoke this subagent in chat or via slash command with clean context isolation:", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            Text(verbatim: "`/\(agent.name) <task description>`")
                .themedCode(.small)
                .padding(8)
                .background(RoundedRectangle(cornerRadius: AppMetrics.controlRadius).fill(.appElevated))
        }
    }
}

/// The capsule used for an agent's scope and source in the list and detail.
struct AgentBadge: View {
    let text: Text
    let color: Color

    var body: some View {
        text
            .themedFont(.tiny, weight: .medium)
            .padding(.horizontal, 6)
            .padding(.vertical, 2)
            .background(Capsule().fill(color.opacity(0.15)))
            .foregroundStyle(color)
    }

    static func scopeColor(_ scope: AppAgentScope) -> Color {
        switch scope {
        case .builtIn: return .blue
        case .userGlobal: return .purple
        case .project: return .green
        case .plugin: return .orange
        }
    }
}
