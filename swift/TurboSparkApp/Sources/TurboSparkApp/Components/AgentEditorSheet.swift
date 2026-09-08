import SwiftUI

/// Modal sheet for creating a new agent or editing an existing user- or
/// project-scoped agent definition. Built-ins and plugin agents are not
/// editable here (they are code and plugin-owned files respectively), which
/// the pane enforces by not offering the button; the save path refuses them
/// again, so a stale sheet cannot write either.
public struct AgentEditorSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var agentToEdit: AppAgentDefinition?

    @State private var name: String = ""
    @State private var displayNameText: String = ""
    @State private var descriptionText: String = ""
    @State private var allowedToolsText: String = ""
    @State private var disallowedToolsText: String = ""
    @State private var maxTurns: Int = 5
    @State private var isProjectScope: Bool = false
    @State private var systemPrompt: String = ""

    public init(model: AppModel, agentToEdit: AppAgentDefinition? = nil) {
        self.model = model
        self.agentToEdit = agentToEdit
    }

    /// The scope a SAVE would write into. Fixed while editing: the file's
    /// location is part of its identity, and moving scopes is a delete plus
    /// a create, not an edit.
    private var effectiveScope: AppAgentScope {
        if agentToEdit != nil { return agentToEdit?.scope ?? .userGlobal }
        return isProjectScope ? .project : .userGlobal
    }

    public var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(agentToEdit == nil
                    ? "Create New Agent"
                    : "Edit Agent: \(name)", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Button {
                    dismiss()
                } label: {
                    Text("Cancel", bundle: .module)
                }
                .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(Color(nsColor: .windowBackgroundColor))

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Agent Name", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField("e.g. deploy-auditor, doc-writer", text: $name)
                                .textFieldStyle(.roundedBorder)
                                .disabled(agentToEdit != nil)
                            Text("The name the `agent` tool and /slash commands resolve. Fixed once created.",
                                 bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.secondary)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Scope", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            Picker("Scope", selection: $isProjectScope) {
                                Text("User Scope (~/.turbospark/agents)", bundle: .module).tag(false)
                                Text("Project Scope (.turbospark/agents)", bundle: .module).tag(true)
                            }
                            .pickerStyle(.segmented)
                            .disabled(agentToEdit != nil || model.selectedProject == nil)
                            if isProjectScope && model.selectedProject == nil && agentToEdit == nil {
                                Text("Open a project to write a project-scoped agent.", bundle: .module)
                                    .themedFont(.tiny)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }

                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Display Name", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField("e.g. Deploy Auditor", text: $displayNameText)
                                .textFieldStyle(.roundedBorder)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Stepper(value: $maxTurns, in: 1...AppAgentDefinition.maxTurnsCeiling) {
                                Text("Max Turns: \(maxTurns)", bundle: .module)
                                    .themedFont(.small)
                            }
                            Text("Upper bound on the agent's autonomous turns (1 to \(AppAgentDefinition.maxTurnsCeiling)).",
                                 bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text("Description (when to use)", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                        TextField("Shown to the model in the agent listing and in Settings; state when this agent should be picked...", text: $descriptionText)
                            .textFieldStyle(.roundedBorder)
                    }

                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Allowed Tools (Comma-separated)", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField("e.g. FileRead, Grep, Bash", text: $allowedToolsText)
                                .textFieldStyle(.roundedBorder)
                            Text("Leave empty for the default tool set.", bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.secondary)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Disallowed Tools (Comma-separated)", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField("e.g. FileWrite, FileEdit, agent", text: $disallowedToolsText)
                                .textFieldStyle(.roundedBorder)
                            Text("Refused even if the allowlist would permit them.", bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.secondary)
                        }
                    }

                    VStack(alignment: .leading, spacing: 4) {
                        Text("System Prompt", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                        TextEditor(text: $systemPrompt)
                            .themedCode(.base)
                            .frame(minHeight: 180)
                            .padding(4)
                            .background(Color(nsColor: .textBackgroundColor))
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(TurboSparkTheme.hairlineColor, lineWidth: 1)
                            )
                    }
                }
                .padding(20)
            }

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            HStack {
                Spacer()
                Button {
                    saveAction()
                } label: {
                    Text(agentToEdit == nil ? "Create Agent" : "Save", bundle: .module)
                }
                .buttonStyle(.borderedProminent)
                .tint(TurboSparkTheme.accentColor)
                .disabled(!canSave)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .frame(minWidth: 620, minHeight: 520)
        .onAppear(perform: loadEditingAgent)
    }

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    private func loadEditingAgent() {
        guard let agent = agentToEdit else { return }
        name = agent.name
        displayNameText = agent.displayName
        descriptionText = agent.agentDescription
        allowedToolsText = (agent.tools ?? []).joined(separator: ", ")
        disallowedToolsText = (agent.disallowedTools ?? []).joined(separator: ", ")
        maxTurns = agent.maxTurns
        isProjectScope = agent.scope == .project
        systemPrompt = agent.systemPrompt
    }

    private func parseToolList(_ text: String) -> [String]? {
        let items = text
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        return items.isEmpty ? nil : items
    }

    private func saveAction() {
        if var agent = agentToEdit {
            agent.displayName = displayNameText.isEmpty ? agent.name.capitalized : displayNameText
            agent.agentDescription = descriptionText
            agent.tools = parseToolList(allowedToolsText)
            agent.disallowedTools = parseToolList(disallowedToolsText)
            agent.maxTurns = maxTurns
            agent.systemPrompt = systemPrompt
            model.updateAgent(agent)
        } else {
            model.createAgent(
                name: name,
                displayName: displayNameText.isEmpty ? nil : displayNameText,
                description: descriptionText,
                systemPrompt: systemPrompt,
                tools: parseToolList(allowedToolsText),
                disallowedTools: parseToolList(disallowedToolsText),
                maxTurns: maxTurns,
                scope: effectiveScope)
        }
        dismiss()
    }
}
