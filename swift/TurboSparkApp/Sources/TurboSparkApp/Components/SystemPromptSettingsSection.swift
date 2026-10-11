import SwiftUI

/// The reusable app-wide system-prompt library in Engine settings.
@MainActor
struct SystemPromptSettingsSection: View {
    @ObservedObject var model: AppModel
    @State private var showingAddSheet = false
    @State private var editingPrompt: AppSystemPrompt?

    var body: some View {
        Section(header: Text("System Prompt", bundle: .module)) {
            Picker(selection: selectedPromptBinding) {
                Text("None", bundle: .module).tag(UUID?.none)
                ForEach(model.systemPrompts) { prompt in
                    Text(verbatim: prompt.name).tag(Optional(prompt.id))
                }
            } label: { Text("Load Prompt", bundle: .module) }

            if let prompt = model.selectedSystemPrompt {
                Text(verbatim: prompt.instructions)
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)

                HStack(spacing: AppMetrics.actionSpacing) {
                    Button {
                        editingPrompt = prompt
                    } label: { Text("Edit Selected", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    Button(role: .destructive) {
                        model.deleteSystemPrompt(prompt.id)
                    } label: { Text("Delete Selected", bundle: .module) }
                    .appAction(.destructive, size: .small)
                }
            }

            Button {
                showingAddSheet = true
            } label: { Text("Add Prompt", bundle: .module) }
            .appAction(.secondary, size: .small)
        }
        .settingsControl("System Prompt", pane: .engine, timing: .nextTurn)
        .sheet(isPresented: $showingAddSheet) {
            SystemPromptEditorSheet(title: Text("Add System Prompt", bundle: .module)) { name, instructions in
                model.addSystemPrompt(name: name, instructions: instructions)
            }
        }
        .sheet(item: $editingPrompt) { prompt in
            SystemPromptEditorSheet(title: Text("Edit System Prompt", bundle: .module), prompt: prompt) { name, instructions in
                model.updateSystemPrompt(id: prompt.id, name: name, instructions: instructions)
            }
        }
    }

    private var selectedPromptBinding: Binding<UUID?> {
        Binding(
            get: { model.selectedSystemPromptID },
            set: { model.selectSystemPrompt($0) })
    }
}

/// Adds or changes a reusable app-wide system prompt.
@MainActor
private struct SystemPromptEditorSheet: View {
    @Environment(\.dismiss) private var dismiss
    let title: Text
    let onSave: (String, String) -> Void

    @State private var name: String
    @State private var instructions: String

    init(
        title: Text,
        prompt: AppSystemPrompt? = nil,
        onSave: @escaping (String, String) -> Void
    ) {
        self.title = title
        self.onSave = onSave
        _name = State(initialValue: prompt?.name ?? "")
        _instructions = State(initialValue: prompt?.instructions ?? "")
    }

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                title
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
            }
            .padding()

            Divider()

            Form {
                TextField(text: $name) {
                    Text("Name", bundle: .module)
                }
                VStack(alignment: .leading, spacing: 6) {
                    Text("Instructions", bundle: .module)
                    TextEditor(text: $instructions)
                        .themedCode(.base)
                        .appEditorSurface(minHeight: 160)
                        .accessibilityLabel(Text("Instructions", bundle: .module))
                }
            }
            .formStyle(.grouped)

            Divider()

            HStack {
                Spacer()
                Button {
                    onSave(name, instructions)
                    dismiss()
                } label: { Text("Save Prompt", bundle: .module) }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
                .disabled(!canSave)
            }
            .padding()
        }
        .frame(width: 540, height: 360)
    }
}
