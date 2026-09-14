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
                Text(verbatim: "None").tag(UUID?.none)
                ForEach(model.systemPrompts) { prompt in
                    Text(prompt.name).tag(Optional(prompt.id))
                }
            } label: { Text("Load Prompt", bundle: .module) }

            if let prompt = model.selectedSystemPrompt {
                Text(prompt.instructions)
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)

                HStack {
                    Button {
                        editingPrompt = prompt
                    } label: { Text("Edit Selected", bundle: .module) }
                    Button(role: .destructive) {
                        model.deleteSystemPrompt(prompt.id)
                    } label: { Text("Delete Selected", bundle: .module) }
                }
            }

            Button {
                showingAddSheet = true
            } label: { Text("Add Prompt", bundle: .module) }
        }
        .settingsControl("System Prompt", pane: .engine, timing: .nextTurn)
        .sheet(isPresented: $showingAddSheet) {
            SystemPromptEditorSheet(title: "Add System Prompt") { name, instructions in
                model.addSystemPrompt(name: name, instructions: instructions)
            }
        }
        .sheet(item: $editingPrompt) { prompt in
            SystemPromptEditorSheet(title: "Edit System Prompt", prompt: prompt) { name, instructions in
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
    let title: String
    let onSave: (String, String) -> Void

    @State private var name: String
    @State private var instructions: String

    init(
        title: String,
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
                Text(verbatim: title)
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
            }
            .padding()

            Divider()

            Form {
                TextField("Name", text: $name)
                VStack(alignment: .leading, spacing: 6) {
                    Text(verbatim: "Instructions")
                    TextEditor(text: $instructions)
                        .themedCode(.base)
                        .frame(minHeight: 160)
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
                .keyboardShortcut(.defaultAction)
                .disabled(!canSave)
            }
            .padding()
        }
        .frame(width: 540, height: 360)
    }
}
