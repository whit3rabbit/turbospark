import SwiftUI

/// The reusable app-wide system-prompt library in Engine settings.
@MainActor
struct SystemPromptSettingsSection: View {
    @ObservedObject var model: AppModel
    @State private var showingAddSheet = false
    @State private var editingPrompt: AppSystemPrompt?

    var body: some View {
        Section("System Prompt") {
            Picker("Load Prompt", selection: selectedPromptBinding) {
                Text(verbatim: "None").tag(UUID?.none)
                ForEach(model.systemPrompts) { prompt in
                    Text(prompt.name).tag(Optional(prompt.id))
                }
            }

            if let prompt = model.selectedSystemPrompt {
                Text(prompt.instructions)
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)

                HStack {
                    Button("Edit Selected") {
                        editingPrompt = prompt
                    }
                    Button("Delete Selected", role: .destructive) {
                        model.deleteSystemPrompt(prompt.id)
                    }
                }
            }

            Button("Add Prompt") {
                showingAddSheet = true
            }
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
                Button("Cancel") { dismiss() }
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
                Button("Save Prompt") {
                    onSave(name, instructions)
                    dismiss()
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!canSave)
            }
            .padding()
        }
        .frame(width: 540, height: 360)
    }
}
