import SwiftUI

/// The app-wide response-style library in the Engine settings pane.
///
/// Personalities are deliberately simple prompt rows. They do not become
/// agent definitions, projects, or CLI settings, so the selected wording is
/// the only context cost beyond the user's existing system prompt.
@MainActor
struct PersonalitySettingsSection: View {
    @ObservedObject var model: AppModel
    @State private var showingAddSheet = false

    var body: some View {
        Section(header: Text("Personality", bundle: .module)) {
            Picker(selection: selectedPersonalityBinding) {
                Text(verbatim: "None").tag(UUID?.none)
                ForEach(model.personalities) { personality in
                    Text(personality.name).tag(Optional(personality.id))
                }
            } label: { Text("Personality", bundle: .module) }

            if let personality = model.selectedPersonality {
                Text(personality.instructions)
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)

                Button(role: .destructive) {
                    model.deletePersonality(personality.id)
                } label: { Text("Remove Selected", bundle: .module) }
            }

            Button {
                showingAddSheet = true
            } label: { Text("Add Personality", bundle: .module) }
        }
        .settingsControl("Personality", pane: .engine, timing: .nextTurn)
        .sheet(isPresented: $showingAddSheet) {
            PersonalityEditorSheet { name, instructions in
                model.addPersonality(name: name, instructions: instructions)
            }
        }
    }

    private var selectedPersonalityBinding: Binding<UUID?> {
        Binding(
            get: { model.selectedPersonalityID },
            set: { model.selectPersonality($0) })
    }
}

/// Creates one compact, user-authored response style.
@MainActor
private struct PersonalityEditorSheet: View {
    @Environment(\.dismiss) private var dismiss
    let onSave: (String, String) -> Void

    @State private var name = ""
    @State private var instructions = ""

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(verbatim: "Add Personality")
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
                        .frame(minHeight: 140)
                }
            }
            .formStyle(.grouped)

            Divider()

            HStack {
                Spacer()
                Button {
                    onSave(name, instructions)
                    dismiss()
                } label: { Text("Save", bundle: .module) }
                .keyboardShortcut(.defaultAction)
                .disabled(!canSave)
            }
            .padding()
        }
        .frame(width: 500, height: 330)
    }
}
