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
    @State private var isConfirmingDelete = false

    var body: some View {
        Section(header: Text("Personality", bundle: .module)) {
            Picker(selection: selectedPersonalityBinding) {
                Text("None", bundle: .module).tag(UUID?.none)
                ForEach(model.personalities) { personality in
                    Text(verbatim: personality.name).tag(Optional(personality.id))
                }
            } label: { Text("Personality", bundle: .module) }
            .labelsHidden()

            if let personality = model.selectedPersonality {
                Text(verbatim: personality.instructions)
                    .themedCode(.small)
                    .foregroundStyle(.appSecondary)

                Button(role: .destructive) {
                    isConfirmingDelete = true
                } label: { Text("Remove Selected", bundle: .module) }
                .appAction(.destructive, size: .small)
                .confirmationDialog(
                    Text("Delete Personality", bundle: .module),
                    isPresented: $isConfirmingDelete,
                    titleVisibility: .visible
                ) {
                    Button(role: .destructive) {
                        model.deletePersonality(personality.id)
                    } label: {
                        Text("Delete Personality", bundle: .module)
                    }
                    Button(role: .cancel) {} label: {
                        Text("Cancel", bundle: .module)
                    }
                } message: {
                    Text("Are you sure you want to delete this personality? This action cannot be undone.", bundle: .module)
                }
            }

            Button {
                showingAddSheet = true
            } label: { Text("Add Personality", bundle: .module) }
            .appAction(.secondary, size: .small)
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
    @FocusState private var isNameFocused: Bool

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !instructions.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            AppAdaptiveHeader {
                Text("Add Personality", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
            } actions: {
                Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
            }
            .padding()

            Divider()

            Form {
                TextField(text: $name) {
                    Text("Name", bundle: .module)
                }
                .focused($isNameFocused)

                VStack(alignment: .leading, spacing: 6) {
                    Text("Instructions", bundle: .module)
                    TextEditor(text: $instructions)
                        .themedCode(.base)
                        .appEditorSurface(minHeight: 140)
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
                } label: { Text("Save", bundle: .module) }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
                .disabled(!canSave)
                .help(canSave
                    ? Text("Save", bundle: .module)
                    : Text("A name and instructions are required to save.", bundle: .module))
            }
            .padding()
        }
        .frame(minWidth: 440, idealWidth: 500, minHeight: 310, idealHeight: 350)
        .onAppear {
            isNameFocused = true
        }
    }
}
