import SwiftUI
import UniformTypeIdentifiers

/// The SOUL.md library in Soul settings: enable toggle, saved-entry picker,
/// editor, and external imports. SOUL is off until explicitly enabled;
/// detected external files are import sources, never auto-consumed.
@MainActor
struct SoulSettingsSection: View {
    @ObservedObject var model: AppModel
    @State private var content = ""
    @State private var statusMessage: String?
    @State private var isImportingFile = false
    @State private var showingSaveAsNewSheet = false

    private var hermesDetected: Bool {
        model.detectedSoulImportSources.contains { $0.kind == .hermes }
    }

    var body: some View {
        Section(header: Text(verbatim: "SOUL.md")) {
            Toggle(isOn: $model.soulPromptEnabled) {
                Text("Enable SOUL.md", bundle: .module)
            }
            .onChange(of: model.soulPromptEnabled) { _, _ in
                model.persistSettingsDebounced()
            }
            .settingsControl("Enable SOUL.md", pane: .soul, timing: .nextTurn)

            HStack(alignment: .center, spacing: 18) {
                TSIdlingSoulSparkView(size: 80)

                VStack(alignment: .leading, spacing: 6) {
                    Text("Global personality and communication guidance for the agent.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)

                    Text(verbatim: statusDescription)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
            .padding(.vertical, 4)

            Picker(selection: selectedSoulBinding) {
                Text(verbatim: "None").tag(UUID?.none)
                ForEach(model.soulPrompts) { soul in
                    Text(soul.name).tag(Optional(soul.id))
                }
            } label: {
                Text("Active Soul", bundle: .module)
            }
            .disabled(!model.soulPromptEnabled)
            .settingsControl("Active Soul", pane: .soul, timing: .nextTurn)

            TextEditor(text: $content)
                .themedCode(.base)
                .frame(minHeight: 150)
                .disabled(!model.soulPromptEnabled)
                .accessibilityLabel(Text(verbatim: "SOUL.md content"))

            HStack {
                Button {
                    save()
                } label: {
                    Text("Save", bundle: .module)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!model.soulPromptEnabled || model.selectedSoulPrompt == nil)

                Button {
                    showingSaveAsNewSheet = true
                } label: {
                    Text("Save As New...", bundle: .module)
                }
                .disabled(!model.soulPromptEnabled)

                if model.selectedSoulPrompt != nil {
                    Button(role: .destructive) {
                        deleteSelected()
                    } label: {
                        Text("Delete Selected", bundle: .module)
                    }
                    .disabled(!model.soulPromptEnabled)
                }

                ForEach(model.detectedSoulImportSources) { source in
                    Button {
                        importSource(source)
                    } label: {
                        importLabel(for: source.kind)
                    }
                }

                if !hermesDetected {
                    Button {
                        createHermes()
                    } label: {
                        Text("Create Hermes SOUL.md", bundle: .module)
                            .settingsControl("Create Hermes SOUL.md", pane: .soul, timing: .nextTurn)
                    }
                }

                Spacer()
            }

            Button {
                isImportingFile = true
            } label: {
                Text("Load File...", bundle: .module)
            }
            .settingsControl("Load File...", pane: .soul, timing: .nextTurn)

            if let statusMessage {
                Text(statusMessage)
                    .themedFont(.small)
                    .foregroundStyle(.appAccent)
            }
        }
        .settingsControl("SOUL.md", pane: .soul, timing: .nextTurn)
        .onAppear(perform: reload)
        .onChange(of: model.selectedSoulPromptID) { _, _ in
            reload()
        }
        .fileImporter(
            isPresented: $isImportingFile,
            allowedContentTypes: [.plainText, .data],
            allowsMultipleSelection: false,
            onCompletion: handleFileImport)
        .sheet(isPresented: $showingSaveAsNewSheet) {
            SoulNameSheet(content: content) { name in
                let prompt = model.addSoulPrompt(name: name, content: content)
                statusMessage = "Saved \(prompt.name)."
            }
        }
    }

    private var statusDescription: String {
        if !model.soulPromptEnabled {
            return "SOUL.md is disabled. Imports stay available and take effect when enabled."
        }
        guard let selected = model.selectedSoulPrompt else {
            return "Enabled. No soul selected."
        }
        return "Using saved soul \"\(selected.name)\"."
    }

    private var selectedSoulBinding: Binding<UUID?> {
        Binding(
            get: { model.selectedSoulPromptID },
            set: { model.selectSoulPrompt($0) })
    }

    private func reload() {
        content = model.selectedSoulPrompt?.content ?? ""
        statusMessage = nil
    }

    private func save() {
        guard let selected = model.selectedSoulPrompt else { return }
        model.updateSoulPrompt(id: selected.id, content: content)
        statusMessage = "SOUL.md saved."
    }

    private func deleteSelected() {
        guard let selected = model.selectedSoulPrompt else { return }
        model.deleteSoulPrompt(selected.id)
    }

    @ViewBuilder
    private func importLabel(for kind: SoulPromptImportKind) -> some View {
        switch kind {
        case .hermes:
            Text("Import Hermes", bundle: .module)
                .settingsControl("Import Hermes", pane: .soul, timing: .nextTurn)
        case .openClaw:
            Text("Import OpenClaw", bundle: .module)
                .settingsControl("Import OpenClaw", pane: .soul, timing: .nextTurn)
        }
    }

    private func importSource(_ source: SoulPromptImportSource) {
        do {
            try model.importSoul(from: source)
            reload()
            statusMessage = model.soulPromptEnabled
                ? "Imported \(source.displayName) SOUL.md."
                : "Imported \(source.displayName) SOUL.md. Enable SOUL.md to use it."
        } catch {
            statusMessage = "Could not import SOUL.md: \(error.localizedDescription)"
        }
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        do {
            guard let url = try result.get().first else { return }
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            let name = url.deletingPathExtension().lastPathComponent
            try model.importSoul(from: url, name: name)
            reload()
            statusMessage = model.soulPromptEnabled
                ? "Loaded \(name) SOUL.md."
                : "Loaded \(name) SOUL.md. Enable SOUL.md to use it."
        } catch {
            statusMessage = "Could not load SOUL.md: \(error.localizedDescription)"
        }
    }

    private func createHermes() {
        do {
            try model.createHermesSoul(content: content)
            statusMessage = "Created Hermes SOUL.md."
        } catch {
            statusMessage = "Could not create SOUL.md: \(error.localizedDescription)"
        }
    }
}

/// Names a new saved soul from the current editor content.
@MainActor
private struct SoulNameSheet: View {
    @Environment(\.dismiss) private var dismiss
    let content: String
    let onSave: (String) -> Void

    @State private var name = ""

    private var canSave: Bool {
        !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && !content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text(verbatim: "Save Soul As New")
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                    .keyboardShortcut(.cancelAction)
            }
            .padding()

            Divider()

            Form {
                TextField("Name", text: $name)
            }
            .formStyle(.grouped)

            Divider()

            HStack {
                Spacer()
                Button {
                    onSave(name)
                    dismiss()
                } label: { Text("Save", bundle: .module) }
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canSave)
            }
            .padding()
        }
        .frame(width: 360, height: 190)
    }
}
