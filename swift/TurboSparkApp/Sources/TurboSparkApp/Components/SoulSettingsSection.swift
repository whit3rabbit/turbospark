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

                    statusDescription
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
            .padding(.vertical, 4)

            Picker(selection: selectedSoulBinding) {
                Text("None", bundle: .module).tag(UUID?.none)
                ForEach(model.soulPrompts) { soul in
                    Text(verbatim: soul.name).tag(Optional(soul.id))
                }
            } label: {
                Text("Active Soul", bundle: .module)
            }
            .disabled(!model.soulPromptEnabled)
            .settingsControl("Active Soul", pane: .soul, timing: .nextTurn)

            TextEditor(text: $content)
                .themedCode(.base)
                .appEditorSurface(minHeight: 150)
                .disabled(!model.soulPromptEnabled)
                .accessibilityLabel(Text("SOUL.md content", bundle: .module))

            editActions
            importActions

            if let statusMessage {
                Text(verbatim: statusMessage)
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
                statusMessage = AppStrings.localized("Saved \(prompt.name).")
            }
        }
    }

    /// What the editor writes back: a fixed set, so the row cannot grow.
    private var editActions: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            Button {
                save()
            } label: {
                Text("Save", bundle: .module)
            }
            .appAction(.primary, size: .small)
            .keyboardShortcut(.defaultAction)
            .disabled(!model.soulPromptEnabled || model.selectedSoulPrompt == nil)

            Button {
                showingSaveAsNewSheet = true
            } label: {
                Text("Save As New...", bundle: .module)
            }
            .appAction(.secondary, size: .small)
            .disabled(!model.soulPromptEnabled)

            if model.selectedSoulPrompt != nil {
                Button(role: .destructive) {
                    deleteSelected()
                } label: {
                    Text("Delete Selected", bundle: .module)
                }
                .appAction(.destructive, size: .small)
                .disabled(!model.soulPromptEnabled)
            }

            Spacer()
        }
    }

    /// Every way in, behind one labeled menu. The detected sources used to be
    /// one button each in the save row, which grew with every app found on
    /// the machine. The items keep their own search anchors; a menu item is
    /// not mounted until the menu opens, so the menu carries one too.
    private var importActions: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            Menu {
                ForEach(model.detectedSoulImportSources) { source in
                    Button {
                        importSource(source)
                    } label: {
                        importLabel(for: source.kind)
                    }
                }
                if !model.detectedSoulImportSources.isEmpty {
                    Divider()
                }
                Button {
                    isImportingFile = true
                } label: {
                    Text("Load File...", bundle: .module)
                        .settingsControl("Load File...", pane: .soul, timing: .nextTurn)
                }
            } label: {
                Label { Text("Import", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
            }
            .appMenuAction(size: .small)
            .help(Text("Import SOUL.md from a detected app or a file", bundle: .module))
            .settingsControl("Import", pane: .soul, timing: .nextTurn)

            if !hermesDetected {
                Button {
                    createHermes()
                } label: {
                    Text("Create Hermes SOUL.md", bundle: .module)
                        .settingsControl("Create Hermes SOUL.md", pane: .soul, timing: .nextTurn)
                }
                .appAction(.secondary, size: .small)
            }

            Spacer()
        }
    }

    private var statusDescription: Text {
        if !model.soulPromptEnabled {
            return Text("SOUL.md is disabled. Imports stay available and take effect when enabled.", bundle: .module)
        }
        guard let selected = model.selectedSoulPrompt else {
            return Text("Enabled. No soul selected.", bundle: .module)
        }
        return Text("Using saved soul \"\(selected.name)\".", bundle: .module)
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
        statusMessage = AppStrings.localized("SOUL.md saved.")
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
            // Hermes and OpenClaw are product names; only the frame translates.
            let sourceName = source.displayName
            statusMessage = model.soulPromptEnabled
                ? AppStrings.localized("Imported \(sourceName) SOUL.md.")
                : AppStrings.localized("Imported \(sourceName) SOUL.md. Enable SOUL.md to use it.")
        } catch {
            statusMessage = AppStrings.localized("Could not import SOUL.md: \(error.localizedDescription)")
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
                ? AppStrings.localized("Loaded \(name) SOUL.md.")
                : AppStrings.localized("Loaded \(name) SOUL.md. Enable SOUL.md to use it.")
        } catch {
            statusMessage = AppStrings.localized("Could not load SOUL.md: \(error.localizedDescription)")
        }
    }

    private func createHermes() {
        do {
            try model.createHermesSoul(content: content)
            statusMessage = AppStrings.localized("Created Hermes SOUL.md.")
        } catch {
            statusMessage = AppStrings.localized("Could not create SOUL.md: \(error.localizedDescription)")
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
                Text("Save Soul As New", bundle: .module)
                    .themedFont(.base, weight: .semibold)
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
            }
            .formStyle(.grouped)

            Divider()

            HStack {
                Spacer()
                Button {
                    onSave(name)
                    dismiss()
                } label: { Text("Save", bundle: .module) }
                    .appAction(.primary)
                    .keyboardShortcut(.defaultAction)
                    .disabled(!canSave)
            }
            .padding()
        }
        .frame(width: 360, height: 190)
    }
}
