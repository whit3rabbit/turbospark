import SwiftUI
import UniformTypeIdentifiers

/// Settings for the Hermes-compatible global SOUL.md personality file.
@MainActor
struct SoulSettingsSection: View {
    @ObservedObject var model: AppModel
    @State private var content = ""
    @State private var statusMessage: String?
    @State private var isImportingFile = false

    private var resolution: SoulPromptResolution {
        model.soulPromptResolution
    }

    var body: some View {
        Section(header: Text(verbatim: "SOUL.md")) {
            HStack(alignment: .center, spacing: 18) {
                TSIdlingSoulSparkView(size: 80)

                VStack(alignment: .leading, spacing: 6) {
                    Text("Global personality and communication guidance for the agent.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)

                    Text(sourceDescription)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                }
            }
            .padding(.vertical, 4)

            if let readError = resolution.readError {
                Text(verbatim: "Could not read SOUL.md: \(readError)")
                    .themedFont(.small)
                    .foregroundStyle(.red)
            }

            TextEditor(text: $content)
                .themedCode(.base)
                .frame(minHeight: 150)
                .accessibilityLabel(Text(verbatim: "SOUL.md content"))

            if !model.detectedSoulImportSources.isEmpty {
                Text("Detected SOUL.md files", bundle: .module)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appSecondary)
            }

            HStack {
                Button {
                    save()
                } label: {
                    Text("Save", bundle: .module)
                }
                .keyboardShortcut(.defaultAction)

                ForEach(model.detectedSoulImportSources) { source in
                    Button {
                        importSource(source)
                    } label: {
                        importLabel(for: source.kind)
                    }
                }

                if !resolution.hermesFileExists {
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
        .fileImporter(
            isPresented: $isImportingFile,
            allowedContentTypes: [.plainText, .data],
            allowsMultipleSelection: false,
            onCompletion: handleFileImport)
    }

    private var sourceDescription: String {
        switch resolution.source {
        case .hermes:
            return "Using Hermes file: \(resolution.hermesFileURL.path)"
        case .turboSpark:
            return "Using the TurboSpark profile fallback. Hermes file not found at \(resolution.hermesFileURL.path)"
        }
    }

    private func reload() {
        content = resolution.content
        statusMessage = nil
    }

    private func save() {
        do {
            try model.saveSoulPrompt(content)
            statusMessage = "SOUL.md saved."
        } catch {
            statusMessage = "Could not save SOUL.md: \(error.localizedDescription)"
        }
    }

    @ViewBuilder
    private func importLabel(for kind: SoulPromptImportKind) -> some View {
        switch kind {
        case .hermes:
            Text("Import Hermes", bundle: .module)
                .settingsControl("Import Hermes", pane: .engine, timing: .nextTurn)
        case .openClaw:
            Text("Import OpenClaw", bundle: .module)
                .settingsControl("Import OpenClaw", pane: .engine, timing: .nextTurn)
        }
    }

    private func importSource(_ source: SoulPromptImportSource) {
        do {
            try model.importSoul(from: source)
            reload()
            statusMessage = "Imported \(source.displayName) SOUL.md into the TurboSpark fallback."
        } catch {
            statusMessage = "Could not import SOUL.md: \(error.localizedDescription)"
        }
    }

    private func handleFileImport(_ result: Result<[URL], Error>) {
        do {
            guard let url = try result.get().first else { return }
            let access = url.startAccessingSecurityScopedResource()
            defer { if access { url.stopAccessingSecurityScopedResource() } }
            try model.importSoul(from: url)
            reload()
            statusMessage = "Loaded SOUL.md into the TurboSpark fallback."
        } catch {
            statusMessage = "Could not load SOUL.md: \(error.localizedDescription)"
        }
    }

    private func createHermes() {
        do {
            try model.createHermesSoul(content: content)
            reload()
            statusMessage = "Created Hermes SOUL.md."
        } catch {
            statusMessage = "Could not create SOUL.md: \(error.localizedDescription)"
        }
    }
}
