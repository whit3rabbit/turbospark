import AppKit
import SwiftUI
import TurboSpark

/// "Install from Files" (Discover header): Inkling 2 Mini from a
/// LookAtMyQuant release folder or an Inkling 2 GGUF already on this Mac.
/// The engine decides what a path is (`TurboSparkCatalog.localInstallKind`),
/// so this view never guesses from a file name.
@MainActor
struct LocalModelInstallSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var path: String?
    @State private var kind: LocalInstallKind?
    /// True when the chosen path could not be read at all.
    @State private var unreadable = false
    @State private var tokenizerFolder: String?
    @State private var alias = ""
    /// The library the user picked for this install. Nil follows the default.
    @State private var destination: String?

    private var chosenLibrary: String? {
        model.installDestinations.libraryForInstall(selection: destination)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 12) {
                    InstallDestinationPicker(model: model, selection: $destination)
                    sourceRow
                    status
                    if kind?.needsTokenizer == true {
                        tokenizerRow
                    }
                    TextField(text: $alias, prompt: Text("Local alias (e.g. my-model)", bundle: .module)) {
                        Text("Local Alias", bundle: .module)
                    }
                    .textFieldStyle(.roundedBorder)
                }
                .padding(14)
            }
            Divider()
            footer
        }
        .themedFont(.small)
        .frame(minWidth: 520, minHeight: 320)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            Text("Install from Files", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            Text("Inkling 2 Mini from a LookAtMyQuant release folder or an Inkling 2 GGUF file.", bundle: .module)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(14)
    }

    private var footer: some View {
        HStack {
            Spacer()
            Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                .appAction(.secondary)
                .keyboardShortcut(.cancelAction)
            installButton
        }
        .padding(12)
    }

    private var sourceRow: some View {
        HStack(spacing: 8) {
            Button {
                chooseSource()
            } label: { Text("Choose Model Files...", bundle: .module) }
            .appAction(.secondary, size: .small)
            if let path {
                SettingsPathText(path)
            }
        }
    }

    @ViewBuilder
    private var status: some View {
        if kind?.kind == "lamq" {
            Label {
                Text("LookAtMyQuant release folder", bundle: .module)
            } icon: {
                Image(systemName: "checkmark.circle")
            }
            .foregroundStyle(.appSuccess)
        } else if kind?.kind == "gguf" {
            Label {
                Text("Inkling 2 GGUF: choose a folder with its tokenizer files, such as a download of thinkingmachines/Inkling-2-Mini.", bundle: .module)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "checkmark.circle")
            }
            .foregroundStyle(.appSuccess)
        } else if path != nil && (kind != nil || unreadable) {
            Label {
                Text("This is neither an Inkling 2 release folder nor an Inkling 2 GGUF.", bundle: .module)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "exclamationmark.triangle")
            }
            .foregroundStyle(.appWarning)
        }
    }

    private var tokenizerRow: some View {
        HStack(spacing: 8) {
            Button {
                chooseTokenizerFolder()
            } label: { Text("Choose Tokenizer Folder...", bundle: .module) }
            .appAction(.secondary, size: .small)
            if let tokenizerFolder {
                SettingsPathText(tokenizerFolder)
            }
        }
    }

    /// Everything the install needs is chosen.
    private var isReady: Bool {
        guard let path, let kind, kind.kind != nil else { return false }
        if kind.needsTokenizer && tokenizerFolder == nil { return false }
        let trimmed = alias.trimmingCharacters(in: .whitespacesAndNewlines)
        return !path.isEmpty && AppModel.isSafeInstallAlias(trimmed)
    }

    private var installButton: some View {
        Button {
            guard let path else { return }
            model.installLocalFiles(
                path: path, alias: alias,
                tokenizerFrom: kind?.needsTokenizer == true ? tokenizerFolder : nil,
                library: chosenLibrary)
            dismiss()
        } label: { Text("Install", bundle: .module) }
        .appAction(.primary)
        .keyboardShortcut(.defaultAction)
        .disabled(!isReady || model.isRunning || !model.canInstall(alias: alias))
        .help(Text("Choose the model files, a tokenizer folder for a GGUF, and an alias first.", bundle: .module))
    }

    private func chooseSource() {
        let panel = NSOpenPanel()
        panel.title = AppStrings.localized("Choose Model Files")
        panel.message = AppStrings.localized("Choose a LookAtMyQuant release folder or an Inkling 2 GGUF file.")
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.resolvesAliases = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        let chosen = url.path
        path = chosen
        kind = nil
        unreadable = false
        tokenizerFolder = nil
        if alias.isEmpty {
            alias = Self.suggestedAlias(for: url)
        }
        // A GGUF header can carry a large vocabulary; read it off the main
        // thread.
        Task {
            let result = await Task.detached { () -> LocalInstallKind? in
                try? TurboSparkCatalog.localInstallKind(path: chosen)
            }.value
            guard path == chosen else { return }
            kind = result
            unreadable = result == nil
        }
    }

    private func chooseTokenizerFolder() {
        let panel = NSOpenPanel()
        panel.title = AppStrings.localized("Choose Tokenizer Folder")
        panel.message = AppStrings.localized("Choose the folder holding tokenizer.json.")
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.canCreateDirectories = false
        panel.resolvesAliases = true
        guard panel.runModal() == .OK, let url = panel.url else { return }
        tokenizerFolder = url.path
    }

    /// The file or folder name, reduced to the characters an alias allows.
    nonisolated static func suggestedAlias(for url: URL) -> String {
        let base = url.hasDirectoryPath ? url.lastPathComponent : url.deletingPathExtension().lastPathComponent
        let mapped = base.lowercased().unicodeScalars.map { scalar -> Character in
            let ok = scalar.value < 128
                && (CharacterSet.alphanumerics.contains(scalar) || "._-".unicodeScalars.contains(scalar))
            return ok ? Character(scalar) : "-"
        }
        let alias = String(mapped).trimmingCharacters(in: CharacterSet(charactersIn: "-."))
        return AppModel.isSafeInstallAlias(alias) ? alias : ""
    }
}
