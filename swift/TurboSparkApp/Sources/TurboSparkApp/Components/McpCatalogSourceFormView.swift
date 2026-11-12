import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Where an MCP catalog is read from: the kind, the source, the git ref and the
/// sparse paths. A dedicated subview taking bindings, per Gotcha 15.
@MainActor
struct McpCatalogSourceFormView: View {
    @Binding var sourceKind: McpImportSheet.SourceKind
    @Binding var sourceText: String
    @Binding var gitRef: String
    @Binding var sparsePathsText: String

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Catalog Source", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)

            Picker(selection: $sourceKind) {
                ForEach(McpImportSheet.SourceKind.allCases) { kind in
                    kind.title.tag(kind)
                }
            } label: { Text("Source", bundle: .module) }
            .labelsHidden()
            .pickerStyle(.segmented)

            sourceField
            if sourceKind != .directory {
                gitRefField
                sparsePathsField
            }
        }
    }

    private var sourceField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Source", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            HStack(spacing: 6) {
                // Placeholders are example identifiers and paths: verbatim.
                TextField(text: $sourceText, prompt: Text(verbatim: placeholder)) {
                    Text("Source", bundle: .module)
                }
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                if sourceKind == .directory {
                    Button { chooseFolder() } label: { Text("Choose...", bundle: .module) }
                        .appAction(.secondary, size: .small)
                }
            }
        }
    }

    private var placeholder: String {
        switch sourceKind {
        case .github: return "owner/repo"
        case .git: return "https://example.com/org/catalog.git"
        case .directory: return "/path/to/catalog"
        }
    }

    private var gitRefField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Git Ref", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            TextField(text: $gitRef, prompt: Text(verbatim: "main")) {
                Text("Git Ref", bundle: .module)
            }
            .labelsHidden()
            .textFieldStyle(.roundedBorder)
        }
    }

    private var sparsePathsField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Sparse Paths (one per line, optional)", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            TextEditor(text: $sparsePathsText)
                .themedCode(.small)
                .appEditorSurface(minHeight: 50)
                .accessibilityLabel(Text("Sparse Paths (one per line, optional)", bundle: .module))
            Text("Checks out only these subtrees, for a repository too large to clone whole.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = AppStrings.localized("Choose")
        panel.message = AppStrings.localized("Choose a folder holding an MCP catalog manifest.")
        if panel.runModal() == .OK, let url = panel.url {
            sourceText = url.path
        }
    }
}
