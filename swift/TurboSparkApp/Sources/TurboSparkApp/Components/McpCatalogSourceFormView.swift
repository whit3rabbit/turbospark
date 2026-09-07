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

            Picker("Source", selection: $sourceKind) {
                ForEach(McpImportSheet.SourceKind.allCases) { kind in
                    Text(kind.rawValue).tag(kind)
                }
            }
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
                .foregroundStyle(.secondary)
            HStack(spacing: 6) {
                TextField(placeholder, text: $sourceText)
                    .textFieldStyle(.roundedBorder)
                if sourceKind == .directory {
                    Button("Choose...") { chooseFolder() }
                        .buttonStyle(.bordered)
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
                .foregroundStyle(.secondary)
            TextField("main", text: $gitRef)
                .textFieldStyle(.roundedBorder)
        }
    }

    private var sparsePathsField: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text("Sparse Paths (one per line, optional)", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.secondary)
            TextEditor(text: $sparsePathsText)
                .themedCode(.small)
                .frame(height: 50)
                .padding(4)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(Color.secondary.opacity(0.3), lineWidth: 0.5))
            Text("Checks out only these subtrees, for a repository too large to clone whole.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.secondary)
        }
    }

    private func chooseFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Choose a folder holding an MCP catalog manifest."
        if panel.runModal() == .OK, let url = panel.url {
            sourceText = url.path
        }
    }
}
