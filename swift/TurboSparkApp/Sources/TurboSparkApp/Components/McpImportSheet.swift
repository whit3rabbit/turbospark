import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Browses an MCP server catalog and installs entries from it.
@MainActor
public struct McpImportSheet: View {
    @ObservedObject var model: AppModel

    /// Where a catalog is read from. The user picks this explicitly rather than
    /// having it guessed from the text, which is what `SkillImportSheet` does:
    /// its heuristic routes anything containing a colon to `.git`, so
    /// `git@github.com:owner/repo.git` and a bare `owner/repo` land in
    /// different cases for a reason the user cannot see.
    public enum SourceKind: String, CaseIterable, Identifiable {
        case github = "GitHub Repo"
        case git = "Git URL"
        case directory = "Local Folder"

        public var id: String { rawValue }
    }

    let onDismiss: () -> Void

    @State private var sourceKind: SourceKind = .github
    @State private var sourceText: String = ""
    @State private var gitRef: String = "main"
    @State private var sparsePathsText: String = ""
    @State private var installToProjectScope: Bool = false

    @State private var isFetching: Bool = false
    @State private var manifest: McpMarketplaceManifest?
    @State private var fetchError: String?
    @State private var installedNames: Set<String> = []
    @State private var entryErrors: [String: String] = [:]

    public init(model: AppModel, onDismiss: @escaping () -> Void) {
        self.model = model
        self.onDismiss = onDismiss
    }

    public var body: some View {
        VStack(spacing: 0) {
            header
            Divider()
            scopeBar
            Divider()
            ScrollView {
                VStack(alignment: .leading, spacing: 18) {
                    sourceForm
                    if let fetchError {
                        errorRow(fetchError)
                    }
                    if let manifest {
                        manifestSection(manifest)
                    }
                }
                .padding(20)
            }
            Divider()
            footer
        }
        .frame(width: 620, height: 660)
    }

    // MARK: - Chrome

    private var header: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Add MCP Marketplace")
                    .font(.headline)
                Text("Install servers from a catalog published in a Git repository or a local folder.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") { onDismiss() }
                .buttonStyle(.plain)
                .foregroundStyle(.secondary)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    private var scopeBar: some View {
        HStack(spacing: 10) {
            Text("Install Into:")
                .font(.caption.weight(.semibold))
            Picker("Scope", selection: $installToProjectScope) {
                Text("All Projects (Global)").tag(false)
                Text("This Project Only").tag(true)
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)
            .disabled(model.selectedProject == nil)
            Spacer()
            if model.selectedProject == nil {
                Text("Open a project to install into one.")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.4))
    }

    private var footer: some View {
        HStack {
            Button("Done") { onDismiss() }
                .keyboardShortcut(.cancelAction)
            Spacer()
            Button {
                Task { await fetchCatalog() }
            } label: {
                if isFetching {
                    ProgressView().scaleEffect(0.6).frame(width: 16, height: 16)
                } else {
                    Label("Fetch Catalog", systemImage: "arrow.down.circle")
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(isFetching || sourceText.trimmingCharacters(in: .whitespaces).isEmpty)
            .keyboardShortcut(.defaultAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 14)
    }

    // MARK: - Source form

    private var sourceForm: some View {
        McpCatalogSourceFormView(
            sourceKind: $sourceKind,
            sourceText: $sourceText,
            gitRef: $gitRef,
            sparsePathsText: $sparsePathsText)
    }

    // MARK: - Results

    private func manifestSection(_ manifest: McpMarketplaceManifest) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 6) {
                Text(manifest.name)
                    .font(.subheadline.weight(.semibold))
                if let owner = manifest.owner {
                    Text(owner)
                        .font(.caption2)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.secondary.opacity(0.12))
                        .clipShape(Capsule())
                }
                Spacer()
                Text("\(manifest.servers.count) servers")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if let description = manifest.manifestDescription {
                Text(description)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if manifest.servers.isEmpty {
                Text("This catalog lists no servers.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(manifest.servers) { entry in
                    entryRow(entry, marketplaceName: manifest.name)
                }
            }
        }
    }

    private func entryRow(_ entry: McpMarketplaceEntry, marketplaceName: String) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 8) {
                VStack(alignment: .leading, spacing: 3) {
                    HStack(spacing: 6) {
                        Text(entry.name)
                            .font(.callout.weight(.semibold))
                        if let version = entry.version {
                            Text(version)
                                .font(.caption2.monospaced())
                                .foregroundStyle(.secondary)
                        }
                        if let category = entry.category {
                            Text(category)
                                .font(.caption2)
                                .padding(.horizontal, 6)
                                .padding(.vertical, 2)
                                .background(TurboSparkTheme.accentColor.opacity(0.12))
                                .clipShape(Capsule())
                        }
                    }
                    if !entry.entryDescription.isEmpty {
                        Text(entry.entryDescription)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    // **THE COMMAND IS SHOWN BEFORE THE BUTTON, ON PURPOSE.**
                    // A catalog entry names a binary this app will spawn. What
                    // the user is approving is that command line, not a name in
                    // somebody else's repository.
                    Text(entry.commandSummary)
                        .font(.caption2.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                        .padding(6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.7))
                        .clipShape(RoundedRectangle(cornerRadius: 5))
                }
                Spacer(minLength: 8)
                installButton(entry, marketplaceName: marketplaceName)
            }

            if let message = entryErrors[entry.id] {
                Text(message)
                    .font(.caption2)
                    .foregroundStyle(.red)
            }
        }
        .padding(10)
        .background(Color(nsColor: .controlBackgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(Color.secondary.opacity(0.15), lineWidth: 0.5))
    }

    private func installButton(
        _ entry: McpMarketplaceEntry, marketplaceName: String
    ) -> some View {
        Group {
            if installedNames.contains(entry.id) {
                Label("Added", systemImage: "checkmark.circle.fill")
                    .font(.caption)
                    .foregroundStyle(.green)
            } else {
                Button("Install") {
                    install(entry, marketplaceName: marketplaceName)
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private func errorRow(_ message: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Image(systemName: "exclamationmark.triangle.fill")
                .font(.caption)
                .foregroundStyle(.orange)
            Text(message)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(8)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(Color.orange.opacity(0.1))
        .clipShape(RoundedRectangle(cornerRadius: 6))
    }

    // MARK: - Actions

    /// The source the form describes. Built from the segment the user picked,
    /// never inferred from the text.
    private var resolvedSource: MarketplaceSource {
        let trimmed = sourceText.trimmingCharacters(in: .whitespacesAndNewlines)
        let ref = gitRef.trimmingCharacters(in: .whitespaces)
        let sparse = sparsePathsText.components(separatedBy: .newlines)
            .map { $0.trimmingCharacters(in: .whitespaces) }
            .filter { !$0.isEmpty }
        switch sourceKind {
        case .github:
            return .github(
                repo: trimmed,
                ref: ref.isEmpty ? nil : ref,
                path: nil,
                sparsePaths: sparse.isEmpty ? nil : sparse)
        case .git:
            return .git(
                url: trimmed,
                ref: ref.isEmpty ? nil : ref,
                path: nil,
                sparsePaths: sparse.isEmpty ? nil : sparse)
        case .directory:
            return .directory(path: trimmed)
        }
    }

    private func fetchCatalog() async {
        isFetching = true
        fetchError = nil
        manifest = nil
        entryErrors = [:]
        installedNames = []

        let source = resolvedSource
        do {
            let fetched = try await McpMarketplaceManager.shared.fetchMarketplace(source: source)
            manifest = fetched
            McpMarketplaceManager.shared.register(name: fetched.name, source: source)
        } catch {
            fetchError = error.localizedDescription
        }
        isFetching = false
    }

    private func install(_ entry: McpMarketplaceEntry, marketplaceName: String) {
        entryErrors[entry.id] = nil
        let project = model.selectedProject
        let useProjectScope = installToProjectScope && project != nil

        // A project server collides with the GLOBAL names too: `executeMcpCall`
        // resolves over `global + project` and a global wins (state#61), so a
        // project entry taking a global name could never be dialled.
        let existingNames: [String] = useProjectScope
            ? (project?.mcpServers.map(\.name) ?? []) + model.globalMcpServers.map(\.name)
            : model.globalMcpServers.map(\.name)

        do {
            let config = try McpMarketplaceManager.shared.makeServerConfig(
                from: entry, marketplaceName: marketplaceName, existingNames: existingNames)
            if useProjectScope, let projectID = project?.id {
                model.addProjectMcpServer(projectID: projectID, config: config)
            } else {
                model.addGlobalMcpServer(config)
            }
            installedNames.insert(entry.id)
            model.showToast(
                "Added '\(config.name)'. It is disabled until you enable it.", style: .success)
        } catch {
            entryErrors[entry.id] = error.localizedDescription
        }
    }
}
