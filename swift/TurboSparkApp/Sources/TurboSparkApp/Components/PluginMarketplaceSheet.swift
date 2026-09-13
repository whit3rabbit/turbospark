import SwiftUI

/// Marketplace browser for plugins: add and refresh marketplace sources,
/// browse their entries, install at user or project scope. Modeled on the
/// skills and MCP import sheets; the manifest format it reads is Claude
/// Code's `.claude-plugin/marketplace.json`.
@MainActor
struct PluginMarketplaceSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss
    @Environment(\.appTheme) private var theme

    @State private var showsSources = false
    @State var projectID: UUID? = nil
    @State private var marketplaces: [String: MarketplaceSource] = [:]
    @State private var selectedMarketplaceName: String? = nil
    @State private var entries: [PluginManifestParser.MarketplaceEntry] = []
    @State private var checkoutDirectory: URL? = nil
    @State private var isLoading = false
    @State private var loadError: String? = nil
    @State private var installedIDs: Set<String> = []

    // Add-marketplace form
    @State private var newMarketplaceName = ""
    @State private var newSourceKind = "github"
    @State private var newGitHubRepo = ""
    @State private var newGitURL = ""
    @State private var newDirectoryPath = ""

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                Text("Plugin Marketplaces", bundle: .module)
                    .font(theme.ui(.title3, weight: .semibold))
                Spacer()
                Button("Done") { dismiss() }
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)

            ExtensionScopePicker(model: model, projectID: $projectID)
            Button { showsSources = true } label: { Text("Manage sources", bundle: .module) }
            Text("Removing a source keeps installed items.", bundle: .module).themedFont(.small)
            Rectangle().fill(.appBorder).frame(height: 1)

            HStack(spacing: 0) {
                marketplaceList
                    .frame(width: 250)

                Rectangle().fill(.appBorder).frame(width: 1)

                entryList
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(minWidth: 760, minHeight: 520)
        .sheet(isPresented: $showsSources, onDismiss: { marketplaces = model.marketplaceSources(kind: .plugins, projectID: projectID) }) {
            MarketplaceSourcesView(model: model, kind: .plugins, projectID: projectID) { source in
                if let match = model.marketplaceSources(kind: .plugins, projectID: projectID).first(where: { $0.value == source }) {
                    marketplaces = model.marketplaceSources(kind: .plugins, projectID: projectID)
                    selectedMarketplaceName = match.key
                    Task { await loadSelected() }
                }
            }
        }
        .onChange(of: projectID) { _, _ in
            refreshInstalledIDs()
            marketplaces = model.marketplaceSources(kind: .plugins, projectID: projectID)
            selectedMarketplaceName = nil; entries = []; checkoutDirectory = nil
        }
        .task {
            marketplaces = model.marketplaceSources(kind: .plugins, projectID: projectID)
            refreshInstalledIDs()
            if selectedMarketplaceName == nil {
                selectedMarketplaceName = marketplaces.keys.sorted().first
                await loadSelected()
            }
        }
    }

    private var scopedProject: AppProject? { model.projects.first { $0.id == projectID } }
    private var installScope: PluginInstallScope { scopedProject.map(PluginInstallScope.project) ?? .user }

    // MARK: - Marketplace list

    private var marketplaceList: some View {
        VStack(alignment: .leading, spacing: 0) {
            Text("Sources", bundle: .module)
                .font(theme.ui(.small, weight: .semibold))
                .foregroundStyle(.appSecondary)
                .padding([.horizontal, .top], 14)

            ForEach(marketplaces.keys.sorted(), id: \.self) { name in
                Button {
                    selectedMarketplaceName = name
                    Task { await loadSelected() }
                } label: {
                    HStack {
                        Text(name)
                            .font(theme.ui(.base))
                            .foregroundStyle(
                                name == selectedMarketplaceName ? Color.accentColor : Color.primary)
                        Spacer()
                        if isLoading && name == selectedMarketplaceName {
                            ProgressView().controlSize(.small)
                        }
                    }
                    .padding(.horizontal, 14)
                    .padding(.vertical, 8)
                    .contentShape(Rectangle())
                }
                .buttonStyle(.plain)
            }

            Spacer()
            Divider()
            addMarketplaceForm
                .padding(14)
        }
    }

    private var addMarketplaceForm: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Add marketplace", bundle: .module)
                .font(theme.ui(.small, weight: .semibold))
            TextField("Name (e.g. my-plugins)", text: $newMarketplaceName)
                .textFieldStyle(.roundedBorder)
            Picker("", selection: $newSourceKind) {
                Text("GitHub", bundle: .module).tag("github")
                Text("Git URL", bundle: .module).tag("git")
                Text("Directory", bundle: .module).tag("directory")
            }
            .labelsHidden()
            .pickerStyle(.segmented)
            switch newSourceKind {
            case "github":
                TextField("owner/repo", text: $newGitHubRepo)
                    .textFieldStyle(.roundedBorder)
            case "git":
                TextField("https://... .git", text: $newGitURL)
                    .textFieldStyle(.roundedBorder)
            default:
                TextField("/path/to/marketplace", text: $newDirectoryPath)
                    .textFieldStyle(.roundedBorder)
            }
            Button("Add") { addMarketplace() }
                .disabled(newMarketplaceName.trimmingCharacters(in: .whitespaces).isEmpty)
        }
    }

    private func addMarketplace() {
        let name = newMarketplaceName.trimmingCharacters(in: .whitespaces)
        let source: MarketplaceSource
        switch newSourceKind {
        case "github":
            let repo = newGitHubRepo.trimmingCharacters(in: .whitespaces)
            guard repo.contains("/") else {
                loadError = "A GitHub source is owner/repo, e.g. anthropics/claude-plugins-official."
                return
            }
            source = .github(repo: repo, ref: "main", path: nil, sparsePaths: nil)
        case "git":
            let url = newGitURL.trimmingCharacters(in: .whitespaces)
            guard url.hasPrefix("https://") || url.hasPrefix("http://") || url.hasPrefix("ssh://") else {
                loadError = "A git source needs an https:// or ssh:// URL."
                return
            }
            source = .git(url: url, ref: nil, path: nil, sparsePaths: nil)
        default:
            let path = newDirectoryPath.trimmingCharacters(in: .whitespaces)
            guard FileManager.default.fileExists(atPath: path) else {
                loadError = "Directory not found: \(path)"
                return
            }
            source = .directory(path: path)
        }
        do {
            try model.saveMarketplace(name: name, source: source, kind: .plugins, projectID: projectID)
            marketplaces = model.marketplaceSources(kind: .plugins, projectID: projectID)
            newMarketplaceName = ""
            newGitHubRepo = ""
            newGitURL = ""
            newDirectoryPath = ""
            selectedMarketplaceName = name
            Task { await loadSelected() }
        } catch {
            loadError = error.localizedDescription
        }
    }

    // MARK: - Entries

    private var entryList: some View {
        VStack(spacing: 0) {
            HStack {
                Text(selectedMarketplaceName ?? "Select a marketplace")
                    .font(theme.ui(.base, weight: .semibold))
                Spacer()
                if let name = selectedMarketplaceName {
                    Button("Refresh") {
                        Task { await loadSelected(force: true) }
                    }
                    Button("Remove", role: .destructive) {
                        do { try model.removeMarketplace(name: name, kind: .plugins, projectID: projectID) }
                        catch {
                            model.showToast(error.localizedDescription, style: .error)
                            return
                        }
                        marketplaces = model.marketplaceSources(kind: .plugins, projectID: projectID)
                        selectedMarketplaceName = marketplaces.keys.sorted().first
                        entries = []
                        Task { await loadSelected() }
                    }
                }
            }
            .padding(.horizontal, 16)
            .padding(.vertical, 10)

            Rectangle().fill(.appBorder).frame(height: 1)

            if let loadError {
                Label(loadError, systemImage: "exclamationmark.triangle")
                    .foregroundStyle(.orange)
                    .font(theme.ui(.small))
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(12)
            }

            ScrollView {
                VStack(spacing: 8) {
                    ForEach(entries, id: \.name) { entry in
                        entryRow(entry)
                    }
                    if entries.isEmpty && !isLoading {
                        Text("No plugin entries loaded.", bundle: .module)
                            .themedFont(.base)
                            .foregroundStyle(.tertiary)
                            .padding(.top, 30)
                    }
                }
                .padding(14)
            }
        }
    }

    private func entryRow(_ entry: PluginManifestParser.MarketplaceEntry) -> some View {
        let installed = installedIDs.contains("\(entry.name)@\(selectedMarketplaceName ?? "")")
        return HStack(alignment: .top, spacing: 12) {
            VStack(alignment: .leading, spacing: 3) {
                HStack(spacing: 6) {
                    Text(entry.name)
                        .font(theme.ui(.base, weight: .medium))
                    if let version = entry.version {
                        Text(version)
                            .font(theme.ui(.small))
                            .foregroundStyle(.appSecondary)
                    }
                    if !entry.strict {
                        Text("non-strict", bundle: .module)
                            .themedFont(.tiny, weight: .medium)
                            .padding(.horizontal, 5)
                            .padding(.vertical, 1)
                            .background(Capsule().fill(Color.orange.opacity(0.15)))
                            .foregroundStyle(.orange)
                    }
                }
                if let description = entry.descriptionText, !description.isEmpty {
                    Text(description)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                        .lineLimit(2)
                }
            }
            Spacer()
            if installed {
                Label("Installed", systemImage: "checkmark.circle.fill")
                    .foregroundStyle(.green)
                    .font(theme.ui(.small))
                Button(role: .destructive) {
                    uninstall(entry: entry, scope: installScope)
                } label: { Text("Uninstall", bundle: .module) }
                .help("Removes only the installation in the selected scope.")
            } else {
                Button { install(entry: entry, scope: installScope) } label: {
                    Text("Install", bundle: .module)
                }
            }
        }
        .padding(10)
        .background(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(.appSurface))
    }

    // MARK: - Actions

    private func loadSelected(force: Bool = false) async {
        guard let name = selectedMarketplaceName,
            let source = marketplaces[name]
        else { return }
        let targetProjectID = projectID
        isLoading = true
        loadError = nil
        defer { if targetProjectID == projectID && name == selectedMarketplaceName { isLoading = false } }
        do {
            // Refresh re-clones/pulls; a plain selection reuses the cache
            // when the source is already checked out.
            let checkout = try await PluginMarketplaceManager.shared.fetchMarketplace(
                name: name, source: source)
            guard targetProjectID == projectID && name == selectedMarketplaceName else { return }
            entries = checkout.manifest.entries
            checkoutDirectory = checkout.directory
        } catch {
            guard targetProjectID == projectID && name == selectedMarketplaceName else { return }
            loadError = error.localizedDescription
            entries = []
        }
    }

    private func refreshInstalledIDs() {
        let records = PluginLedgerStore(root: nil).load().plugins
        let scope = installScope
        installedIDs = Set(records.keys.filter { id in
            records[id]?.contains { $0.scope == scope.ledgerValue && $0.projectPath == scope.projectRootURL?.standardizedFileURL.path } == true
        })
    }

    private func install(entry: PluginManifestParser.MarketplaceEntry, scope: PluginInstallScope) {
        guard projectID == nil || scopedProject != nil else {
            model.showToast(String(localized: "The target project no longer exists.", bundle: .module), style: .error)
            return
        }
        guard let marketplaceName = selectedMarketplaceName else { return }
        let targetCheckout = checkoutDirectory
        guard let marketplaceSource = marketplaces[marketplaceName] else { return }
        isLoading = true
        Task {
            defer { isLoading = false }
            let outcome = await model.installPlugin(
                entry: entry,
                marketplaceName: marketplaceName,
                checkoutDirectory: targetCheckout,
                marketplaceSource: marketplaceSource,
                scope: scope)
            if outcome != nil {
                refreshInstalledIDs()
                model.reloadPlugins()
            }
        }
    }

    private func uninstall(entry: PluginManifestParser.MarketplaceEntry, scope: PluginInstallScope) {
        guard let name = selectedMarketplaceName else { return }
        model.uninstallPluginID("\(entry.name)@\(name)", scope: scope)
        refreshInstalledIDs()
    }
}
