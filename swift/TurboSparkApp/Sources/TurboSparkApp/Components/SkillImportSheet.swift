import AppKit
import SwiftUI

/// Modal sheet for installing skills from remote Git/HTTPS marketplaces.
///
/// Local cross-agent import used to live here; it moved to
/// `AgentContentImportSheet`, the unified wizard that covers skills,
/// agents, and MCP servers from other agent tools.
public struct SkillImportSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    @State private var showsSources = false
    @State private var importToProjectScope: Bool = false

    // Remote marketplace state
    @State private var remoteInput: String = "whit3rabbit/agent-skills"
    @State private var isFetchingRemote: Bool = false
    @State private var remoteManifest: MarketplaceManifest? = nil
    @State private var remoteError: String? = nil
    @State private var installingSkillNames: Set<String> = []

    private let capturedProjectID: UUID?
    public init(model: AppModel, projectID: UUID? = nil) {
        self.model = model
        self.capturedProjectID = projectID ?? model.selectedProject?.id
        self._importToProjectScope = State(initialValue: projectID != nil)
    }
    private var project: AppProject? { model.projects.first { $0.id == capturedProjectID } }

    public var body: some View {
        VStack(spacing: 0) {
            headerBar

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            Button { showsSources = true } label: { Text("Manage sources", bundle: .module) }
            scopeBar

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            remoteMarketplaceContent
        }
        .frame(minWidth: 640, minHeight: 480)
        .sheet(isPresented: $showsSources) {
            MarketplaceSourcesView(model: model, kind: .skills, projectID: importToProjectScope ? capturedProjectID : nil) { source in
                fetchRemoteMarketplace(source: source)
            }
        }
    }

    // MARK: - Header & Scope Bars

    private var headerBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Skill Marketplace", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                Text("Install skills from remote Git/HTTPS marketplaces. Skills from other agent tools on this Mac are imported through Settings > General > Import from Other Agents.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Button {
                dismiss()
            } label: { Text("Close", bundle: .module) }
                .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(.appPage)
    }

    private var scopeBar: some View {
        HStack {
            Spacer()

            Text("Target Scope:", bundle: .module)
                .themedFont(.small, weight: .semibold)
            Picker(selection: $importToProjectScope) {
                Text("User Scope (~/.turbospark/skills)", bundle: .module).tag(false)
                Text("Project Scope (.turbospark/skills)", bundle: .module).tag(true)
            } label: { Text("Scope", bundle: .module) }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)
            .disabled(project == nil && !importToProjectScope)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(.appSurface.opacity(0.4))
    }

    // MARK: - Remote Marketplace Content

    private var remoteMarketplaceContent: some View {
        VStack(spacing: 0) {
            // Search / fetch bar
            HStack(spacing: 10) {
                TextField("GitHub repo (owner/name), Git URL, or HTTPS marketplace.json", text: $remoteInput)
                    .textFieldStyle(.roundedBorder)
                    .themedFont(.base)

                Button(action: { fetchRemoteMarketplace() }) {
                    if isFetchingRemote {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Fetch", bundle: .module)
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(TurboSparkTheme.accentColor)
                .disabled(remoteInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isFetchingRemote)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(.appSurface.opacity(0.2))

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            if let error = remoteError {
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .themedFont(.title2)
                        .foregroundStyle(.red)
                    Text(error)
                        .themedFont(.base)
                        .foregroundStyle(.appSecondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(20)
            } else if let manifest = remoteManifest {
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(manifest.name)
                                .themedFont(.base, weight: .semibold)
                            if let desc = manifest.description {
                                Text(desc)
                                    .themedFont(.small)
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                        Spacer()
                        Text(verbatim: "\(manifest.skills.count) skills available")
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(.appPage)

                    Rectangle()
                        .fill(.appBorder)
                        .frame(height: 1)

                    List(manifest.skills) { entry in
                        HStack(spacing: 12) {
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text(entry.name)
                                        .themedFont(.small, weight: .semibold)
                                    if let v = entry.version {
                                        Text(v)
                                            .themedFont(.tiny)
                                            .foregroundStyle(.appSecondary)
                                    }
                                    if let cat = entry.category {
                                        Text(cat)
                                            .themedFont(.tiny)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 1)
                                            .background(Color.blue.opacity(0.12))
                                            .clipShape(Capsule())
                                    }
                                }
                                Text(entry.description)
                                    .themedFont(.small)
                                    .foregroundStyle(.appSecondary)
                            }

                            Spacer()

                            let isInstalling = installingSkillNames.contains(entry.name)
                            Button(action: { installRemoteEntry(entry) }) {
                                if isInstalling {
                                    ProgressView().controlSize(.small)
                                } else {
                                    Text("Install", bundle: .module)
                                }
                            }
                            .buttonStyle(.bordered)
                            .disabled(isInstalling)
                        }
                        .padding(.vertical, 4)
                    }
                }
            } else {
                VStack(spacing: 12) {
                    Image(systemName: "globe.badge.chevron.backward")
                        .themedFont(.display)
                        .foregroundStyle(.appSecondary)
                    Text("Enter a GitHub repository (e.g. 'whit3rabbit/agent-skills') or direct HTTPS URL to browse and install remote skills.", bundle: .module)
                        .themedFont(.base)
                        .foregroundStyle(.appSecondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 440)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(30)
            }
        }
    }

    // MARK: - Actions

    private func fetchRemoteMarketplace(source explicitSource: MarketplaceSource? = nil) {
        let input = remoteInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard explicitSource != nil || !input.isEmpty else { return }

        isFetchingRemote = true
        remoteError = nil
        remoteManifest = nil

        let source: MarketplaceSource
        if let explicitSource { source = explicitSource }
        else if input.hasPrefix("http://") || input.hasPrefix("https://") {
            source = .url(url: input, headers: nil)
        } else if input.contains("/") && !input.contains(":") {
            source = .github(repo: input, ref: "main", path: "marketplace.json", sparsePaths: nil)
        } else {
            source = .git(url: input, ref: nil, path: nil, sparsePaths: nil)
        }

        let sourceProjectID = importToProjectScope ? capturedProjectID : nil
        Task {
            do {
                let manifest = try await SkillMarketplaceManager.shared.fetchMarketplace(source: source)
                try model.saveMarketplace(name: manifest.name, source: source, kind: .skills, projectID: sourceProjectID)
                await MainActor.run {
                    self.remoteManifest = manifest
                    self.isFetchingRemote = false
                }
            } catch {
                await MainActor.run {
                    self.remoteError = "Failed to load marketplace: \(error.localizedDescription)"
                    self.isFetchingRemote = false
                }
            }
        }
    }

    private func installRemoteEntry(_ entry: MarketplaceSkillEntry) {
        guard !importToProjectScope || project?.rootDirectoryURL != nil else {
            model.showToast(String(localized: "The target project no longer exists.", bundle: .module), style: .error)
            return
        }
        let targetScope: SkillScope
        let projectURL = importToProjectScope ? project?.rootDirectoryURL : nil
        if importToProjectScope, let path = project?.rootDirectoryPath {
            targetScope = .projectLocal(projectPath: path)
        } else {
            targetScope = .userGlobal
        }

        installingSkillNames.insert(entry.name)
        Task {
            do {
                let skill = try await SkillMarketplaceManager.shared.installSkill(
                    entry: entry,
                    targetScope: targetScope,
                    projectRootURL: projectURL
                )
                await MainActor.run {
                    self.installingSkillNames.remove(entry.name)
                    self.model.reloadSkills()
                    self.model.showToast("Installed '\(skill.name)' to \(targetScope.label).", style: .info)
                }
            } catch {
                await MainActor.run {
                    self.installingSkillNames.remove(entry.name)
                    self.model.showToast("Installation failed: \(error.localizedDescription)", style: .error)
                }
            }
        }
    }
}
