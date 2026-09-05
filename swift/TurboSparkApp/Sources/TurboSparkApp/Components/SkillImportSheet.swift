import AppKit
import SwiftUI

/// Modal sheet for discovering skills from other agent harnesses and importing
/// or installing skills from remote Git/HTTPS marketplaces into TurboSpark.
public struct SkillImportSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    public enum ImportMode: String, CaseIterable, Identifiable {
        case localHarnesses = "Local Agents"
        case remoteMarketplace = "Remote / Marketplace"

        public var id: String { rawValue }
    }

    struct ImportableSkillCandidate: Identifiable {
        let id = UUID()
        let skill: AppSkill
        let agent: SkillSourceAgent
        let sourceLocationDescription: String
        var isSelected: Bool = false
    }

    @State private var mode: ImportMode = .localHarnesses
    @State private var candidates: [ImportableSkillCandidate] = []
    @State private var isLoading: Bool = true
    @State private var importToProjectScope: Bool = false

    // Remote marketplace state
    @State private var remoteInput: String = "whit3rabbit/agent-skills"
    @State private var isFetchingRemote: Bool = false
    @State private var remoteManifest: MarketplaceManifest? = nil
    @State private var remoteError: String? = nil
    @State private var installingSkillNames: Set<String> = []

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        VStack(spacing: 0) {
            headerBar

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            modeSelectorBar

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            switch mode {
            case .localHarnesses:
                localHarnessesContent
            case .remoteMarketplace:
                remoteMarketplaceContent
            }
        }
        .frame(minWidth: 640, minHeight: 480)
        .onAppear {
            scanCandidates()
        }
    }

    // MARK: - Header & Mode Bars

    private var headerBar: some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text("Skills Import & Marketplace")
                    .font(.headline)
                Text("Acquire skills from local agent harnesses (Claude, Cursor, Codex) or remote Git/HTTPS marketplaces.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Button("Close") {
                dismiss()
            }
            .keyboardShortcut(.cancelAction)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 12)
        .background(Color(nsColor: .windowBackgroundColor))
    }

    private var modeSelectorBar: some View {
        HStack {
            Picker("Mode", selection: $mode) {
                ForEach(ImportMode.allCases) { m in
                    Text(m.rawValue).tag(m)
                }
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)

            Spacer()

            Text("Target Scope:")
                .font(.caption.weight(.semibold))
            Picker("Scope", selection: $importToProjectScope) {
                Text("User Scope (~/.turbospark/skills)").tag(false)
                Text("Project Scope (.turbospark/skills)").tag(true)
            }
            .pickerStyle(.segmented)
            .frame(maxWidth: 320)
            .disabled(model.selectedProject == nil && !importToProjectScope)
        }
        .padding(.horizontal, 20)
        .padding(.vertical, 8)
        .background(Color(nsColor: .controlBackgroundColor).opacity(0.4))
    }

    // MARK: - Local Harness Content

    @ViewBuilder
    private var localHarnessesContent: some View {
        if isLoading {
            VStack(spacing: 12) {
                ProgressView()
                Text("Scanning agent skill directories...")
                    .font(.callout)
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
        } else if candidates.isEmpty {
            VStack(spacing: 16) {
                Image(systemName: "folder.badge.questionmark")
                    .font(.system(size: 36))
                    .foregroundStyle(.secondary)
                Text("No external skills found in standard agent locations.")
                    .font(.callout)
                    .foregroundStyle(.secondary)

                Button("Choose Custom Folder...") {
                    selectCustomFolder()
                }
                .buttonStyle(.bordered)
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .padding(40)
        } else {
            VStack(spacing: 0) {
                List {
                    ForEach($candidates) { $cand in
                        HStack(spacing: 12) {
                            Toggle("", isOn: $cand.isSelected)
                                .labelsHidden()
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text(cand.skill.name)
                                        .font(.subheadline.weight(.semibold))
                                    Text(cand.agent.displayName)
                                        .font(.caption2)
                                        .padding(.horizontal, 6)
                                        .padding(.vertical, 1)
                                        .background(Color.secondary.opacity(0.15))
                                        .clipShape(Capsule())
                                }
                                Text(cand.skill.skillDescription)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                Text(cand.sourceLocationDescription)
                                    .font(.caption2)
                                    .foregroundStyle(.tertiary)
                            }
                            Spacer()
                        }
                        .padding(.vertical, 2)
                    }
                }

                Rectangle()
                    .fill(TurboSparkTheme.hairlineColor)
                    .frame(height: 1)

                HStack {
                    Button("Select All") {
                        for i in candidates.indices { candidates[i].isSelected = true }
                    }
                    .buttonStyle(.plain)
                    .font(.caption)

                    Text("|").foregroundStyle(.tertiary)

                    Button("Deselect All") {
                        for i in candidates.indices { candidates[i].isSelected = false }
                    }
                    .buttonStyle(.plain)
                    .font(.caption)

                    Spacer()

                    let count = candidates.filter { $0.isSelected }.count
                    Button("Import Selected (\(count))") {
                        importSelectedSkills()
                    }
                    .buttonStyle(.borderedProminent)
                    .tint(TurboSparkTheme.accentColor)
                    .disabled(count == 0)
                }
                .padding(.horizontal, 20)
                .padding(.vertical, 10)
                .background(Color(nsColor: .windowBackgroundColor))
            }
        }
    }

    // MARK: - Remote Marketplace Content

    private var remoteMarketplaceContent: some View {
        VStack(spacing: 0) {
            // Search / fetch bar
            HStack(spacing: 10) {
                TextField("GitHub repo (owner/name), Git URL, or HTTPS marketplace.json", text: $remoteInput)
                    .textFieldStyle(.roundedBorder)
                    .font(.callout)

                Button(action: { fetchRemoteMarketplace() }) {
                    if isFetchingRemote {
                        ProgressView().controlSize(.small)
                    } else {
                        Text("Fetch")
                    }
                }
                .buttonStyle(.borderedProminent)
                .tint(TurboSparkTheme.accentColor)
                .disabled(remoteInput.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || isFetchingRemote)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color(nsColor: .controlBackgroundColor).opacity(0.2))

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            if let error = remoteError {
                VStack(spacing: 8) {
                    Image(systemName: "exclamationmark.triangle")
                        .font(.title2)
                        .foregroundStyle(.red)
                    Text(error)
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(20)
            } else if let manifest = remoteManifest {
                VStack(alignment: .leading, spacing: 0) {
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(manifest.name)
                                .font(.headline)
                            if let desc = manifest.description {
                                Text(desc)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                        Spacer()
                        Text("\(manifest.skills.count) skills available")
                            .font(.caption2)
                            .foregroundStyle(.tertiary)
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color(nsColor: .windowBackgroundColor))

                    Rectangle()
                        .fill(TurboSparkTheme.hairlineColor)
                        .frame(height: 1)

                    List(manifest.skills) { entry in
                        HStack(spacing: 12) {
                            VStack(alignment: .leading, spacing: 2) {
                                HStack {
                                    Text(entry.name)
                                        .font(.subheadline.weight(.semibold))
                                    if let v = entry.version {
                                        Text(v)
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    if let cat = entry.category {
                                        Text(cat)
                                            .font(.caption2)
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 1)
                                            .background(Color.blue.opacity(0.12))
                                            .clipShape(Capsule())
                                    }
                                }
                                Text(entry.description)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }

                            Spacer()

                            let isInstalling = installingSkillNames.contains(entry.name)
                            Button(action: { installRemoteEntry(entry) }) {
                                if isInstalling {
                                    ProgressView().controlSize(.small)
                                } else {
                                    Text("Install")
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
                        .font(.system(size: 40))
                        .foregroundStyle(.secondary)
                    Text("Enter a GitHub repository (e.g. 'whit3rabbit/agent-skills') or direct HTTPS URL to browse and install remote skills.")
                        .font(.callout)
                        .foregroundStyle(.secondary)
                        .multilineTextAlignment(.center)
                        .frame(maxWidth: 440)
                }
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .padding(30)
            }
        }
    }

    // MARK: - Actions

    private func scanCandidates() {
        isLoading = true
        let projectURL = model.selectedProject?.rootDirectoryURL
        DispatchQueue.global(qos: .userInitiated).async {
            let manager = SkillManager.shared
            var found: [ImportableSkillCandidate] = []
            let home = FileManager.default.homeDirectoryForCurrentUser

            for (agent, relPath) in manager.knownUserAgentSkillRoots where agent != .turboSpark {
                let url = home.appendingPathComponent(relPath, isDirectory: true)
                guard FileManager.default.fileExists(atPath: url.path) else { continue }
                let skills = manager.scanDirectory(url, scope: .userGlobal, defaultAgent: agent)
                for skill in skills {
                    found.append(ImportableSkillCandidate(
                        skill: skill,
                        agent: agent,
                        sourceLocationDescription: "~/\(relPath)/\(skill.sourceURL.lastPathComponent)",
                        isSelected: true
                    ))
                }
            }

            if let projectURL {
                for (agent, relPath) in manager.knownProjectSkillSubdirectories where agent != .turboSpark {
                    let url = projectURL.appendingPathComponent(relPath, isDirectory: true)
                    guard FileManager.default.fileExists(atPath: url.path) else { continue }
                    let skills = manager.scanDirectory(url, scope: .projectLocal(projectPath: projectURL.path), defaultAgent: agent)
                    for skill in skills {
                        found.append(ImportableSkillCandidate(
                            skill: skill,
                            agent: agent,
                            sourceLocationDescription: "<project>/\(relPath)/\(skill.sourceURL.lastPathComponent)",
                            isSelected: true
                        ))
                    }
                }
            }

            DispatchQueue.main.async {
                self.candidates = found
                self.isLoading = false
            }
        }
    }

    private func selectCustomFolder() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        panel.prompt = "Import Skill"

        if panel.runModal() == .OK, let selectedURL = panel.url {
            let targetScope: SkillScope
            if importToProjectScope, let path = model.selectedProject?.rootDirectoryPath {
                targetScope = .projectLocal(projectPath: path)
            } else {
                targetScope = .userGlobal
            }
            model.importSkill(from: selectedURL, targetScope: targetScope)
            dismiss()
        }
    }

    private func importSelectedSkills() {
        let targetScope: SkillScope
        if importToProjectScope, let path = model.selectedProject?.rootDirectoryPath {
            targetScope = .projectLocal(projectPath: path)
        } else {
            targetScope = .userGlobal
        }

        let selected = candidates.filter { $0.isSelected }
        for cand in selected {
            let source = cand.skill.isDirectoryBased ? (cand.skill.skillDirectoryURL ?? cand.skill.sourceURL) : cand.skill.sourceURL
            model.importSkill(from: source, targetScope: targetScope)
        }
        dismiss()
    }

    private func fetchRemoteMarketplace() {
        let input = remoteInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !input.isEmpty else { return }

        isFetchingRemote = true
        remoteError = nil
        remoteManifest = nil

        let source: MarketplaceSource
        if input.hasPrefix("http://") || input.hasPrefix("https://") {
            source = .url(url: input, headers: nil)
        } else if input.contains("/") && !input.contains(":") {
            source = .github(repo: input, ref: "main", path: "marketplace.json", sparsePaths: nil)
        } else {
            source = .git(url: input, ref: nil, path: nil, sparsePaths: nil)
        }

        Task {
            do {
                let manifest = try await SkillMarketplaceManager.shared.fetchMarketplace(source: source)
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
        let targetScope: SkillScope
        let projectURL = model.selectedProject?.rootDirectoryURL
        if importToProjectScope, let path = model.selectedProject?.rootDirectoryPath {
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
