import AppKit
import SwiftUI

/// Modal sheet for discovering skills from other agent harnesses and importing them into TurboSpark.
public struct SkillImportSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    struct ImportableSkillCandidate: Identifiable {
        let id = UUID()
        let skill: AppSkill
        let agent: SkillSourceAgent
        let sourceLocationDescription: String
        var isSelected: Bool = false
    }

    @State private var candidates: [ImportableSkillCandidate] = []
    @State private var isLoading: Bool = true
    @State private var importToProjectScope: Bool = false

    public init(model: AppModel) {
        self.model = model
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Import Skills from Other Agents")
                        .font(.headline)
                    Text("Discover and import skills from Claude Code, Cursor, Antigravity, OpenCode, Pi, and other tools.")
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
            .padding(.vertical, 14)
            .background(Color(nsColor: .windowBackgroundColor))

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            // Content
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
                    // Scope picker
                    HStack {
                        Text("Import Destination:")
                            .font(.caption.weight(.semibold))
                        Picker("Destination", selection: $importToProjectScope) {
                            Text("User Scope (~/.turbospark/skills)").tag(false)
                            Text("Project Scope (.turbospark/skills)").tag(true)
                        }
                        .pickerStyle(.segmented)
                        .frame(maxWidth: 360)
                        .disabled(model.selectedProject == nil && !importToProjectScope)

                        Spacer()

                        Button("Select Folder...") {
                            selectCustomFolder()
                        }
                        .buttonStyle(.bordered)
                        .font(.caption)
                    }
                    .padding(.horizontal, 20)
                    .padding(.vertical, 10)
                    .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

                    Divider()

                    // Candidate List
                    List {
                        ForEach($candidates) { $candidate in
                            HStack(spacing: 12) {
                                Toggle("", isOn: $candidate.isSelected)
                                    .labelsHidden()

                                VStack(alignment: .leading, spacing: 2) {
                                    HStack(spacing: 6) {
                                        Text(candidate.skill.name)
                                            .font(.headline)
                                        Text(candidate.agent.displayName)
                                            .font(.caption2.weight(.semibold))
                                            .padding(.horizontal, 6)
                                            .padding(.vertical, 2)
                                            .background(Color.secondary.opacity(0.15))
                                            .clipShape(Capsule())
                                    }

                                    Text(candidate.skill.skillDescription)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)

                                    Text(candidate.sourceLocationDescription)
                                        .font(.system(size: 10))
                                        .foregroundStyle(.tertiary)
                                }

                                Spacer()
                            }
                            .padding(.vertical, 4)
                        }
                    }
                }
            }

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            // Footer
            HStack {
                let selectedCount = candidates.filter { $0.isSelected }.count
                Text("\(selectedCount) skill(s) selected")
                    .font(.caption)
                    .foregroundStyle(.secondary)

                Spacer()

                Button("Import Selected (\(selectedCount))") {
                    importSelectedSkills()
                }
                .buttonStyle(.borderedProminent)
                .tint(TurboSparkTheme.accentColor)
                .disabled(selectedCount == 0)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(Color(nsColor: .windowBackgroundColor))
        }
        .frame(minWidth: 580, minHeight: 440)
        .onAppear {
            scanCandidates()
        }
    }

    private func scanCandidates() {
        isLoading = true
        let projectURL = model.selectedProject?.rootDirectoryURL
        DispatchQueue.global(qos: .userInitiated).async {
            let manager = SkillManager.shared
            var found: [ImportableSkillCandidate] = []
            let home = FileManager.default.homeDirectoryForCurrentUser

            // Scan user agent dirs (skip turbospark itself)
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

            // Scan active project dirs if open
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
}
