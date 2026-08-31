import AppKit
import SwiftUI

/// Settings pane for discovering, creating, editing, and managing User- and Project-scoped skills.
public struct SkillsSettingsPaneView: View {
    @ObservedObject var model: AppModel

    public enum SkillScopeFilter: String, CaseIterable, Identifiable {
        case all = "All Skills"
        case user = "User Scope"
        case project = "Project Scope"

        public var id: String { rawValue }
    }

    @State private var scopeFilter: SkillScopeFilter = .all
    @State private var searchQuery: String = ""
    @State private var selectedSkillID: UUID? = nil
    @State private var isCreatingSkill: Bool = false
    @State private var isEditingSkill: Bool = false
    @State private var isImportingSkill: Bool = false
    @State private var skillToDelete: AppSkill? = nil
    @State private var isConfirmingDelete: Bool = false

    public init(model: AppModel) {
        self.model = model
    }

    private var filteredSkills: [AppSkill] {
        let list: [AppSkill]
        switch scopeFilter {
        case .all:
            list = model.allManagedSkills
        case .user:
            list = model.userSkills
        case .project:
            list = model.projectSkills
        }

        if searchQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return list
        }
        let q = searchQuery.lowercased()
        return list.filter {
            $0.name.lowercased().contains(q) ||
            $0.skillDescription.lowercased().contains(q) ||
            $0.agentOrigin.displayName.lowercased().contains(q)
        }
    }

    private var selectedSkill: AppSkill? {
        if let id = selectedSkillID {
            return model.allManagedSkills.first { $0.id == id }
        }
        return filteredSkills.first
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header bar
            headerControlBar
                .padding(.horizontal, 20)
                .padding(.vertical, 12)
                .background(Color(nsColor: .windowBackgroundColor))

            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 1)

            if model.allManagedSkills.isEmpty {
                emptyStateView
            } else {
                HStack(spacing: 0) {
                    // Left list
                    skillsListView
                        .frame(width: 300)
                        .background(Color(nsColor: .controlBackgroundColor).opacity(0.5))

                    Rectangle()
                        .fill(TurboSparkTheme.hairlineColor)
                        .frame(width: 1)

                    // Right detail
                    if let current = selectedSkill {
                        skillDetailView(skill: current)
                            .frame(maxWidth: .infinity, maxHeight: .infinity)
                    } else {
                        VStack(spacing: 12) {
                            Image(systemName: "wand.and.stars")
                                .font(.system(size: 36))
                                .foregroundStyle(.tertiary)
                            Text("Select a skill to inspect instructions and parameters.")
                                .font(.callout)
                                .foregroundStyle(.secondary)
                        }
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                    }
                }
            }
        }
        .sheet(isPresented: $isCreatingSkill) {
            SkillEditorSheet(model: model, skillToEdit: nil, defaultScope: scopeFilter == .project ? .projectLocal(projectPath: model.selectedProject?.rootDirectoryPath ?? "") : .userGlobal)
        }
        .sheet(isPresented: $isEditingSkill) {
            if let skill = selectedSkill {
                SkillEditorSheet(model: model, skillToEdit: skill, defaultScope: skill.scope)
            }
        }
        .sheet(isPresented: $isImportingSkill) {
            SkillImportSheet(model: model)
        }
        .confirmationDialog(
            "Delete Skill",
            isPresented: $isConfirmingDelete,
            presenting: skillToDelete
        ) { skill in
            Button("Delete '\(skill.name)'", role: .destructive) {
                model.removeSkill(skill)
                if selectedSkillID == skill.id {
                    selectedSkillID = nil
                }
            }
            Button("Cancel", role: .cancel) {}
        } message: { skill in
            Text("Are you sure you want to delete '\(skill.name)' from disk? This action cannot be undone.")
        }
    }

    // MARK: - Header Bar
    private var headerControlBar: some View {
        HStack(spacing: 12) {
            Picker("Scope", selection: $scopeFilter) {
                ForEach(SkillScopeFilter.allCases) { filter in
                    Text(filter.rawValue).tag(filter)
                }
            }
            .pickerStyle(.segmented)
            .frame(width: 280)

            HStack(spacing: 6) {
                Image(systemName: "magnifyingglass")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                TextField("Search skills...", text: $searchQuery)
                    .textFieldStyle(.plain)
                    .font(.callout)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 4)
            .background(Color(nsColor: .controlBackgroundColor))
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .overlay(
                RoundedRectangle(cornerRadius: 6)
                    .stroke(TurboSparkTheme.hairlineColor, lineWidth: 1)
            )

            Spacer()

            Button {
                model.reloadSkills()
            } label: {
                Image(systemName: "arrow.clockwise")
            }
            .buttonStyle(.bordered)
            .help("Refresh skills from disk")

            Button {
                isImportingSkill = true
            } label: {
                Label("Import...", systemImage: "square.and.arrow.down")
            }
            .buttonStyle(.bordered)
            .help("Import skills from Claude, Cursor, Antigravity, or OpenCode")

            Button {
                isCreatingSkill = true
            } label: {
                Label("New Skill", systemImage: "plus")
            }
            .buttonStyle(.borderedProminent)
            .tint(TurboSparkTheme.accentColor)
        }
    }

    // MARK: - Skills List View
    private var skillsListView: some View {
        List(selection: $selectedSkillID) {
            ForEach(filteredSkills) { skill in
                skillRowView(skill: skill)
                    .tag(skill.id)
                    .padding(.vertical, 4)
            }
        }
        .listStyle(.sidebar)
    }

    private func skillRowView(skill: AppSkill) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Image(systemName: "wand.and.stars")
                    .foregroundStyle(skill.isEnabled ? TurboSparkTheme.accentColor : Color.secondary)
                    .font(.subheadline)

                Text(skill.name)
                    .font(.headline)
                    .foregroundStyle(skill.isEnabled ? Color.primary : Color.secondary)

                Spacer()

                // Scope badge
                scopeBadgeView(scope: skill.scope)
            }

            Text(skill.skillDescription)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(2)

            HStack(spacing: 8) {
                Label(skill.agentOrigin.displayName, systemImage: "cpu")
                    .font(.caption2)
                    .foregroundStyle(.tertiary)

                if !skill.manifest.allowedTools.isEmpty {
                    Text("\(skill.manifest.allowedTools.count) tools")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }

                if !skill.referenceFiles.isEmpty {
                    Text("\(skill.referenceFiles.count) files")
                        .font(.caption2)
                        .foregroundStyle(.tertiary)
                }
            }
            .padding(.top, 2)
        }
        .contentShape(Rectangle())
    }

    private func scopeBadgeView(scope: SkillScope) -> some View {
        HStack(spacing: 3) {
            Image(systemName: scope.badgeIcon)
                .font(.system(size: 8))
            Text(scope.label)
                .font(.system(size: 9, weight: .semibold))
        }
        .padding(.horizontal, 6)
        .padding(.vertical, 2)
        .background(
            scope.isProjectScope
                ? Color.orange.opacity(0.15)
                : Color.blue.opacity(0.15)
        )
        .foregroundStyle(
            scope.isProjectScope ? Color.orange : Color.blue
        )
        .clipShape(Capsule())
    }

    // MARK: - Skill Detail View
    private func skillDetailView(skill: AppSkill) -> some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                // Header card
                VStack(alignment: .leading, spacing: 8) {
                    HStack(alignment: .top) {
                        VStack(alignment: .leading, spacing: 4) {
                            HStack(spacing: 8) {
                                Text(skill.name)
                                    .font(.title2.weight(.bold))
                                scopeBadgeView(scope: skill.scope)
                            }
                            Text(skill.skillDescription)
                                .font(.body)
                                .foregroundStyle(.secondary)
                        }

                        Spacer()

                        HStack(spacing: 8) {
                            Button {
                                isEditingSkill = true
                            } label: {
                                Label("Edit", systemImage: "pencil")
                            }
                            .buttonStyle(.bordered)

                            Button {
                                NSWorkspace.shared.activateFileViewerSelecting([skill.sourceURL])
                            } label: {
                                Label("Reveal", systemImage: "folder")
                            }
                            .buttonStyle(.bordered)

                            Button(role: .destructive) {
                                skillToDelete = skill
                                isConfirmingDelete = true
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.bordered)
                        }
                    }

                    Divider()
                        .padding(.vertical, 4)

                    // Metadata badges
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: 12)], spacing: 8) {
                        metaItemView(title: "Agent Origin", value: skill.agentOrigin.displayName, icon: "terminal")
                        metaItemView(title: "Execution Context", value: skill.manifest.context.rawValue.capitalized, icon: "arrow.triangle.branch")
                        metaItemView(title: "Shell Type", value: skill.manifest.shell.rawValue, icon: "apple.terminal")
                        metaItemView(title: "Source Path", value: skill.sourceURL.lastPathComponent, icon: "doc.text")
                    }
                }
                .padding(16)
                .background(Color(nsColor: .controlBackgroundColor))
                .clipShape(RoundedRectangle(cornerRadius: 8))
                .overlay(
                    RoundedRectangle(cornerRadius: 8)
                        .stroke(TurboSparkTheme.hairlineColor, lineWidth: 1)
                )

                // Allowed Tools Section
                if !skill.manifest.allowedTools.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Allowed Tools & Permissions")
                            .font(.headline)
                        FlowLayout(spacing: 6, lineSpacing: 6) {
                            ForEach(skill.manifest.allowedTools, id: \.self) { tool in
                                HStack(spacing: 4) {
                                    Image(systemName: "checkmark.shield")
                                        .font(.caption2)
                                    Text(tool)
                                        .font(.caption.monospaced())
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.green.opacity(0.12))
                                .foregroundStyle(.green)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }
                    }
                }

                // Path Triggers Section
                if !skill.manifest.paths.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Activation Path Triggers")
                            .font(.headline)
                        FlowLayout(spacing: 6, lineSpacing: 6) {
                            ForEach(skill.manifest.paths, id: \.self) { pathPattern in
                                HStack(spacing: 4) {
                                    Image(systemName: "arrow.triangle.turn.up.right.diamond")
                                        .font(.caption2)
                                    Text(pathPattern)
                                        .font(.caption.monospaced())
                                }
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .background(Color.purple.opacity(0.12))
                                .foregroundStyle(.purple)
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }
                    }
                }

                // Reference Files Section
                if !skill.referenceFiles.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("Directory Reference Files (\(skill.referenceFiles.count))")
                            .font(.headline)
                        VStack(spacing: 4) {
                            ForEach(skill.referenceFiles, id: \.self) { fileName in
                                HStack(spacing: 6) {
                                    Image(systemName: "doc")
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                    Text(fileName)
                                        .font(.caption.monospaced())
                                    Spacer()
                                }
                                .padding(.horizontal, 10)
                                .padding(.vertical, 6)
                                .background(Color(nsColor: .controlBackgroundColor).opacity(0.6))
                                .clipShape(RoundedRectangle(cornerRadius: 4))
                            }
                        }
                    }
                }

                // Instructions Markdown Content
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Instruction Body (SKILL.md)")
                            .font(.headline)
                        Spacer()
                        Button {
                            NSPasteboard.general.clearContents()
                            NSPasteboard.general.setString(skill.content, forType: .string)
                        } label: {
                            Label("Copy", systemImage: "doc.on.doc")
                                .font(.caption)
                        }
                        .buttonStyle(.borderless)
                    }

                    Text(skill.content)
                        .font(.system(.body, design: .monospaced))
                        .padding(12)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .background(Color(nsColor: .textBackgroundColor))
                        .clipShape(RoundedRectangle(cornerRadius: 6))
                        .overlay(
                            RoundedRectangle(cornerRadius: 6)
                                .stroke(TurboSparkTheme.hairlineColor, lineWidth: 1)
                        )
                }
            }
            .padding(20)
        }
    }

    private func metaItemView(title: String, value: String, icon: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .font(.caption)
                .foregroundStyle(.secondary)
                .frame(width: 16)
            VStack(alignment: .leading, spacing: 1) {
                Text(title)
                    .font(.system(size: 10))
                    .foregroundStyle(.tertiary)
                Text(value)
                    .font(.caption.weight(.medium))
                    .lineLimit(1)
            }
        }
    }

    // MARK: - Empty State
    private var emptyStateView: some View {
        VStack(spacing: 16) {
            Image(systemName: "wand.and.stars")
                .font(.system(size: 48))
                .foregroundStyle(.secondary)

            Text("No Skills Installed")
                .font(.title2.weight(.bold))

            Text("Skills allow you to package and inject specialized prompts, workflow instructions, reference scripts, and tool permissions into conversations.")
                .font(.callout)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)

            HStack(spacing: 12) {
                Button {
                    isCreatingSkill = true
                } label: {
                    Label("Create First Skill", systemImage: "plus")
                }
                .buttonStyle(.borderedProminent)
                .tint(TurboSparkTheme.accentColor)

                Button {
                    isImportingSkill = true
                } label: {
                    Label("Import from Agents...", systemImage: "square.and.arrow.down")
                }
                .buttonStyle(.bordered)
            }
            .padding(.top, 8)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .padding(40)
    }
}
