import SwiftUI

/// Modal sheet for creating a new skill or editing an existing skill definition.
public struct SkillEditorSheet: View {
    @ObservedObject var model: AppModel
    @Environment(\.dismiss) private var dismiss

    var skillToEdit: AppSkill?
    var defaultScope: SkillScope
    private let capturedProjectPath: String?

    @State private var name: String = ""
    @State private var descriptionText: String = ""
    @State private var allowedToolsText: String = ""
    @State private var pathsText: String = ""
    @State private var content: String = ""
    @State private var isProjectScope: Bool = false
    @State private var context: SkillExecutionContext = .inline
    @State private var shell: SkillShellType = .bash

    public init(model: AppModel, skillToEdit: AppSkill? = nil, defaultScope: SkillScope = .userGlobal) {
        self.model = model
        self.skillToEdit = skillToEdit
        self.defaultScope = defaultScope
        self.capturedProjectPath = defaultScope.projectRootURL?.path ?? model.selectedProject?.rootDirectoryPath
    }

    public var body: some View {
        VStack(spacing: 0) {
            // Header
            HStack {
                Text(skillToEdit == nil ? "Create New Skill" : "Edit Skill: \(name)")
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Button {
                    dismiss()
                } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 14)
            .background(.appPage)

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            // Form Content
            ScrollView {
                VStack(alignment: .leading, spacing: 16) {
                    // Name & Scope
                    HStack(spacing: 16) {
                        VStack(alignment: .leading, spacing: 4) {
                            Text("Skill Name", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            TextField("e.g. git-workflow, jupyter-skill", text: $name)
                                .textFieldStyle(.roundedBorder)
                        }

                        VStack(alignment: .leading, spacing: 4) {
                            Text("Scope", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            Picker(selection: $isProjectScope) {
                                Text("User Scope (~/.turbospark/skills)", bundle: .module).tag(false)
                                Text("Project Scope (.turbospark/skills)", bundle: .module).tag(true)
                            } label: { Text("Scope", bundle: .module) }
                            .pickerStyle(.segmented)
                            .disabled(capturedProjectPath == nil && !isProjectScope)
                        }
                    }

                    // Description
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Description", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                        TextField("Brief summary of what this skill does and when to activate it...", text: $descriptionText)
                            .textFieldStyle(.roundedBorder)
                    }

                    // Allowed Tools
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Allowed Tools (Comma-separated)", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                        TextField("e.g. Bash(git:*), Read, write_file, search_code", text: $allowedToolsText)
                            .textFieldStyle(.roundedBorder)
                        Text("Tool permissions allowed during execution of this skill.", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                    }

                    // Path Triggers
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Activation Path Triggers (Comma-separated)", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                        TextField("e.g. *.ipynb, src/**/*.ts, Cargo.toml", text: $pathsText)
                            .textFieldStyle(.roundedBorder)
                        Text("Glob patterns that will trigger activation when editing matching files.", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                    }

                    // Markdown Content
                    VStack(alignment: .leading, spacing: 4) {
                        HStack {
                            Text("Instruction Body (Markdown)", bundle: .module)
                                .themedFont(.small, weight: .semibold)
                            Spacer()
                            Text("Supports ${arg_name}, ${SKILL_DIR}, and ${SESSION_ID}", bundle: .module)
                                .themedFont(.tiny)
                                .foregroundStyle(.appSecondary)
                        }

                        TextEditor(text: $content)
                            .themedCode(.base)
                            .frame(minHeight: 200)
                            .padding(4)
                            .background(.appElevated)
                            .clipShape(RoundedRectangle(cornerRadius: 6))
                            .overlay(
                                RoundedRectangle(cornerRadius: 6)
                                    .stroke(.appBorder, lineWidth: 1)
                            )
                    }
                }
                .padding(20)
            }

            Rectangle()
                .fill(.appBorder)
                .frame(height: 1)

            // Footer
            HStack {
                Spacer()
                Button {
                    saveAction()
                } label: { Text("Save Skill", bundle: .module) }
                .buttonStyle(.borderedProminent)
                .tint(TurboSparkTheme.accentColor)
                .disabled(name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                .keyboardShortcut(.defaultAction)
            }
            .padding(.horizontal, 20)
            .padding(.vertical, 12)
            .background(.appPage)
        }
        .frame(minWidth: 560, minHeight: 480)
        .onAppear {
            if let skill = skillToEdit {
                name = skill.name
                descriptionText = skill.manifest.description ?? ""
                allowedToolsText = skill.manifest.allowedTools.joined(separator: ", ")
                pathsText = skill.manifest.paths.joined(separator: ", ")
                content = skill.content
                isProjectScope = skill.scope.isProjectScope
                context = skill.manifest.context
                shell = skill.manifest.shell
            } else {
                isProjectScope = defaultScope.isProjectScope
                content = "# Workflow Instructions\n\n1. Analyze the context...\n2. Execute necessary steps..."
            }
        }
    }

    private func saveAction() {
        let toolsList = allowedToolsText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let pathsList = pathsText
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        let targetScope: SkillScope
        if isProjectScope, let path = capturedProjectPath {
            targetScope = .projectLocal(projectPath: path)
        } else {
            targetScope = .userGlobal
        }

        if var skill = skillToEdit {
            skill.manifest.name = name
            skill.manifest.description = descriptionText.isEmpty ? nil : descriptionText
            skill.manifest.allowedTools = toolsList
            skill.manifest.paths = pathsList
            skill.manifest.context = context
            skill.manifest.shell = shell
            skill.content = content
            model.updateSkill(skill)
        } else {
            model.createNewSkill(
                name: name,
                description: descriptionText,
                content: content,
                allowedTools: toolsList,
                paths: pathsList,
                scope: targetScope
            )
        }

        dismiss()
    }
}
