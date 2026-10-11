import AppKit
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// The detail column of the Skills pane: identity, enable state, actions,
/// metadata, path triggers, reference files and the SKILL.md body.
///
/// A dedicated subview rather than an extension of `SkillsSettingsPaneView`
/// so the pane's state stays private (Gotcha 15); the pane passes the
/// selection in and receives Edit and Delete back as closures.
@MainActor
struct SkillDetailView: View {
    @ObservedObject var model: AppModel
    let skill: AppSkill
    let scopedProject: AppProject?
    let onEdit: () -> Void
    let onDelete: () -> Void

    private var isPluginSkill: Bool {
        if case .plugin = skill.scope { return true }
        return false
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 18) {
                headerCard

                // No "Allowed Tools & Permissions" section. `allowed-tools`
                // is enforced by nothing (see the row comment on state#48 in
                // the pane); this section rendered the same field with a
                // green shield, one scroll below the comment saying why not.

                if !skill.manifest.paths.isEmpty {
                    pathTriggers
                }
                if !skill.referenceFiles.isEmpty {
                    referenceFiles
                }
                instructions
            }
            .padding(SettingsLayout.detailInset)
        }
    }

    private var headerCard: some View {
        VStack(alignment: .leading, spacing: 8) {
            VStack(alignment: .leading, spacing: 12) {
                VStack(alignment: .leading, spacing: 4) {
                    HStack(spacing: 8) {
                        Text(skill.name)
                            .themedFont(.title2, weight: .bold)
                        SkillScopeBadge(scope: skill.scope)
                    }
                    Text(skill.skillDescription)
                        .themedFont(.base)
                        .foregroundStyle(.appSecondary)
                }
                controls
            }

            Divider()
                .padding(.vertical, 4)

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 160), spacing: 12)], spacing: 8) {
                metaItem(title: Text("Agent Origin", bundle: .module), value: skill.agentOrigin.displayName, icon: "terminal")
                metaItem(title: Text("Execution Context", bundle: .module), value: skill.manifest.context.rawValue.capitalized, icon: "arrow.triangle.branch")
                metaItem(title: Text("Shell Type", bundle: .module), value: skill.manifest.shell.rawValue, icon: "apple.terminal")
                metaItem(title: Text("Source Path", bundle: .module), value: skill.sourceURL.lastPathComponent, icon: "doc.text")
            }
        }
        .padding(16)
        .background(.appSurface)
        .clipShape(RoundedRectangle(cornerRadius: 8))
        .overlay(
            RoundedRectangle(cornerRadius: 8)
                .stroke(.appBorder, lineWidth: 1)
        )
    }

    private var controls: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let project = scopedProject {
                ExtensionOverridePicker(enabled: Binding(
                    get: { project.enabledSkills[skill.name.lowercased()] },
                    set: { model.setSkillOverride(name: skill.name, enabled: $0, projectID: project.id) }))
            }
            // The list greys a disabled skill and this pane had no way to
            // re-enable it; the only toggle was in the composer's plus menu.
            if scopedProject == nil && !isPluginSkill {
                Toggle(isOn: Binding(
                    get: { skill.isEnabled },
                    set: { _ in model.toggleSkillEnabled(skill) }
                )) {
                    Text("Enabled", bundle: .module)
                }
                .settingsControl("Enabled", pane: .skills, timing: .nextTurn)
                .toggleStyle(.switch)
                .controlSize(.small)
            } else {
                Group {
                    if skill.isEnabled {
                        Text("Enabled", bundle: .module)
                    } else {
                        Text("Disabled", bundle: .module)
                    }
                }
                .themedFont(.small)
            }
            if case .plugin = skill.scope {
                Button { model.openSettings(tab: .plugins) } label: { Text("Manage plugin", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }
            actionRow
        }
    }

    private var actionRow: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            if SkillManager.shared.ownsSkill(skill) {
                Button(action: onEdit) {
                    Label { Text("Edit", bundle: .module) } icon: { Image(systemName: "pencil") }
                }
                .appAction(.secondary, size: .small)
            } else {
                Menu {
                    Button { _ = model.importSkill(from: skill.skillDirectoryURL ?? skill.sourceURL, targetScope: .userGlobal) } label: { Text("Copy to User", bundle: .module) }
                    if let path = scopedProject?.rootDirectoryPath {
                        Button { _ = model.importSkill(from: skill.skillDirectoryURL ?? skill.sourceURL, targetScope: .projectLocal(projectPath: path)) } label: { Text("Copy to Project", bundle: .module) }
                    }
                } label: {
                    Text("Copy to TurboSpark", bundle: .module)
                        .settingsControl("Copy to TurboSpark", pane: .skills, timing: .nextTurn)
                }
                .appMenuAction(size: .small)
                .help(Text("Skill actions", bundle: .module))
            }
            Button {
                NSWorkspace.shared.activateFileViewerSelecting([skill.sourceURL])
            } label: {
                Label { Text("Reveal", bundle: .module) } icon: { Image(systemName: "folder") }
            }
            .appAction(.secondary, size: .small)

            if SkillManager.shared.ownsSkill(skill) {
                AppIconButton("trash", label: Text("Delete", bundle: .module), role: .destructive, action: onDelete)
                    .appAction(.destructive, size: .small)
            }
        }
    }

    private var pathTriggers: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Activation Path Triggers", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            FlowLayout(spacing: 6, lineSpacing: 6) {
                ForEach(skill.manifest.paths, id: \.self) { pathPattern in
                    HStack(spacing: 4) {
                        Image(systemName: "arrow.triangle.turn.up.right.diamond")
                            .themedFont(.tiny)
                            .accessibilityHidden(true)
                        Text(pathPattern)
                            .themedCode(.small)
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

    private var referenceFiles: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Directory Reference Files (\(skill.referenceFiles.count))", bundle: .module)
                .themedFont(.base, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            VStack(spacing: 4) {
                ForEach(skill.referenceFiles, id: \.self) { fileName in
                    HStack(spacing: 6) {
                        Image(systemName: "doc")
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .accessibilityHidden(true)
                        Text(fileName)
                            .themedCode(.small)
                        Spacer()
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 6)
                    .background(.appSurface.opacity(0.6))
                    .clipShape(RoundedRectangle(cornerRadius: 4))
                }
            }
        }
    }

    private var instructions: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("Instruction Body (SKILL.md)", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Button {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(skill.content, forType: .string)
                } label: {
                    Label { Text("Copy", bundle: .module) } icon: { Image(systemName: "doc.on.doc") }
                        .themedFont(.small)
                }
                .appAction(.quiet, size: .small)
            }

            Text(skill.content)
                .themedCode(.base)
                .padding(12)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(.appElevated)
                .clipShape(RoundedRectangle(cornerRadius: 6))
                .overlay(
                    RoundedRectangle(cornerRadius: 6)
                        .stroke(.appBorder, lineWidth: 1)
                )
        }
    }

    private func metaItem(title: Text, value: String, icon: String) -> some View {
        HStack(spacing: 6) {
            Image(systemName: icon)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .frame(width: 16)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                title
                    .themedFont(.tiny)
                    .foregroundStyle(.tertiary)
                Text(value)
                    .themedFont(.small, weight: .medium)
                    .lineLimit(1)
            }
        }
        .accessibilityElement(children: .combine)
    }
}

/// The scope capsule on a skill row and in the skill detail header.
/// `SkillScope.label` is fixed English from the model layer, so it is looked
/// up in the catalog here; every value it can take is a catalog key.
struct SkillScopeBadge: View {
    let scope: SkillScope

    var body: some View {
        HStack(spacing: 3) {
            Image(systemName: scope.badgeIcon)
                .themedFont(.micro)
                .accessibilityHidden(true)
            Text(LocalizedStringKey(scope.label), bundle: .module)
                .themedFont(.micro, weight: .semibold)
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
}
