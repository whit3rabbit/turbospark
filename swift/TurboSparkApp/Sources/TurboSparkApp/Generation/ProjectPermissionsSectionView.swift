import AppKit
import SwiftUI

/// Option for project-level Forge Tool-Call Guardrails override.
enum AppProjectGuardrailsOption: String, CaseIterable, Identifiable {
    case auto = "auto"
    case enabled = "enabled"
    case disabled = "disabled"

    var id: String { rawValue }

    var asOptionalBool: Bool? {
        switch self {
        case .auto: return nil
        case .enabled: return true
        case .disabled: return false
        }
    }

    static func from(optionalBool: Bool?) -> AppProjectGuardrailsOption {
        guard let b = optionalBool else { return .auto }
        return b ? .enabled : .disabled
    }
}

/// Settings section for project tool execution permissions, presets, and guardrails policy.
struct ProjectPermissionsSectionView: View {
    @Binding var permissionMode: AppPermissionMode
    @Binding var fileReadPermission: AppToolPermission
    @Binding var fileWritePermission: AppToolPermission
    @Binding var terminalPermission: AppToolPermission
    @Binding var webPermission: AppToolPermission
    @Binding var mcpPermission: AppToolPermission
    @Binding var automationPermission: AppToolPermission
    @Binding var guardrailsOption: AppProjectGuardrailsOption

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Tool Permissions & Risk Policy", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Menu("Presets") {
                    Button("Auto (Recommended)") { applyPreset(.auto) }
                    Button("Always Ask") { applyPreset(.alwaysAsk) }
                    Button("Permissive (Allow All)") { applyPreset(.permissive) }
                    Button("Read Only") { applyPreset(.readOnly) }
                }
                .menuStyle(.borderlessButton)
                .themedFont(.small)
                .accessibilityLabel("Permission presets")
                .accessibilityHint("Applies a recommended set of tool permissions")
            }

            // Mode Selector
            VStack(alignment: .leading, spacing: 6) {
                Text("Execution Mode", bundle: .module)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.secondary)

                Picker("Execution Mode", selection: $permissionMode) {
                    ForEach(AppPermissionMode.allCases) { mode in
                        Label(mode.shortLabel, systemImage: mode.systemImage).tag(mode)
                    }
                }
                .pickerStyle(.segmented)
                .accessibilityLabel("Permission execution mode")

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: permissionMode.systemImage)
                        .foregroundStyle(TurboSparkTheme.accentColor)
                        .themedFont(.small)
                    Text(permissionMode.descriptionText)
                        .themedFont(.tiny)
                        .foregroundStyle(.secondary)
                }
                .padding(8)
                .frame(maxWidth: .infinity, alignment: .leading)
                .background(Color.primary.opacity(0.03), in: RoundedRectangle(cornerRadius: 6))
            }

            // Granular Category Overrides
            VStack(spacing: 8) {
                permissionRow(title: "File Reading", desc: "List directories and inspect code files", selection: $fileReadPermission)
                permissionRow(title: "File Writing", desc: "Create, edit, or modify files", selection: $fileWritePermission)
                permissionRow(title: "Terminal Commands", desc: "Execute shell commands in project root", selection: $terminalPermission)
                permissionRow(title: "Web Requests", desc: "Fetch web documentation and search", selection: $webPermission)
                permissionRow(title: "MCP External Tools", desc: "Invoke external MCP server tools", selection: $mcpPermission)
                permissionRow(title: "Automation & Crons", desc: "Schedule background tasks and monitors", selection: $automationPermission)
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))

            // Forge Guardrails project option
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Forge Tool-Call Guardrails", bundle: .module)
                        .themedFont(.base, weight: .medium)
                    Text("Repair malformed dialect calls and validate schemas", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.secondary)
                }
                Spacer()
                Picker("", selection: $guardrailsOption) {
                    Text("Auto (Model Default)", bundle: .module).tag(AppProjectGuardrailsOption.auto)
                    Text("Always Enabled", bundle: .module).tag(AppProjectGuardrailsOption.enabled)
                    Text("Always Disabled", bundle: .module).tag(AppProjectGuardrailsOption.disabled)
                }
                .pickerStyle(.menu)
                .frame(width: 175)
                .accessibilityLabel("Forge Guardrails project preference")
            }
            .padding(12)
            .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private func permissionRow(title: String, desc: String, selection: Binding<AppToolPermission>) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .themedFont(.base, weight: .medium)
                Text(desc)
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
            }
            Spacer()
            Picker("", selection: selection) {
                ForEach(AppToolPermission.allCases) { perm in
                    Text(perm.label).tag(perm)
                }
            }
            .pickerStyle(.menu)
            .frame(width: 175)
            .accessibilityLabel("\(title) permission")
        }
    }

    private func applyPreset(_ permissions: AppProjectPermissions) {
        permissionMode = permissions.mode
        fileReadPermission = permissions.fileRead
        fileWritePermission = permissions.fileWrite
        terminalPermission = permissions.terminal
        webPermission = permissions.web
        mcpPermission = permissions.mcp
        automationPermission = permissions.automation
    }
}
