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
    @Binding var browserPermission: AppToolPermission
    @Binding var browserOriginAllowlist: [String]
    @Binding var guardrailsOption: AppProjectGuardrailsOption
    @StateObject private var browserPermissionModel: BrowserProjectPermissionsSettingsViewModel

    init(
        permissionMode: Binding<AppPermissionMode>,
        fileReadPermission: Binding<AppToolPermission>,
        fileWritePermission: Binding<AppToolPermission>,
        terminalPermission: Binding<AppToolPermission>,
        webPermission: Binding<AppToolPermission>,
        mcpPermission: Binding<AppToolPermission>,
        automationPermission: Binding<AppToolPermission>,
        browserPermission: Binding<AppToolPermission>,
        browserOriginAllowlist: Binding<[String]>,
        guardrailsOption: Binding<AppProjectGuardrailsOption>
    ) {
        _permissionMode = permissionMode
        _fileReadPermission = fileReadPermission
        _fileWritePermission = fileWritePermission
        _terminalPermission = terminalPermission
        _webPermission = webPermission
        _mcpPermission = mcpPermission
        _automationPermission = automationPermission
        _browserPermission = browserPermission
        _browserOriginAllowlist = browserOriginAllowlist
        _guardrailsOption = guardrailsOption
        _browserPermissionModel = StateObject(wrappedValue: BrowserProjectPermissionsSettingsViewModel(
            permission: browserPermission.wrappedValue,
            originAllowlist: browserOriginAllowlist.wrappedValue
        ) { updatedPermission, updatedOrigins in
            browserPermission.wrappedValue = updatedPermission
            browserOriginAllowlist.wrappedValue = updatedOrigins
        })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Tool Permissions & Risk Policy", bundle: .module)
                    .themedFont(.small, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Menu {
                    Button { applyPreset(.auto) } label: { Text("Auto (Recommended)", bundle: .module) }
                    Button { applyPreset(.alwaysAsk) } label: { Text("Always Ask", bundle: .module) }
                    Button { applyPreset(.permissive) } label: { Text("Permissive (Allow All)", bundle: .module) }
                    Button { applyPreset(.readOnly) } label: { Text("Read Only", bundle: .module) }
                } label: {
                    Text("Presets", bundle: .module)
                }
                .menuStyle(.borderlessButton)
                .themedFont(.small)
                .accessibilityLabel(Text("Permission presets", bundle: .module))
                .accessibilityHint(Text("Applies a recommended set of tool permissions", bundle: .module))
            }

            // Mode Selector
            VStack(alignment: .leading, spacing: 6) {
                Text("Execution Mode", bundle: .module)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appSecondary)

                Picker(selection: $permissionMode) {
                    ForEach(AppPermissionMode.allCases) { mode in
                        // Every short label is a catalog key.
                        Label {
                            Text(LocalizedStringKey(mode.shortLabel), bundle: .module)
                        } icon: {
                            Image(systemName: mode.systemImage)
                        }
                        .tag(mode)
                    }
                } label: { Text("Execution Mode", bundle: .module) }
                .pickerStyle(.menu)
                .accessibilityLabel(Text("Permission execution mode", bundle: .module))

                HStack(alignment: .top, spacing: 8) {
                    Image(systemName: permissionMode.systemImage)
                        .foregroundStyle(.appAccent)
                        .themedFont(.small)
                    permissionMode.approvalDescriptionText
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
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
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))

            browserPermissionsSection

            // Forge Guardrails project option
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Forge Tool-Call Guardrails", bundle: .module)
                        .themedFont(.base, weight: .medium)
                    Text("Repair malformed dialect calls and validate schemas", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                Spacer()
                Picker(selection: $guardrailsOption) {
                    Text("Auto (Model Default)", bundle: .module).tag(AppProjectGuardrailsOption.auto)
                    Text("Always Enabled", bundle: .module).tag(AppProjectGuardrailsOption.enabled)
                    Text("Always Disabled", bundle: .module).tag(AppProjectGuardrailsOption.disabled)
                } label: {
                    Text("Forge Guardrails project preference", bundle: .module)
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .frame(width: 175)
            }
            .padding(12)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))
        }
    }

    private var browserPermissionsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Browser access", bundle: .module)
                .themedFont(.base, weight: .medium)

            Picker(
                selection: Binding(
                    get: { browserPermissionModel.permission },
                    set: { browserPermissionModel.setPermission($0) }
                )
            ) {
                ForEach(AppToolPermission.allCases) { permission in
                    Text(LocalizedStringKey(permission.label), bundle: .module).tag(permission)
                }
            } label: {
                Text("Browser access", bundle: .module)
            }
            .pickerStyle(.menu)
            .accessibilityLabel(Text("Browser access", bundle: .module))

            Text("Site grants match exact HTTP(S) origins and are limited to 256 per project. Actions ask by default.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)

            Text("Allowed site origins", bundle: .module)
                .themedFont(.small, weight: .semibold)

            if !browserPermissionModel.originAllowlist.isEmpty {
                VStack(alignment: .leading, spacing: 6) {
                    ForEach(browserPermissionModel.originAllowlist, id: \.self) { origin in
                        HStack(spacing: 8) {
                            Text(origin)
                                .themedFont(.tiny, systemDesign: .monospaced)
                                .lineLimit(1)
                                .truncationMode(.middle)
                            Spacer(minLength: 4)
                            Button(role: .destructive) {
                                browserPermissionModel.revokeOrigin(origin)
                            } label: {
                                Text("Remove", bundle: .module)
                            }
                            .appAction(.quiet, size: .small)
                        }
                    }
                }
            }

            HStack(spacing: 8) {
                // The prompt is an example origin, not prose.
                TextField(
                    text: $browserPermissionModel.originDraft,
                    prompt: Text(verbatim: "https://example.com")
                ) {
                    Text("Exact HTTP(S) origin", bundle: .module)
                }
                    .textFieldStyle(.roundedBorder)
                    .accessibilityLabel(Text("Exact HTTP(S) origin", bundle: .module))
                Button {
                    browserPermissionModel.addOrigin()
                } label: {
                    Text("Add", bundle: .module)
                }
                .appAction(.secondary)
            }

            if let feedback = browserPermissionModel.feedback {
                Text(LocalizedStringKey(feedback.localizationKey), bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.red)
                    .accessibilityAddTraits(.updatesFrequently)
            }
        }
        .padding(12)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 8))
    }

    /// `title` and `desc` are catalog keys; the picker carries `title` as
    /// its accessible name, so VoiceOver says which category it sets.
    private func permissionRow(
        title: LocalizedStringKey, desc: LocalizedStringKey, selection: Binding<AppToolPermission>
    ) -> some View {
        HStack {
            VStack(alignment: .leading, spacing: 2) {
                Text(title, bundle: .module)
                    .themedFont(.base, weight: .medium)
                Text(desc, bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            Picker(selection: selection) {
                ForEach(AppToolPermission.allCases) { perm in
                    Text(LocalizedStringKey(perm.label), bundle: .module).tag(perm)
                }
            } label: {
                Text(title, bundle: .module)
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .frame(width: 175)
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
