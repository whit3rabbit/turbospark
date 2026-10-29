import SwiftUI

/// Settings pane for tool profiles: which tools a chat or project turn offers.
///
/// The composer's Tools chip picks among these and flips single tools for one
/// session. This pane is where profiles are made, edited and removed. The
/// built-in profiles are code-defined, so they are read-only here and
/// "Duplicate" makes an editable copy.
struct ToolsSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @State private var editingID: UUID = AppToolProfile.defaultID
    @State private var pendingDelete: AppToolProfile?

    private var editing: AppToolProfile {
        model.allToolProfiles.first { $0.id == editingID } ?? .default
    }

    var body: some View {
        Form {
            Text("Choose which tools the model is offered. Default keeps every tool. Smaller sets suit smaller local models.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)

            profilesSection
            editorSection
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
        .confirmationDialog(
            Text("Delete this profile?", bundle: .module),
            isPresented: Binding(
                get: { pendingDelete != nil }, set: { if !$0 { pendingDelete = nil } }),
            presenting: pendingDelete
        ) { profile in
            Button(role: .destructive) {
                model.deleteToolProfile(profile.id)
                if editingID == profile.id { editingID = AppToolProfile.defaultID }
            } label: {
                Text("Delete", bundle: .module)
            }
        } message: { profile in
            Text("\(profile.displayName) will be removed and Default will be used.", bundle: .module)
        }
    }

    // MARK: Profiles

    private var profilesSection: some View {
        Section {
            ForEach(model.allToolProfiles) { profile in
                profileRow(profile)
            }
            HStack {
                Button {
                    if let id = model.duplicateToolProfile(AppToolProfile.defaultID) { editingID = id }
                } label: {
                    Label { Text("New Profile", bundle: .module) } icon: { Image(systemName: "plus") }
                }
                .settingsControl("New Profile", pane: .tools, timing: .nextTurn)
            }
        } header: {
            Text("Profiles", bundle: .module)
        } footer: {
            Text("Applies to chat and project turns. Subagents and scheduled tasks keep the tools their own definitions list.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
        }
    }

    private func profileRow(_ profile: AppToolProfile) -> some View {
        let isActive = model.selectedToolProfile.id == profile.id
        let isEditing = editingID == profile.id
        return HStack(spacing: 10) {
            Image(systemName: isActive ? "checkmark.circle.fill" : "circle")
                .foregroundStyle(isActive ? .appAccentText : .appSecondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 1) {
                Text(profile.displayName).fontWeight(isEditing ? .semibold : .regular)
                Text(countSummary(profile))
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            Spacer()
            if !isActive {
                Button { model.selectToolProfile(profile.id) } label: {
                    Text("Use", bundle: .module)
                }
                .settingsControl("Use", pane: .tools, timing: .nextTurn)
            }
            Button {
                if let id = model.duplicateToolProfile(profile.id) { editingID = id }
            } label: {
                Text("Duplicate", bundle: .module)
            }
            .settingsControl("Duplicate", pane: .tools, timing: .nextTurn)
            if !profile.isBuiltIn {
                Button(role: .destructive) { pendingDelete = profile } label: {
                    Text("Delete", bundle: .module)
                }
                .settingsControl("Delete", pane: .tools, timing: .nextTurn)
            }
        }
        .contentShape(Rectangle())
        .onTapGesture { editingID = profile.id }
        .accessibilityElement(children: .contain)
        .accessibilityAddTraits(isActive ? .isSelected : [])
    }

    private func countSummary(_ profile: AppToolProfile) -> String {
        let total = AppToolCatalog.profileToolCount()
        guard let allowed = profile.allowedTools else {
            return AppStrings.localized("All \(total) tools")
        }
        return AppStrings.localized("\(min(allowed.count, total)) of \(total) tools")
    }

    // MARK: Editor

    @ViewBuilder
    private var editorSection: some View {
        let profile = editing
        let locked = profile.isBuiltIn
        Section {
            if locked {
                Text("Built-in profiles cannot be edited. Duplicate one to change it.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            } else {
                TextField(text: Binding(
                    get: { profile.name },
                    set: { save(profile, name: $0) })) {
                    Text("Name", bundle: .module)
                }
                .settingsControl("Name", pane: .tools, timing: .nextTurn)
            }

            ForEach(AppToolCatalog.profileGroups()) { group in
                VStack(alignment: .leading, spacing: 4) {
                    Text(LocalizedStringKey(group.title), bundle: .module)
                        .font(theme.ui(.small, weight: .semibold))
                        .foregroundStyle(.appSecondary)
                    ForEach(group.entries) { entry in
                        Toggle(isOn: Binding(
                            get: { profile.allowedTools?.contains(entry.id) ?? true },
                            set: { setTool(entry.id, $0, in: profile) })) {
                            VStack(alignment: .leading, spacing: 1) {
                                Text(entry.name)
                                Text(entry.summary)
                                    .font(theme.ui(.small))
                                    .foregroundStyle(.appSecondary)
                            }
                        }
                        .toggleStyle(.checkbox)
                        .disabled(locked)
                    }
                }
            }

            Toggle(isOn: Binding(
                get: { profile.includesMcpAndCustom },
                set: { save(profile, includesMcpAndCustom: $0) })) {
                VStack(alignment: .leading, spacing: 1) {
                    Text("MCP servers and custom tools", bundle: .module)
                    Text("Tools from connected servers, scripts and your own tool files.", bundle: .module)
                        .font(theme.ui(.small))
                        .foregroundStyle(.appSecondary)
                }
            }
            .toggleStyle(.checkbox)
            .disabled(locked)
            .settingsControl("MCP servers and custom tools", pane: .tools, timing: .nextTurn)
        } header: {
            Text(profile.displayName)
        }
    }

    private func setTool(_ key: String, _ enabled: Bool, in profile: AppToolProfile) {
        let universe = AppToolCatalog.profileGroups().flatMap { $0.entries.map(\.id) }
        var keys = Set(profile.allowedTools ?? universe)
        if enabled { keys.insert(key) } else { keys.remove(key) }
        // Every tool listed is the same thing as no allowlist, so a new tool
        // added later still reaches this profile.
        let allowed: [String]? = Set(universe).isSubset(of: keys) ? nil : Array(keys)
        model.updateToolProfile(
            id: profile.id, name: profile.name, allowedTools: allowed,
            includesMcpAndCustom: profile.includesMcpAndCustom)
    }

    private func save(_ profile: AppToolProfile, name: String? = nil, includesMcpAndCustom: Bool? = nil) {
        model.updateToolProfile(
            id: profile.id, name: name ?? profile.name, allowedTools: profile.allowedTools,
            includesMcpAndCustom: includesMcpAndCustom ?? profile.includesMcpAndCustom)
    }
}
