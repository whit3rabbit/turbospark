import SwiftUI

/// The Profiles settings pane: create, rename, delete, switch between, and
/// back up the users of this installation. Switching is a save-and-relaunch
/// (`AppModel.switchToProfile`); this pane says so in as many words rather
/// than surprising someone with an app restart, and both destructive doors
/// (delete and switch) confirm before acting. Backups run without either
/// door: export reads the profile's own folder and import restores into a
/// NEW identity, so neither one can damage the profile this run belongs to.
struct ProfilesSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var newProfileName: String = ""
    @State private var renameTarget: UserProfile?
    @State private var renameText: String = ""
    @State private var deleteTarget: UserProfile?
    @State private var switchTarget: UserProfile?
    @State private var exportTarget: UserProfile?
    @State private var exportSelection: Set<String> = []
    @State private var importOffer: AppModel.ProfileBackupImportOffer?
    @State private var importName: String = ""

    var body: some View {
        Form {
            usersSection
            addSection
            restoreSection
            notesSection
        }
        .formStyle(.grouped)
        .padding(16)
        .sheet(item: $renameTarget) { profile in
            renameSheet(profile)
        }
        .sheet(item: $importOffer) { offer in
            importSheet(offer)
        }
        .sheet(item: $exportTarget) { profile in
            exportSheet(profile)
        }
        .confirmationDialog(
            "Delete Profile",
            isPresented: Binding(
                get: { deleteTarget != nil },
                set: { if !$0 { deleteTarget = nil } }),
            titleVisibility: .visible
        ) {
            Button("Delete \"\(deleteTarget?.name ?? "")\"", role: .destructive) {
                if let target = deleteTarget {
                    model.deleteProfile(target)
                }
                deleteTarget = nil
            }
            Button(role: .cancel) {
                deleteTarget = nil
            } label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("This removes the user's settings, chat history, projects, agents, global MCP servers and marketplaces, skills, plugins, custom tools, hooks, memory, and model favorites. The folder moves to the Trash, but restoring it does not bring the profile back; it is only for recovering files by hand. Shared and untouched: downloaded models, the install registry, the Keychain server API key, and the UI language.", bundle: .module)
        }
        .confirmationDialog(
            Text("Switch Profile", bundle: .module),
            isPresented: Binding(
                get: { switchTarget != nil },
                set: { if !$0 { switchTarget = nil } }),
            titleVisibility: .visible
        ) {
            Button {
                if let target = switchTarget {
                    model.switchToProfile(target)
                }
                switchTarget = nil
            } label: { Text("Switch and Relaunch", bundle: .module) }
            Button(role: .cancel) {
                switchTarget = nil
            } label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("The app saves everything and relaunches as this user. Each user has separate settings, chat history, skills, MCP servers, agents, and plugins; work stays with the user who created it.", bundle: .module)
        }
    }

    // MARK: - Sections (separate properties: a Form holding every section
    // inline is one type-checker expression and walks into the solver
    // budget, the reason `AppSettingsView.engineSettingsTab` is split too.)

    private var usersSection: some View {
        Section(header: Text("Users on this Mac", bundle: .module)) {
            defaultProfileRow
            ForEach(model.profiles) { profile in
                additionalProfileRow(profile)
            }
            if model.profiles.isEmpty {
                Text("Only the Default user exists. Add one below to give it its own settings, chats, and skills.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
    }

    private var defaultProfileRow: some View {
        profileRow(
            name: UserProfileStore.defaultProfile.name,
            isCurrent: model.isDefaultProfileActive,
            subtitle: "Built in. Shares the ~/.turbospark skills, agents, and tools with other apps.",
            profile: nil)
            .contextMenu {
                defaultProfileActions()
            }
    }

    private func additionalProfileRow(_ profile: UserProfile) -> some View {
        profileRow(
            name: profile.name,
            isCurrent: profile.id == model.currentProfile.id,
            subtitle: "Self-contained settings, chats, skills, plugins, and MCP servers.",
            profile: profile)
        .contextMenu {
            profileActions(profile)
        }
    }

    /// One row for either kind of user. `profile` is nil for the built-in
    /// Default user, which cannot be renamed or deleted (it backs up through
    /// `defaultProfileActions` instead) and switches through the fixed
    /// `defaultProfile` row rather than a registry lookup.
    private func profileRow(
        name: String,
        isCurrent: Bool,
        subtitle: String,
        profile: UserProfile?
    ) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "person.crop.circle")
                .font(theme.ui(.title3))
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    Text(name)
                        .font(theme.ui(.base, weight: .medium))
                    if isCurrent {
                        Text("Current", bundle: .module)
                            .font(theme.ui(.tiny, weight: .semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(Color.accentColor.opacity(0.15), in: Capsule())
                            .foregroundStyle(Color.accentColor)
                    }
                }
                Text(subtitle)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            // Combining only the text keeps the Switch button and the actions
            // menu individually reachable to VoiceOver; combining the whole
            // row flattened them into the label.
            .accessibilityElement(children: .combine)
            .accessibilityLabel("\(name)\(isCurrent ? ", current profile" : "")")
            Spacer()
            if !isCurrent {
                Button {
                    switchTarget = profile ?? UserProfileStore.defaultProfile
                } label: { Text("Switch", bundle: .module) }
                .disabled(!model.canSwitchProfile)
                .help("Saves everything and relaunches the app as this user")
            }
            Menu {
                if let profile {
                    profileActions(profile)
                } else {
                    defaultProfileActions()
                }
            } label: {
                Image(systemName: "ellipsis.circle")
                    .font(theme.ui(.base))
                    .foregroundStyle(.appSecondary)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .accessibilityLabel("More actions for \(name)")
        }
    }

    /// The rename/delete pair, shared by the context menu and the row's
    /// ellipsis menu so neither surface can drift from the other.
    @ViewBuilder
    private func profileActions(_ profile: UserProfile) -> some View {
        Button {
            renameTarget = profile
            renameText = profile.name
        } label: { Text("Rename...", bundle: .module) }
        exportBackupAction(profile)
        if profile.id != model.currentProfile.id {
            Button(role: .destructive) {
                deleteTarget = profile
            } label: { Text("Delete...", bundle: .module) }
        }
    }

    /// The built-in Default user's actions: it cannot be renamed or deleted,
    /// but its setup (stores at the machine root plus the ~/.turbospark
    /// content, minus the shared model downloads) is exactly what a backup
    /// of it captures.
    @ViewBuilder
    private func defaultProfileActions() -> some View {
        exportBackupAction(UserProfileStore.defaultProfile)
    }

    /// One shared Export Backup button so every menu surface offers the same
    /// action for the same profile. It opens the category sheet; the archive
    /// only runs after the sheet's Continue.
    private func exportBackupAction(_ profile: UserProfile) -> some View {
        Button {
            exportSelection = ProfileBackup.allCategoryIDs
            exportTarget = profile
        } label: { Text("Export Backup...", bundle: .module) }
        .disabled(model.profileBackupInFlight)
        .settingsControl("Export Backup...", pane: .profiles, timing: .immediate)
        .help("Writes this user's settings, chats, skills, and tools to a .zip backup")
    }

    private var addSection: some View {
        Section(header: Text("Add a User", bundle: .module)) {
            HStack {
                TextField("Profile name", text: $newProfileName)
                    .onSubmit(addProfile)
                Button(action: addProfile) { Text("Add Profile", bundle: .module) }
                    .disabled(newProfileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            Text("New users start empty: their own settings, chats, skills, and MCP servers, isolated from the Default user and the shared ~/.turbospark folders.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
        }
            .settingsControl("Add a User", pane: .profiles, timing: .immediate)
    }

    private func addProfile() {
        let name = newProfileName
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        model.createProfile(named: name)
        newProfileName = ""
    }

    private var restoreSection: some View {
        Section(header: Text("Restore a Backup", bundle: .module)) {
            VStack(alignment: .leading, spacing: 6) {
                Button {
                    Task {
                        if let offer = await model.pickProfileBackupForImport() {
                            importOffer = offer
                            importName = offer.suggestedName
                        }
                    }
                } label: { Text("Import Backup...", bundle: .module) }
                .disabled(model.profileBackupInFlight)
                .help("Restores a profile backup (.zip) as a new user")
                Text("A backup comes back as a new user with a fresh identity and a name of its own, even when it was exported from the Default user. Keychain-stored hook secrets are not part of a backup.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
            .settingsControl("Restore a Backup", pane: .profiles, timing: .immediate)
    }

    private var notesSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                Text("Each profile is a folder under Application Support. Switching saves all work and relaunches the app; the Default user keeps using the shared ~/.turbospark folders, and every other profile is self-contained. Downloaded models are shared by all users.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                Button {
                    model.revealProfilesInFinder()
                } label: { Text("Reveal Profiles Folder in Finder", bundle: .module) }
            }
        }
    }

    // MARK: - Rename sheet

    private func renameSheet(_ profile: UserProfile) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Rename Profile", bundle: .module)
                .font(theme.ui(.title3, weight: .semibold))
                .settingsControl("Rename Profile", pane: .profiles, timing: .immediate)
            TextField("Profile name", text: $renameText)
                .textFieldStyle(.roundedBorder)
            HStack {
                Spacer()
                Button {
                    renameTarget = nil
                } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
                Button {
                    model.renameProfile(profile, to: renameText)
                    renameTarget = nil
                } label: { Text("Rename", bundle: .module) }
                .keyboardShortcut(.defaultAction)
                .disabled(renameText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 340)
    }

    // MARK: - Export sheet

    /// Choose what the backup carries. Every category starts selected; the
    /// Continue button hands the selection to the save panel and the archive.
    private func exportSheet(_ profile: UserProfile) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("What to Include in This Backup", bundle: .module)
                .font(theme.ui(.title3, weight: .semibold))
                .settingsControl(
                    "What to Include in This Backup", pane: .profiles, timing: .immediate)
            Text("Backing up \"\(profile.name)\"", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            ForEach(ProfileBackup.categories) { category in
                Toggle(isOn: Binding(
                    get: { exportSelection.contains(category.id) },
                    set: { isOn in
                        if isOn {
                            exportSelection.insert(category.id)
                        } else {
                            exportSelection.remove(category.id)
                        }
                    }
                )) {
                    categoryHeader(category.id)
                }
            }
            HStack(spacing: 12) {
                Button {
                    exportSelection = ProfileBackup.allCategoryIDs
                } label: { Text("Select All", bundle: .module) }
                .buttonStyle(.link)
                Button {
                    exportSelection = []
                } label: { Text("Clear All", bundle: .module) }
                .buttonStyle(.link)
                Spacer()
            }
            Text("SOUL and personality are stored with Settings. Keychain-stored hook secrets never travel, and the Default user's downloaded models and install registry are never part of a backup.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            HStack {
                Spacer()
                Button {
                    exportTarget = nil
                } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
                Button {
                    let target = exportTarget
                    let selection = exportSelection
                    exportTarget = nil
                    if let target {
                        model.runProfileBackupExport(target, included: selection)
                    }
                } label: { Text("Continue", bundle: .module) }
                .keyboardShortcut(.defaultAction)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    /// Category labels as literal keys: existing catalog entries ("Settings",
    /// "Skills", ...) are reused, the rest carry their own keys. The id is
    /// the manifest's category id, never shown raw.
    @ViewBuilder
    private func categoryHeader(_ id: String) -> some View {
        switch id {
        case "settings": Text("Settings", bundle: .module)
        case "chats": Text("Chat history", bundle: .module)
        case "projects": Text("Projects", bundle: .module)
        case "models": Text("Model favorites and scan paths", bundle: .module)
        case "mcp": Text("MCP servers and marketplaces", bundle: .module)
        case "skills": Text("Skills", bundle: .module)
        case "agents": Text("Agents", bundle: .module)
        case "tools": Text("Custom tools", bundle: .module)
        case "plugins": Text("Plugins and marketplaces", bundle: .module)
        case "hooks": Text("Hooks", bundle: .module)
        case "memory": Text("Memory", bundle: .module)
        default: Text("Automation and observations", bundle: .module)
        }
    }

    // MARK: - Import sheet

    private func importSheet(_ offer: AppModel.ProfileBackupImportOffer) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Import Profile Backup", bundle: .module)
                .font(theme.ui(.title3, weight: .semibold))
                .settingsControl("Import Profile Backup", pane: .profiles, timing: .immediate)
            Text("Backup of \"\(offer.manifest.profileName)\" exported \(offer.manifest.exportedAt.formatted(date: .abbreviated, time: .shortened)) with app version \(offer.manifest.appVersion.isEmpty ? "unknown" : offer.manifest.appVersion); \(offer.manifest.contents.count) items inside.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            if offer.manifest.isDefault {
                Text("This is a Default-user backup: it imports as a new named user, not the built-in one.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            TextField("Profile name", text: $importName)
                .textFieldStyle(.roundedBorder)
            HStack {
                Spacer()
                Button {
                    importOffer = nil
                } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
                Button {
                    if model.importProfileBackup(offer, named: importName) {
                        importOffer = nil
                    }
                } label: { Text("Import", bundle: .module) }
                .keyboardShortcut(.defaultAction)
                .disabled(importName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
        }
        .padding(20)
        .frame(width: 380)
    }
}
