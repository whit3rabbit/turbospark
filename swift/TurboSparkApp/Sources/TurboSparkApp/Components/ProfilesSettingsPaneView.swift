import SwiftUI

/// The Profiles settings pane: create, rename, delete, switch between, and
/// back up the profiles of this installation. Switching is a
/// save-and-relaunch (`AppModel.switchToProfile`); this pane says so in as
/// many words rather than surprising someone with an app restart, and both
/// destructive doors (delete and switch) confirm before acting. Backups run
/// without either door: export reads the profile's own folder and import
/// restores into a NEW identity, so neither one can damage the profile this
/// run belongs to.
///
/// Terminology: "profile" is this app's concept everywhere in the pane.
/// The UI audit of 2026-10-11 found it called "user" in half the strings,
/// which read as a macOS account. The sheets live in
/// `ProfileSettingsSheets.swift` and `ProfileBackupSheets.swift`.
@MainActor
struct ProfilesSettingsPaneView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var vaultCoordinator = ProfileVaultCoordinator.shared

    @State private var newProfileName: String = ""
    @State private var renameTarget: UserProfile?
    @State private var deleteTarget: UserProfile?
    @State private var switchTarget: UserProfile?
    @State private var exportTarget: UserProfile?
    @State private var importOffer: AppModel.ProfileBackupImportOffer?
    @State private var securitySheet: ProfileSecuritySheetMode?
    @State private var securityBusy = false
    @State private var encryptedBackupRequest: ProfileEncryptedBackupRequest?

    var body: some View {
        Form {
            Section {
                Text("Manage profiles and credentials on this Mac.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            profilesSection
            securitySection
            addSection
            restoreSection
            notesSection
        }
        .formStyle(.grouped)
        // The grouped Form paints its own system gray, which ignores the theme.
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
        .sheet(item: $renameTarget) { profile in
            ProfileRenameSheet(model: model, profile: profile)
        }
        .sheet(item: $importOffer) { offer in
            ProfileBackupImportSheet(model: model, offer: offer)
        }
        .sheet(item: $exportTarget) { profile in
            ProfileOpenExportSheet(model: model, profile: profile)
        }
        .sheet(item: $securitySheet) { mode in
            ProfileSecuritySheet(model: model, mode: mode)
        }
        .sheet(item: $encryptedBackupRequest) { request in
            ProfileEncryptedBackupSheet(model: model, request: request)
        }
        .confirmationDialog(
            Text("Delete Profile", bundle: .module),
            isPresented: Binding(
                get: { deleteTarget != nil },
                set: { if !$0 { deleteTarget = nil } }),
            titleVisibility: .visible
        ) {
            Button(role: .destructive) {
                if let target = deleteTarget {
                    model.deleteProfile(target)
                }
                deleteTarget = nil
            } label: {
                Text("Delete \"\(deleteTarget?.name ?? "")\"", bundle: .module)
            }
            Button(role: .cancel) {
                deleteTarget = nil
            } label: { Text("Cancel", bundle: .module) }
        } message: {
            Text("This removes the profile's settings, chat history, projects, agents, global MCP servers and marketplaces, skills, plugins, custom tools, hooks, memory, and model favorites. The folder moves to the Trash, but restoring it does not bring the profile back; it is only for recovering files by hand. Shared and untouched: downloaded models, the install registry, the Keychain server API key, and the UI language.", bundle: .module)
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
            Text("The app saves everything and relaunches as this profile. Each profile has separate settings, chat history, skills, MCP servers, agents, and plugins; work stays with the profile that created it.", bundle: .module)
        }
    }

    // MARK: - Sections (separate properties: a Form holding every section
    // inline is one type-checker expression and walks into the solver
    // budget, the reason `EngineSettingsPaneView` is split too.)

    private var profilesSection: some View {
        // The anchor sits on the header because Export Backup and Rename live
        // in each row's menu, which is not mounted until it opens; settings
        // search scrolls here instead (`SettingsControlDescriptor.navigationID`).
        Section(header: Text("Profiles on this Mac", bundle: .module)
            .settingsControl("Profiles on this Mac", pane: .profiles, timing: .immediate)) {
            defaultProfileRow
            ForEach(model.profiles) { profile in
                additionalProfileRow(profile)
            }
            if model.profiles.isEmpty {
                Text("Only the Default profile exists. Add one below to give it its own settings, chats, and skills.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
    }

    private var defaultProfileRow: some View {
        profileRow(
            // The built-in profile's stored name is the English word; show it
            // in the UI language like every other fixed label.
            name: Text("Default", bundle: .module),
            spokenName: AppStrings.localized("Default"),
            isCurrent: model.isDefaultProfileActive,
            subtitle: Text("Built in. Shares the ~/.turbospark skills, agents, and tools with other apps.", bundle: .module),
            profile: nil)
            .contextMenu {
                defaultProfileActions()
            }
    }

    private func additionalProfileRow(_ profile: UserProfile) -> some View {
        profileRow(
            name: Text(profile.name),
            spokenName: profile.name,
            isCurrent: profile.id == model.currentProfile.id,
            subtitle: Text("Self-contained settings, chats, skills, plugins, and MCP servers.", bundle: .module),
            profile: profile)
        .contextMenu {
            profileActions(profile)
        }
    }

    /// One row for either kind of profile. `profile` is nil for the built-in
    /// Default profile, which cannot be renamed or deleted (it backs up
    /// through `defaultProfileActions` instead) and switches through the
    /// fixed `defaultProfile` row rather than a registry lookup.
    private func profileRow(
        name: Text,
        spokenName: String,
        isCurrent: Bool,
        subtitle: Text,
        profile: UserProfile?
    ) -> some View {
        HStack(spacing: 10) {
            Image(systemName: "person.crop.circle")
                .font(theme.ui(.title3))
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                HStack(spacing: 6) {
                    name
                        .font(theme.ui(.base, weight: .medium))
                    if isCurrent {
                        Text("Current", bundle: .module)
                            .font(theme.ui(.tiny, weight: .semibold))
                            .padding(.horizontal, 6)
                            .padding(.vertical, 2)
                            .background(theme.accent.opacity(0.15), in: Capsule())
                            .foregroundStyle(theme.accent)
                    }
                }
                subtitle
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
            // Combining only the text keeps the Switch button and the actions
            // menu individually reachable to VoiceOver; combining the whole
            // row flattened them into the label.
            .accessibilityElement(children: .combine)
            .accessibilityLabel(isCurrent
                ? Text("\(spokenName), current profile", bundle: .module)
                : Text(spokenName))
            Spacer()
            if !isCurrent {
                Button {
                    switchTarget = profile ?? UserProfileStore.defaultProfile
                } label: { Text("Switch", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(!model.canSwitchProfile)
                .help(Text("Saves everything and relaunches the app as this profile", bundle: .module))
            }
            Menu {
                if let profile {
                    profileActions(profile)
                } else {
                    defaultProfileActions()
                }
            } label: {
                Label {
                    Text("More actions for \(spokenName)", bundle: .module)
                } icon: {
                    Image(systemName: "ellipsis.circle")
                        .font(theme.ui(.base))
                        .foregroundStyle(.appSecondary)
                }
                .labelStyle(.iconOnly)
            }
            .menuStyle(.borderlessButton)
            .menuIndicator(.hidden)
            .fixedSize()
            .help(Text("More actions for \(spokenName)", bundle: .module))
        }
    }

    /// The rename/delete pair, shared by the context menu and the row's
    /// ellipsis menu so neither surface can drift from the other.
    @ViewBuilder
    private func profileActions(_ profile: UserProfile) -> some View {
        if !profile.isProtected || profile.id == model.currentProfile.id {
            Button {
                renameTarget = profile
            } label: { Text("Rename...", bundle: .module) }
        }
        exportBackupAction(profile)
        if profile.id != model.currentProfile.id {
            Button(role: .destructive) {
                deleteTarget = profile
            } label: { Text("Delete...", bundle: .module) }
        }
    }

    /// The built-in Default profile's actions: it cannot be renamed or
    /// deleted, but its setup (stores at the machine root plus the
    /// ~/.turbospark content, minus the shared model downloads) is exactly
    /// what a backup of it captures.
    @ViewBuilder
    private func defaultProfileActions() -> some View {
        exportBackupAction(UserProfileStore.defaultProfile)
    }

    /// Private vault exports require the profile to be the active, unlocked
    /// one. An inactive profile may be protected and its key is deliberately
    /// unavailable to this process.
    @ViewBuilder
    private func exportBackupAction(_ profile: UserProfile) -> some View {
        if profile.id == model.currentProfile.id {
            Button {
                encryptedBackupRequest = .export
            } label: { Text("Export Encrypted Backup...", bundle: .module) }
            .disabled(model.profileBackupInFlight)
            .settingsControl("Export Backup...", pane: .profiles, timing: .immediate)
            Button {
                exportTarget = profile
            } label: { Text("Export Open ZIP...", bundle: .module) }
            .disabled(model.profileBackupInFlight)
            .help(Text("Writes explicitly selected private data to a plaintext ZIP", bundle: .module))
        } else {
            Button {} label: { Text("Switch to Export", bundle: .module) }
                .disabled(true)
                .help(Text("Switch to this profile before exporting its credentials.", bundle: .module))
        }
    }

    private var addSection: some View {
        Section(header: Text("Add a Profile", bundle: .module)) {
            HStack {
                TextField(text: $newProfileName, prompt: Text("Profile name", bundle: .module)) {
                    Text("Profile name", bundle: .module)
                }
                .onSubmit(addProfile)
                // The one primary action on this pane: creating a profile.
                Button(action: addProfile) { Text("Add Profile", bundle: .module) }
                    .appAction(.primary)
                    .disabled(newProfileName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
            }
            Text("New profiles start empty: their own settings, chats, skills, and MCP servers, isolated from the Default profile and the shared ~/.turbospark folders.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
        }
            .settingsControl("Add a Profile", pane: .profiles, timing: .immediate)
    }

    private var securitySection: some View {
        Section(header: Text("Profile Privacy", bundle: .module)) {
            LabeledContent {
                vaultCoordinator.isProtected
                    ? Text("Protected", bundle: .module)
                    : Text("Encrypted locally", bundle: .module)
            } label: {
                Text("Private storage", bundle: .module)
            }
            if vaultCoordinator.isProtected {
                Toggle(isOn: Binding(
                    get: { vaultCoordinator.quickUnlockEnabled },
                    set: { updateQuickUnlock($0) }
                )) {
                    Text("Touch ID or Mac login", bundle: .module)
                }
                .disabled(securityBusy || !vaultCoordinator.canUseQuickUnlock)
                .help(vaultCoordinator.canUseQuickUnlock
                    ? Text("Touch ID or Mac login", bundle: .module)
                    : Text("Touch ID or Mac login authentication is not available.", bundle: .module))

                HStack {
                    Button {
                        securitySheet = .changePassphrase
                    } label: { Text("Change Passphrase...", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    // Opens a sheet that asks for the passphrase; nothing is
                    // removed until that sheet's own confirm.
                    Button(role: .destructive) {
                        securitySheet = .disableProtection
                    } label: { Text("Disable Protection...", bundle: .module) }
                    .appAction(.destructive, size: .small)
                    Spacer()
                    Button {
                        vaultCoordinator.lockNow()
                    } label: {
                        Label { Text("Lock Now", bundle: .module) } icon: { Image(systemName: "lock") }
                    }
                    .appAction(.secondary, size: .small)
                }
            } else {
                Button {
                    securitySheet = .protect
                } label: {
                    Label { Text("Protect This Profile...", bundle: .module) } icon: { Image(systemName: "lock.shield") }
                }
                .appAction(.secondary, size: .small)
            }

            Text("Protection covers chats, projects, private settings, managed attachments, and generated images. Models, skills, plugins, hooks, external project files, and tool executables remain ordinary files. FileVault is still recommended for whole-disk protection.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
        }
    }

    private func updateQuickUnlock(_ enabled: Bool) {
        securityBusy = true
        Task {
            do {
                try await vaultCoordinator.setQuickUnlock(enabled: enabled)
            } catch {
                model.showToast(error.localizedDescription, style: .error, duration: 8)
            }
            securityBusy = false
        }
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
                    if let url = model.pickEncryptedProfileBackup() {
                        encryptedBackupRequest = .restore(url)
                    }
                } label: { Text("Restore Encrypted Backup...", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(model.profileBackupInFlight)
                Button {
                    Task {
                        if let offer = await model.pickProfileBackupForImport() {
                            importOffer = offer
                        }
                    }
                } label: { Text("Import Legacy ZIP...", bundle: .module) }
                .appAction(.secondary, size: .small)
                .disabled(model.profileBackupInFlight)
                .help(Text("Restores a version 1 plaintext profile backup", bundle: .module))
                Text("Encrypted backups restore as locked profiles with a fresh identity. Legacy version 1 ZIP restore remains available for older archives. Keychain items are never included.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
        }
            .settingsControl("Restore a Backup", pane: .profiles, timing: .immediate)
    }

    private var notesSection: some View {
        Section {
            VStack(alignment: .leading, spacing: 6) {
                Text("Each profile is a folder under Application Support. Switching saves all work and relaunches the app; the Default profile keeps using the shared ~/.turbospark folders, and every other profile is self-contained. Downloaded models are shared by all profiles.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
                Button {
                    model.revealProfilesInFinder()
                } label: { Text("Reveal Profiles Folder in Finder", bundle: .module) }
                .appAction(.secondary, size: .small)
            }
        }
    }
}
