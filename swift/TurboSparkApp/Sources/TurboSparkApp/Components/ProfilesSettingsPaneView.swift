import SwiftUI

/// The Profiles settings pane: create, rename, delete, switch between, and
/// back up the users of this installation. Switching is a save-and-relaunch
/// (`AppModel.switchToProfile`); this pane says so in as many words rather
/// than surprising someone with an app restart, and both destructive doors
/// (delete and switch) confirm before acting. Backups run without either
/// door: export reads the profile's own folder and import restores into a
/// NEW identity, so neither one can damage the profile this run belongs to.
struct ProfilesSettingsPaneView: View {
    private enum SecuritySheet: String, Identifiable {
        case protect
        case changePassphrase
        case disableProtection

        var id: String { rawValue }
    }

    private enum ExactBackupSheet: String, Identifiable {
        case export
        case restore
        var id: String { rawValue }
    }

    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var vaultCoordinator = ProfileVaultCoordinator.shared

    @State private var newProfileName: String = ""
    @State private var renameTarget: UserProfile?
    @State private var renameText: String = ""
    @State private var deleteTarget: UserProfile?
    @State private var switchTarget: UserProfile?
    @State private var exportTarget: UserProfile?
    @State private var exportSelection: Set<String> = []
    @State private var importOffer: AppModel.ProfileBackupImportOffer?
    @State private var importName: String = ""
    @State private var securitySheet: SecuritySheet?
    @State private var currentPassphrase = ""
    @State private var newPassphrase = ""
    @State private var confirmPassphrase = ""
    @State private var enableQuickUnlock = true
    @State private var securityBusy = false
    @State private var securityError: String?
    @State private var exactBackupSheet: ExactBackupSheet?
    @State private var exactBackupURL: URL?
    @State private var exactBackupPassphrase = ""
    @State private var exactBackupConfirmation = ""
    @State private var exactRestoreName = "Imported Profile"
    @State private var exactBackupError: String?
    @State private var plaintextConfirmed = false
    @State private var exportAuthenticationPassphrase = ""
    @State private var exportError: String?

    var body: some View {
        Form {
            usersSection
            securitySection
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
        .sheet(item: $securitySheet) { mode in
            securitySheetView(mode)
        }
        .sheet(item: $exactBackupSheet) { mode in
            exactBackupSheetView(mode)
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
        if !profile.isProtected || profile.id == model.currentProfile.id {
            Button {
                renameTarget = profile
                renameText = profile.name
            } label: { Text("Rename...", bundle: .module) }
        }
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

    /// Private vault exports require the profile to be the active, unlocked
    /// one. An inactive profile may be protected and its key is deliberately
    /// unavailable to this process.
    @ViewBuilder
    private func exportBackupAction(_ profile: UserProfile) -> some View {
        if profile.id == model.currentProfile.id {
            Button {
                exactBackupPassphrase = ""
                exactBackupConfirmation = ""
                exactBackupError = nil
                exactBackupSheet = .export
            } label: { Text("Export Encrypted Backup...", bundle: .module) }
            .disabled(model.profileBackupInFlight)
            .settingsControl("Export Backup...", pane: .profiles, timing: .immediate)
            Button {
                exportSelection = OpenProfileExport.allCategoryIDs
                plaintextConfirmed = false
                exportAuthenticationPassphrase = ""
                exportError = nil
                exportTarget = profile
            } label: { Text("Export Open ZIP...", bundle: .module) }
            .disabled(model.profileBackupInFlight)
            .help("Writes explicitly selected private data to a plaintext ZIP")
        } else {
            Button("Switch to Export") {}
                .disabled(true)
        }
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

    private var securitySection: some View {
        Section(header: Text("Profile Privacy", bundle: .module)) {
            LabeledContent("Private storage", value: vaultCoordinator.isProtected ? "Protected" : "Encrypted locally")
            if vaultCoordinator.isProtected {
                Toggle(isOn: Binding(
                    get: { vaultCoordinator.quickUnlockEnabled },
                    set: updateQuickUnlock
                )) {
                    Text("Touch ID or Mac login", bundle: .module)
                }
                .disabled(securityBusy || !vaultCoordinator.canUseQuickUnlock)

                HStack {
                    Button {
                        beginSecuritySheet(.changePassphrase)
                    } label: { Text("Change Passphrase...", bundle: .module) }
                    Button(role: .destructive) {
                        beginSecuritySheet(.disableProtection)
                    } label: { Text("Disable Protection...", bundle: .module) }
                    Spacer()
                    Button {
                        vaultCoordinator.lockNow()
                    } label: {
                        Label("Lock Now", systemImage: "lock")
                    }
                }
            } else {
                Button {
                    beginSecuritySheet(.protect)
                } label: {
                    Label("Protect This Profile...", systemImage: "lock.shield")
                }
            }

            Text("Protection covers chats, projects, private settings, managed attachments, and generated images. Models, skills, plugins, hooks, external project files, and tool executables remain ordinary files. FileVault is still recommended for whole-disk protection.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
        }
    }

    private func beginSecuritySheet(_ mode: SecuritySheet) {
        currentPassphrase = ""
        newPassphrase = ""
        confirmPassphrase = ""
        enableQuickUnlock = vaultCoordinator.canUseQuickUnlock
        securityError = nil
        securitySheet = mode
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

    private func securitySheetView(_ mode: SecuritySheet) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(securitySheetTitle(mode))
                .font(theme.ui(.title3, weight: .semibold))

            if mode != .protect {
                SecureField("Current passphrase", text: $currentPassphrase)
                    .textFieldStyle(.roundedBorder)
            }
            if mode != .disableProtection {
                SecureField(mode == .protect ? "Recovery passphrase" : "New passphrase", text: $newPassphrase)
                    .textFieldStyle(.roundedBorder)
                SecureField("Confirm passphrase", text: $confirmPassphrase)
                    .textFieldStyle(.roundedBorder)
                Text("Use at least 15 characters. Spaces and pasted passphrases are allowed.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            if mode == .protect, vaultCoordinator.canUseQuickUnlock {
                Toggle("Enable Touch ID or Mac login", isOn: $enableQuickUnlock)
            }
            if mode == .disableProtection {
                Text("The vault remains encrypted, but its key will be stored locally without requiring authentication.", bundle: .module)
                    .font(theme.ui(.small))
                    .foregroundStyle(.appSecondary)
            }
            if let securityError {
                Text(securityError)
                    .font(theme.ui(.small))
                    .foregroundStyle(.red)
            }
            HStack {
                Spacer()
                Button {
                    securitySheet = nil
                } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
                Button(role: mode == .disableProtection ? .destructive : nil) {
                    applySecurityChange(mode)
                } label: {
                    Text(securityActionTitle(mode))
                }
                .keyboardShortcut(.defaultAction)
                .disabled(securityBusy || !securityFormIsValid(mode))
            }
        }
        .padding(20)
        .frame(width: 420)
    }

    private func securitySheetTitle(_ mode: SecuritySheet) -> String {
        switch mode {
        case .protect: return "Protect This Profile"
        case .changePassphrase: return "Change Recovery Passphrase"
        case .disableProtection: return "Disable Profile Protection"
        }
    }

    private func securityActionTitle(_ mode: SecuritySheet) -> String {
        switch mode {
        case .protect: return "Protect Profile"
        case .changePassphrase: return "Change Passphrase"
        case .disableProtection: return "Disable Protection"
        }
    }

    private func securityFormIsValid(_ mode: SecuritySheet) -> Bool {
        switch mode {
        case .protect:
            return newPassphrase.count >= ProfileVaultCrypto.minimumPassphraseLength
                && newPassphrase == confirmPassphrase
        case .changePassphrase:
            return !currentPassphrase.isEmpty
                && newPassphrase.count >= ProfileVaultCrypto.minimumPassphraseLength
                && newPassphrase == confirmPassphrase
        case .disableProtection:
            return !currentPassphrase.isEmpty
        }
    }

    private func applySecurityChange(_ mode: SecuritySheet) {
        securityBusy = true
        securityError = nil
        Task {
            do {
                switch mode {
                case .protect:
                    try await vaultCoordinator.protect(
                        passphrase: newPassphrase, quickUnlock: enableQuickUnlock)
                case .changePassphrase:
                    try await vaultCoordinator.changePassphrase(
                        current: currentPassphrase, replacement: newPassphrase)
                case .disableProtection:
                    try await vaultCoordinator.disableProtection(passphrase: currentPassphrase)
                }
                securitySheet = nil
                model.showToast("Profile privacy settings updated.")
            } catch {
                securityError = error.localizedDescription
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
                        exactBackupURL = url
                        exactBackupPassphrase = ""
                        exactBackupConfirmation = ""
                        exactRestoreName = "Imported Profile"
                        exactBackupError = nil
                        exactBackupSheet = .restore
                    }
                } label: { Text("Restore Encrypted Backup...", bundle: .module) }
                .disabled(model.profileBackupInFlight)
                Button {
                    Task {
                        if let offer = await model.pickProfileBackupForImport() {
                            importOffer = offer
                            importName = offer.suggestedName
                        }
                    }
                } label: { Text("Import Legacy ZIP...", bundle: .module) }
                .disabled(model.profileBackupInFlight)
                .help("Restores a version 1 plaintext profile backup")
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

    /// The portable export is intentionally separate from exact backup. Its
    /// contents are readable by any ZIP tool and therefore plaintext.
    private func exportSheet(_ profile: UserProfile) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Export a Plaintext ZIP", bundle: .module)
                .font(theme.ui(.title3, weight: .semibold))
                .settingsControl(
                    "What to Include in This Backup", pane: .profiles, timing: .immediate)
            Text("Backing up \"\(profile.name)\"", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            ForEach(OpenProfileExport.categories) { category in
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
                    exportSelection = OpenProfileExport.allCategoryIDs
                } label: { Text("Select All", bundle: .module) }
                .buttonStyle(.link)
                Button {
                    exportSelection = []
                } label: { Text("Clear All", bundle: .module) }
                .buttonStyle(.link)
                Spacer()
            }
            Toggle("I understand this ZIP is not encrypted", isOn: $plaintextConfirmed)
            if vaultCoordinator.isProtected,
               !hasRecentProfileAuthentication {
                SecureField("Recovery passphrase", text: $exportAuthenticationPassphrase)
                    .textFieldStyle(.roundedBorder)
            }
            Text("The ZIP contains only the selected categories. It never fetches remote URLs or includes Keychain secrets, models, skills, plugins, hooks, tools, external repositories, or referenced external files.", bundle: .module)
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            if let exportError {
                Text(exportError)
                    .font(theme.ui(.small))
                    .foregroundStyle(.red)
            }
            HStack {
                Spacer()
                Button {
                    exportTarget = nil
                } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
                Button {
                    let selection = exportSelection
                    Task {
                        if vaultCoordinator.isProtected, !hasRecentProfileAuthentication {
                            guard await vaultCoordinator.authenticateRecently(
                                with: exportAuthenticationPassphrase) else {
                                exportError = "Authentication failed."
                                return
                            }
                        }
                        exportTarget = nil
                        model.runOpenProfileExport(included: selection)
                    }
                } label: { Text("Continue", bundle: .module) }
                .keyboardShortcut(.defaultAction)
                .disabled(!plaintextConfirmed)
            }
        }
        .padding(20)
        .frame(width: 380)
    }

    private var hasRecentProfileAuthentication: Bool {
        guard let date = vaultCoordinator.lastAuthenticationAt else { return false }
        return Date().timeIntervalSince(date) < 300
    }

    private func exactBackupSheetView(_ mode: ExactBackupSheet) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(mode == .export ? "Export Encrypted Backup" : "Restore Encrypted Backup")
                .font(theme.ui(.title3, weight: .semibold))
            if mode == .restore {
                TextField("Profile name", text: $exactRestoreName)
                    .textFieldStyle(.roundedBorder)
            }
            SecureField(mode == .export ? "Export password" : "Backup password",
                        text: $exactBackupPassphrase)
                .textFieldStyle(.roundedBorder)
            if mode == .export {
                SecureField("Confirm export password", text: $exactBackupConfirmation)
                    .textFieldStyle(.roundedBorder)
            }
            Text(mode == .export
                 ? "Use at least 15 characters. The backup contains an authenticated SQLCipher snapshot and encrypted managed assets. The device Keychain item is excluded."
                 : "The restored profile remains protected by this password. Its private name and content stay encrypted.")
                .font(theme.ui(.small))
                .foregroundStyle(.appSecondary)
            if let exactBackupError {
                Text(exactBackupError)
                    .font(theme.ui(.small))
                    .foregroundStyle(.red)
            }
            HStack {
                Spacer()
                Button {
                    exactBackupSheet = nil
                    exactBackupURL = nil
                } label: { Text("Cancel", bundle: .module) }
                .keyboardShortcut(.cancelAction)
                Button {
                    if mode == .export {
                        exactBackupSheet = nil
                        model.runEncryptedProfileBackupExport(passphrase: exactBackupPassphrase)
                    } else if let exactBackupURL,
                              model.importEncryptedProfileBackup(
                                exactBackupURL,
                                named: exactRestoreName,
                                passphrase: exactBackupPassphrase) {
                        exactBackupSheet = nil
                        self.exactBackupURL = nil
                    }
                } label: {
                    Text(mode == .export ? "Choose Destination..." : "Restore")
                }
                .keyboardShortcut(.defaultAction)
                .disabled(!exactBackupFormIsValid(mode))
            }
        }
        .padding(20)
        .frame(width: 430)
    }

    private func exactBackupFormIsValid(_ mode: ExactBackupSheet) -> Bool {
        guard exactBackupPassphrase.count >= ProfileVaultCrypto.minimumPassphraseLength else {
            return false
        }
        if mode == .export { return exactBackupPassphrase == exactBackupConfirmation }
        return !exactRestoreName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            && exactBackupURL != nil
    }

    /// Category labels as literal keys: existing catalog entries ("Settings",
    /// "Skills", ...) are reused, the rest carry their own keys. The id is
    /// the manifest's category id, never shown raw.
    @ViewBuilder
    private func categoryHeader(_ id: String) -> some View {
        switch id {
        case "settings": Text("Settings", bundle: .module)
        case "chats": Text("Chat history", bundle: .module)
        case "generated-images": Text("Generated images", bundle: .module)
        case "attachments": Text("Attachments", bundle: .module)
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
