import SwiftUI

// The Profiles pane's sheets. Each one owns its own form state, so every
// presentation starts clean (the pane used to reset a dozen shared @State
// fields by hand before opening each sheet) and the pane file stays
// readable. They close through `dismiss`, which clears the pane's `item`.

/// Which privacy change the security sheet is making.
enum ProfileSecuritySheetMode: String, Identifiable {
    case protect
    case changePassphrase
    case disableProtection

    var id: String { rawValue }
}

/// An encrypted backup round trip: export the current profile, or restore
/// the archive at `url` into a new one.
enum ProfileEncryptedBackupRequest: Identifiable {
    case export
    case restore(URL)

    var id: String {
        switch self {
        case .export: return "export"
        case .restore(let url): return "restore:" + url.path
        }
    }
}

/// The pieces every Profiles sheet shares, so the sheets in both files keep
/// one look for their fields and their Cancel/confirm pair.
@MainActor
enum ProfileSheetParts {
    /// Private exports of a protected profile require a recent passphrase or
    /// biometric check: an unlocked-but-unattended profile must not be
    /// exportable with a password the person at the keyboard just made up.
    static func hasRecentAuthentication(_ coordinator: ProfileVaultCoordinator) -> Bool {
        guard let date = coordinator.lastAuthenticationAt else { return false }
        return Date().timeIntervalSince(date) < 300
    }

    /// A secure field whose placeholder and accessible name are the same
    /// catalog string.
    static func passphraseField(_ title: Text, text: Binding<String>) -> some View {
        SecureField(text: text, prompt: title) { title }
            .textFieldStyle(.roundedBorder)
    }

    /// The Cancel and confirm pair at the foot of every sheet.
    static func buttons(
        confirm: Text,
        role: AppActionRole = .primary,
        buttonRole: ButtonRole? = nil,
        disabled: Bool,
        help: Text? = nil,
        cancel: @escaping () -> Void,
        action: @escaping () -> Void
    ) -> some View {
        HStack {
            Spacer()
            Button(action: cancel) { Text("Cancel", bundle: .module) }
                .appAction(.secondary)
                .keyboardShortcut(.cancelAction)
            Button(role: buttonRole, action: action) { confirm }
                .appAction(role)
                .keyboardShortcut(.defaultAction)
                .disabled(disabled)
                .help(help ?? confirm)
        }
    }
}

// MARK: - Privacy

@MainActor
struct ProfileSecuritySheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel
    let mode: ProfileSecuritySheetMode
    @ObservedObject private var vaultCoordinator = ProfileVaultCoordinator.shared

    @State private var currentPassphrase = ""
    @State private var newPassphrase = ""
    @State private var confirmPassphrase = ""
    // Seeded from the coordinator on appear: the toggle only shows when
    // quick unlock is available, and `protect` must not ask for it otherwise.
    @State private var enableQuickUnlock = true
    @State private var busy = false
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            title
                .themedFont(.title3, weight: .semibold)

            if mode != .protect {
                ProfileSheetParts.passphraseField(Text("Current passphrase", bundle: .module), text: $currentPassphrase)
            }
            if mode != .disableProtection {
                ProfileSheetParts.passphraseField(
                    mode == .protect
                        ? Text("Recovery passphrase", bundle: .module)
                        : Text("New passphrase", bundle: .module),
                    text: $newPassphrase)
                ProfileSheetParts.passphraseField(Text("Confirm passphrase", bundle: .module), text: $confirmPassphrase)
                Text("Use at least 15 characters. Spaces and pasted passphrases are allowed.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            if mode == .protect, vaultCoordinator.canUseQuickUnlock {
                Toggle(isOn: $enableQuickUnlock) {
                    Text("Enable Touch ID or Mac login", bundle: .module)
                }
            }
            if mode == .disableProtection {
                Text("The vault remains encrypted, but its key will be stored locally without requiring authentication.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            if let error {
                Text(error)
                    .themedFont(.small)
                    .foregroundStyle(.appDanger)
            }
            ProfileSheetParts.buttons(
                confirm: actionTitle,
                role: mode == .disableProtection ? .destructive : .primary,
                buttonRole: mode == .disableProtection ? .destructive : nil,
                disabled: busy || !formIsValid,
                cancel: { dismiss() },
                action: apply)
        }
        .padding(20)
        .frame(minWidth: 420, idealWidth: 420)
        .onAppear { enableQuickUnlock = vaultCoordinator.canUseQuickUnlock }
    }

    private var title: Text {
        switch mode {
        case .protect: return Text("Protect This Profile", bundle: .module)
        case .changePassphrase: return Text("Change Recovery Passphrase", bundle: .module)
        case .disableProtection: return Text("Disable Profile Protection", bundle: .module)
        }
    }

    private var actionTitle: Text {
        switch mode {
        case .protect: return Text("Protect Profile", bundle: .module)
        case .changePassphrase: return Text("Change Passphrase", bundle: .module)
        case .disableProtection: return Text("Disable Protection", bundle: .module)
        }
    }

    private var formIsValid: Bool {
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

    private func apply() {
        busy = true
        error = nil
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
                dismiss()
                model.showToast(AppStrings.localized("Profile privacy settings updated."))
            } catch {
                self.error = error.localizedDescription
            }
            busy = false
        }
    }
}

// MARK: - Rename

@MainActor
struct ProfileRenameSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel
    let profile: UserProfile
    @State private var name: String

    init(model: AppModel, profile: UserProfile) {
        self.model = model
        self.profile = profile
        _name = State(initialValue: profile.name)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Rename Profile", bundle: .module)
                .themedFont(.title3, weight: .semibold)
                .settingsControl("Rename Profile", pane: .profiles, timing: .immediate)
            TextField(text: $name, prompt: Text("Profile name", bundle: .module)) {
                Text("Profile name", bundle: .module)
            }
            .textFieldStyle(.roundedBorder)
            ProfileSheetParts.buttons(
                confirm: Text("Rename", bundle: .module),
                disabled: name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                cancel: { dismiss() },
                action: {
                    model.renameProfile(profile, to: name)
                    dismiss()
                })
        }
        .padding(20)
        .frame(minWidth: 340, idealWidth: 340)
    }
}

// MARK: - Open (plaintext) export

/// The portable export is intentionally separate from exact backup. Its
/// contents are readable by any ZIP tool and therefore plaintext.
@MainActor
struct ProfileOpenExportSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel
    let profile: UserProfile
    @ObservedObject private var vaultCoordinator = ProfileVaultCoordinator.shared

    @State private var selection: Set<String> = OpenProfileExport.allCategoryIDs
    @State private var plaintextConfirmed = false
    @State private var authenticationPassphrase = ""
    @State private var error: String?

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Export a Plaintext ZIP", bundle: .module)
                .themedFont(.title3, weight: .semibold)
                .settingsControl(
                    "What to Include in This Backup", pane: .profiles, timing: .immediate)
            Text("Backing up \"\(profile.name)\"", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            ForEach(OpenProfileExport.categories) { category in
                Toggle(isOn: Binding(
                    get: { selection.contains(category.id) },
                    set: { isOn in
                        if isOn {
                            selection.insert(category.id)
                        } else {
                            selection.remove(category.id)
                        }
                    }
                )) {
                    Self.categoryHeader(category.id)
                }
            }
            HStack(spacing: 12) {
                Button {
                    selection = OpenProfileExport.allCategoryIDs
                } label: { Text("Select All", bundle: .module) }
                .buttonStyle(.link)
                .appLinkTint()
                Button {
                    selection = []
                } label: { Text("Clear All", bundle: .module) }
                .buttonStyle(.link)
                .appLinkTint()
                Spacer()
            }
            Toggle(isOn: $plaintextConfirmed) {
                Text("I understand this ZIP is not encrypted", bundle: .module)
            }
            if vaultCoordinator.isProtected, !ProfileSheetParts.hasRecentAuthentication(vaultCoordinator) {
                ProfileSheetParts.passphraseField(Text("Recovery passphrase", bundle: .module), text: $authenticationPassphrase)
            }
            Text("The ZIP contains only the selected categories. It never fetches remote URLs or includes Keychain secrets, models, skills, plugins, hooks, tools, external repositories, or referenced external files.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            if let error {
                Text(error)
                    .themedFont(.small)
                    .foregroundStyle(.appDanger)
            }
            ProfileSheetParts.buttons(
                confirm: Text("Continue", bundle: .module),
                disabled: !plaintextConfirmed,
                cancel: { dismiss() },
                action: export)
        }
        .padding(20)
        .frame(minWidth: 380, idealWidth: 380)
    }

    private func export() {
        let included = selection
        Task {
            if vaultCoordinator.isProtected, !ProfileSheetParts.hasRecentAuthentication(vaultCoordinator) {
                guard await vaultCoordinator.authenticateRecently(
                    with: authenticationPassphrase) else {
                    error = AppStrings.localized("Authentication failed.")
                    return
                }
            }
            dismiss()
            model.runOpenProfileExport(included: included)
        }
    }

    /// Category labels as literal keys: existing catalog entries ("Settings",
    /// "Skills", ...) are reused, the rest carry their own keys. The id is
    /// the manifest's category id, never shown raw.
    @ViewBuilder
    static func categoryHeader(_ id: String) -> some View {
        switch id {
        case "settings": Text("Settings", bundle: .module)
        case "chats": Text("Chat history", bundle: .module)
        case "generated-images": Text("Generated images", bundle: .module)
        case "attachments": Text("Attachments", bundle: .module)
        case "audio": Text("Audio", bundle: .module)
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
}
