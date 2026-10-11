import SwiftUI

// The Profiles pane's backup sheets: the encrypted round trip and the
// legacy ZIP import. See `ProfileSettingsSheets.swift` for the shared
// pieces and why each sheet owns its own state.

// MARK: - Encrypted backup

@MainActor
struct ProfileEncryptedBackupSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel
    let request: ProfileEncryptedBackupRequest
    @ObservedObject private var vaultCoordinator = ProfileVaultCoordinator.shared

    @State private var passphrase = ""
    @State private var confirmation = ""
    // A default the person can edit; it becomes the new profile's name, so
    // it is written in the UI language.
    @State private var restoreName = AppStrings.localized("Imported Profile")
    @State private var authenticationPassphrase = ""
    @State private var error: String?

    private var isExport: Bool {
        if case .export = request { return true }
        return false
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            title
                .themedFont(.title3, weight: .semibold)
            if !isExport {
                TextField(text: $restoreName, prompt: Text("Profile name", bundle: .module)) {
                    Text("Profile name", bundle: .module)
                }
                .textFieldStyle(.roundedBorder)
            }
            ProfileSheetParts.passphraseField(
                isExport ? Text("Export password", bundle: .module) : Text("Backup password", bundle: .module),
                text: $passphrase)
            if isExport {
                ProfileSheetParts.passphraseField(Text("Confirm export password", bundle: .module), text: $confirmation)
                // Same recent-authentication gate as the plaintext ZIP.
                if vaultCoordinator.isProtected, !ProfileSheetParts.hasRecentAuthentication(vaultCoordinator) {
                    ProfileSheetParts.passphraseField(Text("Recovery passphrase", bundle: .module), text: $authenticationPassphrase)
                }
            }
            explanation
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            if let error {
                Text(error)
                    .themedFont(.small)
                    .foregroundStyle(.red)
            }
            ProfileSheetParts.buttons(
                confirm: isExport
                    ? Text("Choose Destination...", bundle: .module)
                    : Text("Restore", bundle: .module),
                disabled: !formIsValid,
                cancel: { dismiss() },
                action: run)
        }
        .padding(20)
        .frame(minWidth: 430, idealWidth: 430)
    }

    private var title: Text {
        isExport
            ? Text("Export Encrypted Backup", bundle: .module)
            : Text("Restore Encrypted Backup", bundle: .module)
    }

    private var explanation: Text {
        isExport
            ? Text("Use at least 15 characters. The backup contains an authenticated SQLCipher snapshot and encrypted managed assets. The device Keychain item is excluded.", bundle: .module)
            : Text("The restored profile remains protected by this password. Its private name and content stay encrypted.", bundle: .module)
    }

    private var formIsValid: Bool {
        guard passphrase.count >= ProfileVaultCrypto.minimumPassphraseLength else { return false }
        switch request {
        case .export:
            return passphrase == confirmation
        case .restore:
            return !restoreName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        }
    }

    private func run() {
        switch request {
        case .export:
            let exportPassphrase = passphrase
            Task {
                if vaultCoordinator.isProtected, !ProfileSheetParts.hasRecentAuthentication(vaultCoordinator) {
                    guard await vaultCoordinator.authenticateRecently(
                        with: authenticationPassphrase) else {
                        error = AppStrings.localized("Authentication failed.")
                        return
                    }
                }
                dismiss()
                model.runEncryptedProfileBackupExport(passphrase: exportPassphrase)
            }
        case .restore(let url):
            if model.importEncryptedProfileBackup(url, named: restoreName, passphrase: passphrase) {
                dismiss()
            }
        }
    }
}

// MARK: - Legacy ZIP import

@MainActor
struct ProfileBackupImportSheet: View {
    @Environment(\.dismiss) private var dismiss
    @ObservedObject var model: AppModel
    let offer: AppModel.ProfileBackupImportOffer
    @State private var name: String

    init(model: AppModel, offer: AppModel.ProfileBackupImportOffer) {
        self.model = model
        self.offer = offer
        _name = State(initialValue: offer.suggestedName)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text("Import Profile Backup", bundle: .module)
                .themedFont(.title3, weight: .semibold)
                .settingsControl("Import Profile Backup", pane: .profiles, timing: .immediate)
            Text("Backup of \"\(offer.manifest.profileName)\" exported \(exportedAt) with app version \(appVersion); \(offer.manifest.contents.count) items inside.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            if offer.manifest.isDefault {
                Text("This is a Default-profile backup: it imports as a new named profile, not the built-in one.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            TextField(text: $name, prompt: Text("Profile name", bundle: .module)) {
                Text("Profile name", bundle: .module)
            }
            .textFieldStyle(.roundedBorder)
            ProfileSheetParts.buttons(
                confirm: Text("Import", bundle: .module),
                disabled: name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
                cancel: { dismiss() },
                action: {
                    if model.importProfileBackup(offer, named: name) {
                        dismiss()
                    }
                })
        }
        .padding(20)
        .frame(minWidth: 380, idealWidth: 380)
    }

    // Formatted to Strings first: the catalog key takes them as %@.
    private var exportedAt: String {
        offer.manifest.exportedAt.formatted(date: .abbreviated, time: .shortened)
    }

    private var appVersion: String {
        offer.manifest.appVersion.isEmpty ? AppStrings.localized("unknown") : offer.manifest.appVersion
    }
}
