import SwiftUI

/// Settings > Network > Image search: the Brave Search API key behind the
/// `image_search` agent tool.
///
/// Storage follows the app's credential rule (swift/AGENTS.md): the key is
/// held in memory until the app quits, and the Keychain is written or read
/// only after the user turns on "Remember this key". Nothing here runs at
/// launch, and the key never goes into settings.json.
@MainActor
struct ImageSearchSettingsSectionView: View {
    @State private var draft = ""
    @State private var keyIsSet = ImageSearchCredentials.hasSessionKey
    @State private var rememberInKeychain = ImageSearchCredentials.isKeychainEnabled()
    @State private var failure: String?

    var body: some View {
        Section {
            Text("Lets the agent find reference photos with Brave Image Search. It needs your own Brave Search API key, which is sent only to Brave.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 6) {
                SecureField(text: $draft) { Text("Brave Search API key", bundle: .module) }
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
                    .frame(maxWidth: .infinity)
                    .onSubmit(save)
                Button(action: save) {
                    Text("Save", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .disabled(draft.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                Button(action: clear) {
                    Text("Clear", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .disabled(!keyIsSet && !rememberInKeychain)
            }
            .settingsControl("Brave Search API key", pane: .network, timing: .nextTurn)

            if keyIsSet {
                Label {
                    Text("A key is set for this session.", bundle: .module)
                } icon: {
                    Image(systemName: "checkmark.circle")
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }

            VStack(alignment: .leading, spacing: 4) {
                Toggle(isOn: Binding(get: { rememberInKeychain }, set: setRemember)) {
                    Text("Remember this key in the Keychain", bundle: .module)
                }
                .toggleStyle(.checkbox)
                .settingsControl("Remember this key in the Keychain", pane: .network, timing: .immediate)
                if rememberInKeychain {
                    Text("Stored in the Keychain, not settings.json.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                } else {
                    Text("The key is used until the app quits and is not saved.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }

            if let failure {
                Label {
                    Text(verbatim: failure)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .themedFont(.small)
                .foregroundStyle(.appWarning)
            }
        } header: {
            Text("Image search", bundle: .module)
        }
    }

    private func save() {
        let key = draft.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !key.isEmpty else { return }
        ImageSearchCredentials.setSessionKey(key)
        keyIsSet = true
        draft = ""
        failure = nil
        if rememberInKeychain { persistToKeychain(key) }
    }

    private func clear() {
        ImageSearchCredentials.setSessionKey(nil)
        keyIsSet = false
        draft = ""
        failure = nil
        // Only an opted-in user has a Keychain item to remove, so an
        // opted-out user's Clear never touches the Keychain.
        if rememberInKeychain { ImageSearchCredentials.deleteFromKeychain() }
    }

    private func setRemember(_ enabled: Bool) {
        guard enabled != rememberInKeychain else { return }
        failure = nil
        if enabled {
            ImageSearchCredentials.setKeychainEnabled(true)
            rememberInKeychain = true
            // Write the key already in memory, or the one typed but not yet
            // saved; with neither, the next Save writes it.
            let typed = draft.trimmingCharacters(in: .whitespacesAndNewlines)
            if let key = typed.isEmpty ? ImageSearchCredentials.resolveKey(keychainRead: { nil }) : typed {
                ImageSearchCredentials.setSessionKey(key)
                keyIsSet = true
                draft = ""
                persistToKeychain(key)
            }
        } else {
            ImageSearchCredentials.deleteFromKeychain()
            ImageSearchCredentials.setKeychainEnabled(false)
            rememberInKeychain = false
        }
    }

    private func persistToKeychain(_ key: String) {
        let status = ImageSearchCredentials.saveToKeychain(key)
        guard status != errSecSuccess else { return }
        // Not remembered: leave the opt-in off so a later launch does not
        // read an item that was never written.
        ImageSearchCredentials.setKeychainEnabled(false)
        rememberInKeychain = false
        failure = AppStrings.localized("Could not save the image search key to Keychain.")
            + " " + TypeSafeKeychain.describe(status: status)
    }
}
