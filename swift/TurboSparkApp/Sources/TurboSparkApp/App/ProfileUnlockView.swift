import SwiftUI

/// The lock screen shown while the encrypted profile is sealed.
struct ProfileUnlockView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var coordinator: ProfileVaultCoordinator
    @State private var passphrase = ""
    @State private var attemptedQuickUnlock = false
    @FocusState private var isPassphraseFocused: Bool

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "lock.shield")
                .themedFont(.display, weight: .medium)
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            VStack(spacing: 5) {
                Text("Unlock TurboSpark", bundle: .module)
                    .themedFont(.title2, weight: .semibold)
                // The vault writes "Protected Profile" as its default public
                // label; any other value is a name stored in the manifest.
                Group {
                    if coordinator.publicLabel == "Protected Profile" {
                        Text("Protected Profile", bundle: .module)
                    } else {
                        Text(verbatim: coordinator.publicLabel)
                    }
                }
                .foregroundStyle(.appSecondary)
            }
            SecureField(text: $passphrase, prompt: Text("Recovery passphrase", bundle: .module)) {
                Text("Recovery passphrase", bundle: .module)
            }
            .textFieldStyle(.roundedBorder)
            .focused($isPassphraseFocused)
            .frame(maxWidth: 360)
            .onSubmit(unlock)
            HStack(spacing: 10) {
                Button(action: unlock) {
                    Text("Unlock", bundle: .module)
                }
                .appAction(.primary)
                .keyboardShortcut(.defaultAction)
                .disabled(passphrase.isEmpty || coordinator.state == .unlocking)

                if coordinator.quickUnlockEnabled {
                    Button {
                        Task { await coordinator.unlockWithSystemAuthentication() }
                    } label: {
                        Label { Text("Use Touch ID or Mac Login", bundle: .module) } icon: { Image(systemName: "touchid") }
                    }
                    .appAction(.secondary)
                    .disabled(coordinator.state == .unlocking)
                }
            }
            if case .error(let message) = coordinator.state {
                HStack(spacing: 6) {
                    Image(systemName: "exclamationmark.triangle.fill")
                        .foregroundStyle(AppAttention.color)
                    Text(verbatim: message)
                        .themedFont(.callout)
                        .foregroundStyle(.appSecondary)
                }
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
            }
            Text("Your recovery passphrase works without this Mac's Keychain. TurboSpark cannot recover it for you.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
        }
        .padding(36)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.appPage)
        .appThemed()
        .onAppear {
            if !coordinator.quickUnlockEnabled {
                isPassphraseFocused = true
            }
        }
        .task {
            guard coordinator.quickUnlockEnabled, !attemptedQuickUnlock else { return }
            attemptedQuickUnlock = true
            await coordinator.unlockWithSystemAuthentication()
        }
    }

    private func unlock() {
        let submitted = passphrase
        passphrase = ""
        Task { await coordinator.unlock(passphrase: submitted) }
    }
}
