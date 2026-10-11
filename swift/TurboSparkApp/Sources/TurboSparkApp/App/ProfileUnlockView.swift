import SwiftUI

/// The lock screen shown while the encrypted profile is sealed.
struct ProfileUnlockView: View {
    @ObservedObject var coordinator: ProfileVaultCoordinator
    @State private var passphrase = ""
    @State private var attemptedQuickUnlock = false

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "lock.shield")
                .themedFont(.display, weight: .medium)
                .foregroundStyle(.secondary)
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
                Text(verbatim: message)
                    .themedFont(.callout)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)
            }
            Text("Your recovery passphrase works without this Mac's Keychain. TurboSpark cannot recover it for you.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.secondary)
                .multilineTextAlignment(.center)
                .frame(maxWidth: 440)
        }
        .padding(36)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(.background)
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
