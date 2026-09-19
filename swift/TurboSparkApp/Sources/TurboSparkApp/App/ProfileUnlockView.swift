import SwiftUI

struct ProfileUnlockView: View {
    @ObservedObject var coordinator: ProfileVaultCoordinator
    @State private var passphrase = ""
    @State private var attemptedQuickUnlock = false

    var body: some View {
        VStack(spacing: 18) {
            Image(systemName: "lock.shield")
                .font(.system(size: 42, weight: .medium))
                .foregroundStyle(.secondary)
                .accessibilityHidden(true)
            VStack(spacing: 5) {
                Text("Unlock TurboSpark", bundle: .module)
                    .font(.title2.weight(.semibold))
                Text(coordinator.publicLabel)
                    .foregroundStyle(.secondary)
            }
            SecureField("Recovery passphrase", text: $passphrase)
                .textFieldStyle(.roundedBorder)
                .frame(maxWidth: 360)
                .onSubmit(unlock)
            HStack(spacing: 10) {
                Button(action: unlock) {
                    Text("Unlock", bundle: .module)
                }
                .keyboardShortcut(.defaultAction)
                .disabled(passphrase.isEmpty || coordinator.state == .unlocking)

                if coordinator.quickUnlockEnabled {
                    Button {
                        Task { await coordinator.unlockWithSystemAuthentication() }
                    } label: {
                        Label("Use Touch ID or Mac Login", systemImage: "touchid")
                    }
                    .disabled(coordinator.state == .unlocking)
                }
            }
            if case .error(let message) = coordinator.state {
                Text(message)
                    .font(.callout)
                    .foregroundStyle(.red)
                    .multilineTextAlignment(.center)
                    .frame(maxWidth: 440)
            }
            Text("Your recovery passphrase works without this Mac's Keychain. TurboSpark cannot recover it for you.", bundle: .module)
                .font(.caption)
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
