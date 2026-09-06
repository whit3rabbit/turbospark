import SwiftUI

/// The knobs a person only reaches for once they know they need them.
///
/// Everything here is read at START time, so the fields are disabled while a
/// server is running rather than silently ignored -- an editable field that
/// changes nothing is worse than a greyed one that says why.
struct ServerAdvancedSettingsView: View {
    @ObservedObject var model: AppModel
    @State private var portText: String = ""

    private var isRunning: Bool { model.server != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if isRunning {
                Text("These take effect the next time the server starts.")
                    .font(.system(size: 10))
                    .foregroundStyle(.secondary)
            }

            field(
                "API key",
                help: "Required on every route except /health, as x-api-key or a Bearer token. Stored in the Keychain, not settings.json."
            ) {
                SecureField("", text: $model.serverAPIKeyInput)
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
                    .frame(width: 240)
                    .disabled(isRunning)
                    .onChange(of: model.serverAPIKeyInput) { _, _ in
                        model.persistSettingsDebounced()
                    }
            }

            // **NOT DECORATION.** A loopback socket is reachable by every
            // process on this machine, which is not the same as private to
            // this app, and a key of nothing but spaces trims to none.
            if AppModel.serverAPIKey(from: model.serverAPIKeyInput) == nil {
                Label(
                    "With no key, any process on this machine can drive your models.",
                    systemImage: "exclamationmark.triangle")
                    .font(.system(size: 10))
                    .foregroundStyle(.orange)
            }

            field(
                "Port",
                help: "0 lets the system choose. Pin one only if something else holds the number."
            ) {
                HStack(spacing: 6) {
                    TextField("", text: $portText)
                        .textFieldStyle(.roundedBorder)
                        .labelsHidden()
                        .frame(width: 90)
                        .disabled(isRunning)
                        .onChange(of: portText) { _, value in
                            model.serverPinnedPort = UInt16(value) ?? 0
                            model.persistSettingsDebounced()
                        }
                    if model.serverPinnedPort == 0 {
                        Text("automatic")
                            .font(.system(size: 10))
                            .foregroundStyle(.secondary)
                    }
                }
            }

            Divider()

            VStack(alignment: .leading, spacing: 4) {
                Text("What this server does not do")
                    .font(.system(size: 11, weight: .medium))
                // Stated rather than left to be discovered. Each of these is
                // a thing somebody will look for, and the honest answer is
                // cheaper than the search.
                bullet("Binds loopback only. There is no setting here to serve the network.")
                bullet("Serves one turn at a time per model. Concurrent requests queue.")
                bullet("Has no embeddings endpoint, because this engine has no embedding path.")
                bullet("Logs requests, never their bodies. Your prompts stay out of the console.")
            }
        }
        .onAppear {
            portText = model.serverPinnedPort == 0 ? "" : String(model.serverPinnedPort)
        }
    }

    private func field<Content: View>(
        _ label: String, help: String, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).font(.system(size: 11, weight: .medium))
            content()
            Text(help)
                .font(.system(size: 10))
                .foregroundStyle(.secondary)
        }
    }

    private func bullet(_ text: String) -> some View {
        HStack(alignment: .top, spacing: 6) {
            Text("-").font(.system(size: 10)).foregroundStyle(.secondary)
            Text(text).font(.system(size: 10)).foregroundStyle(.secondary)
        }
    }
}
