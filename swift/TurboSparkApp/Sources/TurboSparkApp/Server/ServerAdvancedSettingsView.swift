import SwiftUI
import TurboSpark

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

            field(
                "Embedding model",
                help: "Path or alias to an encoder model (e.g. snowflake-arctic-embed-m) to enable /v1/embeddings."
            ) {
                TextField("snowflake-arctic-embed-m or /path/to/encoder", text: $model.serverEmbeddingModelInput)
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
                    .frame(width: 280)
                    .disabled(isRunning)
                    .onChange(of: model.serverEmbeddingModelInput) { _, _ in
                        model.persistSettingsDebounced()
                    }
            }

            field(
                "HF mirror endpoint",
                help: "Base URL for Hugging Face downloads ($HF_ENDPOINT), e.g. https://hf-mirror.com for restricted regions."
            ) {
                TextField("https://huggingface.co", text: $model.hfEndpointInput)
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
                    .frame(width: 280)
                    .disabled(isRunning)
                    .onChange(of: model.hfEndpointInput) { _, _ in
                        model.persistSettingsDebounced()
                    }
            }

            field(
                "Memory guard tier",
                help: "Model load memory reservation tier (safe/balanced, strict, relaxed, off), matching CLI --memory-guard."
            ) {
                Picker("", selection: $model.runtimeOptions.loadGuard) {
                    Text("Safe / Balanced").tag(AppLoadGuardOption.balanced)
                    Text("Relaxed (Default)").tag(AppLoadGuardOption.relaxed)
                    Text("Strict").tag(AppLoadGuardOption.strict)
                    Text("Off").tag(AppLoadGuardOption.off)
                }
                .pickerStyle(.menu)
                .frame(width: 200)
                .disabled(isRunning)
                .onChange(of: model.runtimeOptions.loadGuard) { _, _ in
                    model.persistSettingsDebounced()
                }
            }

            field(
                "Default reasoning effort",
                help: "Default reasoning effort for incoming requests that do not specify reasoning_effort, matching CLI --reasoning."
            ) {
                Picker("", selection: $model.reasoning) {
                    Text("Off (Fastest)").tag(GenerateOptions.Reasoning.off)
                    Text("Low").tag(GenerateOptions.Reasoning.low)
                    Text("Medium (Balanced)").tag(GenerateOptions.Reasoning.medium)
                    Text("High").tag(GenerateOptions.Reasoning.high)
                    Text("Extra High (Max Thorough)").tag(GenerateOptions.Reasoning.xhigh)
                }
                .pickerStyle(.menu)
                .frame(width: 220)
                .disabled(isRunning)
                .onChange(of: model.reasoning) { _, _ in
                    model.persistSettingsDebounced()
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
                bullet("Serves text embeddings when an embedding model is attached.")
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
