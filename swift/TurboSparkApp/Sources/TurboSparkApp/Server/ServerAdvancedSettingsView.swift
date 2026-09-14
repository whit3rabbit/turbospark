import AppKit
import SwiftUI
import TurboSpark

/// The knobs a person only reaches for once they know they need them.
///
/// Every field but one is read at START time, so it is disabled while a
/// server is running rather than silently ignored -- an editable field that
/// changes nothing is worse than a greyed one that says why. The HF mirror
/// endpoint is the exception: it ALSO feeds `TurboSparkCatalog`'s own HF
/// client, which every install/probe/browse call reads outside server
/// context entirely, so it stays enabled and applies live
/// (`HfEndpointResolution`, `swift/docs/SWIFT_SETTINGS_AUDIT.md`).
struct ServerAdvancedSettingsView: View {
    @ObservedObject var model: AppModel
    @State private var copiedKey: Bool = false

    private var isRunning: Bool { model.server != nil }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if isRunning {
                Text("These take effect the next time the server starts.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }

            field(
                "API key",
                help: "Required on every route except /health, as x-api-key or a Bearer token. Stored in the Keychain, not settings.json."
            ) {
                HStack(spacing: 6) {
                    SecureField("", text: $model.serverAPIKeyInput)
                        .textFieldStyle(.roundedBorder)
                        .labelsHidden()
                        .frame(maxWidth: .infinity)
                        .disabled(isRunning)
                        .onChange(of: model.serverAPIKeyInput) { _, _ in
                            model.persistSettingsDebounced()
                        }

                    // Generate is disabled with the field it fills, because
                    // the key is read at START; copy stays live while the
                    // server runs, since handing the running key to a client
                    // is the reason to have it on this pane at all.
                    Button {
                        model.serverAPIKeyInput = ServerAPIKeyGenerator.generate()
                    } label: {
                        Image(systemName: "wand.and.stars")
                    }
                    .buttonStyle(.borderless)
                    .disabled(isRunning)
                    .help("Generate a random key")
                    .accessibilityLabel("Generate a random API key")

                    Button {
                        guard let key = AppModel.serverAPIKey(from: model.serverAPIKeyInput) else { return }
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(key, forType: .string)
                        copiedKey = true
                        DispatchQueue.main.asyncAfter(deadline: .now() + 1.5) {
                            copiedKey = false
                        }
                    } label: {
                        Image(systemName: copiedKey ? "checkmark" : "doc.on.doc")
                    }
                    .buttonStyle(.borderless)
                    .disabled(AppModel.serverAPIKey(from: model.serverAPIKeyInput) == nil)
                    .help("Copy API key to clipboard")
                    .accessibilityLabel("Copy API key")
                    .accessibilityValue(copiedKey ? "Copied" : "")
                }
            }

            // **NOT DECORATION.** A loopback socket is reachable by every
            // process on this machine, which is not the same as private to
            // this app, and a key of nothing but spaces trims to none.
            if AppModel.serverAPIKey(from: model.serverAPIKeyInput) == nil {
                Label(
                    "With no key, any process on this machine can drive your models.",
                    systemImage: "exclamationmark.triangle")
                    .themedFont(.tiny)
                    .foregroundStyle(.orange)
            }

            field(
                "Embedding model",
                help: "Path or alias to an encoder model (e.g. snowflake-arctic-embed-m) to enable /v1/embeddings."
            ) {
                TextField("snowflake-arctic-embed-m or /path/to/encoder", text: $model.serverEmbeddingModelInput)
                    .textFieldStyle(.roundedBorder)
                    .labelsHidden()
                    .frame(maxWidth: .infinity)
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
                    .frame(maxWidth: .infinity)
                    // NOT `.disabled(isRunning)`, unlike its neighbors: the
                    // catalog effect below applies regardless of whether a
                    // server happens to be running, and disabling it here
                    // would just narrow the window in which the two editors
                    // of this setting disagree rather than closing it.
                    .onChange(of: model.hfEndpointInput) { _, newValue in
                        model.persistSettingsDebounced()
                        // Unlike this pane's other fields, this ONE also
                        // feeds a consumer that has nothing to do with
                        // whether a server is running: `TurboSparkCatalog`'s
                        // own HF client, which every model install, probe
                        // and browse call goes through outside the server
                        // entirely. Persisting alone left that client on the
                        // stale endpoint until the next app launch, while
                        // `HfAuthTokenCardView`'s own mirror-endpoint editor
                        // applied it live -- two editors of the same
                        // setting disagreeing about when it takes effect
                        // (`swift/docs/SWIFT_SETTINGS_AUDIT.md`).
                        try? TurboSparkCatalog.setHfEndpoint(HfEndpointResolution.effectiveEndpoint(from: newValue))
                    }
            }

            field(
                "Memory guard tier",
                help: "How much of the machine a model may commit when it loads, matching CLI --load-guard. The same value as Settings > Models & Storage, where Custom takes a byte ceiling."
            ) {
                // Built from the enum, not restated: a hardcoded four-entry
                // list omitted `.custom`, so after choosing Custom in Models &
                // Storage this picker rendered BLANK and any touch of it
                // discarded the ceiling (swift/CLAUDE.md Gotcha 22).
                Picker("", selection: $model.runtimeOptions.loadGuard) {
                    ForEach(AppLoadGuardOption.allCases) { tier in
                        Text(tier.menuLabel).tag(tier)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
                .disabled(isRunning)
                .onChange(of: model.runtimeOptions.loadGuard) { _, _ in
                    model.persistSettingsDebounced()
                }
            }

            field(
                "Default reasoning effort",
                help: "Default reasoning effort for incoming requests that do not specify reasoning_effort, matching CLI --reasoning."
            ) {
                // Same binding and same option source as the Engine pane.
                // Binding `$model.reasoning` directly skipped `setReasoning`,
                // so the per-model memory (state#96) was never written from
                // here, and hardcoding all five levels offered ones the loaded
                // checkpoint refuses (swift/CLAUDE.md Gotcha 9).
                Picker("", selection: Binding(
                    get: { model.reasoning },
                    set: { model.setReasoning($0) }
                )) {
                    ForEach(model.session == nil
                            ? GenerateOptions.Reasoning.allCases
                            : model.availableReasoningLevels) { level in
                        Text(model.reasoningLabel(for: level)).tag(level)
                    }
                }
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
                .disabled(isRunning)
            }


        }
    }

    private func field<Content: View>(
        _ label: String, help: String, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            Text(label).themedFont(.tiny, weight: .medium)
            content()
            Text(help)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

}
