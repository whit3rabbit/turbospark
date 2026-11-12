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

    private var isRunning: Bool { model.server != nil }

    /// Read per body: the launch buttons need the CLI's absolute path
    /// anyway, so availability and the command come from one lookup.
    private var cliURL: URL? { AgentTerminalLaunch.findCLI() }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if isRunning {
                Text("These take effect the next time the server starts.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }

            field(
                Text("API key", bundle: .module),
                help: Text("Required on every route except /health, as x-api-key or a Bearer token.", bundle: .module)
            ) {
                HStack(spacing: 6) {
                    SecureField(text: $model.serverAPIKeyInput) { Text("API key", bundle: .module) }
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
                    AppIconButton("wand.and.stars", label: Text("Generate a random API key", bundle: .module)) {
                        model.serverAPIKeyInput = ServerAPIKeyGenerator.generate()
                    }
                    .buttonStyle(.borderless)
                    .disabled(isRunning)

                    APICopyButton(label: Text("Copy API key to clipboard", bundle: .module)) {
                        AppModel.serverAPIKey(from: model.serverAPIKeyInput)
                    }
                    .disabled(AppModel.serverAPIKey(from: model.serverAPIKeyInput) == nil)
                }
            }

            // Opt-in: the Keychain is not read or written, and macOS is not
            // asked for access, until this is on. A key stored by an earlier
            // version keeps it on (`ServerKeychain.resolveOptIn`).
            VStack(alignment: .leading, spacing: 4) {
                Toggle(isOn: Binding(
                    get: { model.serverKeyInKeychain },
                    set: { model.setServerKeyInKeychain($0) })
                ) {
                    Text("Remember this key in the Keychain", bundle: .module)
                }
                .toggleStyle(.checkbox)
                .disabled(isRunning)
                if model.serverKeyInKeychain {
                    Text("Stored in the Keychain, not settings.json.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                } else {
                    Text("The key is used until the app quits and is not saved.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }

            // **NOT DECORATION.** A loopback socket is reachable by every
            // process on this machine, which is not the same as private to
            // this app, and a key of nothing but spaces trims to none.
            if AppModel.serverAPIKey(from: model.serverAPIKeyInput) == nil {
                Label {
                    Text("With no key, any process on this machine can drive your models.", bundle: .module)
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                    .themedFont(.tiny)
                    .foregroundStyle(.appWarning)
            }

            field(
                Text("HF mirror endpoint", bundle: .module),
                help: Text("Base URL for Hugging Face downloads ($HF_ENDPOINT), e.g. https://hf-mirror.com for restricted regions.", bundle: .module)
            ) {
                TextField(text: $model.hfEndpointInput, prompt: Text(verbatim: "https://huggingface.co")) {
                    Text("HF mirror endpoint", bundle: .module)
                }
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
                Text("Memory guard tier", bundle: .module),
                help: Text("How much of the machine a model may commit when it loads, matching CLI --load-guard. The same value as Settings > Models & Storage, where Custom takes a byte ceiling.", bundle: .module)
            ) {
                // Built from the enum, not restated: a hardcoded four-entry
                // list omitted `.custom`, so after choosing Custom in Models &
                // Storage this picker rendered BLANK and any touch of it
                // discarded the ceiling (swift/CLAUDE.md Gotcha 22).
                Picker(selection: $model.runtimeOptions.loadGuard) {
                    ForEach(AppLoadGuardOption.allCases) { tier in
                        Text(LocalizedStringKey(tier.menuLabel), bundle: .module).tag(tier)
                    }
                } label: { Text("Memory guard tier", bundle: .module) }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
                .disabled(isRunning)
                .onChange(of: model.runtimeOptions.loadGuard) { _, _ in
                    model.persistSettingsDebounced()
                }
            }

            field(
                Text("Default reasoning effort", bundle: .module),
                help: Text("Default reasoning effort for incoming requests that do not specify reasoning_effort, matching CLI --reasoning.", bundle: .module)
            ) {
                // Same binding and same option source as the Engine pane.
                // Binding `$model.reasoning` directly skipped `setReasoning`,
                // so the per-model memory (state#96) was never written from
                // here, and hardcoding all five levels offered ones the loaded
                // checkpoint refuses (swift/CLAUDE.md Gotcha 9).
                Picker(selection: Binding(
                    get: { model.reasoning },
                    set: { model.setReasoning($0) }
                )) {
                    ForEach(model.session == nil
                            ? GenerateOptions.Reasoning.allCases
                            : model.availableReasoningLevels) { level in
                        Text(LocalizedStringKey(model.reasoningLabel(for: level)), bundle: .module).tag(level)
                    }
                } label: { Text("Default reasoning effort", bundle: .module) }
                .labelsHidden()
                .pickerStyle(.menu)
                .frame(maxWidth: .infinity)
                .disabled(isRunning)
            }

            codingAgents

        }
    }

    /// Launch Claude Code / Codex in Terminal against this server's
    /// configuration.
    ///
    /// **NOT DISABLED WHILE `isRunning`.** The launch routes through the
    /// `turbospark start <agent>` CLI launcher, which manages the
    /// background DAEMON: it reuses one serving the same model, restarts
    /// one serving a different model (announced in the Terminal output),
    /// or starts one when none runs. A running IN-APP server is a
    /// different process the CLI cannot see -- with one up and no daemon,
    /// launching starts a daemon for the model, which pays the model's
    /// memory a second time. The help text says where the launch lands;
    /// the Terminal output says which of the three happened.
    private var codingAgents: some View {
        field(
            Text("Coding agents", bundle: .module),
            help: Text("Opens Terminal and launches the agent against the background server; it is started or reused as needed.", bundle: .module)
        ) {
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Button {
                        launchAgent("claude")
                    } label: {
                        Label {
                            Text("Launch Claude Code", bundle: .module)
                        } icon: {
                            Image(systemName: "terminal")
                        }
                    }
                    .appAction(.secondary, size: .small)
                    .disabled(cliURL == nil)

                    Button {
                        launchAgent("codex")
                    } label: {
                        Label {
                            Text("Launch Codex", bundle: .module)
                        } icon: {
                            Image(systemName: "terminal")
                        }
                    }
                    .appAction(.secondary, size: .small)
                    .disabled(cliURL == nil)
                }
                if cliURL == nil {
                    Label {
                        Text("turbospark CLI not found", bundle: .module)
                    } icon: {
                        Image(systemName: "exclamationmark.triangle")
                    }
                        .themedFont(.tiny)
                        .foregroundStyle(.appWarning)
                }
            }
        }
    }

    /// The model argument is the first model the IN-APP server serves,
    /// under its catalog alias when one exists; with nothing served the
    /// flag is omitted and the CLI connects to whatever the daemon is
    /// already serving (or refuses, saying so, when none runs). The pinned
    /// port rides along so a daemon started BY the launch lands on the
    /// port this pane manages; 0 (automatic) is omitted, matching
    /// `ServerDaemonLaunch`.
    private func launchAgent(_ agent: String) {
        guard let cli = cliURL else { return }
        let modelArgument = model.serverModelRows.first?.displayName
        let command = TurboSparkAgent.cliLaunchCommand(
            for: agent,
            binaryPath: cli.path,
            modelArgument: modelArgument,
            port: model.serverPinnedPort == 0 ? nil : model.serverPinnedPort)
        guard var command else { return }
        // The daemon a launch starts (or the agent talking to a keyed one)
        // needs the key configured above; the CLI reads TURBOSPARK_API_KEY.
        if let key = AppModel.serverAPIKey(from: model.serverAPIKeyInput) {
            do {
                let keyFile = try AgentTerminalLaunch.writeAPIKeyFile(key)
                command = AgentTerminalLaunch.commandExportingAPIKey(command, keyFilePath: keyFile)
            } catch {
                model.showToast(
                    AppStrings.localized("Could not pass the API key to the launch: \(error.localizedDescription)"),
                    style: .error)
                return
            }
        }
        do {
            try AgentTerminalLaunch.openInTerminal(command: command)
            model.showToast(AppStrings.localized("Opening Terminal..."), style: .info)
        } catch {
            AgentTerminalLaunch.copyToClipboard(command)
            model.showToast(
                AppStrings.localized("Could not open Terminal: \(error.localizedDescription). The command was copied to the clipboard instead."),
                style: .error)
        }
    }

    private func field<Content: View>(
        _ label: Text, help: Text, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            label.themedFont(.tiny, weight: .medium)
            content()
            help
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

}
