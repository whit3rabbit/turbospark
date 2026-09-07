import SwiftUI
import TurboSpark

/// Settings pane for engine configuration, sampling parameters, reasoning level, guardrails, and in-process server.
struct EngineSettingsPaneView: View {
    @ObservedObject var model: AppModel

    /// **FIVE PROPERTIES RATHER THAN ONE EXPRESSION.** A `Form` holding
    /// all five `Section`s inline is ONE expression to the type checker,
    /// and it exceeded the solver budget on the SDK 14 toolchain that CI
    /// builds the app bundle with: "unable to type-check this expression
    /// in reasonable time". It compiles on a newer one, which is why
    /// nobody developing here saw it. Splitting is the fix the diagnostic
    /// itself asks for, and it is cheap insurance on any toolchain -- a
    /// SwiftUI container that grows a section at a time walks into this
    /// eventually.
    var body: some View {
        Form {
            systemPromptSection
            generationDefaultsSection
            advancedGenerationSection
            reasoningEffortSection
            guardrailsSection
            speculationSection
            kvBitsSection
            inProcessServerSection
        }
        .formStyle(.grouped)
        .padding(16)
    }

    /// A SEPARATE computed property, like every sibling section, for the
    /// type-checker reason spelled out above `body`.
    private var systemPromptSection: some View {
        Section("Default System Prompt") {
            VStack(alignment: .leading, spacing: 6) {
                TextEditor(text: $model.defaultSystemPrompt)
                    .font(.system(.body, design: .monospaced))
                    .frame(minHeight: 100)
                    .onChange(of: model.defaultSystemPrompt) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text(
                    "Sent as the first system message of every conversation, ahead of any "
                    + "project rules. A chat with its own system prompt uses that instead. "
                    + "Leave empty for none."
                )
                .font(.caption)
                .foregroundStyle(.secondary)
                Text("\(model.defaultSystemPrompt.count) characters")
                    .font(.caption)
                    .monospacedDigit()
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var generationDefaultsSection: some View {
        Section("Generation Defaults") {
            HStack {
                Text("Temperature")
                Spacer()
                Slider(value: $model.temperature, in: 0.0...1.5, step: 0.05)
                    .frame(width: 160)
                    .onChange(of: model.temperature) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text(String(format: "%.2f", model.temperature))
                    .monospacedDigit()
                    .frame(width: 40, alignment: .trailing)
            }

            HStack {
                Text("Top-P Sampling")
                Spacer()
                Toggle("", isOn: $model.topPEnabled)
                    .labelsHidden()
                    .onChange(of: model.topPEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Slider(value: $model.topP, in: 0.1...1.0, step: 0.05)
                    .frame(width: 160)
                    .disabled(!model.topPEnabled)
                    .onChange(of: model.topP) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text(String(format: "%.2f", model.topP))
                    .monospacedDigit()
                    .frame(width: 40, alignment: .trailing)
                    .foregroundStyle(model.topPEnabled ? .primary : .secondary)
            }
        }
    }

    /// The sampling and engine-load knobs that used to live ONLY in the
    /// diagnostics Inspector, which is not a surface a person reads as
    /// "settings". Same `AppModel` properties both places, so the two stay
    /// in sync by construction; a separate computed property like every
    /// sibling section for the type-checker reason spelled out above `body`.
    private var advancedGenerationSection: some View {
        Section("Advanced Generation") {
            HStack {
                Text("Context Window")
                Spacer()
                TextField("auto", value: $model.maxContextTokens, format: .number.grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
                    .onChange(of: model.maxContextTokens) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text(model.maxContextTokens == 0 ? "auto" : "tokens")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 44, alignment: .leading)
            }

            HStack {
                Text("Max New Tokens")
                Spacer()
                TextField("2048", value: $model.maxNewTokens, format: .number.grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
                    .onChange(of: model.maxNewTokens) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text("")
                    .frame(width: 44)
            }

            HStack {
                Text("Top-K Sampling")
                Spacer()
                Toggle("", isOn: $model.topKEnabled)
                    .labelsHidden()
                    .onChange(of: model.topKEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                TextField("64", value: $model.topK, format: .number.grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
                    .disabled(!model.topKEnabled)
                    .onChange(of: model.topK) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text("")
                    .frame(width: 44)
            }

            HStack {
                Text("Repetition Penalty")
                Spacer()
                Toggle("", isOn: $model.repetitionPenaltyEnabled)
                    .labelsHidden()
                    .onChange(of: model.repetitionPenaltyEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                TextField("1.0", value: $model.repetitionPenalty, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
                    .disabled(!model.repetitionPenaltyEnabled)
                    .onChange(of: model.repetitionPenalty) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text("")
                    .frame(width: 44)
            }

            HStack {
                Text("Fixed Seed")
                Spacer()
                Toggle("", isOn: $model.seedEnabled)
                    .labelsHidden()
                    .onChange(of: model.seedEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                TextField("0", value: $model.seed, format: .number.grouping(.never))
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
                    .disabled(!model.seedEnabled)
                    .onChange(of: model.seed) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text("")
                    .frame(width: 44)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Stop Sequences")
                TextField("comma-separated", text: $model.stopSequences)
                    .textFieldStyle(.roundedBorder)
                    .onChange(of: model.stopSequences) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text("Generation stops when the model emits any of these comma-separated strings.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Picker("Power Profile", selection: $model.runtimeOptions.powerProfile) {
                ForEach(AppPowerProfileOption.allCases) { profile in
                    Text(profile.menuLabel).tag(profile)
                }
            }
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.powerProfile) { _, _ in
                model.persistSettingsDebounced()
            }

            HStack {
                Text("Output Rate Cap")
                Spacer()
                TextField("Uncapped", value: $model.runtimeOptions.maxTokensPerSec, format: .number)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 110)
                    .onChange(of: model.runtimeOptions.maxTokensPerSec) { _, _ in
                        model.persistSettingsDebounced()
                    }
                Text("tok/s")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(width: 44, alignment: .leading)
            }

            Picker("Memory Load Guard", selection: $model.runtimeOptions.loadGuard) {
                ForEach(AppLoadGuardOption.allCases) { tier in
                    Text(tier.menuLabel).tag(tier)
                }
            }
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.loadGuard) { _, _ in
                model.persistSettingsDebounced()
            }

            Text("Context 0 means the checkpoint's trained context, capped by memory. These mirror the Inspector's engine options and edit the same values.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var reasoningEffortSection: some View {
        Section("Thinking & Reasoning Effort") {
            Picker("Default Reasoning Level", selection: Binding(
                get: { model.reasoning },
                set: { model.setReasoning($0) }
            )) {
                // With a model loaded this is that checkpoint's own set.
                // With none it is the union, because this is the default
                // a FUTURE model inherits and clamping it to nothing
                // would leave a picker with one entry.
                ForEach(model.session == nil
                        ? GenerateOptions.Reasoning.allCases
                        : model.availableReasoningLevels) { level in
                    Text(model.reasoningLabel(for: level)).tag(level)
                }
            }
            .pickerStyle(.menu)

            Text("Controls internal chain-of-thought depth. The accepted levels belong to each checkpoint's own chat template, not to this app: Qwen 3.8 tops out at Extra High and refuses High, while gpt-oss is the other way round. A level a model cannot express is clamped to its nearest one on load, and your choice is remembered per model.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var guardrailsSection: some View {
        Section("Forge Tool-Call Guardrails") {
            Picker("Guardrails Mode", selection: $model.guardrailsMode) {
                ForEach(AppGuardrailsMode.allCases) { mode in
                    Text(mode.label).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .onChange(of: model.guardrailsMode) { _, _ in
                model.persistSettings()
            }

            Text(model.guardrailsMode.descriptionText)
                .font(.caption)
                .foregroundStyle(.secondary)
        }
    }

    private var speculationSection: some View {
        Section("Speculative Decoding") {
            Picker("Speculation Mode", selection: $model.runtimeOptions.speculation) {
                ForEach(AppSpeculationOption.allCases) { opt in
                    Text(opt.menuLabel).tag(opt)
                }
            }
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.speculation) { _, _ in
                model.persistSettingsDebounced()
            }

            if model.runtimeOptions.speculation != .off {
                Picker("Speculative Drafter", selection: $model.runtimeOptions.speculativeDrafter) {
                    ForEach(AppSpeculativeDrafterOption.allCases) { drafter in
                        Text(drafter.menuLabel).tag(drafter)
                    }
                }
                .pickerStyle(.menu)
                .onChange(of: model.runtimeOptions.speculativeDrafter) { _, _ in
                    model.persistSettingsDebounced()
                }
            }
        }
    }

    /// **READS BACK WHAT THE LOADED SESSION ACTUALLY RESOLVED, RATHER THAN
    /// RESTATING THE SETTING.** `Auto` can and does mean "off" on an install
    /// whose `head_dim` or layer mask does not qualify
    /// (`ModelFeatureDescriptor.supportsKvQuant`), and a picker that only
    /// showed the SETTING would look identical whether or not the request
    /// actually reached the engine -- the exact "badge that cannot fail"
    /// shape `swift/CLAUDE.md` Gotcha 22 catalogs. `session.info.kvBits` is
    /// the RESOLVED value (`"off"`, `"4"`, `"3.5 (K3/V4)"`, ...), reported
    /// once at open, so this row can only ever say what actually happened.
    private var kvBitsSection: some View {
        Section("TurboQuant KV-Cache Quantization") {
            Picker("KV-Cache Width", selection: $model.runtimeOptions.kvBits) {
                ForEach(AppKvBitsOption.allCases) { opt in
                    Text(opt.menuLabel).tag(opt)
                }
            }
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.kvBits) { _, _ in
                model.persistSettingsDebounced()
            }

            if let info = model.session?.info {
                HStack {
                    Text("Resolved")
                    Spacer()
                    Text(info.kvBits)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                }
            }

            Text(
                "Quantizes the attention KV cache to shrink its memory footprint at longer "
                    + "contexts, at a small, width-dependent quality cost (docs/TRUBOQUANT.md). "
                    + "Auto asks for 4-bit -- the width with the smallest measured quality "
                    + "impact -- only on checkpoints whose head dimension and layer layout "
                    + "support it, and stays off on every other install. Takes effect on the "
                    + "next model load."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }

    private var inProcessServerSection: some View {
        Section("In-Process Server") {
            HStack {
                Toggle("Enable server", isOn: Binding(
                    get: { model.server != nil },
                    set: { $0 ? model.startServer() : model.stopServer() }
                ))
                .disabled(model.session == nil || model.serverBusy)

                if model.serverBusy {
                    ProgressView()
                        .controlSize(.small)
                        .padding(.leading, 4)
                }
            }

            // Both rows are READ BACK from the running server rather than
            // restated here. The address used to be half asserted (a
            // `127.0.0.1` literal beside the real port) and the auth state
            // was not shown at all, so a key of nothing but spaces trims
            // to empty, starts an unauthenticated server, and looked
            // identical to a key that took.
            if let info = model.serverInfo {
                let rows = ServerStatusRows(info: info, guardrails: model.serverStartedGuardrails)

                HStack {
                    Text("Address")
                    Spacer()
                    Text(rows.address)
                        .font(.caption.monospaced())
                        .foregroundStyle(.secondary)
                        .textSelection(.enabled)
                }

                HStack {
                    Text("Auth")
                    Spacer()
                    Text(rows.authLabel)
                        .font(.caption)
                        .foregroundStyle(rows.authIsWarning ? Color.orange : Color.secondary)
                }
            }

            SecureField("API key (optional)", text: $model.serverAPIKeyInput)
                .disabled(model.server != nil)

            Text(
                "Serves the currently loaded model over OpenAI- and Anthropic-compatible "
                    + "HTTP endpoints on loopback, sharing the same engine this app's chat "
                    + "uses -- not a second copy of the model. Loading a different model, "
                    + "or unloading, stops the server. Loopback keeps it off the network "
                    + "and NOT off this machine: without an API key, any process running "
                    + "here can reach it."
            )
            .font(.caption)
            .foregroundStyle(.secondary)
        }
    }
}
