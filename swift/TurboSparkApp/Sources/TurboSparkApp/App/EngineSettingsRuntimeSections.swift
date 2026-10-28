import SwiftUI
import TurboSpark

/// The Engine pane's runtime sections: reasoning, guardrails, speculation,
/// KV-cache width and the in-process server. One computed property per
/// `Section` for the type-checker reason spelled out above
/// `EngineSettingsPaneView.body`; a separate file only to keep each one
/// readable.
extension EngineSettingsPaneView {
    var reasoningEffortSection: some View {
        Section(header: Text("Thinking & Reasoning Effort", bundle: .module)) {
            Picker(selection: Binding(
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
                    Text(LocalizedStringKey(model.reasoningLabel(for: level)), bundle: .module).tag(level)
                }
            } label: { Text("Default Reasoning Level", bundle: .module) }
            .settingsControl("Default Reasoning Level", pane: .engine, timing: .nextTurn)
            .pickerStyle(.menu)

            Text("Controls internal chain-of-thought depth. The accepted levels belong to each checkpoint's own chat template, not to this app: Qwen 3.8 tops out at Extra High and refuses High, while gpt-oss is the other way round. A level a model cannot express is clamped to its nearest one on load, and your choice is remembered per model.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }

    var guardrailsSection: some View {
        Section(header: Text("Forge Tool-Call Guardrails", bundle: .module)) {
            Picker(selection: $model.guardrailsMode) {
                ForEach(AppGuardrailsMode.allCases) { mode in
                    Text(LocalizedStringKey(mode.label), bundle: .module).tag(mode)
                }
            } label: { Text("Guardrails Mode", bundle: .module) }
            .settingsControl("Guardrails Mode", pane: .engine, timing: .nextTurn)
            .pickerStyle(.menu)
            .onChange(of: model.guardrailsMode) { _, _ in
                model.persistSettings()
            }

            Text(LocalizedStringKey(model.guardrailsMode.descriptionText), bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }

    var speculationSection: some View {
        Section(header: Text("Speculative Decoding", bundle: .module)) {
            Picker(selection: $model.runtimeOptions.speculation) {
                ForEach(AppSpeculationOption.allCases) { opt in
                    Text(LocalizedStringKey(opt.menuLabel), bundle: .module).tag(opt)
                }
            } label: { Text("Speculation Mode", bundle: .module) }
            .settingsControl("Speculation Mode", pane: .engine, timing: .modelReload)
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.speculation) { _, _ in
                model.persistSettingsDebounced()
            }

            if model.runtimeOptions.speculation != .off {
                Picker(selection: $model.runtimeOptions.speculativeDrafter) {
                    ForEach(AppSpeculativeDrafterOption.allCases) { drafter in
                        Text(LocalizedStringKey(drafter.menuLabel), bundle: .module).tag(drafter)
                    }
                } label: { Text("Speculative Drafter", bundle: .module) }
                .settingsControl("Speculative Drafter", pane: .engine, timing: .modelReload)
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
    var kvBitsSection: some View {
        Section(header: Text("TurboQuant KV-Cache Quantization", bundle: .module)) {
            Picker(selection: $model.runtimeOptions.kvBits) {
                ForEach(AppKvBitsOption.allCases) { opt in
                    Text(LocalizedStringKey(opt.menuLabel), bundle: .module).tag(opt)
                }
            } label: { Text("KV-Cache Width", bundle: .module) }
            .settingsControl("KV-Cache Width", pane: .engine, timing: .modelReload)
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.kvBits) { _, _ in
                model.persistSettingsDebounced()
            }

            if let info = model.session?.info {
                LabeledContent {
                    // The engine's own token for "not quantized" is the one
                    // word in this readout; the widths are numbers.
                    Group {
                        if info.kvBits == "off" {
                            Text("Off", bundle: .module)
                        } else {
                            Text(verbatim: info.kvBits)
                        }
                    }
                    .themedFont(.tiny, systemDesign: .monospaced)
                    .foregroundStyle(.appSecondary)
                } label: {
                    Text("Resolved", bundle: .module)
                }
            }

            Text("Quantizes the attention KV cache to shrink its memory footprint at longer contexts, at a small, width-dependent quality cost (docs/TRUBOQUANT.md). Auto asks for 4-bit -- the width with the smallest measured quality impact -- only on checkpoints whose head dimension and layer layout support it, and stays off on every other install. Takes effect on the next model load.", bundle: .module)
            .themedFont(.tiny)
            .foregroundStyle(.appSecondary)
        }
    }

    var anePrefillSection: some View {
        Section(header: Text("ANE+Steel Prefill Split", bundle: .module)) {
            Picker(selection: $model.runtimeOptions.anePrefill) {
                ForEach(AppAnePrefillOption.allCases) { opt in
                    Text(LocalizedStringKey(opt.menuLabel), bundle: .module).tag(opt)
                }
            } label: { Text("Prefill Mode", bundle: .module) }
            .settingsControl("Prefill Mode", pane: .engine, timing: .modelReload)
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.anePrefill) { _, _ in
                model.persistSettingsDebounced()
            }

            if let info = model.session?.info {
                LabeledContent {
                    Group {
                        if let ane = info.anePrefill {
                            if ane.active {
                                HStack(spacing: 4) {
                                    Text("Active", bundle: .module)
                                    Text(verbatim: "(ANE+Steel)")
                                }
                            } else if let reason = ane.reason {
                                Text(verbatim: "Off (\(reason))")
                            } else {
                                Text("Off", bundle: .module)
                            }
                        } else {
                            Text("Off", bundle: .module)
                        }
                    }
                    .themedFont(.tiny, systemDesign: .monospaced)
                    .foregroundStyle(.appSecondary)
                } label: {
                    Text("Resolved", bundle: .module)
                }
            }

            Text("Splits prompt prefill projections across the Apple Neural Engine and Metal GPU for dense Qwen models on Apple Silicon M4 or newer. Auto enables the split when supported. Takes effect on the next model load.", bundle: .module)
            .themedFont(.tiny)
            .foregroundStyle(.appSecondary)
        }
    }

    var inProcessServerSection: some View {
        Section(header: Text("In-Process Server", bundle: .module)) {
            LabeledContent {
                Group {
                    if let name = model.selectedSystemPrompt?.name {
                        Text(name)
                    } else {
                        Text("None", bundle: .module)
                    }
                }
                .themedCode(.small)
                .foregroundStyle(.appSecondary)
            } label: {
                Text("Server Start Prompt", bundle: .module)
            }

            HStack {
                Toggle(isOn: Binding(
                    get: { model.server != nil },
                    set: { $0 ? model.startServer() : model.stopServer() }
                )) {
                    Text("Enable server", bundle: .module)
                }
                .settingsControl("Enable server", pane: .engine, timing: .immediate)
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

                LabeledContent {
                    Text(rows.address)
                        .themedCode(.small)
                        .foregroundStyle(.appSecondary)
                        .textSelection(.enabled)
                } label: {
                    Text("Address", bundle: .module)
                }

                LabeledContent {
                    // `authLabel` is one of two fixed English values
                    // ("API key required", "none"), both catalog keys.
                    Text(LocalizedStringKey(rows.authLabel), bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(rows.authIsWarning ? AnyShapeStyle(.appWarning) : AnyShapeStyle(.appSecondary))
                } label: {
                    Text("Auth", bundle: .module)
                }
            }

            SecureField(text: $model.serverAPIKeyInput, prompt: Text("API key (optional)", bundle: .module)) {
                Text("API key (optional)", bundle: .module)
            }
            .settingsControl("API key (optional)", pane: .engine, timing: .action)
            .disabled(model.server != nil)
            // Without this the key only reached disk on an unrelated save, so a
            // crash before one relaunched the auto-started server with a stale key.
            .onChange(of: model.serverAPIKeyInput) { _, _ in model.persistSettingsDebounced() }

            Text("Serves the currently loaded model over OpenAI- and Anthropic-compatible HTTP endpoints on loopback, sharing the same engine this app's chat uses -- not a second copy of the model. Loading a different model, or unloading, stops the server. Loopback keeps it off the network and NOT off this machine: without an API key, any process running here can reach it.", bundle: .module)
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
        }
    }
}
