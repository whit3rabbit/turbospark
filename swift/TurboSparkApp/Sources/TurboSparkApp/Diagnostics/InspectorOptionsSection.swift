import AppKit
import SwiftUI
import TurboSpark

extension InspectorView {
    var modelSection: some View {
        Section("Model") {
            Picker("Model", selection: Binding(
                get: { model.selected?.alias ?? "" },
                set: { alias in
                    guard let m = model.installed.first(where: { $0.alias == alias }) else { return }
                    model.selectModel(m)
                }
            )) {
                if model.installed.isEmpty {
                    Text("No models installed", bundle: .module).tag("")
                }
                ForEach(model.installed) { m in
                    let visuals = ModelFamilyVisuals.resolve(alias: m.alias, family: m.family, name: m.alias)
                    Label("\(m.alias) (\(visuals.family))", systemImage: visuals.iconSystemName)
                        .tag(m.alias)
                }
            }

            LabeledContent("Path") {
                HStack(spacing: 6) {
                    Text(model.modelPathText)
                        .themedFont(.small)
                        .truncationMode(.middle)
                        .lineLimit(1)
                        .foregroundStyle(.secondary)
                        .help(model.modelPathText)
                    Button {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    } label: {
                        Label("Copy model path", systemImage: "doc.on.doc")
                            .labelStyle(.iconOnly)
                    }
                    .buttonStyle(.borderless)
                    .help("Copy model path")
                    .accessibilityLabel("Copy model path")
                    .accessibilityHint("Copies the on-disk model path to the clipboard")
                }
            }

            Button("Choose Model Folder…") {
                ModelLocationPicker.choose(for: model)
            }
            .disabled(model.opening || model.isRunning)
            .accessibilityHint("Opens a folder picker to choose a model directory")

            if model.canUnloadModel {
                HStack {
                    Button("Reload Model", action: model.reloadModel)
                    Spacer()
                    Button("Unload Model", action: model.unloadModel)
                }
            } else if model.canLoadModel {
                Button("Load Model", action: model.loadModel)
            }

            if let selected = model.selected {
                let visuals = ModelFamilyVisuals.resolve(alias: selected.alias, family: selected.family, name: selected.alias)
                LabeledContent("Installed size") {
                    Text(MetricFormat.storage(selected.installBytes))
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Family") {
                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .themedFont(.tiny)
                            .foregroundStyle(visuals.accentColor)
                            .accessibilityHidden(true)
                        Text(visuals.family)
                    }
                    .themedFont(.small)
                    .foregroundStyle(.secondary)
                    .help("Architecture: \(visuals.family)")
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    var memoryAndPowerSection: some View {
        Section("Open & Architecture Options") {
            // The context window lives in InspectorEssentialsSection's
            // `ContextLadderPicker`, the one place that sets it.
            LabeledContent("Cache Slots") {
                Picker("Slots", selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Text(AppRuntimeOptions.slotsLabel(for: slots)).tag(slots)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                // .labelsHidden() strips the visible label, so VoiceOver would
                // hear "menu, 16" with no idea what 16 means.
                .accessibilityLabel("Cache slots")
            }

            LabeledContent("Power Profile") {
                Picker("Power", selection: $model.runtimeOptions.powerProfile) {
                    ForEach(AppPowerProfileOption.allCases) { profile in
                        Text(profile.menuLabel).tag(profile)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .accessibilityLabel("Power profile")
            }

            LabeledContent("Speculation") {
                Picker("Speculation", selection: $model.runtimeOptions.speculation) {
                    ForEach(AppSpeculationOption.allCases) { spec in
                        Text(spec.menuLabel).tag(spec)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .accessibilityLabel("Speculation")
            }

            if model.runtimeOptions.speculation != .off {
                LabeledContent("Drafter") {
                    Picker("Drafter", selection: $model.runtimeOptions.speculativeDrafter) {
                        ForEach(AppSpeculativeDrafterOption.allCases) { drafter in
                            Text(drafter.menuLabel).tag(drafter)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .fixedSize()
                    .accessibilityLabel("Speculation drafter")
                }
            }

            // `.labelsHidden()` is load-bearing, not cosmetic: on macOS a
            // TextField's title is a VISIBLE LABEL rather than a placeholder,
            // so "Uncapped" was drawn beside the field and clipped to "Un-"
            // by the inspector's width. Every other control in this section
            // already hides its label for the same reason.
            LabeledContent("Rate Cap") {
                HStack(spacing: 6) {
                    TextField("Uncapped", value: $model.runtimeOptions.maxTokensPerSec, format: .number)
                        .labelsHidden()
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 70)
                        .multilineTextAlignment(.trailing)
                        .accessibilityLabel("Rate cap in tokens per second")
                        .accessibilityValue(rateCapIsUncapped ? "Uncapped" : "\(model.runtimeOptions.maxTokensPerSec) tokens per second")
                    // Zero is the uncapped sentinel, and a bare "0" next to
                    // "tok/s" reads as a cap of zero tokens per second.
                    Text(rateCapIsUncapped ? "uncapped" : "tok/s")
                        .themedFont(.small)
                        .foregroundStyle(.secondary)
                        .accessibilityHidden(true)
                }
                .fixedSize()
            }
            .help("Tokens per second ceiling. 0 leaves generation uncapped.")

            if model.canReloadModel {
                Button("Apply Options & Reload") {
                    model.reloadModel()
                }
                .controlSize(.small)
            }
        }
        .disabled(model.isRunning)
    }

    /// Zero is the engine's "no cap" sentinel for `maxTokensPerSec`.
    private var rateCapIsUncapped: Bool {
        model.runtimeOptions.maxTokensPerSec <= 0
    }

    /// The RAW knobs, kept as the expert surface beside the named presets in
    /// Settings > Safety and Steering. What is set here becomes the implicit
    /// "Custom" preset, so a configuration made before presets existed keeps
    /// working.
    var steeringSection: some View {
        Section("Directional Steering") {
            // A family that does not dispatch the edit REFUSES a vector at
            // open, so the knobs below can only produce a failed load there.
            // Disabled with the engine's own reason rather than hidden
            // (swift/CLAUDE.md Gotchas 23 and 33).
            if model.steeringFamilySupported == false {
                Label(
                    model.info?.steering.reason
                        ?? "This model's family does not dispatch the steering edit.",
                    systemImage: "exclamationmark.triangle"
                )
                .themedFont(.small)
                .foregroundStyle(Color.secondary)
            }

            // **STEERING RESOLVES ONCE, AT OPEN.** Editing a field here does
            // nothing to the model already loaded, and until this note
            // existed the app said nothing about that -- a knob that appears
            // to work and does not.
            if model.steeringNeedsReload {
                HStack {
                    Label("Not applied to the loaded model", systemImage: "arrow.clockwise")
                        .themedFont(.small)
                        .foregroundStyle(Color.orange)
                    Spacer()
                    Button("Reload") { model.reloadForSteering() }
                        .themedFont(.small)
                        .disabled(model.selected == nil || model.generating || model.opening)
                }
            }

            TextField("Control vector path (.gguf)", text: Binding(
                get: { model.runtimeOptions.steeringPath ?? "" },
                set: { model.runtimeOptions.steeringPath = $0.isEmpty ? nil : $0 }
            ))
            .textFieldStyle(.roundedBorder)
            .themedFont(.small)
            .accessibilityLabel("Control vector path")
            .accessibilityHint("Path to a GGUF control vector file. Leave blank to disable directional steering.")

            if let path = model.runtimeOptions.steeringPath, !path.isEmpty {
                LabeledContent("Mode") {
                    Picker("Steering Mode", selection: $model.runtimeOptions.steeringMode) {
                        ForEach(AppSteeringModeOption.allCases) { mode in
                            Text(mode.menuLabel).tag(mode)
                        }
                    }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .fixedSize()
                    .accessibilityLabel("Steering mode")
                }

                LabeledContent("Scale") {
                    HStack(spacing: 8) {
                        Slider(value: $model.runtimeOptions.steeringScale, in: 0...3, step: 0.1)
                            .accessibilityLabel("Steering scale")
                            .accessibilityValue(String(format: "%.1f", model.runtimeOptions.steeringScale))
                        Text(model.runtimeOptions.steeringScale, format: .number.precision(.fractionLength(1)))
                            .monospacedDigit()
                            .frame(width: 32, alignment: .trailing)
                    }
                }

                LabeledContent("Layers (Start:End)") {
                    TextField("e.g. 10:30 (all if blank)", text: $model.runtimeOptions.steeringLayers)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 120)
                        .accessibilityLabel("Steering layers")
                        .accessibilityHint("Comma-separated start:end layer range; blank means all layers")
                }

                if model.runtimeOptions.steeringMode == .clamp {
                    LabeledContent("Clamp Target") {
                        // No `in:` range on the Stepper: unlike `steeringScale`
                        // (0...3, documented, already a slider), this is a raw,
                        // control-vector-dependent activation value with no
                        // engine-defined bound, so a hardcoded min/max here
                        // would silently clamp values a real vector needs.
                        HStack(spacing: 4) {
                            TextField("Target", value: $model.runtimeOptions.steeringTarget, format: .number)
                                .textFieldStyle(.roundedBorder)
                                .frame(width: 70)
                            Stepper("Clamp target", value: $model.runtimeOptions.steeringTarget, step: 0.1)
                                .labelsHidden()
                        }
                    }
                }

                LabeledContent("Activation Gate") {
                    HStack(spacing: 4) {
                        TextField("Gate threshold", value: $model.runtimeOptions.steeringGate, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 70)
                        Stepper("Activation gate", value: $model.runtimeOptions.steeringGate, step: 0.1)
                            .labelsHidden()
                    }
                }
            }
        }
        .disabled(model.isRunning)
    }

}
