import AppKit
import SwiftUI
import TurboSpark

extension InspectorView {
    var modelSection: some View {
        Section(header: Text("Model", bundle: .module)) {
            Picker(selection: Binding(
                get: { model.selected?.alias ?? "" },
                set: { alias in
                    guard let m = model.loadableModels.first(where: { $0.alias == alias }) else { return }
                    model.selectModel(m)
                }
            )) {
                // The loadable list only (requirement 3.3).
                if model.loadableModels.isEmpty {
                    if let statement = model.notConnectedStatement {
                        Text(verbatim: statement).tag("")
                    } else {
                        Text("No models installed", bundle: .module).tag("")
                    }
                }
                ForEach(model.loadableModels) { m in
                    let visuals = ModelFamilyVisuals.resolve(alias: m.alias, family: m.family, name: m.alias)
                    Label {
                        Text(verbatim: "\(m.alias) (\(visuals.family))")
                    } icon: {
                        Image(systemName: visuals.iconSystemName)
                    }
                    .tag(m.alias)
                }
            } label: { Text("Model", bundle: .module) }
            // The section header already reads "Model". The label stays as
            // the VoiceOver name; hidden, the picker takes the row's full
            // width instead of floating right of an empty label column.
            .labelsHidden()
            .frame(maxWidth: .infinity, alignment: .leading)

            LabeledContent {
                HStack(spacing: 6) {
                    // Selectable, middle-truncated, full path in the tooltip
                    // (DESIGN.md "Text in layouts").
                    SettingsPathText(model.modelPathText)
                        .foregroundStyle(.appSecondary)
                    AppIconButton("doc.on.doc", label: Text("Copy model path", bundle: .module)) {
                        NSPasteboard.general.clearContents()
                        NSPasteboard.general.setString(model.modelPathText, forType: .string)
                    }
                    .appAction(.quiet, size: .small)
                    .accessibilityHint(Text("Copies the on-disk model path to the clipboard", bundle: .module))
                }
            } label: {
                Text("Path", bundle: .module)
            }

            Button {
                ModelLocationPicker.choose(for: model)
            } label: { Text("Choose Model Folder\u{2026}", bundle: .module) }
            .appAction(.secondary, size: .small)
            .disabled(model.opening || model.isRunning)
            .accessibilityHint(Text("Opens a folder picker to choose a model directory", bundle: .module))

            if model.canUnloadModel {
                HStack {
                    Button(action: model.reloadModel) { Text("Reload Model", bundle: .module) }
                        .appAction(.secondary, size: .small)
                    Spacer()
                    Button(action: model.unloadModel) { Text("Unload Model", bundle: .module) }
                        .appAction(.secondary, size: .small)
                }
            } else if model.canLoadModel {
                Button(action: model.loadModel) { Text("Load Model", bundle: .module) }
                    .appAction(.secondary, size: .small)
            }

            if let selected = model.selected {
                let visuals = ModelFamilyVisuals.resolve(alias: selected.alias, family: selected.family, name: selected.alias)
                LabeledContent {
                    Text(MetricFormat.storage(selected.installBytes))
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                } label: {
                    Text("Installed size", bundle: .module)
                }
                LabeledContent {
                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .themedFont(.tiny)
                            .foregroundStyle(visuals.accentColor)
                            .accessibilityHidden(true)
                        Text(visuals.family)
                    }
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .help(Text("Architecture: \(visuals.family)", bundle: .module))
                } label: {
                    Text("Family", bundle: .module)
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    var memoryAndPowerSection: some View {
        Section(header: Text("Open & Architecture Options", bundle: .module)) {
            // The context window lives in InspectorEssentialsSection's
            // `ContextLadderPicker`, the one place that sets it.
            LabeledContent {
                Picker(selection: $model.runtimeOptions.expertCacheSlots) {
                    ForEach(AppRuntimeOptions.allowedSlotCounts, id: \.self) { slots in
                        Self.slotsText(slots).tag(slots)
                    }
                } label: { Text("Slots", bundle: .module) }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                // .labelsHidden() strips the visible label, so VoiceOver would
                // hear "menu, 16" with no idea what 16 means.
                .accessibilityLabel(Text("Cache slots", bundle: .module))
            } label: {
                Text("Cache Slots", bundle: .module)
            }

            LabeledContent {
                Picker(selection: $model.runtimeOptions.powerProfile) {
                    ForEach(AppPowerProfileOption.allCases) { profile in
                        Text(LocalizedStringKey(profile.menuLabel), bundle: .module).tag(profile)
                    }
                } label: { Text("Power", bundle: .module) }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .accessibilityLabel(Text("Power profile", bundle: .module))
            } label: { Text("Power Profile", bundle: .module) }

            LabeledContent {
                Picker(selection: $model.runtimeOptions.speculation) {
                    ForEach(AppSpeculationOption.allCases) { spec in
                        Text(LocalizedStringKey(spec.menuLabel), bundle: .module).tag(spec)
                    }
                } label: { Text("Speculation", bundle: .module) }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .accessibilityLabel(Text("Speculation", bundle: .module))
            } label: { Text("Speculation", bundle: .module) }

            if model.runtimeOptions.speculation != .off {
                LabeledContent {
                    Picker(selection: $model.runtimeOptions.speculativeDrafter) {
                        ForEach(AppSpeculativeDrafterOption.allCases) { drafter in
                            Text(LocalizedStringKey(drafter.menuLabel), bundle: .module).tag(drafter)
                        }
                    } label: { Text("Drafter", bundle: .module) }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .fixedSize()
                    .accessibilityLabel(Text("Speculation drafter", bundle: .module))
                } label: { Text("Drafter", bundle: .module) }
            }

            // `.labelsHidden()` is load-bearing, not cosmetic: on macOS a
            // TextField's title is a VISIBLE LABEL rather than a placeholder,
            // so "Uncapped" was drawn beside the field and clipped to "Un-"
            // by the inspector's width. Every other control in this section
            // already hides its label for the same reason.
            LabeledContent {
                HStack(spacing: 6) {
                    TextField(value: $model.runtimeOptions.maxTokensPerSec, format: .number) {
                        Text("Uncapped", bundle: .module)
                    }
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 70)
                    .multilineTextAlignment(.trailing)
                    .accessibilityLabel(Text("Rate cap in tokens per second", bundle: .module))
                    .accessibilityValue(rateCapIsUncapped
                        ? Text("Uncapped", bundle: .module)
                        : Text("\(model.runtimeOptions.maxTokensPerSec.formatted()) tokens per second", bundle: .module))
                    // Zero is the uncapped sentinel, and a bare "0" next to
                    // "tok/s" reads as a cap of zero tokens per second.
                    (rateCapIsUncapped ? Text("uncapped", bundle: .module) : Text("tok/s", bundle: .module))
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .accessibilityHidden(true)
                }
                .fixedSize()
            } label: {
                Text("Rate Cap", bundle: .module)
            }
            .help(Text("Tokens per second ceiling. 0 leaves generation uncapped.", bundle: .module))

            if model.canReloadModel {
                Button {
                    model.reloadModel()
                } label: { Text("Apply Options & Reload", bundle: .module) }
                .appAction(.secondary, size: .small)
            }
        }
        .disabled(model.isRunning)
    }

    /// Zero is the engine's "no cap" sentinel for `maxTokensPerSec`.
    private var rateCapIsUncapped: Bool {
        model.runtimeOptions.maxTokensPerSec <= 0
    }

    /// `AppRuntimeOptions.slotsLabel(for:)` spelled as a catalog `Text`; that
    /// one returns English for the places that need a `String`.
    private static func slotsText(_ slots: Int) -> Text {
        slots == 0 ? Text("Auto", bundle: .module) : Text("\(slots) slots", bundle: .module)
    }

    /// The RAW knobs, kept as the expert surface beside the named presets in
    /// Settings > Safety and Steering. What is set here becomes the implicit
    /// "Custom" preset, so a configuration made before presets existed keeps
    /// working.
    var steeringSection: some View {
        Section(header: Text("Directional Steering", bundle: .module)) {
            // A family that does not dispatch the edit REFUSES a vector at
            // open, so the knobs below can only produce a failed load there.
            // Disabled with the engine's own reason rather than hidden
            // (swift/CLAUDE.md Gotchas 23 and 33).
            if model.steeringFamilySupported == false {
                Label {
                    // The engine's own refusal sentence when it gave one.
                    if let reason = model.info?.steering.reason {
                        Text(verbatim: reason)
                    } else {
                        Text("This model's family does not dispatch the steering edit.", bundle: .module)
                    }
                } icon: {
                    Image(systemName: "exclamationmark.triangle")
                }
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            }

            // **STEERING RESOLVES ONCE, AT OPEN.** Editing a field here does
            // nothing to the model already loaded, and until this note
            // existed the app said nothing about that -- a knob that appears
            // to work and does not.
            if model.steeringNeedsReload {
                HStack {
                    Label { Text("Not applied to the loaded model", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                        .themedFont(.small)
                        .foregroundStyle(.appWarning)
                    Spacer()
                    Button { model.reloadForSteering() } label: { Text("Reload", bundle: .module) }
                        .appAction(.secondary, size: .small)
                        .disabled(model.selected == nil || model.generating || model.opening)
                }
            }

            TextField(text: Binding(
                get: { model.runtimeOptions.steeringPath ?? "" },
                set: { model.runtimeOptions.steeringPath = $0.isEmpty ? nil : $0 }
            )) {
                Text("Control vector path (.gguf)", bundle: .module)
            }
            .textFieldStyle(.roundedBorder)
            .themedFont(.small)
            .accessibilityLabel(Text("Control vector path", bundle: .module))
            .accessibilityHint(Text("Path to a GGUF control vector file. Leave blank to disable directional steering.", bundle: .module))

            if let path = model.runtimeOptions.steeringPath, !path.isEmpty {
                LabeledContent {
                    Picker(selection: $model.runtimeOptions.steeringMode) {
                        ForEach(AppSteeringModeOption.allCases) { mode in
                            Text(LocalizedStringKey(mode.menuLabel), bundle: .module).tag(mode)
                        }
                    } label: { Text("Steering Mode", bundle: .module) }
                    .pickerStyle(.menu)
                    .labelsHidden()
                    .fixedSize()
                    .accessibilityLabel(Text("Steering mode", bundle: .module))
                } label: { Text("Mode", bundle: .module) }

                LabeledContent {
                    HStack(spacing: 8) {
                        Slider(value: $model.runtimeOptions.steeringScale, in: 0...3, step: 0.1)
                            .accessibilityLabel(Text("Steering scale", bundle: .module))
                            .accessibilityValue(String(format: "%.1f", model.runtimeOptions.steeringScale))
                        Text(model.runtimeOptions.steeringScale, format: .number.precision(.fractionLength(1)))
                            .monospacedDigit()
                            .frame(width: 32, alignment: .trailing)
                    }
                } label: {
                    Text("Scale", bundle: .module)
                }

                LabeledContent {
                    TextField(text: $model.runtimeOptions.steeringLayers) {
                        Text("e.g. 10:30 (all if blank)", bundle: .module)
                    }
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 120)
                    .accessibilityLabel(Text("Steering layers", bundle: .module))
                    .accessibilityHint(Text("Comma-separated start:end layer range; blank means all layers", bundle: .module))
                } label: {
                    Text("Layers (Start:End)", bundle: .module)
                }

                if model.runtimeOptions.steeringMode == .clamp {
                    LabeledContent {
                        // No `in:` range on the Stepper: unlike `steeringScale`
                        // (0...3, documented, already a slider), this is a raw,
                        // control-vector-dependent activation value with no
                        // engine-defined bound, so a hardcoded min/max here
                        // would silently clamp values a real vector needs.
                        HStack(spacing: 4) {
                            TextField(value: $model.runtimeOptions.steeringTarget, format: .number) {
                                Text("Target", bundle: .module)
                            }
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 70)
                            Stepper(value: $model.runtimeOptions.steeringTarget, step: 0.1) {
                                Text("Clamp target", bundle: .module)
                            }
                            .labelsHidden()
                        }
                    } label: {
                        Text("Clamp Target", bundle: .module)
                    }
                }

                LabeledContent {
                    HStack(spacing: 4) {
                        TextField(value: $model.runtimeOptions.steeringGate, format: .number) {
                            Text("Gate threshold", bundle: .module)
                        }
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 70)
                        Stepper(value: $model.runtimeOptions.steeringGate, step: 0.1) {
                            Text("Activation gate", bundle: .module)
                        }
                        .labelsHidden()
                    }
                } label: {
                    Text("Activation Gate", bundle: .module)
                }
            }
        }
        .disabled(model.isRunning)
    }

}
