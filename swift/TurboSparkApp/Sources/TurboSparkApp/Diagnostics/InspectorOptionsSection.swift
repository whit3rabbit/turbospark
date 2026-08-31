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
                    Text("No models installed").tag("")
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
                        .font(.caption)
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
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Family") {
                    HStack(spacing: 4) {
                        Image(systemName: visuals.iconSystemName)
                            .font(.caption2)
                            .foregroundStyle(visuals.accentColor)
                            .accessibilityHidden(true)
                        Text(visuals.family)
                    }
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .help("Architecture: \(visuals.family)")
                }
            }
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }

    var memoryAndPowerSection: some View {
        Section("Open & Architecture Options") {
            ContextWindowOptionsView(model: model)

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
                        .font(.caption)
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

    var steeringSection: some View {
        Section("Directional Steering") {
            TextField("Control vector path (.gguf)", text: Binding(
                get: { model.runtimeOptions.steeringPath ?? "" },
                set: { model.runtimeOptions.steeringPath = $0.isEmpty ? nil : $0 }
            ))
            .textFieldStyle(.roundedBorder)
            .font(.caption)
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
                        TextField("Target", value: $model.runtimeOptions.steeringTarget, format: .number)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 70)
                    }
                }

                LabeledContent("Activation Gate") {
                    TextField("Gate threshold", value: $model.runtimeOptions.steeringGate, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 70)
                }
            }
        }
        .disabled(model.isRunning)
    }

    var generationSection: some View {
        Section("Generation Sampling") {
            // DISABLED rather than hidden here, unlike the compact chrome
            // controls: this is the panel a user goes looking in, and a
            // missing row reads as a missing feature. The levels are the
            // checkpoint's own and arrive with the session, so there is
            // nothing to offer until one is open.
            LabeledContent("Thinking") {
                Picker("Thinking", selection: Binding(
                    get: { model.reasoning },
                    set: { model.setReasoning($0) }
                )) {
                    ForEach(model.availableReasoningLevels) { level in
                        Text(model.reasoningLabel(for: level)).tag(level)
                    }
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .disabled(!model.reasoningPickerEnabled)
                .accessibilityLabel("Thinking effort")
            }

            if model.session == nil {
                Text("Load a model to see the reasoning levels its chat template accepts.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else if !model.isReasoningSupported {
                Text("This checkpoint ships no reasoning knob, so a level would change nothing.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            LabeledContent("Max New Tokens") {
                Stepper(value: $model.maxNewTokens, in: 64...16384, step: 128) {
                    Text("\(model.maxNewTokens)").monospacedDigit()
                }
                .fixedSize()
                .accessibilityLabel("Max new tokens")
                .accessibilityValue("\(model.maxNewTokens)")
            }

            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0...2, step: 0.05)
                        .accessibilityLabel("Temperature")
                        .accessibilityValue(String(format: "%.2f", model.temperature))
                    Text(model.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }

            Toggle("Top-K", isOn: $model.topKEnabled)
                .toggleStyle(.switch)
            if model.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: $model.topK, in: 1...256, step: 1) {
                        Text("\(model.topK)").monospacedDigit()
                    }
                    .fixedSize()
                    .accessibilityLabel("Top-K value")
                    .accessibilityValue("\(model.topK)")
                }
            }

            Toggle("Top-P", isOn: $model.topPEnabled)
                .toggleStyle(.switch)
            if model.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: $model.topP, in: 0.01...1, step: 0.01)
                            .accessibilityLabel("Top-P value")
                            .accessibilityValue(String(format: "%.2f", model.topP))
                        Text(model.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }

            Toggle("Repetition Penalty", isOn: $model.repetitionPenaltyEnabled)
                .toggleStyle(.switch)
            if model.repetitionPenaltyEnabled {
                LabeledContent("Penalty") {
                    HStack(spacing: 8) {
                        Slider(value: $model.repetitionPenalty, in: 1.0...2.0, step: 0.05)
                            .accessibilityLabel("Repetition penalty")
                            .accessibilityValue(String(format: "%.2f", model.repetitionPenalty))
                        Text(model.repetitionPenalty, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }

            Toggle("Deterministic Seed", isOn: $model.seedEnabled)
                .toggleStyle(.switch)
            if model.seedEnabled {
                LabeledContent("Seed value") {
                    TextField("Seed", value: $model.seed, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 100)
                        .accessibilityLabel("Seed value")
                        .accessibilityValue("\(model.seed)")
                }
            }

            LabeledContent("Stop Sequences") {
                TextField("Comma separated strings", text: $model.stopSequences)
                    .textFieldStyle(.roundedBorder)
                    .font(.caption)
                    .accessibilityLabel("Stop sequences")
                    .accessibilityHint("Comma-separated strings that stop generation when produced")
            }
        }
        .disabled(model.isRunning)
    }
}

private struct ContextWindowOptionsView: View {
    @ObservedObject var model: AppModel
    @State private var isCustom = false

    private var selectionTag: Binding<Int> {
        Binding(
            get: {
                if isCustom { return -1 }
                if AppContextLengthOption(rawValue: model.maxContextTokens) != nil {
                    return model.maxContextTokens
                }
                return -1
            },
            set: { newValue in
                if newValue == -1 {
                    isCustom = true
                    if model.maxContextTokens == 0 {
                        model.maxContextTokens = model.resolvedContextTokens
                    }
                } else {
                    isCustom = false
                    model.maxContextTokens = newValue
                }
            }
        )
    }

    var body: some View {
        Group {
            LabeledContent("Context Window") {
                Picker("Context", selection: selectionTag) {
                    ForEach(AppContextLengthOption.allCases) { option in
                        Text(option.formattedLabel(resolvedAuto: model.resolvedContextTokens))
                            .tag(option.tokens)
                    }
                    Divider()
                    Text("Custom…").tag(-1)
                }
                .pickerStyle(.menu)
                .labelsHidden()
                .fixedSize()
                .accessibilityLabel("Context window")
            }

            if isCustom {
                LabeledContent("Custom Size") {
                    HStack(spacing: 8) {
                        Slider(
                            value: Binding(
                                get: { Double(model.maxContextTokens) },
                                set: {
                                    let raw = Int($0)
                                    let step = raw < 8192 ? 256 : (raw < 32768 ? 512 : 1024)
                                    let snapped = max(512, min(131072, ((raw + step / 2) / step) * step))
                                    model.maxContextTokens = snapped
                                }
                            ),
                            in: 512...131072
                        )
                        .accessibilityLabel("Custom context size")
                        .accessibilityValue("\(model.maxContextTokens) tokens")
                        Text("\(model.maxContextTokens.formatted())")
                            .font(.caption.monospacedDigit())
                            .frame(width: 55, alignment: .trailing)
                    }
                }
                .transition(.opacity.combined(with: .move(edge: .top)))
            }
        }
        .onAppear {
            if model.maxContextTokens > 0 && AppContextLengthOption(rawValue: model.maxContextTokens) == nil {
                isCustom = true
            }
        }
        .animation(.smooth(duration: 0.2), value: isCustom)
    }
}
