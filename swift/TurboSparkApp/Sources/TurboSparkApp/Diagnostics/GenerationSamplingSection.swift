import SwiftUI
import TurboSpark

/// The Inspector's Generation Sampling section, scope-aware: the knobs edit
/// either the app-wide defaults or the selected chat's own override, and the
/// bottom of the section carries the preset registry over the same snapshot.
///
/// Ported from Unsloth Studio's per-conversation sampling controls, with the
/// engagement made EXPLICIT (a scope picker) rather than Studio's implicit
/// "edits belong to the open conversation": Studio's settings sheet sits
/// beside one thread, while this Inspector is always open, and a silent
/// chat-local edit would read to the user as an app-wide one.
///
/// `scopeSettings` is the one binding every knob hangs off. Its GET resolves
/// the visible values (the chat's override when the scope is the chat and one
/// exists, the app-wide settings otherwise) and its SET writes the whole
/// snapshot back through `AppModel+Sampling.swift`, which is where both
/// persistence paths live (settings.json debounced, chat archive debounced).
/// Reasoning stays bound to the app-wide property: a scope override covers
/// sampling only, on purpose (`AppSamplingSettings`'s doc).
@MainActor
struct GenerationSamplingSection: View {
    @ObservedObject var model: AppModel

    private enum Scope: Hashable {
        case appDefaults
        case thisChat
    }

    /// Where the knobs below write. Re-derived whenever the selection moves:
    /// a chat carrying an override opens on "This chat", everything else on
    /// "App defaults".
    @State private var scope: Scope = .appDefaults
    @State private var newPresetName: String = ""

    var body: some View {
        Section("Generation Sampling") {
            scopePicker
            if model.selectedChat.samplingOverride != nil {
                removeOverrideRow
            }

            // DISABLED rather than hidden here, unlike the compact chrome
            // controls: this is the panel a user goes looking in, and a
            // missing row reads as a missing feature. The levels are the
            // checkpoint's own and arrive with the session, so there is
            // nothing to offer until one is open. APP-WIDE, unlike every
            // row below: the scope picker covers sampling only.
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
                Text("Load a model to see the reasoning levels its chat template accepts.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else if !model.isReasoningSupported {
                Text("This checkpoint ships no reasoning knob, so a level would change nothing.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            LabeledContent("Max New Tokens") {
                Stepper(value: scopeSettings.maxNewTokens, in: 64...16384, step: 128) {
                    Text("\(settings.maxNewTokens)", bundle: .module).monospacedDigit()
                }
                .fixedSize()
                .accessibilityLabel("Max new tokens")
                .accessibilityValue("\(settings.maxNewTokens)")
            }

            LabeledContent("Temperature") {
                HStack(spacing: 8) {
                    Slider(value: scopeSettings.temperature, in: 0...2, step: 0.05)
                        .accessibilityLabel("Temperature")
                        .accessibilityValue(String(format: "%.2f", settings.temperature))
                    Text(settings.temperature, format: .number.precision(.fractionLength(2)))
                        .monospacedDigit()
                        .frame(width: 36, alignment: .trailing)
                }
            }

            Toggle("Top-K", isOn: scopeSettings.topKEnabled)
                .toggleStyle(.switch)
            if settings.topKEnabled {
                LabeledContent("K value") {
                    Stepper(value: scopeSettings.topK, in: 1...256, step: 1) {
                        Text("\(settings.topK)", bundle: .module).monospacedDigit()
                    }
                    .fixedSize()
                    .accessibilityLabel("Top-K value")
                    .accessibilityValue("\(settings.topK)")
                }
            }

            Toggle("Top-P", isOn: scopeSettings.topPEnabled)
                .toggleStyle(.switch)
            if settings.topPEnabled {
                LabeledContent("P value") {
                    HStack(spacing: 8) {
                        Slider(value: scopeSettings.topP, in: 0.01...1, step: 0.01)
                            .accessibilityLabel("Top-P value")
                            .accessibilityValue(String(format: "%.2f", settings.topP))
                        Text(settings.topP, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }

            Toggle("Repetition Penalty", isOn: scopeSettings.repetitionPenaltyEnabled)
                .toggleStyle(.switch)
            if settings.repetitionPenaltyEnabled {
                LabeledContent("Penalty") {
                    HStack(spacing: 8) {
                        Slider(value: scopeSettings.repetitionPenalty, in: 1.0...2.0, step: 0.05)
                            .accessibilityLabel("Repetition penalty")
                            .accessibilityValue(String(format: "%.2f", settings.repetitionPenalty))
                        Text(settings.repetitionPenalty, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                            .frame(width: 36, alignment: .trailing)
                    }
                }
            }

            Toggle("Deterministic Seed", isOn: scopeSettings.seedEnabled)
                .toggleStyle(.switch)
            if settings.seedEnabled {
                LabeledContent("Seed value") {
                    TextField("Seed", value: scopeSettings.seed, format: .number)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 100)
                        .accessibilityLabel("Seed value")
                        .accessibilityValue("\(settings.seed)")
                }
            }

            LabeledContent("Stop Sequences") {
                TextField("Comma separated strings", text: scopeSettings.stopSequences)
                    .labelsHidden()
                    .textFieldStyle(.roundedBorder)
                    .themedFont(.small)
                    .accessibilityLabel("Stop sequences")
                    .accessibilityHint("Comma-separated strings that stop generation when produced")
            }

            Divider()
            presetsArea
        }
        .disabled(model.isRunning)
        .onAppear(perform: syncScopeWithSelection)
        .onChange(of: model.selectedChatID) { syncScopeWithSelection() }
    }

    // MARK: - Scope

    private var scopePicker: some View {
        LabeledContent("Edits apply to") {
            Picker("Edits apply to", selection: $scope) {
                Text("App defaults", bundle: .module).tag(Scope.appDefaults)
                Text("This chat", bundle: .module).tag(Scope.thisChat)
            }
            .pickerStyle(.menu)
            .labelsHidden()
            .fixedSize()
            .accessibilityLabel("Sampling edit scope")
            .accessibilityHint("Whether the sampling controls below edit the app-wide defaults or this conversation only")
        }
    }

    /// Shown while the selected chat carries an override, from either scope:
    /// deleting it is the "back to the defaults" action, the sibling of the
    /// system-prompt sheet's "Use Default".
    private var removeOverrideRow: some View {
        LabeledContent {
            Button("Remove Override", role: .destructive) {
                model.removeChatSamplingOverride(id: model.selectedChatID)
                scope = .appDefaults
            }
            .buttonStyle(.bordered)
            .accessibilityLabel("Remove this chat's sampling override")
        } label: {
            Text("This chat's sampling", bundle: .module)
        }
    }

    private func syncScopeWithSelection() {
        scope = model.selectedChat.samplingOverride != nil ? .thisChat : .appDefaults
    }

    /// The visible values: the chat's override when the scope is the chat
    /// and one exists, the app-wide settings otherwise. A chat switched to
    /// "This chat" with no override yet DISPLAYS the app-wide values, and
    /// the first edit below seeds the override from them.
    private var settings: AppSamplingSettings {
        scopeSettings.wrappedValue
    }

    private var scopeSettings: Binding<AppSamplingSettings> {
        Binding(
            get: {
                if scope == .thisChat, let override = model.selectedChat.samplingOverride {
                    return override
                }
                return model.globalSamplingSettings()
            },
            set: { newValue in
                guard scope == .thisChat else {
                    model.applyGlobalSampling(newValue)
                    return
                }
                model.setChatSamplingOverride(id: model.selectedChatID, settings: newValue)
            }
        )
    }

    // MARK: - Presets

    private var presetsArea: some View {
        Group {
            if model.samplingPresets.isEmpty {
                Text(
                    "No sampling presets saved yet. Set the knobs above, then save them under a name.",
                    bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            } else {
                ForEach(model.samplingPresets) { preset in
                    LabeledContent {
                        HStack(spacing: 6) {
                            Button("Apply") {
                                applyPreset(preset)
                            }
                            .buttonStyle(.bordered)
                            .accessibilityLabel("Apply sampling preset \(preset.name)")

                            // A visible delete affordance beside the
                            // right-click one below: this was context-menu
                            // only, unlike "Remove Override" above, whose
                            // destructive action is a plainly visible button.
                            Button(role: .destructive) {
                                model.deleteSamplingPreset(preset.id)
                            } label: {
                                Image(systemName: "trash")
                            }
                            .buttonStyle(.borderless)
                            .accessibilityLabel("Delete sampling preset \(preset.name)")
                        }
                    } label: {
                        Text(preset.name)
                    }
                    .contextMenu {
                        Button("Delete Preset", role: .destructive) {
                            model.deleteSamplingPreset(preset.id)
                        }
                    }
                }
            }

            HStack {
                TextField("Preset name", text: $newPresetName)
                    .textFieldStyle(.roundedBorder)
                    .themedFont(.small)
                    .accessibilityLabel("New sampling preset name")
                Button("Save Preset", action: savePreset)
                    .buttonStyle(.bordered)
                    .disabled(presetName.isEmpty)
                    .accessibilityLabel("Save current sampling as a preset")
            }
        }
    }

    private var presetName: String {
        newPresetName.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    /// Stamps a preset into the scope being edited, never into the other
    /// one: "App defaults" applies app-wide even when the selected chat
    /// carries an override, and vice versa.
    private func applyPreset(_ preset: AppSamplingPreset) {
        if scope == .thisChat {
            model.setChatSamplingOverride(id: model.selectedChatID, settings: preset.settings)
        } else {
            model.applyGlobalSampling(preset.settings)
        }
    }

    private func savePreset() {
        let name = presetName
        guard !name.isEmpty else { return }
        // Saving over an existing NAME replaces that preset, which is the
        // predictable reading of "Save"; a new name mints a new id.
        let existingID = model.samplingPresets.first(where: { $0.name == name })?.id
        let preset = AppSamplingPreset(
            id: existingID ?? UUID(),
            name: name,
            settings: settings)
        model.upsertSamplingPreset(preset)
        newPresetName = ""
    }
}
