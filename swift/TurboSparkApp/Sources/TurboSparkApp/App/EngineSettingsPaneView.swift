import SwiftUI
import TurboSpark

/// Settings pane for engine configuration, sampling parameters, reasoning level, guardrails, and in-process server.
@MainActor
struct EngineSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var fans: FanController = .shared

    /// **ONE PROPERTY PER SECTION RATHER THAN ONE EXPRESSION.** A `Form`
    /// holding every `Section` inline is ONE expression to the type checker,
    /// and it exceeded the solver budget on the SDK 14 toolchain that CI
    /// builds the app bundle with: "unable to type-check this expression
    /// in reasonable time". It compiles on a newer one, which is why
    /// nobody developing here saw it. Splitting is the fix the diagnostic
    /// itself asks for, and it is cheap insurance on any toolchain -- a
    /// SwiftUI container that grows a section at a time walks into this
    /// eventually. The runtime sections live in
    /// `EngineSettingsRuntimeSections.swift` for the same reason and to
    /// keep each file readable.
    var body: some View {
        Form {
            Text("Sampling and prompt changes apply to the next turn. Load settings apply after model reload.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
            SystemPromptSettingsSection(model: model)
            PersonalitySettingsSection(model: model)
            generationDefaultsSection
            advancedGenerationSection
            coolingSection
            reasoningEffortSection
            guardrailsSection
            speculationSection
            kvBitsSection
            anePrefillSection
            inProcessServerSection
        }
        .formStyle(.grouped)
        .scrollContentBackground(.hidden)
        .background(.appPage)
        .padding(SettingsLayout.formInsets)
    }

    // Every control in these rows carries its OWN name through a label
    // closure and hides it with `.labelsHidden()`, because the visible
    // row label is a separate `Text` that holds the search anchor. The old
    // empty-titled toggles and unlabeled sliders read as "checkbox" and
    // "slider" to VoiceOver with no hint of what they set. Widths are min/ideal/max
    // rather than fixed so a large text size or a long translation grows
    // the control instead of clipping it.

    private var generationDefaultsSection: some View {
        Section(header: Text("Generation Defaults", bundle: .module)) {
            LabeledContent {
                HStack(spacing: 8) {
                    Slider(value: $model.temperature, in: 0.0...1.5, step: 0.05) {
                        Text("Temperature", bundle: .module)
                    }
                    .labelsHidden()
                    .frame(minWidth: 120, idealWidth: 160, maxWidth: 220)
                    .accessibilityValue(Text(verbatim: Self.twoDecimals(model.temperature)))
                    .onChange(of: model.temperature) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    readout(Self.twoDecimals(model.temperature))
                }
            } label: {
                Text("Temperature", bundle: .module)
                    .settingsControl("Temperature", pane: .engine, timing: .nextTurn)
            }

            LabeledContent {
                HStack(spacing: 8) {
                    Toggle(isOn: $model.topPEnabled) {
                        Text("Enable Top-P Sampling", bundle: .module)
                    }
                    .labelsHidden()
                    .onChange(of: model.topPEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    Slider(value: $model.topP, in: 0.1...1.0, step: 0.05) {
                        Text("Top-P Sampling", bundle: .module)
                    }
                    .labelsHidden()
                    .frame(minWidth: 120, idealWidth: 160, maxWidth: 220)
                    .disabled(!model.topPEnabled)
                    .accessibilityValue(Text(verbatim: Self.twoDecimals(model.topP)))
                    .onChange(of: model.topP) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    readout(Self.twoDecimals(model.topP), enabled: model.topPEnabled)
                }
            } label: {
                Text("Top-P Sampling", bundle: .module)
                    .settingsControl("Top-P Sampling", pane: .engine, timing: .nextTurn)
            }
        }
    }

    /// The sampling and engine-load knobs that used to live ONLY in the
    /// diagnostics Inspector, which is not a surface a person reads as
    /// "settings". Same `AppModel` properties both places, so the two stay
    /// in sync by construction; a separate computed property like every
    /// sibling section for the type-checker reason spelled out above `body`.
    private var advancedGenerationSection: some View {
        Section(header: Text("Advanced Generation", bundle: .module)) {
            LabeledContent {
                HStack(spacing: 8) {
                    TextField(
                        value: $model.maxContextTokens, format: .number.grouping(.never),
                        prompt: Text("auto", bundle: .module)
                    ) { Text("Context Window", bundle: .module) }
                    .modifier(EngineNumberFieldStyle())
                    .onChange(of: model.maxContextTokens) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    unit(model.maxContextTokens == 0 ? Text("auto", bundle: .module) : Text("tokens", bundle: .module))
                }
            } label: {
                Text("Context Window", bundle: .module)
                    .settingsControl("Context Window", pane: .engine, timing: .modelReload)
            }

            LabeledContent {
                HStack(spacing: 8) {
                    TextField(
                        value: $model.maxNewTokens, format: .number.grouping(.never),
                        prompt: Text(verbatim: "2048")
                    ) { Text("Max New Tokens", bundle: .module) }
                    .modifier(EngineNumberFieldStyle())
                    .onChange(of: model.maxNewTokens) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    unit(Text("tokens", bundle: .module))
                }
            } label: {
                Text("Max New Tokens", bundle: .module)
                    .settingsControl("Max New Tokens", pane: .engine, timing: .nextTurn)
            }

            LabeledContent {
                HStack(spacing: 8) {
                    Toggle(isOn: $model.topKEnabled) {
                        Text("Enable Top-K Sampling", bundle: .module)
                    }
                    .labelsHidden()
                    .onChange(of: model.topKEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    TextField(
                        value: $model.topK, format: .number.grouping(.never),
                        prompt: Text(verbatim: "64")
                    ) { Text("Top-K Sampling", bundle: .module) }
                    .modifier(EngineNumberFieldStyle())
                    .disabled(!model.topKEnabled)
                    .onChange(of: model.topK) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    unitSpacer
                }
            } label: {
                Text("Top-K Sampling", bundle: .module)
                    .settingsControl("Top-K Sampling", pane: .engine, timing: .nextTurn)
            }

            LabeledContent {
                HStack(spacing: 8) {
                    Toggle(isOn: $model.repetitionPenaltyEnabled) {
                        Text("Enable Repetition Penalty", bundle: .module)
                    }
                    .labelsHidden()
                    .onChange(of: model.repetitionPenaltyEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    TextField(
                        value: $model.repetitionPenalty, format: .number,
                        prompt: Text(verbatim: "1.0")
                    ) { Text("Repetition Penalty", bundle: .module) }
                    .modifier(EngineNumberFieldStyle())
                    .disabled(!model.repetitionPenaltyEnabled)
                    .onChange(of: model.repetitionPenalty) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    unitSpacer
                }
            } label: {
                Text("Repetition Penalty", bundle: .module)
                    .settingsControl("Repetition Penalty", pane: .engine, timing: .nextTurn)
            }

            LabeledContent {
                HStack(spacing: 8) {
                    Toggle(isOn: $model.seedEnabled) {
                        Text("Enable Fixed Seed", bundle: .module)
                    }
                    .labelsHidden()
                    .onChange(of: model.seedEnabled) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    TextField(
                        value: $model.seed, format: .number.grouping(.never),
                        prompt: Text(verbatim: "0")
                    ) { Text("Fixed Seed", bundle: .module) }
                    .modifier(EngineNumberFieldStyle())
                    .disabled(!model.seedEnabled)
                    .onChange(of: model.seed) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    unitSpacer
                }
            } label: {
                Text("Fixed Seed", bundle: .module)
                    .settingsControl("Fixed Seed", pane: .engine, timing: .nextTurn)
            }

            VStack(alignment: .leading, spacing: 4) {
                Text("Stop Sequences", bundle: .module)
                    .settingsControl("Stop Sequences", pane: .engine, timing: .nextTurn)
                TextField(text: $model.stopSequences, prompt: Text("comma-separated", bundle: .module)) {
                    Text("Stop Sequences", bundle: .module)
                }
                .labelsHidden()
                .textFieldStyle(.roundedBorder)
                .onChange(of: model.stopSequences) { _, _ in
                    model.persistSettingsDebounced()
                }
                Text("Generation stops when the model emits any of these comma-separated strings.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            Picker(selection: $model.runtimeOptions.powerProfile) {
                ForEach(AppPowerProfileOption.allCases) { profile in
                    Text(LocalizedStringKey(profile.menuLabel), bundle: .module).tag(profile)
                }
            } label: { Text("Power Profile", bundle: .module) }
            .settingsControl("Power Profile", pane: .engine, timing: .modelReload)
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.powerProfile) { _, _ in
                model.persistSettingsDebounced()
            }

            LabeledContent {
                HStack(spacing: 8) {
                    TextField(
                        value: $model.runtimeOptions.maxTokensPerSec, format: .number,
                        prompt: Text("Uncapped", bundle: .module)
                    ) { Text("Output Rate Cap", bundle: .module) }
                    .modifier(EngineNumberFieldStyle())
                    .onChange(of: model.runtimeOptions.maxTokensPerSec) { _, _ in
                        model.persistSettingsDebounced()
                    }
                    unit(Text("tok/s", bundle: .module))
                }
            } label: {
                Text("Output Rate Cap", bundle: .module)
                    .settingsControl("Output Rate Cap", pane: .engine, timing: .modelReload)
            }

            Picker(selection: $model.runtimeOptions.loadGuard) {
                ForEach(AppLoadGuardOption.allCases) { tier in
                    Text(LocalizedStringKey(tier.menuLabel), bundle: .module).tag(tier)
                }
            } label: { Text("Memory Load Guard", bundle: .module) }
            .settingsControl("Memory Load Guard", pane: .engine, timing: .modelReload)
            .pickerStyle(.menu)
            .onChange(of: model.runtimeOptions.loadGuard) { _, _ in
                model.persistSettingsDebounced()
            }

            // `.labelsHidden()` on every TextField above is load-bearing: on macOS a
            // TextField's title is a VISIBLE LABEL rather than a placeholder
            // (swift/CLAUDE.md Gotcha 23), and this pane reintroduced the exact
            // "Uncapped" clipping the Inspector had already fixed. The label
            // closure still names the field for VoiceOver.
            Text("Context 0 means the checkpoint's trained context, capped by memory. Stop sequences, power profile, output rate cap and memory load guard also appear in the Inspector and edit the same values.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }

    private var coolingSection: some View {
        Section(header: Text("Cooling & Fan Control", bundle: .module)) {
            if fans.isAvailable {
                Toggle(isOn: Binding(
                    get: { fans.keepFansPinnedOnQuit },
                    set: { newValue in
                        fans.keepFansPinnedOnQuit = newValue
                        model.keepFansPinnedOnQuit = newValue
                        model.persistSettingsDebounced()
                    }
                )) {
                    Text("Keep fans pinned when TurboSpark quits", bundle: .module)
                    .settingsControl("Keep fans pinned when TurboSpark quits", pane: .engine, timing: .immediate)
                }

                Text("ThermalForge is active. Fan speed can be monitored and pinned to maximum from the status bar fan readout.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
            } else {
                HStack(alignment: .top) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("ThermalForge Not Installed", bundle: .module)
                            .themedFont(.base, weight: .medium)
                        Text("Fan speed cannot be controlled without ThermalForge installed. Install ThermalForge to monitor and pin fan speeds for sustained performance.", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    }
                    Spacer()
                    Link(destination: URL(string: "https://github.com/ProducerGuy/ThermalForge")!) {
                        HStack(spacing: 4) {
                            Text("ThermalForge", bundle: .module)
                            Image(systemName: "arrow.up.forward.app")
                                .accessibilityHidden(true)
                        }
                    }
                    .appAction(.secondary, size: .small)
                }
            }
        }
    }

    // MARK: - Row pieces

    /// Locale-aware two-decimal readout ("0,70" in German), shared by the
    /// visible value and the slider's VoiceOver value so they cannot differ.
    static func twoDecimals(_ value: Double) -> String {
        value.formatted(.number.precision(.fractionLength(2)))
    }

    /// The visible value beside a slider. Hidden from VoiceOver because the
    /// slider already speaks the same value.
    private func readout(_ value: String, enabled: Bool = true) -> some View {
        Text(verbatim: value)
            .monospacedDigit()
            .foregroundStyle(enabled ? AnyShapeStyle(AppThemeColor.appText) : AnyShapeStyle(AppThemeColor.appSecondary))
            .fixedSize()
            .frame(minWidth: 40, alignment: .trailing)
            .accessibilityHidden(true)
    }

    /// The unit column after a number field. A minimum width (not a fixed
    /// one) keeps the fields aligned row to row without clipping a long
    /// translation of "tokens".
    private func unit(_ text: Text) -> some View {
        text
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
            .fixedSize()
            .frame(minWidth: Self.unitColumnWidth, alignment: .leading)
    }

    /// An empty unit column, so a row without a unit lines its field up
    /// with the rows that have one.
    private var unitSpacer: some View {
        Color.clear
            .frame(width: Self.unitColumnWidth, height: 1)
            .accessibilityHidden(true)
    }

    private static let unitColumnWidth: CGFloat = 44
}

/// The shared look of the pane's number fields: label hidden (the row shows
/// it), rounded border, and a width range instead of a fixed width.
private struct EngineNumberFieldStyle: ViewModifier {
    func body(content: Content) -> some View {
        content
            .labelsHidden()
            .textFieldStyle(.roundedBorder)
            .frame(minWidth: 80, idealWidth: 110, maxWidth: 140)
    }
}
