import SwiftUI
import TurboSpark

/// The inspector's Essentials block: the two settings that change what a
/// turn actually does, before any of the engine knobs.
///
/// This exists because the Model Settings pane offered eleven controls in one
/// flat list and the two a new user needs -- how hard the model thinks, and how
/// much conversation it can hold -- sat between "Cache Slots" and
/// "Speculation". Both are answered by the loaded checkpoint and this machine
/// rather than by preference, so both are presented as what is *available*
/// first and a choice second.
///
/// Everything else stays exactly where it was, in this file's sibling
/// `InspectorOptionsSection` ("Open & Architecture Options"). The drop-in
/// this section came from described those as living behind an
/// `AdvancedDisclosure`; no such type was delivered and none exists, so the
/// advanced knobs are still a flat section below this one rather than a
/// collapsed row.
@MainActor
struct InspectorEssentialsSection: View {
    @ObservedObject var model: AppModel

    var body: some View {
        Section(header: Text("Essentials", bundle: .module)) {
            ThinkingLevelControl(model: model)
            ContextLadderPicker(model: model)
        }
        .disabled(model.isRunning || model.isInstallingModel)
    }
}

// MARK: - Thinking level

/// Reasoning effort, offered as a segmented control over the levels THIS
/// checkpoint's template accepts.
///
/// **The set is not ours to decide.** `availableReasoningLevels` comes off
/// `SessionInfo.reasoningEfforts`, probed at open: Qwen 3.8 answers
/// `[.off, .low, .medium, .xhigh]` and RAISES on `.high`, where gpt-oss
/// answers `[.off, .low, .medium, .high]`. A control that offered
/// `Reasoning.allCases` shipped a menu entry that failed the turn with a
/// template error, which is why this renders the model's own list and nothing
/// else -- and why the empty case explains itself rather than showing four
/// dead buttons.
private struct ThinkingLevelControl: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text("Thinking level", bundle: .module)
                    .themedFont(.base, weight: .medium)
                Spacer()
                if model.reasoningPickerEnabled {
                    Text("FROM TEMPLATE", bundle: .module)
                        .themedFont(.micro, weight: .semibold)
                        .tracking(0.5)
                        .foregroundStyle(.appSecondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                        .help(Text("These levels were read from the loaded checkpoint's chat template at open, not from a per-family table.", bundle: .module))
                }
            }

            if model.reasoningPickerEnabled {
                Picker(selection: Binding(
                    get: { model.reasoning },
                    set: { model.setReasoning($0) }
                )) {
                    // Level names and descriptions are fixed English in
                    // `GenerateOptions.Reasoning` and `ReasoningLevelPolicy`;
                    // each one is a catalog key.
                    ForEach(model.availableReasoningLevels) { level in
                        Text(LocalizedStringKey(model.reasoningLabel(for: level)), bundle: .module).tag(level)
                    }
                } label: { Text("Thinking level", bundle: .module) }
                .pickerStyle(.segmented)
                .labelsHidden()
                .accessibilityLabel(Text("Thinking level", bundle: .module))
                .accessibilityValue(Text(LocalizedStringKey(model.reasoningLabel(for: model.reasoning)), bundle: .module))

                // The description belongs to the SELECTED level, so the
                // segmented control does not need a subtitle per segment.
                Text(LocalizedStringKey(model.reasoningDescription(for: model.reasoning)), bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(theme.metadataForeground)
                    .fixedSize(horizontal: false, vertical: true)
                    .tsAnimation(TSMotion.hover, value: model.reasoning)
            } else {
                unavailableExplanation
            }
        }
        .padding(.vertical, 2)
    }

    /// Two distinct reasons, named separately.
    ///
    /// `reasoningPickerEnabled` is `session != nil && isReasoningSupported`,
    /// and the user's next action differs per branch: load the model, or stop
    /// looking for a control this checkpoint does not have. One grey
    /// "unavailable" would cover both and help with neither.
    @ViewBuilder
    private var unavailableExplanation: some View {
        HStack(alignment: .firstTextBaseline, spacing: 7) {
            Image(systemName: model.session == nil ? "arrow.down.circle" : "minus.circle")
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            Group {
                if model.session == nil {
                    Text("Load the model to see which thinking levels it accepts.", bundle: .module)
                } else {
                    Text("This checkpoint's chat template has no reasoning levels.", bundle: .module)
                }
            }
            .themedFont(.small)
            .foregroundStyle(theme.metadataForeground)
            .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.vertical, 6)
        .accessibilityElement(children: .combine)
    }
}

// MARK: - Context window

/// The context window, priced per rung against this machine's memory, in ONE
/// control: a collapsed summary button that opens a popover of every
/// selectable option (each priced rung, Auto, and Custom).
///
/// **THIS USED TO BE TWO CONTROLS.** An always-expanded list of priced rows
/// sat above a separate, plain-text `Picker` (the only thing that could
/// actually express Auto or a custom size) -- the rows looked like the whole
/// control and were not, and the real dropdown was easy to miss below them.
/// Tapping a rung already set `model.maxContextTokens` directly; what moved
/// here is the DISPLAY, collapsing both into one button plus a popover, not
/// the underlying wiring.
///
/// **NOTHING HERE MULTIPLIES A FIGURE.** KV cost is not linear in the window:
/// a sliding-window layer is a ring capped at `sliding_window + 128` and stops
/// growing, while a full-attention layer grows forever. Measured 4K to 128K,
/// Mistral 7B grows 32x and Gemma 4 grows 9x -- so a control that scaled one
/// 4K reading by the context ratio is 3.5x high on Gemma, in the direction
/// that refuses a window the machine affords. `catalog::context_ladder` calls
/// `fit` per rung; this view renders those rungs and computes nothing.
///
/// An empty ladder therefore offers NO priced rows in the popover: never a
/// ladder of zeros, and never one filled from a family baseline. The popover
/// falls back to the fixed `AppContextLengthOption` presets (unpriced) in
/// that case, because Auto and a custom size are not rungs either way and a
/// probe failure should not remove the ability to pick a window at all.
struct ContextLadderPicker: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var ladder: ContextLadder?
    @State private var loadFailed = false
    @State private var isCustom = false

    var body: some View {
        VStack(alignment: .leading, spacing: 7) {
            HStack {
                Text("Context window", bundle: .module)
                    .themedFont(.base, weight: .medium)
                Spacer()
                if let physical = model.telemetry?.physicalMemoryBytes, physical > 0 {
                    Text("\(MetricFormat.memory(physical)) on this Mac", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                        .monospacedDigit()
                }
            }

            ContextSelectionButton(model: model, ladder: ladder, isCustom: $isCustom)

            if isCustom {
                customSizeRow
                    .transition(.opacity.combined(with: .move(edge: .top)))
            }

            if let rungs = ladder?.rungs, !rungs.isEmpty {
                Text("Each row is priced by the engine, not scaled from one figure.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)

                if rungs.contains(where: \.pastTrained) {
                    // Past the trained window WARNS rather than refusing:
                    // RoPE extrapolates rather than failing, and some
                    // checkpoints ship YaRN scaling meant to exceed it.
                    Text("Windows past the trained one still load. Quality beyond it is the checkpoint's business, not this engine's.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else if loadFailed {
                Text("Memory cost per window is unavailable for this model.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
        }
        .padding(.vertical, 2)
        .task(id: Self.ladderKey(path: model.selected?.path, slots: model.activeCacheSlots, loadGuard: model.activeLoadGuard)) {
            await loadLadder()
        }
        .onAppear { syncIsCustom() }
        .onChange(of: model.maxContextTokens) { _, _ in syncIsCustom() }
        .tsAnimation(.smooth(duration: 0.2), value: isCustom)
    }

    private var customSizeRow: some View {
        LabeledContent {
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
                .accessibilityLabel(Text("Custom context size", bundle: .module))
                .accessibilityValue(Text("\(model.maxContextTokens.formatted()) tokens", bundle: .module))
                Text(verbatim: "\(model.maxContextTokens.formatted())")
                    .themedFont(.small).monospacedDigit()
                    .frame(width: 55, alignment: .trailing)
            }
        } label: {
            Text("Custom Size", bundle: .module)
        }
    }

    /// `isCustom` is a stored flag, not a computed one, because choosing
    /// "Custom" from the popover has to WRITE two things as one gesture (the
    /// flag, and a seed value when the current setting is Auto) -- a purely
    /// computed property cannot do that. This keeps it in sync with
    /// `model.maxContextTokens` for every other path: a persisted setting
    /// from before this model was probed, a ladder that just finished
    /// loading, or a rung/preset tapped in the popover. Clears whenever the
    /// current token count matches Auto, a priced rung, or (with no ladder) a
    /// fixed preset, so a real, nameable selection never gets stuck reading
    /// as "Custom".
    private func syncIsCustom() {
        let tokens = model.maxContextTokens
        // Custom is a sticky user choice: dragging the slider across a rung
        // value, or seeding Custom from Auto with a rung-sized value, must not
        // flip it off. Every explicit row tap clears the flag itself.
        if isCustom && tokens != 0 { return }
        if tokens == 0 {
            isCustom = false
            return
        }
        if let rungs = ladder?.rungs, rungs.contains(where: { Int($0.context) == tokens }) {
            isCustom = false
            return
        }
        if (ladder?.rungs.isEmpty ?? true), AppContextLengthOption(rawValue: tokens) != nil {
            isCustom = false
            return
        }
        isCustom = true
    }

    /// Everything the ladder is priced from. `OpenOptions` values are not
    /// Equatable, so their descriptions stand in; without the slots and guard
    /// in the task id, changing either would leave a stale ladder.
    static func ladderKey(
        path: String?, slots: OpenOptions.Sizing, loadGuard: OpenOptions.LoadGuard
    ) -> String {
        "\(path ?? "")|\(slots)|\(loadGuard)"
    }

    /// Reads the INSTALLED model's own manifest through the same trio
    /// `ts_session_open` reads, so the ladder and the open cannot disagree.
    /// Priced with the user's load guard and cache slots so verdicts match
    /// what the loader will decide.
    private func loadLadder() async {
        let slots = model.activeCacheSlots
        let loadGuard = model.activeLoadGuard
        guard let path = model.selected?.path, !path.isEmpty else {
            ladder = nil
            loadFailed = false
            syncIsCustom()
            return
        }
        do {
            // Detached: the FFI call walks the install directory, and this
            // view is MainActor-inferred.
            let found = try await Task.detached(priority: .userInitiated) {
                try TurboSparkCatalog.contextLadder(
                    modelPath: path, expertCacheSlots: slots, loadGuard: loadGuard)
            }.value
            if Task.isCancelled { return }
            ladder = found
            loadFailed = false
        } catch {
            // Reported, not swallowed: a silent nil here is indistinguishable
            // from a model with no ArchConfig.
            FileHandle.standardError.write(
                Data("context ladder unavailable for \(path): \(error)\n".utf8))
            ladder = nil
            loadFailed = true
        }
        syncIsCustom()
    }
}
