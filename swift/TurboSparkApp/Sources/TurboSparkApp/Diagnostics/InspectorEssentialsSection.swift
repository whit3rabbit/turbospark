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
        Section("Essentials") {
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
                        .foregroundStyle(.secondary)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 3)
                        .background(Color.primary.opacity(0.06), in: RoundedRectangle(cornerRadius: 4))
                        .help("These levels were read from the loaded checkpoint's chat template at open, not from a per-family table.")
                }
            }

            if model.reasoningPickerEnabled {
                Picker("Thinking level", selection: Binding(
                    get: { model.reasoning },
                    set: { model.setReasoning($0) }
                )) {
                    ForEach(model.availableReasoningLevels) { level in
                        Text(model.reasoningLabel(for: level)).tag(level)
                    }
                }
                .pickerStyle(.segmented)
                .labelsHidden()
                .accessibilityLabel("Thinking level")
                .accessibilityValue(model.reasoningLabel(for: model.reasoning))

                // The description belongs to the SELECTED level, so the
                // segmented control does not need a subtitle per segment.
                Text(model.reasoningDescription(for: model.reasoning))
                    .themedFont(.small)
                    .foregroundStyle(theme.metadataForeground)
                    .fixedSize(horizontal: false, vertical: true)
                    .animation(TSMotion.hover, value: model.reasoning)
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
                .foregroundStyle(.tertiary)
                .accessibilityHidden(true)
            Text(
                model.session == nil
                    ? "Load the model to see which thinking levels it accepts."
                    : "This checkpoint's chat template has no reasoning levels."
            )
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
private struct ContextLadderPicker: View {
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
                        .foregroundStyle(.secondary)
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
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if rungs.contains(where: \.pastTrained) {
                    // Past the trained window WARNS rather than refusing:
                    // RoPE extrapolates rather than failing, and some
                    // checkpoints ship YaRN scaling meant to exceed it.
                    Text("Windows past the trained one still load. Quality beyond it is the checkpoint's business, not this engine's.", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.secondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
            } else if loadFailed {
                Text("Memory cost per window is unavailable for this model.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(.vertical, 2)
        .task(id: model.selected?.path) { await loadLadder() }
        .onAppear { syncIsCustom() }
        .onChange(of: model.maxContextTokens) { _, _ in syncIsCustom() }
        .animation(.smooth(duration: 0.2), value: isCustom)
    }

    private var customSizeRow: some View {
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
                Text("\(model.maxContextTokens.formatted())", bundle: .module)
                    .themedFont(.small).monospacedDigit()
                    .frame(width: 55, alignment: .trailing)
            }
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

    /// Reads the INSTALLED model's own manifest through the same trio
    /// `ts_session_open` reads, so the ladder and the open cannot disagree.
    private func loadLadder() async {
        guard let path = model.selected?.path, !path.isEmpty else {
            ladder = nil
            loadFailed = false
            syncIsCustom()
            return
        }
        do {
            ladder = try TurboSparkCatalog.contextLadder(modelPath: path)
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

/// The collapsed control: one row showing the current selection, tapping it
/// opens `ContextSelectionPopover`.
private struct ContextSelectionButton: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let ladder: ContextLadder?
    @Binding var isCustom: Bool
    @State private var isPresented = false

    private var currentRung: ContextLadderRung? {
        ladder?.rungs.first { Int($0.context) == model.maxContextTokens }
    }

    var body: some View {
        Button {
            isPresented = true
        } label: {
            HStack(spacing: 6) {
                summary
                Spacer(minLength: 4)
                Image(systemName: "chevron.up.chevron.down")
                    .themedFont(.micro, weight: .semibold)
                    .foregroundStyle(.tertiary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(Color.primary.opacity(0.05), in: RoundedRectangle(cornerRadius: 7, style: .continuous))
            .contentShape(.rect(cornerRadius: 7))
        }
        .buttonStyle(.plain)
        .accessibilityLabel("Context window")
        .accessibilityValue(summaryAccessibilityValue)
        .popover(isPresented: $isPresented, arrowEdge: .bottom) {
            ContextSelectionPopover(
                model: model,
                ladder: ladder,
                isCustom: $isCustom,
                onDismiss: { isPresented = false })
        }
    }

    // `.small`/`.tiny` are the sidebar rows' own roles in the canonical type
    // scale, so this button renders at the same size as the sidebar and chat
    // chrome it sits beside.
    @ViewBuilder
    private var summary: some View {
        if let rung = currentRung {
            let presentation = ModelFitPresentation.of(rung.verdict)
            Circle()
                .fill(presentation.color)
                .frame(width: 7, height: 7)
                .accessibilityHidden(true)
            Text(rung.context.formatted())
                .font(theme.ui(.small, weight: .semibold))
                .monospacedDigit()
            Text(MetricFormat.storage(rung.counted))
                .font(theme.ui(.tiny))
                .foregroundStyle(.secondary)
                .monospacedDigit()
            Text(presentation.compactLabel)
                .font(theme.ui(.tiny))
                .foregroundStyle(presentation.color)
        } else if model.maxContextTokens == 0 {
            Text("Auto (\(model.resolvedContextTokens.formatted()))", bundle: .module)
                .font(theme.ui(.small, weight: .semibold))
        } else if isCustom {
            Text("Custom \u{00B7} \(model.maxContextTokens.formatted())", bundle: .module)
                .font(theme.ui(.small, weight: .semibold))
                .monospacedDigit()
        } else {
            Text(model.maxContextTokens.formatted())
                .font(theme.ui(.small, weight: .semibold))
                .monospacedDigit()
        }
    }

    private var summaryAccessibilityValue: String {
        if let rung = currentRung {
            let presentation = ModelFitPresentation.of(rung.verdict)
            return "\(rung.context.formatted()) tokens, \(MetricFormat.storage(rung.counted)), \(presentation.label)"
        }
        if model.maxContextTokens == 0 {
            return "Auto, \(model.resolvedContextTokens.formatted()) tokens"
        }
        return "\(model.maxContextTokens.formatted()) tokens"
    }
}

/// The popover's contents: Auto, then every priced rung (or, with no ladder,
/// the fixed presets), then Custom. One `onSelect` per row, mirroring
/// `ToolApprovalMenuPopover`'s shape (icon/dot, one or two lines of text, a
/// trailing checkmark on the current selection) -- that view is this app's
/// existing precedent for a rich per-row menu, needed because a native
/// `Picker`/`Menu` on macOS can only show plain text per row
/// (`ModelLoaderControl.swift`'s `.menuStyle(.button)` gotcha).
private struct ContextSelectionPopover: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    let ladder: ContextLadder?
    @Binding var isCustom: Bool
    let onDismiss: () -> Void

    var body: some View {
        VStack(spacing: 2) {
            autoRow

            if let rungs = ladder?.rungs, !rungs.isEmpty {
                ForEach(rungs) { rung in
                    rungRow(rung)
                }
            } else {
                ForEach(AppContextLengthOption.allCases.filter { $0 != .auto }) { option in
                    fixedOptionRow(option)
                }
            }

            customRow
        }
        .padding(6)
        .frame(width: 300)
        .background(TurboSparkTheme.surfaceColor)
    }

    private var autoRow: some View {
        optionRow(
            isSelected: model.maxContextTokens == 0,
            primary: Text("Auto", bundle: .module),
            secondary: Text("\(model.resolvedContextTokens.formatted()) tokens, resolved from the checkpoint and this machine", bundle: .module),
            dotColor: nil,
            disabled: false
        ) {
            isCustom = false
            model.maxContextTokens = 0
            onDismiss()
        }
    }

    private func rungRow(_ rung: ContextLadderRung) -> some View {
        let presentation = ModelFitPresentation.of(rung.verdict)
        let isSelected = !isCustom && Int(rung.context) == model.maxContextTokens
        let isRefused = rung.verdict == .refused
        var secondary = "\(MetricFormat.storage(rung.counted)) \u{00B7} \(presentation.label)"
        if let marker = marker(rung) {
            secondary += " \u{00B7} \(marker)"
        }

        return optionRow(
            isSelected: isSelected,
            primary: Text(rung.context.formatted()),
            secondary: Text(secondary),
            dotColor: presentation.color,
            disabled: isRefused
        ) {
            isCustom = false
            model.maxContextTokens = Int(rung.context)
            onDismiss()
        }
        .help(
            isRefused
                ? "Needs more memory than this Mac has, with this model loaded."
                : "Use a \(rung.context.formatted()) token window (\(MetricFormat.storage(rung.counted)) of KV cache)")
        .accessibilityLabel("\(rung.context.formatted()) tokens")
        .accessibilityValue("\(MetricFormat.storage(rung.counted)), \(presentation.label)"
            + (isRefused ? ", too large for this Mac" : ""))
    }

    private func fixedOptionRow(_ option: AppContextLengthOption) -> some View {
        optionRow(
            isSelected: !isCustom && model.maxContextTokens == option.tokens,
            primary: Text(option.menuLabel),
            secondary: nil,
            dotColor: nil,
            disabled: false
        ) {
            isCustom = false
            model.maxContextTokens = option.tokens
            onDismiss()
        }
    }

    private var customRow: some View {
        optionRow(
            isSelected: isCustom,
            primary: Text("Custom\u{2026}", bundle: .module),
            secondary: Text("Pick any size from 512 to 131,072", bundle: .module),
            dotColor: nil,
            disabled: false
        ) {
            if model.maxContextTokens == 0 {
                model.maxContextTokens = model.resolvedContextTokens
            }
            isCustom = true
            onDismiss()
        }
    }

    /// One short word, or nothing. Mirrors the marker this control used to
    /// show as a separate trailing column on its always-visible rows; here it
    /// is folded into the secondary line since the popover has room for it.
    private func marker(_ rung: ContextLadderRung) -> String? {
        if rung.pastTrained { return "beyond" }
        if rung.isTrainedMax { return "trained" }
        if rung.isLargestFitting { return "max fit" }
        return nil
    }

    @ViewBuilder
    private func optionRow(
        isSelected: Bool,
        primary: Text,
        secondary: Text?,
        dotColor: Color?,
        disabled: Bool,
        onSelect: @escaping () -> Void
    ) -> some View {
        Button(action: onSelect) {
            HStack(spacing: 8) {
                if let dotColor {
                    Circle()
                        .fill(dotColor)
                        .frame(width: 7, height: 7)
                        .accessibilityHidden(true)
                }
                VStack(alignment: .leading, spacing: 1) {
                    primary
                        .font(theme.ui(.small, weight: isSelected ? .semibold : .regular))
                        .foregroundStyle(Color.primary)
                    if let secondary {
                        secondary
                            .font(theme.ui(.tiny))
                            .foregroundStyle(.secondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 4)
                if isSelected {
                    Image(systemName: "checkmark")
                        .themedFont(.tiny, weight: .bold)
                        .foregroundStyle(Color.primary)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(
                isSelected ? theme.accent.opacity(0.12) : Color.clear,
                in: RoundedRectangle(cornerRadius: 6, style: .continuous))
            .contentShape(.rect(cornerRadius: 6))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.99))
        .disabled(disabled)
        .opacity(disabled ? 0.45 : 1)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}
