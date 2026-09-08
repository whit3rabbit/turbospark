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
                        .themedFont(points: 9, weight: .semibold)
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

/// The context window, priced per rung against this machine's memory.
///
/// **NOTHING HERE MULTIPLIES A FIGURE.** KV cost is not linear in the window:
/// a sliding-window layer is a ring capped at `sliding_window + 128` and stops
/// growing, while a full-attention layer grows forever. Measured 4K to 128K,
/// Mistral 7B grows 32x and Gemma 4 grows 9x -- so a control that scaled one
/// 4K reading by the context ratio is 3.5x high on Gemma, in the direction
/// that refuses a window the machine affords. `catalog::context_ladder` calls
/// `fit` per rung; this view renders those rungs and computes nothing.
///
/// An empty ladder therefore renders NO ladder: never a ladder of zeros, and
/// never one filled from a family baseline. `ContextWindowOptionsView` sits
/// under the rungs either way, because `Auto` and a custom size are not rungs
/// and a ladder alone would be a one-way door out of both.
private struct ContextLadderPicker: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var ladder: ContextLadder?
    @State private var loadFailed = false

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

            if let rungs = ladder?.rungs, !rungs.isEmpty {
                VStack(spacing: 1) {
                    ForEach(rungs) { rung in
                        rungRow(rung)
                    }
                }

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

            // ALWAYS present, ladder or no ladder. A rung is a concrete
            // window, so `Auto` (resolve from the checkpoint's trained
            // context and this machine's memory) and an arbitrary custom
            // size have no rung to be -- picking one off the ladder would
            // otherwise be a one-way door out of Auto. When there is no
            // ArchConfig there is no ladder at all, and this is the whole
            // control rather than the escape hatch under it.
            ContextWindowOptionsView(model: model)
        }
        .padding(.vertical, 2)
        .task(id: model.selected?.path) { await loadLadder() }
    }

    /// One rung: window, its measured cost, the engine's verdict, and what is
    /// notable about it. A refused rung is DISABLED -- it is the RAM ceiling
    /// made visible, and offering it would let the user pick a window the
    /// open then refuses.
    @ViewBuilder
    private func rungRow(_ rung: ContextLadderRung) -> some View {
        let presentation = ModelFitPresentation.of(rung.verdict)
        let isCurrent = model.resolvedContextTokens == Int(rung.context)
        let isRefused = rung.verdict == .refused

        Button {
            model.maxContextTokens = Int(rung.context)
        } label: {
            HStack(spacing: 6) {
                // minWidth, NOT a fixed width. A fixed one has to be sized
                // against the widest value the ladder can ever hold, and
                // sizing it against the ladder in front of you is how this
                // got shipped wrapping: a 3-rung gptoss ladder tops out at
                // 16,384, and the full one carries 131,072 and 1,007,616,
                // which broke across two lines at width 52. `fixedSize`
                // is what actually forbids the wrap; the frame only keeps
                // the small values right-aligned with each other.
                Text(rung.context.formatted())
                    .themedFont(.small, weight: isCurrent ? .semibold : .regular)
                    .monospacedDigit()
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                    .frame(minWidth: 52, alignment: .trailing)

                Text(MetricFormat.storage(rung.counted))
                    .themedFont(.small)
                    .monospacedDigit()
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)
                    .frame(minWidth: 56, alignment: .trailing)
                    .foregroundStyle(.secondary)

                Circle()
                    .fill(presentation.color)
                    .frame(width: 7, height: 7)
                    .accessibilityHidden(true)

                // The inspector column is ~260pt and these five cells do not
                // all fit at every text size. The VERDICT is the one that
                // must never wrap or truncate ("Resident" broke across two
                // lines at the shipped width), so it is fixed-size and the
                // marker beside it is what gives way.
                Text(presentation.compactLabel)
                    .themedFont(.small)
                    .foregroundStyle(presentation.color)
                    .lineLimit(1)
                    .fixedSize(horizontal: true, vertical: false)

                Spacer(minLength: 4)

                // Whole or not at all. Truncation leaves "t..." for
                // "trained" and "beyo..." for "beyond", which spends the
                // width and says nothing; the same words are in the row's
                // help text and its accessibility value either way.
                if let marker = marker(rung) {
                    ViewThatFits(in: .horizontal) {
                        Text(marker)
                            .themedFont(.tiny)
                            .foregroundStyle(.tertiary)
                            .lineLimit(1)
                        Color.clear.frame(width: 0, height: 0)
                    }
                    .layoutPriority(-1)
                }
            }
            .padding(.horizontal, 7)
            .padding(.vertical, 5)
            .background {
                RoundedRectangle(cornerRadius: 7, style: .continuous)
                    .fill(isCurrent ? theme.accent.opacity(0.12) : .clear)
            }
            .contentShape(.rect(cornerRadius: 7))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.99))
        .disabled(isRefused)
        .opacity(isRefused ? 0.45 : 1)
        .help(
            isRefused
                ? "Needs more memory than this Mac has, with this model loaded."
                : "Use a \(rung.context.formatted()) token window (\(MetricFormat.storage(rung.counted)) of KV cache)")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("\(rung.context.formatted()) tokens")
        .accessibilityValue(
            "\(MetricFormat.storage(rung.counted)), \(presentation.label)"
                + (isCurrent ? ", current" : "")
                + (isRefused ? ", too large for this Mac" : ""))
        .accessibilityAddTraits(isCurrent ? [.isButton, .isSelected] : .isButton)
    }

    /// One short word, or nothing.
    ///
    /// The inspector row has about 50pt left after the window, the cost and
    /// the verdict, which is one `.tiny` word -- "current" alone truncated to
    /// "c..." at the shipped width. So: CURRENT IS NOT A MARKER HERE. The
    /// highlighted background and the semibold digits already say it, and the
    /// accessibility value below still spells it for VoiceOver, which is the
    /// reader that cannot see either. The other three are the ones a user
    /// cannot infer from the row itself, and they are mutually exclusive in
    /// practice, so the joined form was never really reachable.
    private func marker(_ rung: ContextLadderRung) -> String? {
        if rung.pastTrained { return "beyond" }
        if rung.isTrainedMax { return "trained" }
        if rung.isLargestFitting { return "max fit" }
        return nil
    }

    /// Reads the INSTALLED model's own manifest through the same trio
    /// `ts_session_open` reads, so the ladder and the open cannot disagree.
    private func loadLadder() async {
        guard let path = model.selected?.path, !path.isEmpty else {
            ladder = nil
            loadFailed = false
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
    }
}
