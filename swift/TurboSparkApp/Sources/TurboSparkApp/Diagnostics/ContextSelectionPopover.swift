import SwiftUI
import TurboSpark

/// The collapsed control: one row showing the current selection, tapping it
/// opens `ContextSelectionPopover`. Split out of `InspectorEssentialsSection`
/// (where `ContextLadderPicker` hosts it) to keep that file readable.
struct ContextSelectionButton: View {
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
                    .foregroundStyle(.appSecondary)
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(
                .appSurface,
                in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .contentShape(.rect(cornerRadius: AppMetrics.controlRadius))
        }
        .buttonStyle(.plain)
        .accessibilityLabel(Text("Context window", bundle: .module))
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
                .foregroundStyle(.appSecondary)
                .monospacedDigit()
            // The fit labels are fixed English in `ModelFitPresentation`;
            // each one is a catalog key.
            Text(LocalizedStringKey(presentation.compactLabel), bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(presentation.color)
        } else if model.maxContextTokens == 0 {
            Text("Auto (\(model.resolvedContextTokens.formatted()))", bundle: .module)
                .font(theme.ui(.small, weight: .semibold))
        } else if isCustom {
            (Text("Custom", bundle: .module) + Text(verbatim: " \u{00B7} \(model.maxContextTokens.formatted())"))
                .font(theme.ui(.small, weight: .semibold))
                .monospacedDigit()
        } else {
            Text(model.maxContextTokens.formatted())
                .font(theme.ui(.small, weight: .semibold))
                .monospacedDigit()
        }
    }

    private var summaryAccessibilityValue: Text {
        if let rung = currentRung {
            let presentation = ModelFitPresentation.of(rung.verdict)
            return Text("\(rung.context.formatted()) tokens", bundle: .module)
                + Text(verbatim: ", \(MetricFormat.storage(rung.counted)), ")
                + Text(LocalizedStringKey(presentation.label), bundle: .module)
        }
        if model.maxContextTokens == 0 {
            return Text("Auto", bundle: .module)
                + Text(verbatim: ", ")
                + Text("\(model.resolvedContextTokens.formatted()) tokens", bundle: .module)
        }
        return Text("\(model.maxContextTokens.formatted()) tokens", bundle: .module)
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
        .background(.appSurface)
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
        let storage = MetricFormat.storage(rung.counted)
        let fitLabel = Text(LocalizedStringKey(presentation.label), bundle: .module)
        var secondary = Text(verbatim: "\(storage) \u{00B7} ") + fitLabel
        if let marker = marker(rung) {
            secondary = secondary + Text(verbatim: " \u{00B7} ") + marker
        }

        return optionRow(
            isSelected: isSelected,
            primary: Text(rung.context.formatted()),
            secondary: secondary,
            dotColor: presentation.color,
            disabled: isRefused
        ) {
            isCustom = false
            model.maxContextTokens = Int(rung.context)
            onDismiss()
        }
        .help(
            isRefused
                ? Text("Needs more memory than this Mac has, with this model loaded.", bundle: .module)
                : Text("Use a \(rung.context.formatted()) token window (\(storage) of KV cache)", bundle: .module))
        .accessibilityLabel(Text("\(rung.context.formatted()) tokens", bundle: .module))
        .accessibilityValue(
            Text(verbatim: "\(storage), ") + fitLabel
                + (isRefused ? Text(verbatim: ", ") + Text("too large for this Mac", bundle: .module) : Text(verbatim: "")))
    }

    private func fixedOptionRow(_ option: AppContextLengthOption) -> some View {
        // `.auto` is filtered out above, so every label here is a number.
        optionRow(
            isSelected: !isCustom && model.maxContextTokens == option.tokens,
            primary: Text(verbatim: option.menuLabel),
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
    private func marker(_ rung: ContextLadderRung) -> Text? {
        if rung.pastTrained { return Text("beyond", bundle: .module) }
        if rung.isTrainedMax { return Text("trained", bundle: .module) }
        if rung.isLargestFitting { return Text("max fit", bundle: .module) }
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
                        .foregroundStyle(.appText)
                    if let secondary {
                        secondary
                            .font(theme.ui(.tiny))
                            .foregroundStyle(.appSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                Spacer(minLength: 4)
                if isSelected {
                    Image(systemName: "checkmark")
                        .themedFont(.tiny, weight: .bold)
                        .foregroundStyle(.appText)
                }
            }
            .padding(.horizontal, 8)
            .padding(.vertical, 6)
            .background(
                isSelected ? theme.accent.opacity(0.12) : Color.clear,
                in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .contentShape(.rect(cornerRadius: AppMetrics.controlRadius))
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.99))
        .disabled(disabled)
        .opacity(disabled ? 0.45 : 1)
        .accessibilityAddTraits(isSelected ? [.isButton, .isSelected] : .isButton)
    }
}
