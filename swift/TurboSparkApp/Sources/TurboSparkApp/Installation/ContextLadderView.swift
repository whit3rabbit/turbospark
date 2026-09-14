import SwiftUI
import TurboSpark

/// What a longer context window costs, window by window.
///
/// **THE TABLE EXISTS BECAUSE THE CURVE IS NOT LINEAR.** A sliding-window
/// layer is a ring capped at `sliding_window + 128` and stops growing past
/// it, while a fully-attentive layer grows forever. Measured 4,096 to 131,072
/// on the shipped baselines: Mistral 7B grows 32x (512 MiB to 16,384) and
/// Gemma 4 grows 9x (305 to 2,785). So this renders what the engine computed
/// and never multiplies a figure of its own.
@MainActor
struct ContextLadderView: View {
    let rungs: [ContextLadderRung]
    let trainedContext: UInt32?
    /// The window currently selected, highlighted so the table reads as "you
    /// are here, and here is what moving costs".
    var currentContext: UInt32? = nil

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack {
                Label { Text("Memory by context window", bundle: .module) } icon: { Image(systemName: "arrow.left.and.right") }
                    .themedFont(.small, weight: .semibold)
                    .foregroundStyle(.appSecondary)
                Spacer()
                if let trained = trainedContext {
                    Text("trained to \(trained.formatted())", bundle: .module)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
            }

            ForEach(rungs) { rung in
                row(rung)
            }

            if rungs.contains(where: \.pastTrained) {
                // Past the trained window WARNS rather than refusing: RoPE
                // extrapolates rather than failing, and some checkpoints ship
                // YaRN scaling meant to exceed it.
                Text("Windows past the trained one still load. Quality beyond it is the checkpoint's business, not this engine's.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }

    @ViewBuilder
    private func row(_ rung: ContextLadderRung) -> some View {
        let p = ModelFitPresentation.of(rung.verdict)
        let isCurrent = currentContext == rung.context
        HStack(spacing: 8) {
            Text(rung.context.formatted())
                .themedFont(.small, weight: isCurrent ? .bold : .regular).monospacedDigit()
                .frame(width: 68, alignment: .trailing)
            Text(MetricFormat.storage(rung.counted))
                .themedFont(.small).monospacedDigit()
                .frame(width: 78, alignment: .trailing)
                .foregroundStyle(.appSecondary)
            Circle()
                .fill(p.color)
                .frame(width: 7, height: 7)
                .accessibilityHidden(true)
            Text(p.compactLabel)
                .themedFont(.small)
                .foregroundStyle(p.color)
            Spacer()
            Text(marker(rung, isCurrent: isCurrent))
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
        .padding(.vertical, 1)
        .padding(.horizontal, 4)
        .background(
            isCurrent
                ? Color.accentColor.opacity(0.10) : Color.clear,
            in: RoundedRectangle(cornerRadius: 4)
        )
        .accessibilityElement(children: .combine)
        .accessibilityLabel(
            "\(rung.context.formatted()) tokens, \(MetricFormat.storage(rung.counted)), \(p.label)")
    }

    private func marker(_ rung: ContextLadderRung, isCurrent: Bool) -> String {
        var parts: [String] = []
        if isCurrent { parts.append("current") }
        if rung.isTrainedMax { parts.append("trained max") }
        if rung.isLargestFitting { parts.append("largest that fits") }
        if rung.pastTrained { parts.append("past trained") }
        return parts.joined(separator: ", ")
    }
}

/// A measured decode band, with the silicon it was taken on.
///
/// **THE CHIP IS SHOWN WHENEVER IT IS NOT THIS MACHINE'S.** tok/s does not
/// transfer across silicon, so an unlabelled band would present another
/// machine's number as this one's answer. Showing nothing was the previous
/// behaviour and left every non-M4-Max user with no signal at all.
@MainActor
struct ThroughputRowView: View {
    let band: ThroughputBand

    var body: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(spacing: 6) {
                Text("Decode", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                Text(
                    String(
                        format: "%.1f to %.1f tok/s", band.minTokensPerSecond,
                        band.maxTokensPerSecond)
                )
                .themedFont(.small).monospacedDigit()
                if band.measuredOnThisChip {
                    Text("measured", bundle: .module)
                        .themedFont(.tiny, weight: .semibold)
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.green.opacity(0.16), in: Capsule())
                        .foregroundStyle(.green)
                }
                Spacer()
            }
            if !band.measuredOnThisChip {
                Text("measured on \(band.chip), not this machine. Rates do not transfer across silicon.", bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
