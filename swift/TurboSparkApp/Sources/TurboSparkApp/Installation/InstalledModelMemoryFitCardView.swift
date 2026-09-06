import SwiftUI
import TurboSpark

/// Memory fit card displaying unified memory footprint, working set explanation, and context ladder.
struct InstalledModelMemoryFitCardView: View {
    let fit: ModelRecommendation?
    let descriptor: ModelFeatureDescriptor
    let installedModel: InstalledModel
    let ladder: ContextLadder?
    let activeFitContext: UInt32
    let resolvedSlots: Int?

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Label("Unified Memory & Working Set", systemImage: "memorychip")
                    .font(.subheadline.weight(.semibold))
                Spacer()
                if let fit {
                    ModelFitVerdictPill(verdict: fit.verdict)
                }
            }

            // **THE FIGURES ARE GUARDED ON THEIR SOURCE, NOT ON THEIR VALUE.**
            // A row whose checkpoint header nobody has read reports zeros in
            // every sizing field, and a zero rendered as a figure reads as
            // "fits easily" -- the exact inverse (swift Gotcha 23). This card
            // used to state a hardcoded 2.2 GB for gemma4, 1.7 GB for qwen36
            // and "unknown" for every other MoE family, beside a prose blurb
            // claiming "~1.6 to 2.2 GB peak RAM" for all of them regardless
            // of layer count or expert stride.
            if let fit, fit.countedSource != .unknown {
                allocationRow(fit)
                Text(workingSetExplanation(fit))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)

                if let ladder, !ladder.rungs.isEmpty {
                    Divider().padding(.vertical, 2)
                    ContextLadderView(
                        rungs: ladder.rungs,
                        trainedContext: ladder.trainedContext,
                        currentContext: activeFitContext
                    )
                }
            } else {
                Text("RAM requirement: unknown")
                    .font(.caption.monospacedDigit().weight(.medium))
                    .foregroundStyle(.secondary)
                Text("Nothing has read this checkpoint's header, so its working set is unknown. `turbospark-model recommend --probe` reads it without downloading anything.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .padding(14)
        .background(Color(nsColor: .controlBackgroundColor), in: RoundedRectangle(cornerRadius: 10))
    }

    /// The two byte figures, which answer different questions.
    ///
    /// `countedBytes` is what the engine ALLOCATES, so exceeding memory there
    /// is a failed load. The install on disk is what it READS, and exceeding
    /// memory there is the streaming this engine is built around: it costs
    /// throughput rather than correctness.
    @ViewBuilder
    private func allocationRow(_ fit: ModelRecommendation) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            HStack(spacing: 6) {
                Text("Allocates")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text(MetricFormat.storage(fit.countedBytes))
                    .font(.caption.monospacedDigit().weight(.medium))
                    .foregroundStyle(Color.accentColor)
                // A measured figure and an estimate are different claims and
                // must not read the same.
                if fit.countedSource == .measured {
                    Text("measured")
                        .font(.caption2.weight(.semibold))
                        .padding(.horizontal, 5)
                        .padding(.vertical, 1)
                        .background(Color.green.opacity(0.16), in: Capsule())
                        .foregroundStyle(.green)
                        .help("A frozen benchmark row taken on this chip at this context and slot count.")
                }
                Spacer()
            }
            // A footprint is only meaningful beside its context and slot
            // count: Gemma 4 reads 2,175 MiB at 16 slots and 3,654 at 32.
            Text(configurationCaption(fit))
                .font(.caption2)
                .foregroundStyle(.secondary)
            HStack(spacing: 6) {
                Text("Reads")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Text("\(MetricFormat.storage(installedModel.installBytes)) on disk")
                    .font(.caption.monospacedDigit())
            }
            // **THE BAND MAY BELONG TO OTHER SILICON, AND THEN IT NAMES THE
            // CHIP.** Every frozen row in the catalog was taken on one
            // machine, so a chip-matched lookup answers nothing on any other
            // Mac and this row used to vanish entirely. Reporting a
            // measurement at its own stated configuration is honest;
            // presenting it as this machine's answer is not.
            if let band = fit.throughput {
                ThroughputRowView(band: band)
            }
        }
    }

    private func configurationCaption(_ fit: ModelRecommendation) -> String {
        let slots = resolvedSlots ?? fit.slotCacheSlots
        let context = activeFitContext
        let base = "at \(context.formatted()) tokens"
        return descriptor.isSlotCacheStreaming
            ? "\(base), \(slots) expert-cache slots"
            : base
    }

    private func workingSetExplanation(_ fit: ModelRecommendation) -> String {
        if descriptor.isSlotCacheStreaming {
            return "Mixture-of-Experts. Routed expert weights are read from storage on demand through a working-memory slot cache, so what has to fit is the cache and the KV rather than the whole install."
        }
        return "Dense. Every token touches every weight layer, so the weights are mapped into unified memory. A mapped range is not charged to the process footprint, which is why the allocated figure is far below the size on disk."
    }
}
