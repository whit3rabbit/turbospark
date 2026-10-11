import AppKit
import SwiftUI
import TurboSpark

/// Card displaying hardware compatibility and sizing metrics for a model.
struct ModelHardwareFitCardView: View {
    let recommendation: ModelRecommendation?
    let entry: CatalogEntry

    /// Whether this machine has actually sized the row.
    ///
    /// A `.unknown` verdict means the sizing could not be determined, and the
    /// numeric fields that come with it are zeros rather than measurements.
    private var fitIsKnown: Bool {
        guard let recommendation else { return false }
        return recommendation.verdict != .unknown
    }

    var body: some View {
        if let recommendation, fitIsKnown {
            knownFitCard(recommendation)
        } else {
            unprobedFitCard
        }
    }

    private func knownFitCard(_ rec: ModelRecommendation) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "gauge.with.needle")
                    .themedFont(.small)
                    .foregroundStyle(Color.accentColor)
                    .help(Text("Hardware compatibility estimate", bundle: .module))
                Text("Hardware Compatibility", bundle: .module)
                    .themedFont(.base, weight: .semibold)

                Spacer()

                verdictPill(rec.verdict)
            }

            Text(rec.verdictSummary)
                .themedFont(.base)
                .foregroundStyle(.appSecondary)

            Divider()

            // EVERY CELL IS GUARDED ON BEING NON-ZERO. A zero here is an
            // absent reading, not a measurement of zero, and it rendered as
            // "Zero KB" and "0 tokens" -- which read like facts about the
            // model rather than like missing data.
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 110, maximum: 200), spacing: 12)], spacing: 12) {
                if rec.countedBytes > 0 {
                    metricCell(title: "Unified Memory", value: Text(verbatim: MetricFormat.storage(rec.countedBytes)))
                }
                if rec.installBytes > 0 {
                    metricCell(title: "On-Disk Storage", value: Text(verbatim: MetricFormat.storage(rec.installBytes)))
                }
                if rec.largestContext > 0 {
                    metricCell(title: "Max Context", value: Text("\(rec.largestContext.formatted()) tokens", bundle: .module))
                }
                if rec.slotCacheSlots > 0 {
                    metricCell(title: "Expert Slots", value: Text(verbatim: "\(rec.slotCacheSlots)"))
                }
                if let minS = rec.toksPerSecondMin, let maxS = rec.toksPerSecondMax {
                    metricCell(
                        title: "Expected Speed",
                        value: Text(verbatim: "\(String(format: "%.1f", minS))-\(String(format: "%.1f", maxS)) tok/s"))
                }
            }
        }
        .modelCardStyle()
    }

    /// Shown when the row has no sizing for this machine.
    ///
    /// The previous version rendered the recommendation's zeros through the
    /// same grid, so an unsized row advertised "Unified Memory: Zero KB",
    /// "Max Context: 0 tokens" and "Expert Slots: 16 slots". The last one is
    /// the worst of the three and is a documented trap: with no architecture
    /// read there is no expert stride, `Auto` divides by nothing and returns
    /// `DEFAULT_CACHE_SLOTS`, so 16 is the answer arrived at BY IGNORANCE and
    /// looks identical to a real one (root `CLAUDE.md` Gotcha 58).
    ///
    /// What is shown instead is only what the catalog itself states, plus the
    /// command that would produce the missing numbers. There is deliberately
    /// no Probe button: `ts_recommend_json` is offline-only and this binding
    /// exposes no probing call, so a button here could not do the work.
    private var unprobedFitCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "gauge.with.needle")
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
                Text("Hardware Compatibility", bundle: .module)
                    .themedFont(.base, weight: .semibold)

                Spacer()

                Text("Not sized", bundle: .module)
                    .themedFont(.small, weight: .bold)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.secondary.opacity(0.16), in: Capsule())
                    .foregroundStyle(.appSecondary)
            }

            Text("This row has no memory sizing for this machine, so whether it fits, how much it would pin, and its largest usable context are all unknown.", bundle: .module)
                .themedFont(.base)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            Divider()

            LazyVGrid(columns: [GridItem(.adaptive(minimum: 110, maximum: 200), spacing: 12)], spacing: 12) {
                if entry.installBytes > 0 {
                    metricCell(title: "On-Disk Storage", value: Text(verbatim: MetricFormat.storage(entry.installBytes)))
                }
                if entry.downloadBytes > 0 {
                    metricCell(title: "Download", value: Text(verbatim: MetricFormat.storage(entry.downloadBytes)))
                }
            }

            Text("Read the headers to size it: turbospark-model recommend --probe", bundle: .module)
                .themedCode(.small)
                .foregroundStyle(.tertiary)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .modelCardStyle()
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Hardware compatibility: not sized on this machine", bundle: .module))
    }

    /// `title` is a catalog key; `value` is already a display `Text`.
    private func metricCell(title: LocalizedStringKey, value: Text) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(title, bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
            value
                .themedFont(.base, weight: .medium).monospacedDigit()
        }
    }

    @ViewBuilder
    private func verdictPill(_ verdict: ModelRecommendation.FitVerdict) -> some View {
        ModelFitVerdictPill(verdict: verdict)
    }
}
