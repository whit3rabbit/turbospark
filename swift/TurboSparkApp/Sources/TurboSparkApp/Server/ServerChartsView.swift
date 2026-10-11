import Charts
import SwiftUI

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Three charts over the rolling window, each answering one question.
///
/// **EVERY SERIES IS FROM A MEASUREMENT, AND THE ONE THAT IS NOT MEASURABLE
/// IS ABSENT.** There is no time-to-first-token chart: a caller means
/// "request in, first token out" and nothing inside a generation can see the
/// wait behind the runner's lock, so the honest split is prefill (measured),
/// decode (measured), and queue (the subtraction). That last one is what a
/// person actually wants when a request felt slow.
@MainActor
struct ServerChartsView: View {
    @ObservedObject var model: AppModel

    private var points: [ServerMetricPoint] { model.serverMetrics.points }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("Traffic", bundle: .module)
                .themedFont(.small, weight: .semibold)
                .foregroundStyle(.appSecondary)

            if points.isEmpty {
                emptyState
            } else {
                HStack(alignment: .top, spacing: 14) {
                    chartCard(Text("Decode rate", bundle: .module), subtitle: rateSubtitle) { throughputChart }
                    chartCard(Text("Where the time went", bundle: .module), subtitle: latencySubtitle) { latencyChart }
                }
            }
        }
    }

    private var emptyState: some View {
        Text("No requests yet. Charts fill in as traffic arrives.", bundle: .module)
            .themedFont(.tiny)
            .foregroundStyle(.appSecondary)
            .frame(maxWidth: .infinity, alignment: .leading)
            .padding(.vertical, 18)
    }

    private var rateSubtitle: Text? {
        guard let rate = model.serverMetrics.aggregateTokensPerSecond else { return nil }
        // Tokens over decode seconds across the window, NOT a mean of
        // per-request rates -- that would weight a two-token reply the same
        // as a thousand-token one.
        return Text("\(String(format: "%.1f", rate)) tok/s over the window", bundle: .module)
    }

    private var latencySubtitle: Text? {
        guard let p50 = model.serverMetrics.prefillPercentile(0.5),
            let p95 = model.serverMetrics.prefillPercentile(0.95)
        else { return nil }
        return Text(
            "prefill p50 \(String(format: "%.2fs", p50)), p95 \(String(format: "%.2fs", p95))",
            bundle: .module)
    }

    /// Legend entries are drawn from the series values, so the phase names
    /// are resolved in the UI language rather than plotted as English.
    private var prefillPhase: String { AppStrings.localized("prefill") }
    private var decodePhase: String { AppStrings.localized("decode") }
    private var queuedPhase: String { AppStrings.localized("queued") }

    /// One line per model that has actually SERVED something. Built from
    /// what is present rather than from the attached list, so there is never
    /// a flat empty series a reader has to interpret.
    private var throughputChart: some View {
        Chart {
            ForEach(points.filter { $0.tokensPerSecond != nil }) { point in
                LineMark(
                    x: .value("Request", point.requestID),
                    y: .value("tok/s", point.tokensPerSecond ?? 0))
                    .foregroundStyle(by: .value("Model", point.model))
                    .interpolationMethod(.monotone)
                PointMark(
                    x: .value("Request", point.requestID),
                    y: .value("tok/s", point.tokensPerSecond ?? 0))
                    .foregroundStyle(by: .value("Model", point.model))
                    .symbolSize(18)
            }
        }
        .chartXAxis(.hidden)
        .chartLegend(position: .bottom, alignment: .leading, spacing: 4)
        .frame(height: 120)
    }

    /// Prefill, decode and queue stacked per request, which is the shape
    /// that answers "why was that one slow".
    private var latencyChart: some View {
        Chart {
            ForEach(points) { point in
                BarMark(
                    x: .value("Request", point.requestID),
                    y: .value("Seconds", point.prefillSeconds))
                    .foregroundStyle(by: .value("Phase", prefillPhase))
                BarMark(
                    x: .value("Request", point.requestID),
                    y: .value("Seconds", point.decodeSeconds))
                    .foregroundStyle(by: .value("Phase", decodePhase))
                if let queued = point.queuedSeconds, queued > 0.001 {
                    BarMark(
                        x: .value("Request", point.requestID),
                        y: .value("Seconds", queued))
                        .foregroundStyle(by: .value("Phase", queuedPhase))
                }
            }
        }
        .chartXAxis(.hidden)
        .chartLegend(position: .bottom, alignment: .leading, spacing: 4)
        .frame(height: 120)
    }

    private func chartCard<Content: View>(
        _ title: Text, subtitle: Text?, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            VStack(alignment: .leading, spacing: 1) {
                title.themedFont(.tiny, weight: .medium)
                if let subtitle {
                    subtitle
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .monospacedDigit()
                }
            }
            content()
        }
        .padding(12)
        .frame(maxWidth: .infinity, alignment: .leading)
        .background(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .fill(.appSurface))
        .overlay(
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5))
    }
}
