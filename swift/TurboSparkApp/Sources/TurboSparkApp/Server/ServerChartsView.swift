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
                    chartCard("Decode rate", subtitle: rateSubtitle) { throughputChart }
                    chartCard("Where the time went", subtitle: latencySubtitle) { latencyChart }
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

    private var rateSubtitle: String {
        guard let rate = model.serverMetrics.aggregateTokensPerSecond else { return "" }
        // Tokens over decode seconds across the window, NOT a mean of
        // per-request rates -- that would weight a two-token reply the same
        // as a thousand-token one.
        return String(format: "%.1f tok/s over the window", rate)
    }

    private var latencySubtitle: String {
        guard let p50 = model.serverMetrics.prefillPercentile(0.5),
            let p95 = model.serverMetrics.prefillPercentile(0.95)
        else { return "" }
        return String(format: "prefill p50 %.2fs, p95 %.2fs", p50, p95)
    }

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
                    .foregroundStyle(by: .value("Phase", "prefill"))
                BarMark(
                    x: .value("Request", point.requestID),
                    y: .value("Seconds", point.decodeSeconds))
                    .foregroundStyle(by: .value("Phase", "decode"))
                if let queued = point.queuedSeconds, queued > 0.001 {
                    BarMark(
                        x: .value("Request", point.requestID),
                        y: .value("Seconds", queued))
                        .foregroundStyle(by: .value("Phase", "queued"))
                }
            }
        }
        .chartXAxis(.hidden)
        .chartLegend(position: .bottom, alignment: .leading, spacing: 4)
        .frame(height: 120)
    }

    private func chartCard<Content: View>(
        _ title: String, subtitle: String, @ViewBuilder content: () -> Content
    ) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            VStack(alignment: .leading, spacing: 1) {
                Text(title).themedFont(.tiny, weight: .medium)
                if !subtitle.isEmpty {
                    Text(subtitle)
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
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .fill(.appSurface))
        .overlay(
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5))
    }
}
