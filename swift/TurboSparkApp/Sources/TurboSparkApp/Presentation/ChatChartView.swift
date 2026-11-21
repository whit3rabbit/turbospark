import Charts
import SwiftUI

/// Native, offline rendering of a simple chart fence (`ChatChartSpec`).
///
/// Replaces the CDN-loaded web page for the common case: no network grant,
/// theme colors, and VoiceOver access to every value through per-mark labels,
/// a text summary and an audio-graph chart descriptor.
struct ChatChartView: View {
    let spec: ChatChartSpec
    @Environment(\.appTheme) private var theme
    @Environment(\.locale) private var locale

    /// No categorical palette exists in the design system, so the chart uses
    /// the theme's five non-neutral roles, then the same five at half
    /// strength. Color is never the only cue: bars and lines carry a legend
    /// (lines also a symbol per series) and slices a labelled value list.
    private var palette: [Color] {
        let base = [theme.accent, theme.warning, theme.success, theme.danger, theme.secondaryText]
        return base + base.map { $0.opacity(0.5) }
    }

    private var isSector: Bool { spec.kind == .pie || spec.kind == .donut }

    /// The color domain: slices for a pie, series otherwise.
    private var domain: [String] {
        isSector ? spec.labels : spec.series.indices.map(seriesKey)
    }

    private func seriesKey(_ index: Int) -> String {
        spec.series[index].name ?? ""
    }

    private var kindTitle: String {
        switch spec.kind {
        case .bar: return AppStrings.localized("Bar chart")
        case .line: return AppStrings.localized("Line chart")
        case .pie: return AppStrings.localized("Pie chart")
        case .donut: return AppStrings.localized("Donut chart")
        }
    }

    private var accessibilityTitle: String {
        guard let title = spec.title, !title.isEmpty else { return kindTitle }
        return "\(kindTitle), \(title)"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let title = spec.title, !title.isEmpty {
                Text(verbatim: title)
                    .themedFont(.callout, weight: .semibold)
                    .foregroundStyle(theme.foreground)
                    .accessibilityHidden(true)
            }
            chart
                .chartForegroundStyleScale(domain: domain, range: Array(palette.prefix(domain.count)))
                .chartLegend(isSector || spec.series.count == 1 ? .hidden : .visible)
                .chartLegend(position: .bottom, alignment: .leading, spacing: 6)
                .frame(height: 220)
                .accessibilityLabel(Text(verbatim: accessibilityTitle))
                .accessibilityValue(Text(verbatim: spec.summary(locale: locale)))
                .accessibilityChartDescriptor(
                    ChatChartDescriptor(
                        spec: spec, title: accessibilityTitle,
                        categoryTitle: AppStrings.localized("Category"),
                        valueTitle: AppStrings.localized("Value"),
                        locale: locale))
            if isSector {
                sliceLegend
            }
        }
        .padding(.vertical, 4)
    }

    @ViewBuilder private var chart: some View {
        if isSector {
            Chart {
                ForEach(Array(spec.labels.enumerated()), id: \.offset) { index, label in
                    let value = spec.series[0].values[index]
                    SectorMark(
                        angle: .value("Value", value),
                        innerRadius: .ratio(spec.kind == .donut ? 0.6 : 0),
                        angularInset: 1)
                        .foregroundStyle(by: .value("Category", label))
                        .accessibilityLabel(Text(verbatim: label))
                        .accessibilityValue(Text(verbatim: value.formatted(.number.locale(locale))))
                }
            }
        } else {
            Chart {
                ForEach(Array(spec.series.enumerated()), id: \.offset) { seriesIndex, series in
                    ForEach(Array(spec.labels.enumerated()), id: \.offset) { index, label in
                        mark(series: seriesIndex, label: label, value: series.values[index])
                    }
                }
            }
        }
    }

    @ChartContentBuilder
    private func mark(series index: Int, label: String, value: Double) -> some ChartContent {
        let key = seriesKey(index)
        let axLabel = key.isEmpty ? label : "\(key), \(label)"
        let axValue = value.formatted(.number.locale(locale))
        switch spec.kind {
        case .line:
            LineMark(x: .value("Category", label), y: .value("Value", value))
                .foregroundStyle(by: .value("Series", key))
                .symbol(by: .value("Series", key))
                .accessibilityLabel(Text(verbatim: axLabel))
                .accessibilityValue(Text(verbatim: axValue))
        default:
            if spec.isStacked || spec.series.count == 1 {
                BarMark(x: .value("Category", label), y: .value("Value", value))
                    .foregroundStyle(by: .value("Series", key))
                    .accessibilityLabel(Text(verbatim: axLabel))
                    .accessibilityValue(Text(verbatim: axValue))
            } else {
                // Side by side, Chart.js and ECharts' default for several
                // bar series; Swift Charts would stack them otherwise.
                BarMark(x: .value("Category", label), y: .value("Value", value))
                    .foregroundStyle(by: .value("Series", key))
                    .position(by: .value("Series", key))
                    .accessibilityLabel(Text(verbatim: axLabel))
                    .accessibilityValue(Text(verbatim: axValue))
            }
        }
    }

    /// Each slice's color beside its name, value and share, so a slice is
    /// identified by text and not by hue alone.
    private var sliceLegend: some View {
        let values = spec.series[0].values
        let total = values.reduce(0, +)
        return VStack(alignment: .leading, spacing: 3) {
            ForEach(Array(spec.labels.enumerated()), id: \.offset) { index, label in
                HStack(spacing: 6) {
                    RoundedRectangle(cornerRadius: 2)
                        .fill(palette[index])
                        .frame(width: 10, height: 10)
                    Text(verbatim: label)
                        .foregroundStyle(theme.foreground)
                    Spacer(minLength: 8)
                    Text(verbatim: values[index].formatted(.number.locale(locale)))
                        .foregroundStyle(theme.secondaryText)
                        .monospacedDigit()
                    Text(verbatim: (values[index] / total).formatted(
                        .percent.precision(.fractionLength(0)).locale(locale)))
                        .foregroundStyle(theme.secondaryText)
                        .monospacedDigit()
                        .frame(minWidth: 36, alignment: .trailing)
                }
                .themedFont(.small)
            }
        }
        // The chart's own summary already reads every slice.
        .accessibilityHidden(true)
    }
}

/// The audio graph and data table VoiceOver offers for a chart.
struct ChatChartDescriptor: AXChartDescriptorRepresentable {
    let spec: ChatChartSpec
    let title: String
    let categoryTitle: String
    let valueTitle: String
    let locale: Locale

    func makeChartDescriptor() -> AXChartDescriptor {
        let all = spec.series.flatMap(\.values)
        // Bars and slices grow from zero; a flat series still needs a span.
        var low = min(all.min() ?? 0, 0)
        var high = max(all.max() ?? 1, 0)
        if low == high { low -= 1; high += 1 }
        let format = locale
        let xAxis = AXCategoricalDataAxisDescriptor(title: categoryTitle, categoryOrder: spec.labels)
        let yAxis = AXNumericDataAxisDescriptor(
            title: valueTitle, range: low...high, gridlinePositions: []
        ) { $0.formatted(.number.locale(format)) }
        let series = spec.series.map { series in
            AXDataSeriesDescriptor(
                name: series.name ?? title,
                isContinuous: spec.kind == .line,
                dataPoints: zip(spec.labels, series.values).map { AXDataPoint(x: $0, y: $1) })
        }
        return AXChartDescriptor(
            title: title, summary: spec.summary(locale: locale),
            xAxis: xAxis, yAxis: yAxis, additionalAxes: [], series: series)
    }
}
