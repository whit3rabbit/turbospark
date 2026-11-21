import CoreFoundation
import Foundation

/// The simple chart shapes a chat chart fence can draw natively with Swift
/// Charts: categorical bar, line, pie or donut, with numeric values.
///
/// The fence formats are the ones the code block already previews through
/// the web panel (`CodeBlockContainer.chartHTML`): a Chart.js config for
/// `chartjs`/`chart.js`/`chart`, an ECharts option for `echarts`. No new
/// model-facing format is introduced. Anything outside the simple subset
/// parses to nil and keeps the existing code block plus web preview, because
/// drawing a partial reading of a richer config would misstate the data.
struct ChatChartSpec: Equatable {
    enum Kind: Equatable { case bar, line, pie, donut }

    struct Series: Equatable {
        /// Nil only for a single unnamed series; several series need names
        /// for the legend to tell them apart.
        let name: String?
        let values: [Double]
    }

    let kind: Kind
    let title: String?
    let labels: [String]
    let series: [Series]
    /// Bars of several series stack instead of standing side by side.
    let isStacked: Bool

    /// The theme has five non-neutral roles drawn at two strengths, so more
    /// series or slices than this would repeat a color.
    static let maxCategoricalColors = 10
    /// Bigger bodies are not "simple" and parsing them on every streamed
    /// token would cost render time for nothing.
    static let maxBodyBytes = 64 * 1024
    static let maxCategories = 200

    static func parse(language: String?, code: String) -> ChatChartSpec? {
        guard let kind = language?.lowercased(),
              code.utf8.count <= maxBodyBytes,
              let data = code.data(using: .utf8),
              let root = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
        else { return nil }
        let spec: ChatChartSpec?
        switch kind {
        case "chartjs", "chart.js", "chart": spec = parseChartJS(root)
        case "echarts", "echarts-fulldata": spec = parseECharts(root)
        default: spec = nil
        }
        return spec?.validated()
    }

    // MARK: Chart.js

    private static func parseChartJS(_ root: [String: Any]) -> ChatChartSpec? {
        guard let type = root["type"] as? String,
              let data = root["data"] as? [String: Any],
              let rawLabels = data["labels"] as? [Any],
              let labels = labelStrings(rawLabels),
              let datasets = data["datasets"] as? [[String: Any]]
        else { return nil }
        let kind: Kind
        switch type {
        case "bar": kind = .bar
        case "line": kind = .line
        case "pie": kind = .pie
        case "doughnut": kind = .donut
        default: return nil
        }
        var series: [Series] = []
        for set in datasets {
            // A per-dataset type makes a mixed chart; drawing it as one kind
            // would be wrong.
            if let own = set["type"] as? String, own != type { return nil }
            if set["type"] != nil, !(set["type"] is String) { return nil }
            guard let raw = set["data"] as? [Any], let values = numbers(raw) else { return nil }
            series.append(Series(name: set["label"] as? String, values: values))
        }
        let options = root["options"] as? [String: Any]
        if let axis = options?["indexAxis"], (axis as? String) != "x" { return nil }
        let scales = options?["scales"] as? [String: Any]
        let isStacked = ["x", "y"].contains { key in
            ((scales?[key] as? [String: Any])?["stacked"] as? Bool) == true
        }
        let plugins = options?["plugins"] as? [String: Any]
        let title = ((plugins?["title"] as? [String: Any])?["text"] as? String)
            ?? ((options?["title"] as? [String: Any])?["text"] as? String)
        return ChatChartSpec(kind: kind, title: title, labels: labels, series: series, isStacked: isStacked)
    }

    // MARK: ECharts

    private static func parseECharts(_ root: [String: Any]) -> ChatChartSpec? {
        guard let seriesList = root["series"] as? [[String: Any]], let first = seriesList.first,
              let type = first["type"] as? String,
              seriesList.allSatisfy({ ($0["type"] as? String) == type })
        else { return nil }
        let title = (root["title"] as? [String: Any])?["text"] as? String
        if type == "pie" {
            guard seriesList.count == 1, let items = first["data"] as? [[String: Any]] else { return nil }
            var labels: [String] = []
            var values: [Double] = []
            for item in items {
                guard let name = item["name"] as? String, let value = number(item["value"]) else { return nil }
                labels.append(name)
                values.append(value)
            }
            // A radius pair ["40%", "70%"] is ECharts' donut.
            let isDonut = (first["radius"] as? [Any])?.count == 2
            return ChatChartSpec(
                kind: isDonut ? .donut : .pie, title: title, labels: labels,
                series: [Series(name: first["name"] as? String, values: values)], isStacked: false)
        }
        let kind: Kind
        switch type {
        case "bar": kind = .bar
        case "line": kind = .line
        default: return nil
        }
        // An array of axes, or a category y axis (horizontal bars), is out
        // of the simple subset.
        guard let xAxis = root["xAxis"] as? [String: Any],
              let rawLabels = xAxis["data"] as? [Any],
              let labels = labelStrings(rawLabels)
        else { return nil }
        if let yAxis = root["yAxis"] {
            guard let y = yAxis as? [String: Any], (y["type"] as? String) != "category" else { return nil }
        }
        var series: [Series] = []
        for entry in seriesList {
            guard let raw = entry["data"] as? [Any], let values = numbers(raw) else { return nil }
            series.append(Series(name: entry["name"] as? String, values: values))
        }
        let isStacked = seriesList.contains { $0["stack"] != nil }
        return ChatChartSpec(kind: kind, title: title, labels: labels, series: series, isStacked: isStacked)
    }

    // MARK: Validation

    /// The checks both formats share. Each one guards a way the chart could
    /// silently say something the data does not.
    private func validated() -> ChatChartSpec? {
        guard !labels.isEmpty, labels.count <= Self.maxCategories,
              !series.isEmpty, series.count <= Self.maxCategoricalColors,
              series.allSatisfy({ $0.values.count == labels.count }),
              // Duplicate categories merge on a categorical axis.
              Set(labels).count == labels.count
        else { return nil }
        switch kind {
        case .pie, .donut:
            guard series.count == 1, labels.count <= Self.maxCategoricalColors,
                  series[0].values.allSatisfy({ $0 >= 0 }),
                  series[0].values.reduce(0, +) > 0
            else { return nil }
        case .bar, .line:
            if series.count > 1 {
                let names = series.compactMap(\.name)
                guard names.count == series.count, Set(names).count == names.count else { return nil }
            }
        }
        return self
    }

    private static func number(_ value: Any?) -> Double? {
        // JSON booleans arrive as NSNumber and would read as 1 and 0.
        guard let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() else { return nil }
        let d = n.doubleValue
        return d.isFinite ? d : nil
    }

    private static func numbers(_ raw: [Any]) -> [Double]? {
        var out: [Double] = []
        out.reserveCapacity(raw.count)
        for item in raw {
            guard let d = number(item) else { return nil }
            out.append(d)
        }
        return out
    }

    /// Labels may be strings or numbers (years are often bare numbers).
    private static func labelStrings(_ raw: [Any]) -> [String]? {
        var out: [String] = []
        for item in raw {
            if let s = item as? String {
                out.append(s)
            } else if let d = number(item) {
                out.append(d.rounded() == d && abs(d) < 1e15 ? String(Int64(d)) : String(d))
            } else {
                return nil
            }
        }
        return out
    }

    // MARK: Summary

    /// Every value as text, the VoiceOver summary and the descriptor summary.
    /// Data only, so nothing here needs translating beyond number format.
    func summary(locale: Locale = .current) -> String {
        func pairs(_ values: [Double]) -> String {
            zip(labels, values)
                .map { "\($0) \($1.formatted(.number.locale(locale)))" }
                .joined(separator: ", ")
        }
        if series.count == 1, series[0].name == nil || kind == .pie || kind == .donut {
            return pairs(series[0].values) + "."
        }
        return series
            .map { "\($0.name ?? ""): \(pairs($0.values))." }
            .joined(separator: " ")
    }
}
