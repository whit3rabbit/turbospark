import XCTest
@testable import TurboSparkApp

/// `ChatChartSpec.parse` decides whether a chart fence renders natively with
/// Swift Charts or falls back to the code block plus the web preview. A wrong
/// yes draws a chart that misstates the data, so every unsupported or
/// ambiguous shape must come back nil.
final class ChatChartSpecTests: XCTestCase {
    // MARK: Chart.js

    func testChartJSBarParses() throws {
        let code = #"""
        {"type":"bar","data":{"labels":["Jan","Feb","Mar"],
         "datasets":[{"label":"Sales","data":[10,20.5,30]}]},
         "options":{"plugins":{"title":{"display":true,"text":"Q1"}}}}
        """#
        let spec = try XCTUnwrap(ChatChartSpec.parse(language: "chartjs", code: code))
        XCTAssertEqual(spec.kind, .bar)
        XCTAssertEqual(spec.title, "Q1")
        XCTAssertEqual(spec.labels, ["Jan", "Feb", "Mar"])
        XCTAssertEqual(spec.series, [.init(name: "Sales", values: [10, 20.5, 30])])
        XCTAssertFalse(spec.isStacked)
    }

    func testChartJSMultiSeriesLineAndStackedBar() throws {
        let line = #"{"type":"line","data":{"labels":[2023,2024],"datasets":[{"label":"A","data":[1,2]},{"label":"B","data":[3,4]}]}}"#
        let spec = try XCTUnwrap(ChatChartSpec.parse(language: "chart", code: line))
        XCTAssertEqual(spec.kind, .line)
        XCTAssertEqual(spec.labels, ["2023", "2024"], "numeric labels print without a decimal point")
        XCTAssertEqual(spec.series.map(\.name), ["A", "B"])

        let stacked = #"{"type":"bar","data":{"labels":["x"],"datasets":[{"label":"A","data":[1]},{"label":"B","data":[2]}]},"options":{"scales":{"x":{"stacked":true},"y":{"stacked":true}}}}"#
        XCTAssertEqual(ChatChartSpec.parse(language: "chart.js", code: stacked)?.isStacked, true)
    }

    func testChartJSPieAndDoughnut() throws {
        let pie = #"{"type":"pie","data":{"labels":["a","b"],"datasets":[{"data":[1,3]}]}}"#
        XCTAssertEqual(ChatChartSpec.parse(language: "chartjs", code: pie)?.kind, .pie)
        let donut = #"{"type":"doughnut","data":{"labels":["a","b"],"datasets":[{"data":[1,3]}]}}"#
        let spec = try XCTUnwrap(ChatChartSpec.parse(language: "chartjs", code: donut))
        XCTAssertEqual(spec.kind, .donut)
        XCTAssertNil(spec.series[0].name)
    }

    // MARK: ECharts

    func testEChartsBarParses() throws {
        let code = #"{"title":{"text":"T"},"xAxis":{"type":"category","data":["a","b"]},"yAxis":{"type":"value"},"series":[{"name":"S","type":"bar","data":[5,6]}]}"#
        let spec = try XCTUnwrap(ChatChartSpec.parse(language: "echarts", code: code))
        XCTAssertEqual(spec.kind, .bar)
        XCTAssertEqual(spec.title, "T")
        XCTAssertEqual(spec.labels, ["a", "b"])
        XCTAssertEqual(spec.series, [.init(name: "S", values: [5, 6])])
    }

    func testEChartsStackedLineAndPieRadius() throws {
        let stacked = #"{"xAxis":{"data":["a"]},"series":[{"name":"A","type":"bar","stack":"t","data":[1]},{"name":"B","type":"bar","stack":"t","data":[2]}]}"#
        XCTAssertEqual(ChatChartSpec.parse(language: "echarts", code: stacked)?.isStacked, true)

        let pie = #"{"series":[{"type":"pie","data":[{"name":"x","value":2},{"name":"y","value":3}]}]}"#
        let pieSpec = try XCTUnwrap(ChatChartSpec.parse(language: "echarts", code: pie))
        XCTAssertEqual(pieSpec.kind, .pie)
        XCTAssertEqual(pieSpec.labels, ["x", "y"])
        XCTAssertEqual(pieSpec.series[0].values, [2, 3])

        let ring = #"{"series":[{"type":"pie","radius":["40%","70%"],"data":[{"name":"x","value":2}]}]}"#
        XCTAssertEqual(ChatChartSpec.parse(language: "echarts", code: ring)?.kind, .donut)
    }

    // MARK: Refusals (fall back to the code block and web preview)

    func testRefusesMalformedAndUnsupported() {
        let cases: [(String?, String)] = [
            ("swift", #"{"type":"bar","data":{"labels":["a"],"datasets":[{"data":[1]}]}}"#),
            (nil, #"{"type":"bar","data":{"labels":["a"],"datasets":[{"data":[1]}]}}"#),
            ("chartjs", "{not json"),
            ("chartjs", #"{"type":"bar","data":{"labels":["a"],"datasets":[{"data":[1]}]"#),
            ("chartjs", #"{"type":"radar","data":{"labels":["a"],"datasets":[{"data":[1]}]}}"#),
            // Length mismatch would silently drop or invent points.
            ("chartjs", #"{"type":"bar","data":{"labels":["a","b"],"datasets":[{"data":[1]}]}}"#),
            // JSON booleans are NSNumber; they must not read as 1 and 0.
            ("chartjs", #"{"type":"bar","data":{"labels":["a"],"datasets":[{"data":[true]}]}}"#),
            // Null gaps and point objects are not in the simple schema.
            ("chartjs", #"{"type":"line","data":{"labels":["a","b"],"datasets":[{"data":[1,null]}]}}"#),
            ("chartjs", #"{"type":"line","data":{"labels":["a"],"datasets":[{"data":[{"x":1,"y":2}]}]}}"#),
            // Horizontal bars and mixed per-dataset types are not drawn natively.
            ("chartjs", #"{"type":"bar","data":{"labels":["a"],"datasets":[{"data":[1]}]},"options":{"indexAxis":"y"}}"#),
            ("chartjs", #"{"type":"bar","data":{"labels":["a"],"datasets":[{"label":"A","data":[1]},{"label":"B","type":"line","data":[2]}]}}"#),
            // Several series need distinct names for the legend.
            ("chartjs", #"{"type":"bar","data":{"labels":["a"],"datasets":[{"data":[1]},{"data":[2]}]}}"#),
            // Duplicate categories would merge on a categorical axis.
            ("chartjs", #"{"type":"bar","data":{"labels":["a","a"],"datasets":[{"data":[1,2]}]}}"#),
            // A pie has one series of non-negative values with a positive total.
            ("chartjs", #"{"type":"pie","data":{"labels":["a"],"datasets":[{"label":"A","data":[1]},{"label":"B","data":[2]}]}}"#),
            ("chartjs", #"{"type":"pie","data":{"labels":["a","b"],"datasets":[{"data":[-1,2]}]}}"#),
            ("chartjs", #"{"type":"pie","data":{"labels":["a"],"datasets":[{"data":[0]}]}}"#),
            ("chartjs", #"{"type":"bar","data":{"labels":[],"datasets":[{"data":[]}]}}"#),
            ("echarts", #"{"xAxis":{"data":["a"]},"series":[{"type":"bar","data":[1]},{"type":"line","data":[2]}]}"#),
            ("echarts", #"{"xAxis":[{"data":["a"]}],"series":[{"type":"bar","data":[1]}]}"#),
            ("echarts", #"{"yAxis":{"type":"category","data":["a"]},"xAxis":{"type":"value"},"series":[{"type":"bar","data":[1]}]}"#),
            ("echarts", #"{"xAxis":{"data":["a"]},"series":[{"type":"bar","data":[{"value":1}]}]}"#),
            ("echarts", #"{"series":[{"type":"scatter","data":[[1,2]]}]}"#),
        ]
        for (language, code) in cases {
            XCTAssertNil(ChatChartSpec.parse(language: language, code: code), "\(language ?? "nil"): \(code)")
        }
    }

    func testRefusesMoreSlicesThanThePaletteCanTellApart() {
        let labels = (0...ChatChartSpec.maxCategoricalColors).map { "\"s\($0)\"" }.joined(separator: ",")
        let values = (0...ChatChartSpec.maxCategoricalColors).map { _ in "1" }.joined(separator: ",")
        let code = #"{"type":"pie","data":{"labels":[\#(labels)],"datasets":[{"data":[\#(values)]}]}}"#
        XCTAssertNil(ChatChartSpec.parse(language: "chartjs", code: code))
    }

    // MARK: Text summary

    func testSummaryListsEveryValue() throws {
        let code = #"{"type":"bar","data":{"labels":["a","b"],"datasets":[{"label":"A","data":[1,2]},{"label":"B","data":[3,4.5]}]}}"#
        let spec = try XCTUnwrap(ChatChartSpec.parse(language: "chartjs", code: code))
        XCTAssertEqual(
            spec.summary(locale: Locale(identifier: "en_US")),
            "A: a 1, b 2. B: a 3, b 4.5.")
        let pie = try XCTUnwrap(ChatChartSpec.parse(
            language: "chartjs",
            code: #"{"type":"pie","data":{"labels":["x","y"],"datasets":[{"data":[1,3]}]}}"#))
        XCTAssertEqual(pie.summary(locale: Locale(identifier: "en_US")), "x 1, y 3.")
    }
}
