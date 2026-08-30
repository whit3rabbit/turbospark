import TurboSpark
import XCTest

@testable import TurboSparkApp

/// Unit tests verifying live benchmarks telemetry, CPU sampling, memory metrics, and formatting.
final class StatusBarBenchmarksTests: XCTestCase {
    @MainActor
    func testAppModelLiveCPUSampling() {
        let model = AppModel()
        let cpu = model.currentProcessCPUUsage
        XCTAssertNotNil(cpu, "Process CPU usage should be readable via Darwin Mach task threads")
        if let cpu = cpu {
            XCTAssertGreaterThanOrEqual(cpu, 0.0, "CPU percentage must be non-negative")
        }
    }

    @MainActor
    func testAppModelLiveTokensPerSecond() {
        let model = AppModel()
        XCTAssertEqual(model.liveTokensPerSecond, 0.0, "Tokens per second should be 0 when idle")

        model.liveTokenCount = 50
        model.liveElapsedDecodeSeconds = 2.0
        XCTAssertEqual(model.liveTokensPerSecond, 25.0, "Tokens per second should accurately calculate count / seconds")
    }

    func testMetricFormatRate() {
        XCTAssertEqual(MetricFormat.rate(0.0), "0.0")
        XCTAssertEqual(MetricFormat.rate(42.567), "42.6")
        XCTAssertEqual(MetricFormat.rate(120.0), "120.0")
    }

    func testMetricFormatPercent() {
        XCTAssertEqual(MetricFormat.percent(0.0), "0.0%")
        XCTAssertEqual(MetricFormat.percent(85.42), "85.4%")
        XCTAssertEqual(MetricFormat.percent(100.0), "100.0%")
    }

    func testMetricFormatMemory() {
        XCTAssertEqual(MetricFormat.memory(nil), "\u{2014}")
        let oneGB: UInt64 = 1024 * 1024 * 1024
        let formatted = MetricFormat.memory(oneGB)
        XCTAssertFalse(formatted.isEmpty)
        XCTAssertNotEqual(formatted, "\u{2014}")
    }

    func testMetricFormatSeconds() {
        XCTAssertEqual(MetricFormat.seconds(nil), "\u{2014}")
        XCTAssertEqual(MetricFormat.seconds(0.05), "50 ms")
        XCTAssertEqual(MetricFormat.seconds(2.5), "2.50 s")
    }

    func testStatusBarViewModeEnum() {
        XCTAssertEqual(StatusBarViewMode.text.rawValue, "text")
        XCTAssertEqual(StatusBarViewMode.graphs.rawValue, "graphs")
        XCTAssertEqual(StatusBarViewMode.text.label, "Numbers")
        XCTAssertEqual(StatusBarViewMode.graphs.label, "Live Graphs")
        XCTAssertEqual(StatusBarViewMode.text.systemImage, "number")
        XCTAssertEqual(StatusBarViewMode.graphs.systemImage, "chart.xyaxis.line")
    }

    @MainActor
    func testAppearanceManagerStatusBarViewModeToggle() {
        let manager = AppearanceManager.shared
        manager.statusBarViewMode = .text
        XCTAssertEqual(manager.statusBarViewMode, .text)

        manager.statusBarViewMode = .graphs
        XCTAssertEqual(manager.statusBarViewMode, .graphs)

        // Reset to default
        manager.statusBarViewMode = .text
    }
}
