import SwiftUI
import Combine

// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
/// Slim bottom strip carrying live benchmarks: memory, CPU, context fill, throughput, and thermal telemetry.
/// Supports both compact numeric text and real-time sparkline graph visualization modes.
@MainActor
struct StatusBarView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject private var appearanceManager = AppearanceManager.shared

    /// Polled rather than published: footprint and CPU are read straight from
    /// the mach counters on every poll.

    /// Historical ring buffers for live sparkline graphs (up to 16 data points).
    @State private var throughputHistory: [Double] = []

    private let poll = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 0) {
            // Memory and CPU live in the top bar's ChromeTelemetryView now.
            // The per-token numbers stay here: they belong beside throughput,
            // not in the chrome above the transcript.
            if let contextFill {
                contextReadout(contextFill)
            }
            FanReadoutView(model: model, showsLeadingDivider: contextFill != nil)
            Spacer(minLength: 12)
            throughputReadout
            barDivider
            tokenReadout
            barDivider
            thermalReadout(model.telemetry?.thermalLevel ?? "nominal")
            if let pressure = model.telemetry?.memoryPressure, pressure.lowercased() != "normal" {
                barDivider
                memoryPressureReadout(pressure)
            }
            barDivider
            viewModeToggleButton
        }
        .font(theme.ui(.tiny))
        .foregroundStyle(.appSecondary)
        .padding(.horizontal, 12)
        .frame(height: AppChromeLayout.statusBarHeight)
        .background(TurboSparkTheme.barBackgroundColor)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(.appBorder)
                .frame(height: 0.5)
        }
        .onAppear { refreshMetrics() }
        .onReceive(poll) { _ in refreshMetrics() }
        .onChange(of: model.liveTokenCount) { refreshMetrics() }
    }

    private var barDivider: some View {
        Rectangle()
            .fill(.appBorder)
            .frame(width: 0.5, height: 11)
            .padding(.horizontal, 9)
    }

    private var isGraphMode: Bool {
        appearanceManager.statusBarViewMode == .graphs
    }

    // MARK: - View Mode Switcher

    private var viewModeToggleButton: some View {
        Button {
            withAnimation(.easeInOut(duration: 0.15)) {
                appearanceManager.statusBarViewMode = isGraphMode ? .text : .graphs
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isGraphMode ? "chart.xyaxis.line" : "number")
                    .font(theme.ui(.tiny, weight: .medium))
                    .foregroundStyle(isGraphMode ? TurboSparkTheme.accentColor : Color.secondary)
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .background(isGraphMode ? TurboSparkTheme.accentColor.opacity(0.12) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: 4))
        }
        .buttonStyle(.plain)
        .help(isGraphMode ? "Benchmarks view: Live Graphs (click to switch to Numbers)" : "Benchmarks view: Numbers (click to switch to Live Graphs)")
        .accessibilityLabel("Toggle benchmarks graph view")
        .accessibilityValue(isGraphMode ? "Live Graphs" : "Numbers")
        .appPointerCursor()
    }

    private func refreshMetrics() {
        let rate = model.phase == .decode ? model.liveTokensPerSecond : (model.diagnostics?.tokensPerSecond ?? 0.0)
        appendSample(rate, to: &throughputHistory)

        model.refreshTelemetry()
    }

    private func appendSample(_ value: Double, to array: inout [Double], maxCount: Int = 16) {
        array.append(value)
        if array.count > maxCount {
            array.removeFirst(array.count - maxCount)
        }
    }

    // MARK: - Context

    /// Fraction of the loaded model's context window the transcript occupies.
    private var contextFill: Double? {
        guard model.session != nil else { return nil }
        let limit = model.resolvedContextTokens
        guard limit > 0 else { return nil }
        return min(1.0, Double(model.estimatedContextTokens) / Double(limit))
    }

    private func contextReadout(_ fraction: Double) -> some View {
        let hint = compactionHint(fraction)
        return HStack(spacing: 6) {
            Image(systemName: "text.alignleft")
                .font(theme.ui(.tiny))
                .accessibilityHidden(true)
            MeterBar(fraction: fraction, tint: hint?.urgent == true ? .orange : TurboSparkTheme.accentColor)
                .frame(width: 38)
            Text(
                hint?.text
                    ?? "\(model.estimatedContextTokens.formatted(.number.notation(.compactName))) / \(model.resolvedContextTokens.formatted(.number.notation(.compactName)))"
            )
            .monospacedDigit()
        }
        .help(hint?.help ?? "Estimated conversation length against the loaded context window")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Context used")
        .accessibilityValue(
            hint?.text ?? "\(model.estimatedContextTokens) of \(model.resolvedContextTokens) tokens")
    }

    /// One line of readout text for the context meter while compaction is
    /// relevant, or nil when the plain used/limit numbers should show.
    private struct CompactionHint {
        let text: String
        let help: String
        let urgent: Bool
    }

    /// Claude Code's "% until auto-compact" readout. The trigger mirrors
    /// `AppChatCompaction`: four fifths of the USABLE window (context minus
    /// the reply reservation), which is why the arithmetic runs through the
    /// same constants rather than restating a fraction. Hidden until the
    /// meter is within 40 points of the trigger; while the summarizer runs
    /// it names the state instead of a number.
    private func compactionHint(_ fraction: Double) -> CompactionHint? {
        guard model.session != nil else { return nil }
        if model.isCompacting {
            return CompactionHint(
                text: "Compacting conversation",
                help: "Summarizing older turns so the conversation keeps fitting the window",
                urgent: true)
        }
        guard model.autoCompactEnabled else { return nil }
        let limit = Double(model.resolvedContextTokens)
        let usable = limit - Double(max(1, model.maxNewTokens))
        guard limit > 0, usable > 0 else { return nil }
        let trigger =
            usable * Double(AppChatCompaction.triggerNumerator)
            / Double(AppChatCompaction.triggerDenominator)
        let used = fraction * limit
        let percentLeft = Int(((trigger - used) / trigger * 100).rounded())
        guard percentLeft <= 40 else { return nil }
        return CompactionHint(
            text: "\(max(0, percentLeft))% until auto-compact",
            help: "Older turns are summarized automatically once this reaches 0%",
            urgent: percentLeft <= 15)
    }

    // MARK: - Throughput

    private var throughputReadout: some View {
        HStack(spacing: 5) {
            Image(systemName: "gauge.with.dots.needle.bottom.50percent")
                .font(theme.ui(.tiny))
                .accessibilityHidden(true)
            if isGraphMode {
                MiniSparklineView(
                    samples: throughputHistory,
                    tint: TurboSparkTheme.accentColor
                )
            }
            Text(rateText)
                .monospacedDigit()
                .foregroundStyle(.appText)
            Text("tok/s", bundle: .module)
                .foregroundStyle(.tertiary)
        }
        .help("Live or last run decoding throughput (tokens per second)")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Decode rate")
        .accessibilityValue("\(rateText) tokens per second")
    }

    private var tokenReadout: some View {
        HStack(spacing: 5) {
            Text(tokensText)
                .monospacedDigit()
                .foregroundStyle(.appText)
            Text("tokens", bundle: .module)
                .foregroundStyle(.tertiary)
        }
        .help("Total tokens generated in this turn or previous run")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Tokens generated")
        .accessibilityValue(tokensText)
    }

    private var rateText: String {
        if model.phase == .decode { return MetricFormat.rate(model.liveTokensPerSecond) }
        if let diagnostics = model.diagnostics { return MetricFormat.rate(diagnostics.tokensPerSecond) }
        return "\u{2014}"
    }

    private var tokensText: String {
        if model.isRunning { return "\(model.liveTokenCount)" }
        if let diagnostics = model.diagnostics { return "\(diagnostics.generatedTokens)" }
        return "\u{2014}"
    }

    // MARK: - Thermal / Temp

    private func thermalReadout(_ level: String) -> some View {
        let isAbnormal = level.lowercased() != "nominal"
        let tintColor: Color = {
            switch level.lowercased() {
            case "critical": return .red
            case "serious", "fair": return .orange
            default: return .secondary
            }
        }()

        return HStack(spacing: 5) {
            Image(systemName: isAbnormal ? "thermometer.high" : "thermometer.medium")
                .font(theme.ui(.tiny))
                .foregroundStyle(tintColor)
                .accessibilityHidden(true)
            Text(level.capitalized)
                .foregroundStyle(isAbnormal ? tintColor : .primary)
            Text("Temp", bundle: .module)
                .foregroundStyle(.tertiary)
        }
        .help(thermalHelp(level))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Thermal status")
        .accessibilityValue(level)
    }

    private func thermalHelp(_ level: String) -> String {
        if level.lowercased() == "nominal" {
            return "Thermal status is Nominal: machine is operating at normal temperature and full performance."
        }
        return "Thermal pressure is \(level.capitalized); engine throughput and power figures may throttle down."
    }

    // MARK: - Memory Pressure

    /// The kernel's own memory-pressure verdict.
    private func memoryPressureReadout(_ level: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: "exclamationmark.triangle")
                .font(theme.ui(.tiny))
                .accessibilityHidden(true)
            Text("Memory \(level.capitalized)", bundle: .module)
        }
        .foregroundStyle(level.lowercased() == "critical" ? .red : .orange)
        .help(
            "The system is short of memory. Generation is being paced down; "
                + "closing an idle model or another app will relieve it."
        )
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Memory pressure")
        .accessibilityValue(level)
    }
}

/// Mini sparkline chart rendering live historical data samples with gradient fill area and line curve.
private struct MiniSparklineView: View {
    let samples: [Double]
    let tint: Color
    var fixedMax: Double? = nil
    var minRange: Double = 1.0

    var body: some View {
        GeometryReader { proxy in
            let w = proxy.size.width
            let h = proxy.size.height
            let validSamples = samples.isEmpty ? [0.0, 0.0] : (samples.count == 1 ? [samples[0], samples[0]] : samples)
            let minVal = 0.0
            let calculatedMax = validSamples.max() ?? 1.0
            let maxVal = max(fixedMax ?? calculatedMax, minVal + minRange)

            let points: [CGPoint] = validSamples.enumerated().map { index, val in
                let x = w * CGFloat(index) / CGFloat(max(1, validSamples.count - 1))
                let normalized = max(0.0, min(1.0, (val - minVal) / maxVal))
                let y = h - (CGFloat(normalized) * (h - 3) + 1.5)
                return CGPoint(x: x, y: y)
            }

            ZStack {
                // Gradient fill area under curve
                Path { path in
                    guard let first = points.first else { return }
                    path.move(to: CGPoint(x: first.x, y: h))
                    path.addLine(to: first)
                    for pt in points.dropFirst() {
                        path.addLine(to: pt)
                    }
                    if let last = points.last {
                        path.addLine(to: CGPoint(x: last.x, y: h))
                    }
                    path.closeSubpath()
                }
                .fill(
                    LinearGradient(
                        colors: [tint.opacity(0.35), tint.opacity(0.04)],
                        startPoint: .top,
                        endPoint: .bottom
                    )
                )

                // Sparkline stroke
                Path { path in
                    guard let first = points.first else { return }
                    path.move(to: first)
                    for pt in points.dropFirst() {
                        path.addLine(to: pt)
                    }
                }
                .stroke(tint, style: StrokeStyle(lineWidth: 1.2, lineCap: .round, lineJoin: .round))

                // End pulse indicator
                if let last = points.last {
                    Circle()
                        .fill(tint)
                        .frame(width: 2.5, height: 2.5)
                        .position(last)
                }
            }
        }
        .frame(width: 38, height: 13)
        .clipped()
        .accessibilityHidden(true)
    }
}

/// Two-tone horizontal meter used for the memory and context readouts.
private struct MeterBar: View {
    let fraction: Double
    let tint: Color

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule()
                    .fill(Color.primary.opacity(0.1))
                Capsule()
                    .fill(tint)
                    .frame(width: max(2, geometry.size.width * fraction))
            }
        }
        .frame(height: 4)
        .accessibilityHidden(true)
    }
}
