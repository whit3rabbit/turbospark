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
    @State private var sampleGate = SampleRateGate()


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
        .background(TurboSparkTheme.barBackgroundColor(isDark: theme.isDark))
        .overlay(alignment: .top) {
            Rectangle()
                .fill(.appBorder)
                .frame(height: 0.5)
        }
        .onAppear { refreshMetrics() }
        .task {
            // Lifetime follows view identity, not struct re-creation: a Timer
            // publisher stored on the struct restarted on every parent render
            // (every ~25 ms while generating), so the 2 s poll never fired.
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                if Task.isCancelled { break }
                refreshMetrics()
            }
        }
        // Per token this only feeds the throughput sparkline, rate-limited.
        // The telemetry FFI/JSON round trip stays on the 2 s timer: thermal
        // and memory pressure move on a scale of seconds, not per token.
        .onChange(of: model.liveTokenCount) {
            if sampleGate.admit(now: Date()) { appendThroughputSample() }
        }
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
            TSMotion.animate(TSMotion.select) {
                appearanceManager.statusBarViewMode = isGraphMode ? .text : .graphs
            }
        } label: {
            HStack(spacing: 4) {
                Image(systemName: isGraphMode ? "chart.xyaxis.line" : "number")
                    .font(theme.ui(.tiny, weight: .medium))
                    .foregroundStyle(isGraphMode ? theme.accent : theme.secondaryText)
            }
            .padding(.horizontal, 4)
            .padding(.vertical, 2)
            .background(isGraphMode ? theme.accent.opacity(0.12) : Color.clear)
            .clipShape(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        }
        .buttonStyle(.plain)
        .help(isGraphMode
            ? Text("Benchmarks view: Live Graphs (click to switch to Numbers)", bundle: .module)
            : Text("Benchmarks view: Numbers (click to switch to Live Graphs)", bundle: .module))
        .accessibilityLabel(Text("Toggle benchmarks graph view", bundle: .module))
        .accessibilityValue(isGraphMode ? Text("Live Graphs", bundle: .module) : Text("Numbers", bundle: .module))
        .appPointerCursor()
    }

    private func refreshMetrics() {
        _ = sampleGate.admit(now: Date(), minInterval: 0)
        appendThroughputSample()
        model.refreshTelemetry()
    }

    private func appendThroughputSample() {
        let rate = model.phase == .decode ? model.liveTokensPerSecond : (model.diagnostics?.tokensPerSecond ?? 0.0)
        appendSample(rate, to: &throughputHistory)
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
            MeterBar(fraction: fraction, tint: hint?.urgent == true ? theme.warning : theme.accent)
                .frame(width: 38)
            (hint?.text
                ?? Text(verbatim: "\(model.estimatedContextTokens.formatted(.number.notation(.compactName))) / \(model.resolvedContextTokens.formatted(.number.notation(.compactName)))"))
            .monospacedDigit()
        }
        .help(hint?.help ?? Text("Estimated conversation length against the loaded context window", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Context used", bundle: .module))
        .accessibilityValue(
            hint?.text
                ?? Text("\(model.estimatedContextTokens) of \(model.resolvedContextTokens) tokens", bundle: .module))
    }

    /// One line of readout text for the context meter while compaction is
    /// relevant, or nil when the plain used/limit numbers should show.
    private struct CompactionHint {
        let text: Text
        let help: Text
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
                text: Text("Compacting conversation...", bundle: .module),
                help: Text("Summarizing older turns so the conversation keeps fitting the window", bundle: .module),
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
            text: Text("\(max(0, percentLeft))% until auto-compact", bundle: .module),
            help: Text("Older turns are summarized automatically once this reaches zero.", bundle: .module),
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
                    tint: theme.accent
                )
            }
            Text(verbatim: rateText)
                .monospacedDigit()
                .foregroundStyle(.appText)
            Text("tok/s", bundle: .module)
                .foregroundStyle(.tertiary)
        }
        .help(Text("Live or last run decoding throughput (tokens per second)", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Decode rate", bundle: .module))
        .accessibilityValue(Text("\(rateText) tokens per second", bundle: .module))
    }

    private var tokenReadout: some View {
        HStack(spacing: 5) {
            Text(verbatim: tokensText)
                .monospacedDigit()
                .foregroundStyle(.appText)
            Text("tokens", bundle: .module)
                .foregroundStyle(.tertiary)
        }
        .help(Text("Total tokens generated in this turn or previous run", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Tokens generated", bundle: .module))
        .accessibilityValue(Text(verbatim: tokensText))
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
            case "critical", "serious", "fair": return theme.warning
            default: return theme.secondaryText
            }
        }()

        return HStack(spacing: 5) {
            Image(systemName: isAbnormal ? "thermometer.high" : "thermometer.medium")
                .font(theme.ui(.tiny))
                .foregroundStyle(tintColor)
                .accessibilityHidden(true)
            Text(verbatim: thermalLevelName(level))
                .foregroundStyle(isAbnormal ? tintColor : theme.foreground)
            Text("Temp", bundle: .module)
                .foregroundStyle(.appSecondary)
        }
        .help(thermalHelp(level))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Thermal status", bundle: .module))
        .accessibilityValue(Text(verbatim: thermalLevelName(level)))
    }

    /// The engine reports `ThermalLevel`'s lowercased case name, so these four
    /// are the whole vocabulary; anything else is shown as received.
    private func thermalLevelName(_ level: String) -> String {
        switch level.lowercased() {
        case "nominal": return AppStrings.localized("Nominal")
        case "fair": return AppStrings.localized("Fair")
        case "serious": return AppStrings.localized("Serious")
        case "critical": return AppStrings.localized("Critical")
        default: return level.capitalized
        }
    }

    private func thermalHelp(_ level: String) -> Text {
        if level.lowercased() == "nominal" {
            return Text("Thermal status is Nominal: machine is operating at normal temperature and full performance.", bundle: .module)
        }
        return Text("Thermal pressure is \(thermalLevelName(level)); engine throughput and power figures may throttle down.", bundle: .module)
    }

    // MARK: - Memory Pressure

    /// The kernel's own memory-pressure verdict. `MemoryPressure` has three
    /// cases and `normal` never reaches this readout, so anything short of
    /// critical is the warn level.
    private func memoryPressureReadout(_ level: String) -> some View {
        let isCritical = level.lowercased() == "critical"
        return HStack(spacing: 5) {
            Image(systemName: "exclamationmark.triangle")
                .font(theme.ui(.tiny))
                .accessibilityHidden(true)
            isCritical ? Text("Memory Critical", bundle: .module) : Text("Memory Warning", bundle: .module)
        }
        .foregroundStyle(AppAttention.color)
        .help(Text("The system is short of memory. Generation is being paced down; closing an idle model or another app will relieve it.", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Memory pressure", bundle: .module))
        .accessibilityValue(isCritical ? Text("Critical", bundle: .module) : Text("Warning", bundle: .module))
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
                    .fill(.appText.opacity(0.1))
                Capsule()
                    .fill(tint)
                    .frame(width: max(2, geometry.size.width * fraction))
            }
        }
        .frame(height: 4)
        .accessibilityHidden(true)
    }
}

/// Admits at most one sample per `minInterval`, so a per-token trigger yields
/// a sparkline on a real time base (about 2 samples a second) instead of one
/// point per token.
struct SampleRateGate {
    private var lastAdmitted: Date?
    static let defaultInterval: TimeInterval = 0.5

    /// Returns true (and records `now`) when enough time passed since the
    /// last admitted sample.
    mutating func admit(now: Date, minInterval: TimeInterval = SampleRateGate.defaultInterval) -> Bool {
        if let lastAdmitted, now.timeIntervalSince(lastAdmitted) < minInterval { return false }
        lastAdmitted = now
        return true
    }
}
