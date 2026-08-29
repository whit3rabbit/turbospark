import SwiftUI

/// Slim bottom strip carrying live memory, context fill and throughput.
///
/// Memory lives here rather than in the top bar because it is a background
/// reading: it should be glanceable without competing with the model loader.
struct StatusBarView: View {
    @ObservedObject var model: AppModel

    /// Polled rather than published: the footprint is read straight from the
    /// mach counter through the FFI on every call, so nothing mutates a
    /// `@Published` property when it changes and no view would refresh.
    @State private var memoryBytes: UInt64?

    private let poll = Timer.publish(every: 2, on: .main, in: .common).autoconnect()

    var body: some View {
        HStack(spacing: 0) {
            memoryReadout
            if let contextFill {
                barDivider
                contextReadout(contextFill)
            }
            Spacer(minLength: 12)
            if showsThroughput {
                throughputReadout
                barDivider
                tokenReadout
            }
            if let thermal = model.telemetry?.thermalLevel, thermal.lowercased() != "nominal" {
                barDivider
                thermalReadout(thermal)
            }
            // Shown only when abnormal, exactly as the thermal readout is: a
            // badge that is always present carries no information, and this
            // strip is meant to be glanceable.
            if let pressure = model.telemetry?.memoryPressure, pressure.lowercased() != "normal" {
                barDivider
                memoryPressureReadout(pressure)
            }
        }
        .font(.system(size: 10.5))
        .foregroundStyle(.secondary)
        .padding(.horizontal, 12)
        .frame(height: AppChromeLayout.statusBarHeight)
        .background(TurboSparkTheme.barBackgroundColor)
        .overlay(alignment: .top) {
            Rectangle()
                .fill(TurboSparkTheme.hairlineColor)
                .frame(height: 0.5)
        }
        .onAppear { refreshMemory() }
        .onReceive(poll) { _ in refreshMemory() }
        .onChange(of: model.liveTokenCount) { refreshMemory() }
    }

    private var barDivider: some View {
        Rectangle()
            .fill(TurboSparkTheme.hairlineColor)
            .frame(width: 0.5, height: 11)
            .padding(.horizontal, 9)
    }

    // MARK: - Memory

    private var memoryReadout: some View {
        HStack(spacing: 6) {
            Image(systemName: "memorychip")
                .font(.system(size: 10))
                .accessibilityHidden(true)
            Text(MetricFormat.memory(memoryBytes))
                .monospacedDigit()
                .foregroundStyle(.primary)
            if let fraction = memoryFraction {
                MeterBar(fraction: fraction, tint: memoryTint(fraction))
                    .frame(width: 44)
                Text(MetricFormat.percent(fraction * 100))
                    .monospacedDigit()
                    .foregroundStyle(.tertiary)
            }
        }
        .help(memoryHelp)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Process memory")
        .accessibilityValue(memoryAccessibilityValue)
    }

    private var memoryFraction: Double? {
        guard let memoryBytes,
              let physical = model.telemetry?.physicalMemoryBytes,
              physical > 0
        else { return nil }
        return min(1.0, Double(memoryBytes) / Double(physical))
    }

    private func memoryTint(_ fraction: Double) -> Color {
        if fraction > 0.85 { return .red }
        if fraction > 0.65 { return .orange }
        return TurboSparkTheme.accentColor
    }

    private var memoryHelp: String {
        var text = "Physical footprint of this process (phys_footprint)."
        if let physical = model.telemetry?.physicalMemoryBytes {
            text += " Machine has \(MetricFormat.memory(physical))."
        }
        return text
    }

    private var memoryAccessibilityValue: String {
        guard let fraction = memoryFraction else { return MetricFormat.memory(memoryBytes) }
        return "\(MetricFormat.memory(memoryBytes)), \(MetricFormat.percent(fraction * 100)) of system memory"
    }

    private func refreshMemory() {
        memoryBytes = model.currentProcessMemoryBytes
        // Refreshed on the SAME poll rather than only when the model list is
        // reloaded, which is what used to set it: memory pressure is live
        // machine state, and a reading taken once at startup would say
        // nothing about the moment a user is looking at.
        model.refreshTelemetry()
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
        HStack(spacing: 6) {
            Image(systemName: "text.alignleft")
                .font(.system(size: 10))
                .accessibilityHidden(true)
            MeterBar(fraction: fraction, tint: fraction > 0.9 ? .orange : TurboSparkTheme.accentColor)
                .frame(width: 44)
            Text("\(model.estimatedContextTokens.formatted(.number.notation(.compactName))) / \(model.resolvedContextTokens.formatted(.number.notation(.compactName)))")
                .monospacedDigit()
        }
        .help("Estimated conversation length against the loaded context window")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Context used")
        .accessibilityValue("\(model.estimatedContextTokens) of \(model.resolvedContextTokens) tokens")
    }

    // MARK: - Throughput

    private var showsThroughput: Bool {
        model.isRunning || model.diagnostics != nil
    }

    private var throughputReadout: some View {
        HStack(spacing: 5) {
            Text(rateText)
                .monospacedDigit()
                .foregroundStyle(.primary)
            Text("tok/s")
                .foregroundStyle(.tertiary)
        }
        .help("Live decoding rate (tokens per second)")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Decode rate")
        .accessibilityValue("\(rateText) tokens per second")
    }

    private var tokenReadout: some View {
        HStack(spacing: 5) {
            Text(tokensText)
                .monospacedDigit()
                .foregroundStyle(.primary)
            Text("tokens")
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

    /// The kernel's own memory-pressure verdict.
    ///
    /// **This is the machine's state, not this process's.** Another app can
    /// put the machine under pressure, and the engine's response is the same
    /// either way: it paces its own decode rate down. It does not unload
    /// anything, because a session belongs to this app rather than to the
    /// engine -- deciding to close an idle one is a decision for the person
    /// reading this strip.
    private func memoryPressureReadout(_ level: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: "exclamationmark.triangle")
                .font(.system(size: 10))
                .accessibilityHidden(true)
            Text("Memory \(level.capitalized)")
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

    private func thermalReadout(_ level: String) -> some View {
        HStack(spacing: 5) {
            Image(systemName: "thermometer.medium")
                .font(.system(size: 10))
                .accessibilityHidden(true)
            Text(level.capitalized)
        }
        .foregroundStyle(.orange)
        .help("Thermal pressure has left Nominal; throughput and power figures are not comparable to a clean run.")
        .accessibilityElement(children: .ignore)
        .accessibilityLabel("Thermal pressure")
        .accessibilityValue(level)
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
