import SwiftUI

/// Memory, CPU and GPU wait, in the top bar's centre.
///
/// Polled on a 2-second timer -- the same cadence `StatusBarView` used for
/// these three -- so this bar never redraws per token. The readouts are the
/// same accessors, moved rather than restated: `currentProcessMemoryBytes`,
/// `currentProcessCPUUsage`, and `diagnostics?.phases?.gpuWaitMs`.
///
/// **GPU wait, not "GPU %".** There is no GPU-utilisation figure anywhere in
/// this app's telemetry, and inventing a percentage from `gpuWaitMs` would be
/// a number arrived at by ignorance (`swift/CLAUDE.md`'s rule about the
/// unsized model's "16 slots"). GPU wait is what the engine actually
/// measures, so GPU wait is what this shows -- and it shows an em dash until a
/// run has produced phase counters, rather than a zero.
@MainActor
struct ChromeTelemetryView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    @State private var memoryBytes: UInt64?
    @State private var cpuPercent: Double?


    var body: some View {
        HStack(spacing: 11) {
            memoryCell
            divider
            cpuCell
            divider
            gpuCell
        }
        .padding(.horizontal, 12)
        .padding(.vertical, 5)
        .background(.appText.opacity(0.04), in: RoundedRectangle(cornerRadius: 9, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 9, style: .continuous)
                .stroke(.appBorder, lineWidth: 0.5)
        }
        .onAppear(perform: refresh)
        .task {
            // Lifetime follows view identity, not struct re-creation: a Timer
            // publisher stored on the struct restarted on every parent render
            // (every ~25 ms while generating), so the 2 s poll never fired.
            while !Task.isCancelled {
                try? await Task.sleep(for: .seconds(2))
                if Task.isCancelled { break }
                refresh()
            }
        }
    }

    private var divider: some View {
        Rectangle()
            .fill(.appBorder)
            .frame(width: 1, height: 13)
    }

    private var memoryCell: some View {
        HStack(spacing: 6) {
            Image(systemName: "memorychip")
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            Text(verbatim: MetricFormat.memory(memoryBytes))
                .font(theme.code(.callout, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(.appText)
            if let fraction = memoryFraction {
                TSMeterBar(fraction: fraction, tint: memoryTint(fraction))
                    .frame(width: 34)
            }
        }
        .help(memoryHelp)
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Process memory", bundle: .module))
        .accessibilityValue(memoryAccessibilityValue)
    }

    private var cpuCell: some View {
        HStack(spacing: 6) {
            Image(systemName: "cpu")
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            Text(verbatim: cpuText)
                .font(theme.code(.callout, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(.appText)
            Text("CPU", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
        }
        .help(Text("Current CPU utilization of the TurboSpark process", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("Process CPU usage", bundle: .module))
        .accessibilityValue(Text(verbatim: cpuText))
    }

    private var gpuCell: some View {
        HStack(spacing: 6) {
            Image(systemName: "chart.line.uptrend.xyaxis")
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
                .accessibilityHidden(true)
            Text(verbatim: gpuWaitText)
                .font(theme.code(.callout, weight: .medium))
                .monospacedDigit()
                .foregroundStyle(.appText)
            Text("GPU wait", bundle: .module)
                .font(theme.ui(.tiny))
                .foregroundStyle(.appSecondary)
        }
        .help(Text("Metal GPU wait per pass from the last run's phase counters. Updates after a generation pass completes; TurboSpark does not measure a live GPU percentage.", bundle: .module))
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(Text("GPU wait per pass", bundle: .module))
        .accessibilityValue(Text(verbatim: gpuWaitText))
    }

    // MARK: - Values

    private var memoryFraction: Double? {
        guard let memoryBytes,
              let physical = model.telemetry?.physicalMemoryBytes,
              physical > 0
        else { return nil }
        return min(1.0, Double(memoryBytes) / Double(physical))
    }

    private func memoryTint(_ fraction: Double) -> Color {
        if fraction > 0.65 { return AppAttention.color }
        return theme.accent
    }

    private var memoryHelp: Text {
        let footprint = Text("Physical footprint of this process (phys_footprint).", bundle: .module)
        guard let physical = model.telemetry?.physicalMemoryBytes else { return footprint }
        return footprint + Text(verbatim: " ")
            + Text("Machine has \(MetricFormat.memory(physical)).", bundle: .module)
    }

    private var memoryAccessibilityValue: Text {
        guard let fraction = memoryFraction else { return Text(verbatim: MetricFormat.memory(memoryBytes)) }
        return Text(
            "\(MetricFormat.memory(memoryBytes)), \(MetricFormat.percent(fraction * 100)) of system memory",
            bundle: .module)
    }

    private var cpuText: String {
        guard let cpuPercent else { return "\u{2014}" }
        return MetricFormat.percent(cpuPercent)
    }

    /// An em dash until a run has produced phase counters. `calls > 0` is the
    /// same guard `RunnerDiagnosticsSection` uses before showing this figure.
    private var gpuWaitText: String {
        guard let phases = model.diagnostics?.phases, phases.calls > 0 else { return "\u{2014}" }
        return String(format: "%.1f ms", phases.gpuWaitMs)
    }

    private func refresh() {
        memoryBytes = model.currentProcessMemoryBytes
        cpuPercent = model.currentProcessCPUUsage
        model.refreshTelemetry()
    }
}

/// Two-tone horizontal meter.
///
/// Declared here rather than reused: `StatusBarView`'s `MeterBar` is
/// `private`, and widening it to share would couple the top bar to a file it
/// otherwise has nothing to do with. Six lines is cheaper than that coupling.
private struct TSMeterBar: View {
    let fraction: Double
    let tint: Color

    var body: some View {
        GeometryReader { geometry in
            ZStack(alignment: .leading) {
                Capsule().fill(.appText.opacity(0.1))
                Capsule()
                    .fill(tint)
                    .frame(width: max(2, geometry.size.width * fraction))
            }
        }
        .frame(height: 4)
        .accessibilityHidden(true)
    }
}
