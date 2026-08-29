import SwiftUI

struct RunnerDiagnosticsSection: View {
    let diagnostics: AppDiagnostics?

    var body: some View {
        Section("Last run") {
            if let diagnostics {
                groupLabel("Result")
                DiagnosticRow("Prompt tokens", "\(diagnostics.promptTokens)")
                DiagnosticRow("Output tokens", "\(diagnostics.generatedTokens)")
                DiagnosticRow("Stop reason", diagnostics.stopReason.rawValue)

                groupLabel("Performance")
                DiagnosticRow("Prefill time", MetricFormat.seconds(diagnostics.prefillSeconds))
                DiagnosticRow("Decode time", MetricFormat.seconds(diagnostics.decodeSeconds))
                DiagnosticRow("Decode rate", "\(MetricFormat.rate(diagnostics.tokensPerSecond)) tok/s")
                if let peak = diagnostics.peakMemoryBytes {
                    DiagnosticRow("Peak memory", MetricFormat.memory(peak))
                }

                if let phases = diagnostics.phases, phases.calls > 0 {
                    groupLabel("Phase Breakdown (Metal)")
                    DiagnosticRow("Passes / calls", "\(phases.calls)")
                    DiagnosticRow("Time / pass", String(format: "%.2f ms", phases.totalMsPerCall))
                    DiagnosticRow("GPU wait", String(format: "%.2f ms", phases.gpuWaitMs))
                    DiagnosticRow("Router", String(format: "%.2f ms", phases.routerMs))
                    DiagnosticRow("Expert I/O", String(format: "%.2f ms", phases.expertIoMs))
                    DiagnosticRow("CB1 GPU", String(format: "%.2f ms", phases.cb1GpuMs))
                    if let hitRate = phases.expertHitRate {
                        DiagnosticRow("Expert cache hit", String(format: "%.1f%%", hitRate * 100))
                    }
                }

            } else {
                Text("No runs yet")
                    .font(.caption)
                    .foregroundStyle(.tertiary)
            }
        }
    }

    private func groupLabel(_ title: String) -> some View {
        Text(title)
            .font(.caption)
            .textCase(.uppercase)
            .foregroundStyle(.tertiary)
            .accessibilityHeading(.h3)
    }
}

private struct DiagnosticRow: View {
    let label: String
    let value: String

    init(_ label: String, _ value: String) {
        self.label = label
        self.value = value
    }

    var body: some View {
        LabeledContent(label) {
            Text(value)
                .font(.caption)
                .monospacedDigit()
                .foregroundStyle(.secondary)
                .textSelection(.enabled)
                .multilineTextAlignment(.trailing)
        }
    }
}
