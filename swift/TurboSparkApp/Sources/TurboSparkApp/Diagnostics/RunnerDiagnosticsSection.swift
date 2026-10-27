import SwiftUI
import TurboSpark

/// Diagnostics inspector section presenting prefill/decode timing, token counts, tok/s speed, and GPU phase breakdown.
struct RunnerDiagnosticsSection: View {
    let diagnostics: AppDiagnostics?
    var recoveryEvents: [StreamRecoveryEvent] = []

    var body: some View {
        Section(header: Text("Last run", bundle: .module)) {
            if let diagnostics {
                groupLabel("Result")
                DiagnosticRow("Prompt tokens", "\(diagnostics.promptTokens)")
                DiagnosticRow("Output tokens", "\(diagnostics.generatedTokens)")
                DiagnosticRow("Stop reason", value: Self.stopReasonText(diagnostics.stopReason))
                if let event = diagnostics.compactionBoundaryEvent {
                    ForEach(Self.compactionBoundaryRows(for: event), id: \.label) { row in
                        DiagnosticRow(verbatim: row.label, row.value)
                    }
                }

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
                    // The engine's own phase name (command buffer 1), not prose.
                    DiagnosticRow(verbatim: "CB1 GPU", String(format: "%.2f ms", phases.cb1GpuMs))
                    if let hitRate = phases.expertHitRate {
                        DiagnosticRow("Expert cache hit", String(format: "%.1f%%", hitRate * 100))
                    }
                }
            } else {
                Text("No runs yet", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }

            if !recoveryEvents.isEmpty {
                groupLabel("Recovery events")
                ForEach(Array(recoveryEvents.enumerated()), id: \.offset) { entry in
                    StreamRecoveryEventNoticeView(event: entry.element, occurrence: entry.offset)
                }
            }
        }
    }

    private func groupLabel(_ title: LocalizedStringKey) -> some View {
        Text(title, bundle: .module)
            .themedFont(.small)
            .textCase(.uppercase)
            .foregroundStyle(.appSecondary)
            .accessibilityHeading(.h3)
    }

    /// The stop reason in words. The raw values (`endOfTurn`, `maxTokens`)
    /// are wire identifiers, so they were English camelCase in every language.
    static func stopReasonText(_ reason: GenerationResult.StopReason) -> Text {
        switch reason {
        case .endOfTurn: return Text("End of turn", bundle: .module)
        case .toolCalls: return Text("Tool calls", bundle: .module)
        case .eos: return Text("End of sequence", bundle: .module)
        case .stopString: return Text("Stop string", bundle: .module)
        case .maxTokens: return Text("Max tokens", bundle: .module)
        case .cancelled: return Text("Cancelled", bundle: .module)
        case .unknown: return Text("Unknown", bundle: .module)
        }
    }

    /// Labels are values rather than views so the request-boundary test can
    /// read them; `AppStrings` keeps them on the in-app language like the
    /// `Text` rows around them.
    static func compactionBoundaryRows(for event: CompactionBoundaryEvent) -> [RunnerDiagnosticValue] {
        [
            RunnerDiagnosticValue(
                label: AppStrings.localized("Rows shortened"),
                value: "\(event.rowsCleared)"),
            RunnerDiagnosticValue(
                label: AppStrings.localized("Estimated tokens saved"),
                value: "\(event.estimatedTokensSaved)"),
        ]
    }
}

struct RunnerDiagnosticValue: Equatable {
    let label: String
    let value: String
}

private struct DiagnosticRow: View {
    let label: Text
    let value: Text

    /// A catalog label beside a readout with nothing to translate (a count,
    /// a duration, a rate).
    init(_ label: LocalizedStringKey, _ value: String) {
        self.label = Text(label, bundle: .module)
        self.value = Text(verbatim: value)
    }

    init(_ label: LocalizedStringKey, value: Text) {
        self.label = Text(label, bundle: .module)
        self.value = value
    }

    /// A label that is already localized, or an identifier with nothing to
    /// translate.
    init(verbatim label: String, _ value: String) {
        self.label = Text(verbatim: label)
        self.value = Text(verbatim: value)
    }

    var body: some View {
        LabeledContent {
            value
                .themedFont(.small).monospacedDigit()
                .foregroundStyle(.appSecondary)
        } label: {
            label
        }
    }
}
