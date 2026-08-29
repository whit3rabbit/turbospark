import AppKit
import SwiftUI
import TurboSpark

extension InspectorView {
    var telemetrySection: some View {
        Section("Session Telemetry & Introspection") {
            if let t = model.telemetry {
                LabeledContent("Physical RAM") {
                    Text(MetricFormat.storage(t.physicalMemoryBytes))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let rec = t.recommendedWorkingSetBytes {
                    LabeledContent("Recommended max") {
                        Text(MetricFormat.storage(rec))
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                if let chip = t.chip {
                    LabeledContent("Chip") {
                        Text(chip)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Low Power Mode") {
                    Text(t.lowPowerMode ? "Enabled" : "Disabled")
                        .font(.caption)
                        .foregroundStyle(t.lowPowerMode ? .orange : .secondary)
                }
                LabeledContent("Thermal Level") {
                    Text(t.thermalLevel.capitalized)
                        .font(.caption)
                        .foregroundStyle(t.thermalLevel == "nominal" ? Color.secondary : Color.orange)
                }
            }

            if let info = model.info {
                LabeledContent("Dialect") {
                    Text(info.dialect)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Vocab Size") {
                    Text("\(info.vocabSize)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Resolved Context") {
                    Text("\(info.maxContext)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let trained = info.trainedContext {
                    LabeledContent("Trained Context") {
                        Text("\(trained)")
                            .font(.caption.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
                if info.pastTrainedContext {
                    Text("Warning: Context window exceeds trained context")
                        .font(.caption)
                        .foregroundStyle(.orange)
                }
                LabeledContent("Resolved Slots") {
                    Text("\(info.expertCacheSlots)")
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
                if let block = info.speculation.block {
                    LabeledContent("Speculative") {
                        Text("\(info.speculation.drafter?.rawValue ?? "on") x\(block)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                } else if let reason = info.speculation.reason {
                    LabeledContent("Speculative") {
                        Text("Off (\(reason))")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if info.steering.active {
                    LabeledContent("Active Steering") {
                        Text(info.steering.summary ?? (info.steering.mode ?? "Active"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }
                if !info.specialTokens.stopTokenIds.isEmpty {
                    LabeledContent("Stop Token IDs") {
                        Text(info.specialTokens.stopTokenIds.map { "\($0)" }.joined(separator: ", "))
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }
}
