import AppKit
import SwiftUI
import TurboSpark

extension InspectorView {
    /// Inspector sidebar section displaying hardware RAM, chip details, thermal pressure, and model architecture metadata.
    var telemetrySection: some View {
        Section("Session Telemetry & Introspection") {
            if let t = model.telemetry {
                LabeledContent("Physical RAM") {
                    Text(MetricFormat.storage(t.physicalMemoryBytes))
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                if let rec = t.recommendedWorkingSetBytes {
                    LabeledContent("Recommended max") {
                        Text(MetricFormat.storage(rec))
                            .themedFont(.small).monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
                if let chip = t.chip {
                    LabeledContent("Chip") {
                        Text(chip)
                            .themedFont(.small)
                            .foregroundStyle(.secondary)
                    }
                }
                LabeledContent("Low Power Mode") {
                    Text(t.lowPowerMode ? "Enabled" : "Disabled")
                        .themedFont(.small)
                        .foregroundStyle(t.lowPowerMode ? .orange : .secondary)
                }
                LabeledContent("Thermal Level") {
                    Text(t.thermalLevel.capitalized)
                        .themedFont(.small)
                        .foregroundStyle(t.thermalLevel == "nominal" ? Color.secondary : Color.orange)
                }
            }

            if let info = model.info {
                LabeledContent("Dialect") {
                    Text(info.dialect)
                        .themedFont(.small)
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Vocab Size") {
                    Text("\(info.vocabSize)", bundle: .module)
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                LabeledContent("Resolved Context") {
                    Text("\(info.maxContext)", bundle: .module)
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                if let trained = info.trainedContext {
                    LabeledContent("Trained Context") {
                        Text("\(trained)", bundle: .module)
                            .themedFont(.small).monospacedDigit()
                            .foregroundStyle(.secondary)
                    }
                }
                if info.pastTrainedContext {
                    Text("Warning: Context window exceeds trained context", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.orange)
                }
                LabeledContent("Resolved Slots") {
                    Text("\(info.expertCacheSlots)", bundle: .module)
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.secondary)
                }
                if let block = info.speculation.block {
                    LabeledContent("Speculative") {
                        Text("\(info.speculation.drafter?.rawValue ?? "on") x\(block)", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.secondary)
                    }
                } else if let reason = info.speculation.reason {
                    LabeledContent("Speculative") {
                        Text("Off (\(reason))", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.secondary)
                    }
                }
                if info.steering.active {
                    LabeledContent("Active Steering") {
                        Text(info.steering.summary ?? (info.steering.mode ?? "Active"))
                            .themedFont(.small)
                            .foregroundStyle(.secondary)
                    }
                }
                if !info.specialTokens.stopTokenIds.isEmpty {
                    LabeledContent("Stop Token IDs") {
                        Text(info.specialTokens.stopTokenIds.map { "\($0)" }.joined(separator: ", "))
                        .themedFont(.tiny).monospacedDigit()
                        .foregroundStyle(.secondary)
                    }
                }
            }
        }
    }
}
