import AppKit
import SwiftUI
import TurboSpark

extension InspectorView {
    /// Inspector sidebar section displaying hardware RAM, chip details, thermal pressure, and model architecture metadata.
    var telemetrySection: some View {
        Section(header: Text("Session Telemetry & Introspection", bundle: .module)) {
            if let t = model.telemetry {
                LabeledContent {
                    Text(MetricFormat.storage(t.physicalMemoryBytes))
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                } label: {
                    Text("Physical RAM", bundle: .module)
                }
                if let rec = t.recommendedWorkingSetBytes {
                    LabeledContent {
                        Text(MetricFormat.storage(rec))
                            .themedFont(.small).monospacedDigit()
                            .foregroundStyle(.appSecondary)
                    } label: {
                        Text("Recommended max", bundle: .module)
                    }
                }
                if let chip = t.chip {
                    LabeledContent {
                        Text(chip)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    } label: {
                        Text("Chip", bundle: .module)
                    }
                }
                LabeledContent {
                    if t.lowPowerMode {
                        Text("Enabled", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.orange)
                    } else {
                        Text("Disabled", bundle: .module)
                            .themedFont(.small)
                            .foregroundStyle(.secondary)
                    }
                } label: {
                    Text("Low Power Mode", bundle: .module)
                }
                LabeledContent {
                    Self.thermalLevelText(t.thermalLevel)
                        .themedFont(.small)
                        .foregroundStyle(t.thermalLevel == "nominal" ? Color.secondary : Color.orange)
                } label: {
                    Text("Thermal Level", bundle: .module)
                }
            }

            if let info = model.info {
                LabeledContent {
                    Text(info.dialect)
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                } label: {
                    Text("Dialect", bundle: .module)
                }
                LabeledContent {
                    Text(verbatim: "\(info.vocabSize)")
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                } label: {
                    Text("Vocab Size", bundle: .module)
                }
                LabeledContent {
                    Text(verbatim: "\(info.maxContext)")
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                } label: {
                    Text("Resolved Context", bundle: .module)
                }
                if let trained = info.trainedContext {
                    LabeledContent {
                        Text(verbatim: "\(trained)")
                            .themedFont(.small).monospacedDigit()
                            .foregroundStyle(.appSecondary)
                    } label: { Text("Trained Context", bundle: .module) }
                }
                if info.pastTrainedContext {
                    Text("Warning: Context window exceeds trained context", bundle: .module)
                        .themedFont(.small)
                        .foregroundStyle(.orange)
                }
                LabeledContent {
                    Text(verbatim: "\(info.expertCacheSlots)")
                        .themedFont(.small).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                } label: {
                    Text("Resolved Slots", bundle: .module)
                }
                if let block = info.speculation.block {
                    LabeledContent {
                        // The drafter name is the engine's identifier; only
                        // the "on" fallback is a word.
                        ((info.speculation.drafter.map { Text(verbatim: $0.rawValue) } ?? Text("On", bundle: .module))
                            + Text(verbatim: " x\(block)"))
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    } label: {
                        Text("Speculative", bundle: .module)
                    }
                } else if let reason = info.speculation.reason {
                    LabeledContent {
                        // `reason` is the engine's own sentence, shown as sent.
                        (Text("Off", bundle: .module) + Text(verbatim: " (\(reason))"))
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                    } label: {
                        Text("Speculative", bundle: .module)
                    }
                }
                if info.steering.active {
                    LabeledContent {
                        Group {
                            // Summary and mode come from the engine as sent;
                            // only the fallback is ours to translate.
                            if let summary = info.steering.summary ?? info.steering.mode {
                                Text(verbatim: summary)
                            } else {
                                Text("Active", bundle: .module)
                            }
                        }
                        .themedFont(.small)
                        .foregroundStyle(.appSecondary)
                    } label: {
                        Text("Active Steering", bundle: .module)
                    }
                }
                if !info.specialTokens.stopTokenIds.isEmpty {
                    LabeledContent {
                        Text(verbatim: info.specialTokens.stopTokenIds.map { "\($0)" }.joined(separator: ", "))
                        .themedFont(.tiny).monospacedDigit()
                        .foregroundStyle(.appSecondary)
                    } label: {
                        Text("Stop Token IDs", bundle: .module)
                    }
                }
            }
        }
    }

    /// The thermal state in words. The engine reports `ProcessInfo`'s level
    /// as a lowercase identifier, so capitalizing it gave English everywhere;
    /// an unknown future level still shows as sent.
    private static func thermalLevelText(_ level: String) -> Text {
        switch level.lowercased() {
        case "nominal": return Text("Nominal", bundle: .module)
        case "fair": return Text("Fair", bundle: .module)
        case "serious": return Text("Serious", bundle: .module)
        case "critical": return Text("Critical", bundle: .module)
        default: return Text(verbatim: level.capitalized)
        }
    }
}
