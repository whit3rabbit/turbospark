import SwiftUI
import TurboSpark

/// Hardware scan summary card showing detected Apple Silicon hardware, RAM, and working set.
struct HardwareScanCard: View {
    let telemetry: SystemTelemetry?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "memorychip")
                    .themedFont(.base, weight: .semibold)
                    .foregroundStyle(.appAccent)
                    .help(Text("Apple Silicon unified memory and hardware analysis", bundle: .module))
                    .accessibilityHidden(true)
                Text("Hardware & Memory Analysis", bundle: .module)
                    .themedFont(.base, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Text("Detected", bundle: .module)
                    .themedFont(.tiny, weight: .bold)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.green.opacity(0.18), in: Capsule())
                    .foregroundStyle(.green)
                    .accessibilityHidden(true)
            }

            Divider()

            HStack(spacing: 16) {
                statItem(
                    label: "Apple Silicon Chip",
                    value: telemetry?.chip ?? "Apple Silicon",
                    icon: "cpu"
                )

                Divider()

                statItem(
                    label: "Physical RAM",
                    value: MetricFormat.memory(telemetry?.physicalMemoryBytes),
                    icon: "chart.bar.fill"
                )

                if let ws = telemetry?.recommendedWorkingSetBytes {
                    Divider()
                    statItem(
                        label: "Working Set Budget",
                        value: MetricFormat.memory(ws),
                        icon: "gauge.with.needle"
                    )
                }
            }
        }
        .padding(16)
        .background(
            .appSurface,
            in: RoundedRectangle(cornerRadius: 14)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(.appBorder.opacity(0.6), lineWidth: 0.5)
        )
    }

    /// `label` is a catalog key; `value` is a reading and renders as written.
    private func statItem(label: LocalizedStringKey, value: String, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
                Text(label, bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
            }
            Text(verbatim: value)
                .themedFont(.base, weight: .semibold)
                .lineLimit(1)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }
}

/// Spotlight card explaining MoE streaming advantages and memory efficiency.
struct MoESpotlightCard: View {
    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                Image(systemName: "sparkles")
                    .themedFont(.base, weight: .semibold)
                    .foregroundStyle(Color.purple)
                    .help(Text("Mixture of Experts architecture details", bundle: .module))
                    .accessibilityHidden(true)
                Text("MoE (Mixture of Experts) Architecture Spotlight", bundle: .module)
                    .themedFont(.small, weight: .bold)
                    .foregroundStyle(.appText)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Text("RAM Saver", bundle: .module)
                    .themedFont(.tiny, weight: .bold)
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.purple.opacity(0.2), in: Capsule())
                    .foregroundStyle(Color.purple)
                    .accessibilityHidden(true)
            }

            Text("MoE models stream weights directly from fast NVMe storage and only cache active experts in **~2-4 GB RAM**. This delivers **26B-35B parameter intelligence** on Apple Silicon without exhausting memory.", bundle: .module)
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)

            HStack(spacing: 12) {
                featurePill(icon: "bolt.fill", text: "Streaming Slot Cache (~2-4 GB RAM)")
                featurePill(icon: "brain.head.profile", text: "High Parameter Quality (20B-35B)")
                featurePill(icon: "hare.fill", text: "Fast Local Metal Decodes")
            }
            .padding(.top, 2)
        }
        .padding(16)
        .background(
            LinearGradient(
                colors: [
                    Color.purple.opacity(0.12),
                    Color.clear,
                ],
                startPoint: .topLeading,
                endPoint: .bottomTrailing
            ),
            in: RoundedRectangle(cornerRadius: 14)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.purple.opacity(0.3), lineWidth: 0.8)
        )
    }

    private func featurePill(icon: String, text: LocalizedStringKey) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .themedFont(.tiny)
                .foregroundStyle(Color.purple)
                .accessibilityHidden(true)
            Text(text, bundle: .module)
                .themedFont(.tiny, weight: .medium)
                .foregroundStyle(.appText)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(.appPage.opacity(0.7), in: RoundedRectangle(cornerRadius: 6))
    }
}

/// Progress card displayed when a model installation/download is active.
struct ActiveInstallCard: View {
    @ObservedObject var model: AppModel

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Group {
                        if let stage = model.installStageText {
                            Text(stage)
                        } else {
                            Text("Downloading model files", bundle: .module)
                        }
                    }
                    .themedFont(.base, weight: .semibold)
                }
                Spacer()
                Button(action: model.cancelInstall) { Text("Cancel", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .help(Text("Cancel model download", bundle: .module))
                    .accessibilityLabel(Text("Cancel Download", bundle: .module))
                    .accessibilityHint(Text(
                        "Cancels the download; nothing is kept, and the model can be re-installed after the cancelled walk exits",
                        bundle: .module))
            }

            if let fraction = model.installProgressFraction {
                ProgressView(value: fraction)
                    .tint(TurboSparkTheme.accentColor)
                    .accessibilityLabel(Text("Download progress", bundle: .module))
                    .accessibilityValue(Text(verbatim: MetricFormat.percent(fraction * 100)))

                HStack {
                    if let downloaded = model.installDownloadedBytes, let total = model.installTotalBytes {
                        Text("\(MetricFormat.storage(downloaded)) of \(MetricFormat.storage(total))", bundle: .module)
                    }
                    Spacer()
                    Text(verbatim: MetricFormat.percent(fraction * 100))
                }
                .themedFont(.small).monospacedDigit()
                .foregroundStyle(.appSecondary)
            }
        }
        .padding(16)
        .background(Color.accentColor.opacity(0.1), in: RoundedRectangle(cornerRadius: 14))
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color.accentColor.opacity(0.35), lineWidth: 1)
        )
    }
}
