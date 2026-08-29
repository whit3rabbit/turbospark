import SwiftUI
import TurboSpark

/// Hardware scan summary card showing detected Apple Silicon hardware, RAM, and working set.
struct HardwareScanCard: View {
    let telemetry: SystemTelemetry?

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Image(systemName: "memorychip")
                    .font(.headline)
                    .foregroundStyle(TurboSparkTheme.accentColor)
                    .accessibilityHidden(true)
                Text("Hardware & Memory Analysis")
                    .font(.headline)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Text("Detected")
                    .font(.caption2.weight(.bold))
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
            Color(nsColor: .controlBackgroundColor).opacity(0.8),
            in: RoundedRectangle(cornerRadius: 14)
        )
        .overlay(
            RoundedRectangle(cornerRadius: 14)
                .stroke(Color(nsColor: .separatorColor).opacity(0.6), lineWidth: 0.5)
        )
    }

    private func statItem(label: String, value: String, icon: String) -> some View {
        VStack(alignment: .leading, spacing: 3) {
            HStack(spacing: 4) {
                Image(systemName: icon)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(label)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            Text(value)
                .font(.callout.weight(.semibold))
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
                    .font(.headline)
                    .foregroundStyle(Color.purple)
                    .accessibilityHidden(true)
                Text("MoE (Mixture of Experts) Architecture Spotlight")
                    .font(.subheadline.weight(.bold))
                    .foregroundStyle(.primary)
                    .accessibilityAddTraits(.isHeader)
                Spacer()
                Text("RAM Saver")
                    .font(.caption2.weight(.bold))
                    .padding(.horizontal, 6)
                    .padding(.vertical, 2)
                    .background(Color.purple.opacity(0.2), in: Capsule())
                    .foregroundStyle(Color.purple)
                    .accessibilityHidden(true)
            }

            Text("MoE models stream weights directly from fast NVMe storage and only cache active experts in **~2-4 GB RAM**. This delivers **26B-35B parameter intelligence** on Apple Silicon without exhausting memory.")
                .font(.caption)
                .foregroundStyle(.secondary)
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
                    Color(nsColor: .controlBackgroundColor).opacity(0.6),
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

    private func featurePill(icon: String, text: String) -> some View {
        HStack(spacing: 4) {
            Image(systemName: icon)
                .font(.caption2)
                .foregroundStyle(Color.purple)
                .accessibilityHidden(true)
            Text(text)
                .font(.caption2.weight(.medium))
                .foregroundStyle(.primary)
        }
        .padding(.horizontal, 8)
        .padding(.vertical, 4)
        .background(Color(nsColor: .windowBackgroundColor).opacity(0.7), in: RoundedRectangle(cornerRadius: 6))
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
                    Text(model.installStageText ?? "Downloading model...")
                        .font(.callout.weight(.semibold))
                }
                Spacer()
                Button("Cancel", action: model.cancelInstall)
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .accessibilityLabel("Cancel download")
                    .accessibilityHint("Stops the model download and keeps any verified progress")
            }

            if let fraction = model.installProgressFraction {
                ProgressView(value: fraction)
                    .tint(TurboSparkTheme.accentColor)
                    .accessibilityLabel("Download progress")
                    .accessibilityValue(MetricFormat.percent(fraction * 100))

                HStack {
                    if let downloaded = model.installDownloadedBytes, let total = model.installTotalBytes {
                        Text("\(MetricFormat.storage(downloaded)) of \(MetricFormat.storage(total))")
                    }
                    Spacer()
                    Text(MetricFormat.percent(fraction * 100))
                }
                .font(.caption.monospacedDigit())
                .foregroundStyle(.secondary)
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
