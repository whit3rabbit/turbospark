import SwiftUI

/// One bounded progress card, driven by the install rather than a toast timer.
@MainActor
struct ModelInstallToastView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    private var title: LocalizedStringKey {
        if model.isCancellingModelInstall { return "Cancelling" }
        if model.isInstallPaused { return "Paused" }
        if model.opening { return "Loading model" }
        if let fraction = model.installProgressFraction, fraction < 1 {
            return "Downloading"
        }
        return "Installing"
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 10) {
                Image(systemName: model.isInstallPaused ? "pause.circle" : "arrow.down.circle")
                    .themedFont(.base, weight: .semibold)
                    .foregroundStyle(.appAccent)
                    .accessibilityHidden(true)

                VStack(alignment: .leading, spacing: 3) {
                    Text(title, bundle: .module)
                        .themedFont(.base, weight: .semibold)
                    if let alias = model.installingAlias {
                        Text(alias)
                            .themedFont(.small)
                            .foregroundStyle(.appSecondary)
                            .lineLimit(2)
                            .truncationMode(.middle)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }
                .frame(maxWidth: .infinity, alignment: .leading)

                if !model.opening && !model.isCancellingModelInstall {
                    Button {
                        if model.isInstallPaused {
                            model.resumeModelDownload()
                        } else {
                            model.pauseModelDownload()
                        }
                    } label: {
                        if model.isInstallPaused {
                            Text("Resume", bundle: .module)
                        } else {
                            Text("Pause", bundle: .module)
                        }
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .disabled(!model.isInstallPaused && !model.canPauseModelDownload)
                    .fixedSize()

                    Button(action: model.cancelActiveModelDownload) {
                        Text("Cancel", bundle: .module)
                    }
                    .buttonStyle(.bordered)
                    .controlSize(.small)
                    .fixedSize()
                    .accessibilityLabel(Text("Cancel Download", bundle: .module))
                }
            }

            if let fraction = model.installProgressFraction, fraction < 1, !model.opening {
                ProgressView(value: fraction)
                    .tint(theme.accent)
                    .accessibilityLabel(Text(title, bundle: .module))
                    .accessibilityValue(MetricFormat.percent(fraction * 100))

                HStack(spacing: 8) {
                    if let downloaded = model.installDownloadedBytes,
                       let total = model.installTotalBytes, total > 0 {
                        Text("\(MetricFormat.storage(downloaded)) of \(MetricFormat.storage(total))", bundle: .module)
                            .lineLimit(1)
                    }
                    Spacer(minLength: 0)
                    TimelineView(.periodic(from: .now, by: 1)) { _ in
                        if let speed = model.modelDownloadSpeed() {
                            Text("Speed: \(MetricFormat.transferBytes(speed))/s", bundle: .module)
                        } else {
                            Text(MetricFormat.percent(fraction * 100))
                        }
                    }
                    .fixedSize()
                }
                .themedFont(.small)
                .monospacedDigit()
                .foregroundStyle(.appSecondary)
            } else if model.isInstallPaused {
                ProgressView(value: model.installProgressFraction ?? 0)
                    .tint(theme.accent)
                    .accessibilityLabel(Text("Paused", bundle: .module))
            } else {
                ProgressView()
                    .progressViewStyle(.linear)
                    .tint(theme.accent)
                    .accessibilityLabel(Text(title, bundle: .module))
            }
            if let downloaded = model.installDownloadedBytes,
               model.installTotalBytes == nil || model.installTotalBytes == 0 {
                Text(MetricFormat.storage(downloaded))
                    .themedFont(.small)
                    .monospacedDigit()
                    .foregroundStyle(.appSecondary)
            }
            if model.installProgressFraction == nil || (model.installProgressFraction ?? 0) >= 1 {
                TimelineView(.periodic(from: .now, by: 1)) { _ in
                    if let speed = model.modelDownloadSpeed() {
                        Text("Speed: \(MetricFormat.transferBytes(speed))/s", bundle: .module)
                            .themedFont(.small)
                            .monospacedDigit()
                            .foregroundStyle(.appSecondary)
                    }
                }
            }
        }
        .padding(14)
        .foregroundStyle(.appText)
        .background {
            RoundedRectangle(cornerRadius: 14)
                .fill(.appPage)
                .shadow(color: .black.opacity(0.18), radius: 12, x: 0, y: 4)
                .overlay {
                    RoundedRectangle(cornerRadius: 14)
                        .stroke(.appBorder, lineWidth: 0.5)
                }
        }
        .onAppear {
            let message = String(localized: "Installing", bundle: .module)
                + ": " + (model.installingAlias ?? "")
            _ = AccessibilityNotification.Announcement.post(.init(message))
        }
    }
}
