import SwiftUI

/// The downloads panel: current work and profile history across launches.
/// It opens under the top bar, where `DownloadsTopBarButton` lives, and the
/// root view hosts it. Anything that sets `isDownloadManagerExpanded` (the
/// Humanize and Audio model bars) opens it the same way.
@MainActor
struct DownloadManagerView: View {
    @ObservedObject var model: AppModel
    @State private var listHeight: CGFloat = 300

    var body: some View {
        VStack(alignment: .trailing, spacing: 10) {
            if model.isDownloadManagerExpanded {
                panel.transition(.move(edge: .top).combined(with: .opacity))
            }
        }
        .foregroundStyle(.appText)
        .frame(maxWidth: 420, alignment: .trailing)
    }

    private var panel: some View {
        VStack(alignment: .leading, spacing: 12) {
            AppAdaptiveHeader {
                Text("Downloads", bundle: .module)
            } actions: {
                Button { model.clearFinishedDownloads() } label: {
                    Text("Clear", bundle: .module)
                }
                .disabled(!model.modelDownloads.contains { $0.status.isTerminal })
                .appAction(.quiet, size: .small)
                .themedFont(.small)
            }

            if model.modelDownloads.isEmpty {
                AppEmptyState(
                    systemImage: "arrow.down.circle",
                    title: Text("No downloads yet", bundle: .module)
                )
                // AppEmptyState fills its container, which is right for a
                // workspace. In this floating panel it stretched the empty
                // card up the window's whole trailing edge, over the
                // inspector (RootWindowLayoutTests).
                .fixedSize(horizontal: false, vertical: true)
                .padding(.vertical, 12)
            } else {
                ScrollView {
                    VStack(spacing: 10) {
                        ForEach(model.modelDownloads) { download in
                            if download.id == model.activeModelDownloadID && model.hasActiveModelDownload {
                                ModelInstallToastView(model: model)
                            } else {
                                historyRow(download)
                            }
                        }
                    }
                    .padding(1)
                    .background {
                        GeometryReader { geometry in
                            Color.clear.preference(
                                key: DownloadListHeightKey.self, value: geometry.size.height)
                        }
                    }
                }
                .onPreferenceChange(DownloadListHeightKey.self) { listHeight = $0 }
                // Fit short histories while allowing the shared overlay to
                // shrink the scroll area in a short window.
                .frame(maxHeight: min(300, listHeight))
                .defaultScrollAnchor(.top)
            }

            Text(historyFooter, bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(14)
        .appGlassSurface(.rect(cornerRadius: AppMetrics.cardRadius))
    }

    private var historyFooter: LocalizedStringKey {
        if !model.downloadHistoryWritable { return "Download history could not be loaded." }
        if model.downloadHistorySaveFailed { return "Download history could not be saved." }
        return "History is saved across launches. Retry reuses verified ranges for pinned models; conversion restarts."
    }

    private func historyRow(_ download: ModelDownload) -> some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: download.status == .completed ? "checkmark.circle" : "arrow.down.circle")
                .foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 4) {
                Text(verbatim: download.request.alias)
                    .themedFont(.small, weight: .medium)
                    .lineLimit(2)
                    .truncationMode(.middle)
                    .fixedSize(horizontal: false, vertical: true)
                Text(statusLabel(download.status), bundle: .module)
                    .themedFont(.tiny)
                    .foregroundStyle(.appSecondary)
                if download.downloadedBytes > 0 {
                    if download.status == .completed {
                        Text(verbatim: MetricFormat.storage(download.downloadedBytes))
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                    } else if let total = download.totalBytes, total > 0 {
                        Text("Last progress: \(MetricFormat.storage(download.downloadedBytes)) of \(MetricFormat.storage(total))", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                    } else {
                        Text("Last progress: \(MetricFormat.storage(download.downloadedBytes))", bundle: .module)
                            .themedFont(.tiny)
                            .foregroundStyle(.appSecondary)
                    }
                }
                if download.updatedAt != .distantPast {
                    Text(download.updatedAt, format: .dateTime.month(.abbreviated).day().hour().minute())
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                }
                if let failure = download.failure {
                    Text(verbatim: failure)
                        .themedFont(.tiny)
                        .foregroundStyle(.appSecondary)
                        .lineLimit(2)
                        .help(failure)
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            if download.status.canRetry {
                Button { model.retryModelDownload(download) } label: {
                    Text("Retry", bundle: .module)
                }
                .appAction(.secondary, size: .small)
                .disabled(!model.canRetryModelDownload(download))
                .help(!model.canRetryModelDownload(download)
                    ? Text("Download cannot be retried at this time.", bundle: .module)
                    : Text("Retry", bundle: .module))
            }
        }
        .padding(10)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius))
    }

    private func statusLabel(_ status: ModelDownload.Status) -> LocalizedStringKey {
        switch status {
        case .queued: return "Queued"
        case .running: return "Downloading"
        case .paused: return "Paused"
        case .packing: return "Packing"
        case .verifying: return "Verifying"
        case .loading: return "Loading model"
        case .cancelling: return "Cancelling"
        case .completed: return "Completed"
        case .cancelled: return "Cancelled"
        case .failed: return "Failed"
        case .interrupted: return "Interrupted"
        }
    }
}

private struct DownloadListHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}

/// The top bar's downloads control. It replaced a floating pill in the
/// window's bottom corner, and the strip of empty space every pane gave up to
/// keep that pill clear of the composer. Idle it is one quiet icon; while a
/// model downloads it grows a percentage (or a bar when the size is unknown)
/// and an accent tint, so progress is visible from every section.
struct DownloadsTopBarButton: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    @ScaledMetric private var buttonSize: CGFloat = 26

    var body: some View {
        let active = model.hasActiveModelDownload
        let lit = active || model.isDownloadManagerExpanded
        Button {
            TSMotion.animate(TSMotion.pane) { model.isDownloadManagerExpanded.toggle() }
        } label: {
            HStack(spacing: 5) {
                Image(systemName: model.isInstallPaused ? "pause.circle" : "arrow.down.circle")
                    .accessibilityHidden(true)
                if active { progress }
            }
            .font(theme.ui(.callout, weight: .medium))
            .padding(.horizontal, active ? 8 : 0)
            .frame(minWidth: buttonSize, minHeight: buttonSize)
            .foregroundStyle(lit ? AnyShapeStyle(.appAccentText) : AnyShapeStyle(.appSecondary))
            .background(
                lit ? theme.accent.opacity(0.14) : .clear,
                in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
            .contentShape(Rectangle())
        }
        .buttonStyle(TSPressScaleStyle(scale: 0.9))
        .tsAnimation(TSMotion.hover, value: lit)
        .tsAnimation(TSMotion.hover, value: active)
        .help(Text("Downloads", bundle: .module))
        .accessibilityLabel(Text("Downloads", bundle: .module))
        .accessibilityValue(progressValue)
        .accessibilityAddTraits(model.isDownloadManagerExpanded ? .isSelected : [])
    }

    @ViewBuilder private var progress: some View {
        if model.isCancellingModelInstall {
            Text("Cancelling", bundle: .module).font(theme.ui(.small, weight: .medium))
        } else if model.isInstallPaused {
            Text("Paused", bundle: .module).font(theme.ui(.small, weight: .medium))
        } else if let fraction = model.installProgressFraction {
            Text(verbatim: MetricFormat.percent(fraction * 100))
                .font(theme.ui(.small, weight: .medium)).monospacedDigit()
        } else {
            AppProgressBar(value: nil, label: Text("Downloading", bundle: .module), size: .compact)
                .frame(width: 36)
        }
    }

    private var progressValue: Text {
        guard model.hasActiveModelDownload else { return Text(verbatim: "") }
        if model.isCancellingModelInstall { return Text("Cancelling", bundle: .module) }
        if model.isInstallPaused { return Text("Paused", bundle: .module) }
        if let fraction = model.installProgressFraction { return Text(verbatim: MetricFormat.percent(fraction * 100)) }
        return Text("Downloading", bundle: .module)
    }
}
