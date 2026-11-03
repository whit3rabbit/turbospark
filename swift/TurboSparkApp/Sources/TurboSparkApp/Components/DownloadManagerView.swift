import SwiftUI

/// Corner access to current work and profile history across launches.
@MainActor
struct DownloadManagerView: View {
    @ObservedObject var model: AppModel
    @State private var listHeight: CGFloat = 300

    var body: some View {
        VStack(alignment: .trailing, spacing: 10) {
            if model.isDownloadManagerExpanded {
                panel
            }

            Button {
                model.isDownloadManagerExpanded.toggle()
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: model.isInstallPaused ? "pause.circle" : "arrow.down.circle")
                    Text("Downloads", bundle: .module)
                    if model.hasActiveModelDownload {
                        if model.isCancellingModelInstall {
                            Text("Cancelling", bundle: .module)
                                .foregroundStyle(.appSecondary)
                        } else if model.isInstallPaused {
                            Text("Paused", bundle: .module)
                                .foregroundStyle(.appSecondary)
                        } else if let fraction = model.installProgressFraction {
                            Text(verbatim: MetricFormat.percent(fraction * 100))
                                .monospacedDigit()
                        } else {
                            AppProgressBar(value: nil, label: Text("Downloading", bundle: .module), size: .compact)
                                .frame(width: 48)
                        }
                    }
                    Image(systemName: model.isDownloadManagerExpanded ? "chevron.down" : "chevron.up")
                        .accessibilityHidden(true)
                }
                .themedFont(.small, weight: .medium)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .background(.appPage, in: Capsule())
                .overlay { Capsule().stroke(.appBorder, lineWidth: 1) }
                .contentShape(Capsule())
            }
            .buttonStyle(.plain)
            .accessibilityLabel(Text("Downloads", bundle: .module))
            .background {
                GeometryReader { geometry in
                    Color.clear.preference(
                        key: DownloadControlHeightKey.self, value: geometry.size.height)
                }
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
        .background {
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius)
                .fill(.appPage)
                .shadow(color: .black.opacity(0.2), radius: 14, x: 0, y: 4)
                .overlay {
                    RoundedRectangle(cornerRadius: AppMetrics.cardRadius).stroke(.appBorder, lineWidth: 1)
                }
        }
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

struct DownloadControlHeightKey: PreferenceKey {
    static let defaultValue: CGFloat = 0

    static func reduce(value: inout CGFloat, nextValue: () -> CGFloat) {
        value = max(value, nextValue())
    }
}
