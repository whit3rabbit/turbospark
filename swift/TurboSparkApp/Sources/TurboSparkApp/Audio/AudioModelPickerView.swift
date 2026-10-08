import SwiftUI
import TurboSpark

@MainActor
struct AudioModelPickerView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController

    private var download: ModelDownload? {
        controller.selectedProfile.flatMap { model.audioDownload(for: $0) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label { Text("Audio model", bundle: .module) } icon: { Image(systemName: "cpu") }
                    .themedFont(.base, weight: .semibold)
                Spacer()
                Button { controller.refreshModels() } label: { Image(systemName: "arrow.clockwise") }
                    .buttonStyle(.borderless)
                    .disabled(controller.isRefreshingModels)
                    .help(Text("Refresh", bundle: .module))
                    .accessibilityLabel(Text("Refresh", bundle: .module))
            }
            Picker(selection: Binding(get: { controller.recipe.modelID }, set: controller.selectModel)) {
                Text("Choose a model", bundle: .module).tag(Optional<String>.none)
                ForEach(controller.availableProfiles) { profile in
                    Text(profile.displayName).tag(Optional(profile.identity.alias))
                }
            } label: { Text("Audio model", bundle: .module) }
            .labelsHidden().frame(maxWidth: .infinity, alignment: .leading)
            .disabled(controller.isBusy || controller.hasRecordingActivity)

            if controller.isRefreshingModels && controller.profiles.isEmpty {
                ProgressView().controlSize(.small)
            } else if let error = controller.modelCatalogError {
                Text(error).themedFont(.small).foregroundStyle(.appSecondary).textSelection(.enabled)
                Button { controller.refreshModels() } label: { Text("Retry", bundle: .module) }
            }

            if let profile = controller.selectedProfile {
                modelSummary(profile)
                Divider()
                installControls(profile)
                if !controller.selectedProfileIsRunnable {
                    VStack(alignment: .leading, spacing: 6) {
                        Label { Text("Runtime unavailable", bundle: .module) }
                            icon: { Image(systemName: "wrench.and.screwdriver") }
                            .themedFont(.small, weight: .medium)
                        if let reason = profile.capabilities.unavailableReason {
                            Text(reason).themedFont(.small).foregroundStyle(.appSecondary)
                                .fixedSize(horizontal: false, vertical: true)
                        }
                    }
                } else if profile.readiness != "qualified" {
                    Label { Text("Experimental model", bundle: .module) } icon: { Image(systemName: "flask") }
                        .themedFont(.small).foregroundStyle(.appSecondary)
                }
                modelDetails(profile)
            } else if !controller.isRefreshingModels && controller.modelCatalogError == nil {
                Text("Choose a model", bundle: .module).foregroundStyle(.appSecondary)
            }
        }
        .themedFont(.small)
    }

    private func modelSummary(_ profile: AudioProfile) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(profile.displayName).themedFont(.base, weight: .medium).fixedSize(horizontal: false, vertical: true)
            if !profile.identity.repository.isEmpty {
                Text(profile.identity.repository).foregroundStyle(.appSecondary)
                    .textSelection(.enabled).lineLimit(2).truncationMode(.middle)
            }
            HStack(spacing: 8) {
                if profile.downloadBytes > 0 {
                    Label { Text(MetricFormat.storage(profile.downloadBytes)) }
                        icon: { Image(systemName: "internaldrive") }
                }
                Spacer(minLength: 0)
                if controller.modelInstalled {
                    Label { Text("Installed", bundle: .module) } icon: { Image(systemName: "checkmark.circle") }
                        .foregroundStyle(.appAccent)
                }
            }.themedFont(.tiny).foregroundStyle(.appSecondary)
        }
    }

    @ViewBuilder
    private func installControls(_ profile: AudioProfile) -> some View {
        if let download, !download.status.isTerminal {
            downloadProgress(download)
        } else if !controller.modelInstalled && !profile.identity.repository.isEmpty {
            if let download, download.status.canRetry {
                if let failure = download.failure {
                    Text(failure).foregroundStyle(.appSecondary).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Button { model.retryModelDownload(download) } label: {
                    Label { Text("Retry", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(!model.canRetryModelDownload(download))
            } else {
                Button { controller.installSelectedModel() } label: {
                    Label { Text("Download model", bundle: .module) } icon: { Image(systemName: "arrow.down.circle") }
                        .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent).controlSize(.large)
                .disabled(!model.canQueueModelDownload(.audio(identity: profile.identity)) || controller.isBusy)
            }
        }
        if let folder = controller.selectedLocalModelURL {
            Label { Text(folder.lastPathComponent).lineLimit(2).truncationMode(.middle) }
                icon: { Image(systemName: "folder") }
                .help(folder.path)
            if controller.installedPaths[profile.identity.alias] != nil {
                Button { controller.useDownloadedModel() } label: { Text("Use downloaded model", bundle: .module) }
                    .disabled(controller.isBusy || controller.hasRecordingActivity)
            }
        }
        Button { controller.chooseModelFolder() } label: {
            Label { Text("Choose local model folder", bundle: .module) } icon: { Image(systemName: "folder") }
        }
        .buttonStyle(.borderless)
        .disabled(controller.isBusy || controller.hasRecordingActivity)
    }

    private func downloadProgress(_ download: ModelDownload) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(LocalizedStringKey(download.audioStatusTitle), bundle: .module)
                    .themedFont(.small, weight: .medium)
                Spacer()
                if let total = download.totalBytes, total > 0 {
                    Text(min(1, Double(download.downloadedBytes) / Double(total)), format: .percent.precision(.fractionLength(0)))
                        .monospacedDigit()
                }
            }
            if let total = download.totalBytes, total > 0 {
                ProgressView(value: min(1, Double(download.downloadedBytes) / Double(total)))
                Text("\(MetricFormat.storage(download.downloadedBytes)) of \(MetricFormat.storage(total))", bundle: .module)
                    .themedFont(.tiny).foregroundStyle(.appSecondary)
            } else {
                ProgressView().progressViewStyle(.linear)
            }
            HStack {
                if download.status == .queued {
                    Button { model.cancelQueuedAudioDownload(download) } label: { Text("Cancel", bundle: .module) }
                } else if download.id == model.activeModelDownloadID && !model.isCancellingModelInstall {
                    Button {
                        if model.isInstallPaused { model.resumeModelDownload() }
                        else { model.pauseModelDownload() }
                    } label: { Text(model.isInstallPaused ? "Resume" : "Pause", bundle: .module) }
                    .disabled(!model.isInstallPaused && !model.canPauseModelDownload)
                    Button { model.cancelActiveModelDownload() } label: { Text("Cancel", bundle: .module) }
                }
                Spacer(minLength: 0)
                Button { model.isDownloadManagerExpanded = true } label: { Text("Downloads", bundle: .module) }
            }.controlSize(.small)
        }
    }

    private func modelDetails(_ profile: AudioProfile) -> some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 10) {
                if !profile.identity.revision.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Pinned revision", bundle: .module).themedFont(.small, weight: .medium)
                        Text(profile.identity.revision).themedCode(.tiny).textSelection(.enabled)
                    }
                }
                Text(profile.family).foregroundStyle(.appSecondary)
                Text(profile.capabilities.backend).foregroundStyle(.appSecondary)
                if let limit = profile.capabilities.maxInputSeconds {
                    HStack {
                        Text("Input limit (seconds)", bundle: .module)
                        Text(verbatim: String(limit))
                    }
                }
                ForEach(profile.evidence, id: \.self) { evidence in
                    Text(evidence).foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
                }
            }.padding(.top, 10)
        } label: { Text("Model details", bundle: .module) }
    }
}

private extension ModelDownload {
    var audioStatusTitle: String {
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
