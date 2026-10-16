import SwiftUI
import TurboSpark

@MainActor
struct AudioModelPickerView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController

    @State private var confirmingDelete = false
    /// A profile whose install terms are on screen awaiting acceptance.
    @State private var acknowledging: AudioProfile?

    private var download: ModelDownload? {
        controller.selectedProfile.flatMap { model.audioDownload(for: $0) }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack {
                Label { Text("Audio model", bundle: .module) } icon: { Image(systemName: "cpu") }
                    .themedFont(.base, weight: .semibold)
                Spacer()
                AppIconButton("arrow.clockwise", label: Text("Refresh", bundle: .module)) { controller.refreshModels() }
                    .appAction(.quiet, size: .small)
                    .disabled(controller.isRefreshingModels)
            }
            Picker(selection: Binding(get: { controller.recipe.modelID }, set: { controller.selectModel($0) })) {
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
                    .appAction(.secondary, size: .small)
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
                installManagement(profile)
                modelDetails(profile)
            } else if !controller.isRefreshingModels && controller.modelCatalogError == nil {
                Text("Choose a model", bundle: .module).foregroundStyle(.appSecondary)
            }
        }
        .themedFont(.small)
        .confirmationDialog(
            Text("Delete Model", bundle: .module), isPresented: $confirmingDelete
        ) {
            Button(role: .destructive) { controller.deleteSelectedModel() } label: {
                Text("Delete Model", bundle: .module)
            }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            if let name = controller.selectedProfile?.displayName { Text(name) }
        }
    }

    /// Adopt an install the receipt store does not know yet, delete a managed
    /// one, and list installs that exist but cannot be used.
    @ViewBuilder
    private func installManagement(_ profile: AudioProfile) -> some View {
        if controller.selectedAdoptable != nil {
            Button { controller.adoptSelectedModel() } label: {
                Label { Text("Adopt existing install", bundle: .module) }
                    icon: { Image(systemName: "checkmark.seal") }
                    .frame(maxWidth: .infinity)
            }
            .appAction(.primary, size: .large)
            .disabled(controller.isManagingInstall || controller.isBusy)
        }
        if controller.canDeleteSelectedModel {
            Button(role: .destructive) { confirmingDelete = true } label: {
                Label { Text("Delete Model", bundle: .module) } icon: { Image(systemName: "trash") }
            }
            .appAction(.destructive, size: .small)
        }
        if !controller.incompatibleInstalls.isEmpty {
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(controller.incompatibleInstalls, id: \.record.alias) { row in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(row.record.alias).themedFont(.small, weight: .medium)
                            Text(row.error).themedFont(.tiny).foregroundStyle(.appSecondary)
                                .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                        }
                    }
                }.padding(.top, 8)
            } label: { Text("Installs that need attention", bundle: .module) }
        }
    }

    private func modelSummary(_ profile: AudioProfile) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if controller.selectedLocalModelURL == nil {
                Text(profile.displayName).themedFont(.base, weight: .medium).fixedSize(horizontal: false, vertical: true)
            }
            if controller.selectedLocalModelURL == nil, !profile.identity.repository.isEmpty {
                Text(profile.identity.repository).foregroundStyle(.appSecondary)
                    .textSelection(.enabled).lineLimit(2).truncationMode(.middle)
            }
            HStack(spacing: 8) {
                if controller.selectedLocalModelURL == nil, profile.downloadBytes > 0 {
                    Label { Text(MetricFormat.storage(profile.downloadBytes)) }
                        icon: { Image(systemName: "internaldrive") }
                }
                Spacer(minLength: 0)
                if controller.isModelLoaded {
                    Label { Text("Loaded", bundle: .module) } icon: { Image(systemName: "bolt.circle.fill") }
                        .foregroundStyle(.appAccent)
                } else if controller.modelInstalled {
                    Label { Text("Installed", bundle: .module) } icon: { Image(systemName: "checkmark.circle") }
                        .foregroundStyle(.appAccent)
                }
            }.themedFont(.tiny).foregroundStyle(.appSecondary)
        }
    }

    @ViewBuilder
    private func installControls(_ profile: AudioProfile) -> some View {
        if controller.modelInstalled {
            if controller.isModelLoading {
                HStack(spacing: 8) {
                    ProgressView().controlSize(.small)
                    Text("Loading model", bundle: .module)
                        .themedFont(.small, weight: .medium)
                    Spacer()
                    Button { controller.cancel() } label: {
                        Text("Cancel", bundle: .module)
                    }
                    .appAction(.quiet, size: .small)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
            } else if controller.canUnloadModel {
                Button { controller.unloadModel() } label: {
                    Label { Text("Unload", bundle: .module) } icon: { Image(systemName: "eject.fill") }
                        .frame(maxWidth: .infinity)
                }
                .appAction(.secondary, size: .large)
                .help(Text("Eject model and release unified memory", bundle: .module))
                .accessibilityLabel(Text("Unload Model", bundle: .module))
            } else if controller.canLoadModel {
                Button { controller.loadModel() } label: {
                    Label { Text("Load Model", bundle: .module) } icon: { Image(systemName: "bolt.fill") }
                        .frame(maxWidth: .infinity)
                }
                .appAction(.secondary, size: .large)
                .help(Text("Load model into memory slot cache for inference", bundle: .module))
                .accessibilityLabel(Text("Load Model", bundle: .module))
            }
        } else if let download, !download.status.isTerminal {
            downloadProgress(download)
        } else if !profile.identity.repository.isEmpty {
            if let download, download.status.canRetry {
                if let failure = download.failure {
                    Text(failure).foregroundStyle(.appSecondary).textSelection(.enabled)
                        .fixedSize(horizontal: false, vertical: true)
                }
                Button { model.retryModelDownload(download) } label: {
                    Label { Text("Retry", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                        .frame(maxWidth: .infinity)
                }
                .appAction(.primary, size: .large)
                .disabled(!model.canRetryModelDownload(download))
            } else {
                Button {
                    if profile.requiredAcknowledgments.isEmpty { controller.installSelectedModel() }
                    else { acknowledging = profile }
                } label: {
                    Label { Text("Download model", bundle: .module) } icon: { Image(systemName: "arrow.down.circle") }
                        .frame(maxWidth: .infinity)
                }
                .appAction(.primary, size: .large)
                .disabled(!model.canQueueModelDownload(.audio(identity: profile.identity)) || controller.isBusy)
                .alert(Text("Before downloading", bundle: .module),
                       isPresented: Binding(get: { acknowledging != nil }, set: { if !$0 { acknowledging = nil } }),
                       presenting: acknowledging) { terms in
                    Button { controller.installSelectedModel(acknowledged: terms.requiredAcknowledgments.map(\.id)) } label: {
                        Text("Accept and download", bundle: .module)
                    }
                    Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
                } message: { terms in
                    Text(Self.acknowledgmentText(terms))
                }
            }
        }
        if let folder = controller.selectedLocalModelURL {
            Label { Text(folder.lastPathComponent).lineLimit(2).truncationMode(.middle) }
                icon: { Image(systemName: "folder") }
                .help(folder.path)
            if controller.installedPaths[profile.identity.alias] != nil {
                Button { controller.useDownloadedModel() } label: { Text("Use downloaded model", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(controller.isBusy || controller.hasRecordingActivity)
            }
        }
        Button { controller.chooseModelFolder() } label: {
            Label { Text("Choose local model folder", bundle: .module) } icon: { Image(systemName: "folder") }
        }
        .appAction(.secondary, size: .small)
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
                AppProgressBar(value: min(1, Double(download.downloadedBytes) / Double(total)),
                               label: Text("Download progress", bundle: .module),
                               animated: download.status != .paused)
                Text("\(MetricFormat.storage(download.downloadedBytes)) of \(MetricFormat.storage(total))", bundle: .module)
                    .themedFont(.tiny).foregroundStyle(.appSecondary)
            } else {
                AppProgressBar(value: nil, label: Text("Download progress", bundle: .module),
                               animated: download.status != .paused)
            }
            HStack {
                // The same treatment as the install toast's controls:
                // Pause/Resume secondary, Cancel destructive.
                if download.status == .queued {
                    Button(role: .destructive) { model.cancelQueuedAudioDownload(download) } label: { Text("Cancel", bundle: .module) }
                        .appAction(.destructive, size: .small)
                } else if download.id == model.activeModelDownloadID && !model.isCancellingModelInstall {
                    Button {
                        if model.isInstallPaused { model.resumeModelDownload() }
                        else { model.pauseModelDownload() }
                    } label: {
                        model.isInstallPaused ? Text("Resume", bundle: .module) : Text("Pause", bundle: .module)
                    }
                    .appAction(.secondary, size: .small)
                    .disabled(!model.isInstallPaused && !model.canPauseModelDownload)
                    Button(role: .destructive) { model.cancelActiveModelDownload() } label: { Text("Cancel", bundle: .module) }
                        .appAction(.destructive, size: .small)
                }
                Spacer(minLength: 0)
                Button { model.isDownloadManagerExpanded = true } label: { Text("Downloads", bundle: .module) }
                    .appAction(.quiet, size: .small)
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

extension AudioModelPickerView {
    /// The terms a profile asks the user to accept, localized for ids this
    /// app knows and the catalog's own summary otherwise.
    static func acknowledgmentText(_ profile: AudioProfile) -> String {
        profile.requiredAcknowledgments.map { term in
            switch term.id {
            case "megatts3_community_encoder":
                return AppStrings.localized("The voice-cloning encoder in this model is a community re-upload. Its origin and license have not been verified.")
            case "omnivoice_higgs_tokenizer":
                return AppStrings.localized("The bundled audio tokenizer is community-licensed, not Apache-licensed: it requires attribution and caps commercial use above 100,000 annual active users.")
            default:
                return term.summary
            }
        }.joined(separator: "\n\n")
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
