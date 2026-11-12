import SwiftUI
import TurboSpark

/// Download, retry, or the running download's progress and Cancel for one
/// pinned audio model, with the profile's terms first when it has any.
/// Shared by the Speech-to-Text models popover and Installed > Audio.
@MainActor
struct AudioModelDownloadControls: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let profile: AudioProfile
    var size: ControlSize = .small
    /// Count the destination's free space before the download starts, as the
    /// model card does: a blocked install is disabled with the reason, a tight
    /// one asks first. Off for the surfaces that never did.
    var checksSpace = false
    /// Called when the user presses Download, before the terms or the install.
    var onStart: (() -> Void)?

    /// A profile whose install terms are on screen awaiting acceptance.
    @State private var acknowledging: AudioProfile?
    @State private var confirmingLowSpace = false

    private var download: ModelDownload? { model.audioDownload(for: profile) }
    /// The default library for downloads, nil for This Mac (requirement 2.2).
    private var library: String? { model.installDestinations.libraryForInstall(selection: nil) }

    var body: some View {
        if let download, !download.status.isTerminal {
            progress(download)
        } else if !AudioRuntimeProvider(profile).canRun(profile) {
            Label {
                Text(verbatim: profile.capabilities.unavailableReason ?? AppStrings.localized("Runtime unavailable"))
                    .fixedSize(horizontal: false, vertical: true)
            } icon: { Image(systemName: "wrench.and.screwdriver").accessibilityHidden(true) }
            .themedFont(.small).foregroundStyle(.appSecondary)
        } else if !profile.identity.repository.isEmpty {
            if let download, download.status.canRetry {
                VStack(alignment: .leading, spacing: 6) {
                    if let failure = download.failure {
                        Text(failure).themedFont(.small).foregroundStyle(.appSecondary)
                            .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    }
                    Button { model.retryModelDownload(download) } label: {
                        Label { Text("Retry", bundle: .module) } icon: { Image(systemName: "arrow.clockwise") }
                    }
                    .appAction(.secondary, size: size)
                    .disabled(!model.canRetryModelDownload(download))
                }
            } else {
                downloadButton
            }
        }
    }

    private var decision: ModelInstallDecision {
        checksSpace ? model.audioInstallDecision(for: profile, library: library) : .allowed
    }

    private func request() {
        onStart?()
        switch decision {
        case .blocked: return
        case .confirm: confirmingLowSpace = true
        case .allowed: proceed()
        }
    }

    private func proceed() {
        if profile.requiredAcknowledgments.isEmpty { controller.install(profile, library: library) }
        else { acknowledging = profile }
    }

    @ViewBuilder private var downloadButton: some View {
        VStack(alignment: .leading, spacing: 6) {
            if let reason = decision.reason {
                Label {
                    Text(verbatim: reason).fixedSize(horizontal: false, vertical: true)
                } icon: {
                    Image(systemName: decision.isBlocked ? "xmark.octagon.fill" : "exclamationmark.triangle.fill")
                        .accessibilityHidden(true)
                }
                .themedFont(.small)
                .foregroundStyle(decision.isBlocked ? AnyShapeStyle(.appDanger) : AnyShapeStyle(AppAttention.color))
            }
            downloadAction
        }
        .alert(
            Text("Install anyway?", bundle: .module), isPresented: $confirmingLowSpace
        ) {
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
            Button { proceed() } label: { Text("Install", bundle: .module) }
        } message: {
            Text(verbatim: decision.reason ?? "")
        }
    }

    private var downloadAction: some View {
        Button(action: request) {
            Label {
                if profile.downloadBytes > 0 {
                    Text("Download model", bundle: .module) + Text(verbatim: " (\(MetricFormat.storage(profile.downloadBytes)))")
                } else {
                    Text("Download model", bundle: .module)
                }
            } icon: { Image(systemName: "arrow.down.circle") }
        }
        .appAction(.secondary, size: size)
        .disabled(!model.canQueueModelDownload(.audio(identity: profile.identity)) || controller.isBusy || decision.isBlocked)
        .help(
            controller.isBusy
                ? Text("Cannot change model while audio is active or recording.", bundle: .module)
                : Text("Download model", bundle: .module)
        )
        .alert(Text("Before downloading", bundle: .module),
               isPresented: Binding(get: { acknowledging != nil }, set: { if !$0 { acknowledging = nil } }),
               presenting: acknowledging) { terms in
            Button { controller.install(terms, acknowledged: terms.requiredAcknowledgments.map(\.id), library: library) } label: {
                Text("Accept and download", bundle: .module)
            }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: { terms in
            Text(AudioModelPickerView.acknowledgmentText(terms))
        }
    }

    private func progress(_ download: ModelDownload) -> some View {
        HStack(spacing: 8) {
            AppProgressBar(value: download.fractionDone, label: Text("Download progress", bundle: .module),
                           animated: download.status != .paused)
                .frame(minWidth: 80, maxWidth: .infinity)
            if download.status == .queued {
                cancelButton { model.cancelQueuedAudioDownload(download) }
            } else if download.id == model.activeModelDownloadID && !model.isCancellingModelInstall {
                cancelButton { model.cancelActiveModelDownload() }
            }
        }
    }

    private func cancelButton(_ action: @escaping () -> Void) -> some View {
        // The install toast's treatment for the same control.
        Button(role: .destructive, action: action) { Text("Cancel Download", bundle: .module) }
            .appAction(.destructive, size: size)
    }
}
