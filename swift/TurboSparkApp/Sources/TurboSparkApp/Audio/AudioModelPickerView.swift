import SwiftUI
import TurboSpark

@MainActor
struct AudioModelPickerView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController

    @State private var confirmingDelete = false
    /// A profile whose install terms are on screen awaiting acceptance.
    @State private var acknowledging: AudioProfile?
    /// A profile whose low-space confirmation is on screen (requirement 2.3).
    @State private var confirmingLowSpace: AudioProfile?
    /// The library the user picked for the download. Nil follows the default
    /// (requirement 2.2). Per install: not remembered across launches.
    @State private var destination: String?

    /// The `library:` the download takes for the menu's choice: nil for This Mac.
    private var chosenLibrary: String? {
        model.installDestinations.libraryForInstall(selection: destination)
    }

    /// What the gate says about downloading `profile` into the chosen library:
    /// that library's room against the profile's size, and the connect statement
    /// when the library is not there. Audio had no free-space check before.
    private func downloadDecision(_ profile: AudioProfile) -> ModelInstallDecision {
        model.audioInstallDecision(for: profile, library: chosenLibrary)
    }

    /// The download button: the gate first, then the terms the profile asks for,
    /// then the install. The acknowledgement flow is unchanged.
    private func requestDownload(_ profile: AudioProfile) {
        switch downloadDecision(profile) {
        case .blocked: return
        case .confirm: confirmingLowSpace = profile
        case .allowed: continueDownload(profile)
        }
    }

    private func continueDownload(_ profile: AudioProfile) {
        if profile.requiredAcknowledgments.isEmpty {
            controller.installSelectedModel(library: chosenLibrary)
        } else {
            acknowledging = profile
        }
    }
    @Environment(\.locale) private var locale

    private var download: ModelDownload? {
        controller.selectedProfile.flatMap { model.audioDownload(for: $0) }
    }

    /// A row names the model and its engine, and a single-language model
    /// also its language in the UI's own language, so a Japanese or
    /// Mandarin recognizer is recognizable before it is chosen.
    private func rowLabel(_ profile: AudioProfile) -> Text {
        var label = Text(profile.displayName) + Text(verbatim: " - ")
            + Text(LocalizedStringKey(AudioEngineLabel(profile).title), bundle: .module)
        if let code = AudioModelLanguage.singleCode(for: profile) {
            label = label + Text(verbatim: " - " + AudioModelLanguage.name(for: code, locale: locale))
        }
        return label
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            AppAdaptiveHeader(
                leading: {
                    Label { Text("Audio model", bundle: .module) } icon: { Image(systemName: "cpu") }
                        .themedFont(.base, weight: .semibold)
                },
                actions: {
                    AppIconButton("arrow.clockwise", label: Text("Refresh", bundle: .module)) { controller.refreshModels() }
                        .appAction(.quiet, size: .small)
                        .disabled(controller.isRefreshingModels)
                }
            )
            // The menu lists what is on this Mac. A selection that is not listed
            // reads as no choice, so the picker never shows a blank for an unknown tag.
            let listed = controller.pickerProfiles
            Picker(selection: Binding(
                get: { controller.recipe.modelID.flatMap { id in listed.contains { $0.identity.alias == id } ? id : nil } },
                set: { controller.selectModel($0) })
            ) {
                Text("Choose a model", bundle: .module).tag(Optional<String>.none)
                ForEach(listed) { profile in
                    pickerRow(profile).tag(Optional(profile.identity.alias))
                }
            } label: { Text("Audio model", bundle: .module) }
            .labelsHidden().frame(maxWidth: .infinity, alignment: .leading)
            .help(
                (controller.isBusy || controller.hasRecordingActivity)
                    ? Text("Cannot change model while audio is active or recording.", bundle: .module)
                    : Text("Audio model", bundle: .module)
            )
            .disabled(controller.isBusy || controller.hasRecordingActivity)

            if listed.isEmpty && !controller.isRefreshingModels && controller.modelCatalogError == nil {
                noModelInstalled
            }

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
            } else if !listed.isEmpty && !controller.isRefreshingModels && controller.modelCatalogError == nil {
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
        .modelMoveSheet(model: model, kinds: [.audio])
        .alert(
            Text("Install anyway?", bundle: .module),
            isPresented: Binding(
                get: { confirmingLowSpace != nil }, set: { if !$0 { confirmingLowSpace = nil } }),
            presenting: confirmingLowSpace
        ) { profile in
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
            Button { continueDownload(profile) } label: { Text("Install", bundle: .module) }
        } message: { profile in
            Text(verbatim: downloadDecision(profile).reason ?? "")
        }
        .task {
            if controller.installedPaths.isEmpty || controller.profiles.isEmpty {
                controller.refreshModels()
            }
        }
    }

    /// Nothing is installed for this task. Models are downloaded in Discover, so the
    /// pane says so and opens it.
    private var noModelInstalled: some View {
        HStack(spacing: 8) {
            Text("No model installed", bundle: .module).foregroundStyle(.appSecondary)
            Spacer()
            Button { model.openModelHub() } label: { Text("Discover Models", bundle: .module) }
                .appAction(.secondary, size: .small)
        }
    }

    /// A row of the model menu. An install on a drive that is away reads "Not
    /// connected" with a glyph; it is neither incompatible nor missing.
    @ViewBuilder
    private func pickerRow(_ profile: AudioProfile) -> some View {
        let label = AudioOfflineInstalls.pickerLabel(
            displayName: profile.displayName, status: controller.installStatus(for: profile))
        if let glyph = label.glyph {
            Label {
                rowLabel(profile) + Text(verbatim: ", ") + Text("Not connected", bundle: .module)
            } icon: {
                Image(systemName: glyph)
            }
        } else {
            rowLabel(profile)
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
        // An installed model with a receipt can move to another library (task 6.11).
        // The sheet says why it cannot while the model is resident or served.
        if let subject = controller.moveSubject(for: profile) {
            Button { model.presentMoveSheet(for: subject) } label: {
                Label { Text("Move to...", bundle: .module) } icon: { Image(systemName: "arrow.left.arrow.right.circle") }
            }
            .appAction(.secondary, size: .small)
            .disabled(controller.isManagingInstall || model.modelsBeingChanged.contains(subject.installId))
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
            AudioEngineSummary(profile: profile)
            if profile.family == "fish_drama3" {
                // The Fish Audio Community License asks a product that uses
                // the model to show this attribution, in these words.
                Text(verbatim: "Built with Fish Audio").foregroundStyle(.appSecondary)
            }
            HStack(spacing: 8) {
                if controller.selectedLocalModelURL == nil, profile.downloadBytes > 0 {
                    Label { Text(MetricFormat.storage(profile.downloadBytes)) }
                        icon: { Image(systemName: "internaldrive") }
                }
                Spacer(minLength: 0)
                if controller.isProfileLoaded(profile) {
                    Label { Text("Loaded", bundle: .module) } icon: { Image(systemName: "bolt.circle.fill") }
                        .foregroundStyle(.appAccent)
                } else if controller.modelInstalled {
                    Label { Text("Installed", bundle: .module) } icon: { Image(systemName: "checkmark.circle") }
                        .foregroundStyle(.appAccent)
                } else if case .notConnected = controller.installStatus(for: profile) {
                    // A shape and a word: the state survives color blindness.
                    Label { Text("Not connected", bundle: .module) } icon: { Image(systemName: LibraryRowStatus.notConnected.glyph) }
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
            } else if controller.isProfileLoaded(profile) {
                Button { controller.unloadModel() } label: {
                    Label { Text("Unload", bundle: .module) } icon: { Image(systemName: "eject.fill") }
                        .frame(maxWidth: .infinity)
                }
                .appAction(.secondary, size: .large)
                .disabled(!controller.canUnloadModel)
                .help(Text("Eject model and release unified memory", bundle: .module))
                .accessibilityLabel(Text("Unload Model", bundle: .module))
            } else {
                Button { controller.loadModel() } label: {
                    Label { Text("Load Model", bundle: .module) } icon: { Image(systemName: "bolt.fill") }
                        .frame(maxWidth: .infinity)
                }
                .appAction(.secondary, size: .large)
                .disabled(!controller.canLoadModel)
                .help(Text("Load model into memory slot cache for inference", bundle: .module))
                .accessibilityLabel(Text("Load Model", bundle: .module))
            }
        } else if controller.isRefreshingModels && controller.installedPaths[profile.identity.alias] == nil {
            HStack(spacing: 8) {
                ProgressView().controlSize(.small)
                Text("Checking model status...", bundle: .module)
                    .themedFont(.small, weight: .medium)
                    .foregroundStyle(.appSecondary)
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        } else if let download, !download.status.isTerminal {
            downloadProgress(download)
        } else if let statement = AudioOfflineInstalls.statement(for: controller.installStatus(for: profile)),
            !controller.modelInstalled
        {
            // Installed on a drive that is away: the download button is replaced
            // by this statement, because a download would be refused as
            // `alias_elsewhere` (task 6.10).
            Label {
                Text(verbatim: statement).fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: LibraryRowStatus.notConnected.glyph)
            }
            .foregroundStyle(.appSecondary)
            .accessibilityElement(children: .combine)
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
                // Where the model goes: only when there is a choice (requirement
                // 2.1), and the gate below counts that library's room.
                InstallDestinationPicker(model: model, selection: $destination)
                let decision = downloadDecision(profile)
                if let reason = decision.reason {
                    Label(
                        reason,
                        systemImage: decision.isBlocked ? "xmark.octagon.fill" : "exclamationmark.triangle.fill"
                    )
                    .themedFont(.small)
                    .foregroundStyle(
                        decision.isBlocked
                            ? AnyShapeStyle(.appDanger)
                            : AnyShapeStyle(AppAttention.color)
                    )
                    .fixedSize(horizontal: false, vertical: true)
                }
                Button {
                    requestDownload(profile)
                } label: {
                    Label { Text("Download model", bundle: .module) } icon: { Image(systemName: "arrow.down.circle") }
                        .frame(maxWidth: .infinity)
                }
                .appAction(.primary, size: .large)
                .disabled(
                    !model.canQueueModelDownload(.audio(identity: profile.identity)) || controller.isBusy
                        || decision.isBlocked)
                .alert(Text("Before downloading", bundle: .module),
                       isPresented: Binding(get: { acknowledging != nil }, set: { if !$0 { acknowledging = nil } }),
                       presenting: acknowledging) { terms in
                    Button {
                        controller.installSelectedModel(
                            acknowledged: terms.requiredAcknowledgments.map(\.id), library: chosenLibrary)
                    } label: {
                        Text("Accept and download", bundle: .module)
                    }
                    Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
                } message: { terms in
                    Text(Self.acknowledgmentText(terms))
                }
            }
        }
        if let folder = controller.selectedLocalModelURL {
            HStack(spacing: 6) {
                Image(systemName: "folder").foregroundStyle(.appSecondary)
                SettingsPathText(folder.path)
            }
            if controller.installedPaths[profile.identity.alias] != nil {
                Button { controller.useDownloadedModel() } label: { Text("Use downloaded model", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .help(
                        (controller.isBusy || controller.hasRecordingActivity)
                            ? Text("Cannot change model while audio is active or recording.", bundle: .module)
                            : Text("Use downloaded model", bundle: .module)
                    )
                    .disabled(controller.isBusy || controller.hasRecordingActivity)
            }
        }
        // FluidAudio models load only from a verified download.
        if AudioRuntimeProvider(profile).acceptsLocalFolder {
            Button { controller.chooseModelFolder() } label: {
                Label { Text("Choose local model folder", bundle: .module) } icon: { Image(systemName: "folder") }
            }
            .appAction(.secondary, size: .small)
            .help(
                (controller.isBusy || controller.hasRecordingActivity)
                    ? Text("Cannot change model while audio is active or recording.", bundle: .module)
                    : Text("Choose local model folder", bundle: .module)
            )
            .disabled(controller.isBusy || controller.hasRecordingActivity)
        }
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
                VStack(alignment: .leading, spacing: 4) {
                    Text("Engine", bundle: .module).themedFont(.small, weight: .medium)
                    Text(LocalizedStringKey(AudioEngineLabel(profile).title), bundle: .module)
                        .foregroundStyle(.appSecondary)
                }
                VStack(alignment: .leading, spacing: 4) {
                    Text("Provider", bundle: .module).themedFont(.small, weight: .medium)
                    Text(verbatim: AudioEngineLabel.provider(for: profile)).foregroundStyle(.appSecondary)
                }
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
            case "funasr_model_license":
                return AppStrings.localized("Paraformer is released under Alibaba's FunASR Model Open Source License. You may use, change, and share it if you credit the source and keep the model name. Alibaba can revise the license, and it ends for anyone who disparages the model.")
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
