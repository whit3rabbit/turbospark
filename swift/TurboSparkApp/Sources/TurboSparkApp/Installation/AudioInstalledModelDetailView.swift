import SwiftUI
import TurboSpark

/// The selected audio model in Installed: what it is, its state and the
/// action that state allows, the Speech-to-Text modes that use it, and its
/// catalog facts.
@MainActor
struct AudioInstalledModelDetailView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let entry: AudioInstalledEntry
    @State private var confirmingDelete = false

    private var profile: AudioProfile? { entry.profile }
    private var state: AudioModelState {
        controller.entryState(entry, download: profile.flatMap { model.audioDownload(for: $0) })
    }
    private var localURL: URL? { entry.group == .local ? controller.localModelURL(entry.alias) : nil }
    private var location: String? { localURL?.path ?? entry.path ?? profile.flatMap { controller.installedPaths[$0.identity.alias] } }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 20) {
                header
                actions
                notice
                if let profile, !SpeechToTextJob.allCases.filter({ $0.accepts(profile) }).isEmpty {
                    AudioInstalledUsedByCard(controller: controller, profile: profile)
                }
                facts
            }
            .padding(24)
            .frame(maxWidth: 820, alignment: .leading)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .confirmationDialog(deleteTitle, isPresented: $confirmingDelete) {
            Button(role: .destructive) { delete() } label: {
                entry.group == .local ? Text("Remove", bundle: .module) : Text("Delete Model", bundle: .module)
            }
            Button(role: .cancel) {} label: { Text("Cancel", bundle: .module) }
        } message: {
            entry.group == .local
                ? Text("Its folder stays where it is.", bundle: .module)
                : Text("This removes the model from this Mac. A mode that uses it moves to another installed model. Transcripts you made stay as they are.", bundle: .module)
        }
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 6) {
            Text(verbatim: entry.name).themedFont(.title2, weight: .semibold).accessibilityAddTraits(.isHeader)
            if let repository = profile?.identity.repository, !repository.isEmpty {
                Text(verbatim: repository).themedCode(.tiny).foregroundStyle(.appSecondary)
                    .textSelection(.enabled).lineLimit(1).truncationMode(.middle).help(Text(verbatim: repository))
            }
            FlowLayout(spacing: 8, lineSpacing: 4) {
                if let profile { AudioPinnedBadge(profile: profile, isLocalFolder: entry.group == .local) }
                AudioModelStateLabel(state: state)
                if let profile {
                    Text(LocalizedStringKey(AudioEngineLabel(profile).title), bundle: .module)
                        .themedFont(.tiny).foregroundStyle(.appSecondary)
                }
            }
        }
    }

    private var actions: some View {
        HStack(alignment: .top, spacing: 10) {
            FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: 6) {
                lifecycleAction
                if let profile, state.isOnDisk, SpeechToTextJob.allCases.contains(where: { $0.accepts(profile) }) {
                    Button { model.openAudio(.transcribe) } label: {
                        Label { Text("Open Speech-to-Text", bundle: .module) } icon: { Image(systemName: "waveform") }
                    }
                    .appAction(.quiet)
                }
                if let location {
                    Button { ModelStorageManager.revealInFinder(path: location) } label: {
                        Label { Text("Reveal", bundle: .module) } icon: { Image(systemName: "folder") }
                    }
                    .appAction(.quiet)
                    .help(Text("Reveal in Finder", bundle: .module) + Text(verbatim: ": \(location)"))
                    .accessibilityLabel(Text("Reveal in Finder", bundle: .module))
                }
            }
            Spacer(minLength: 0)
            if canDelete {
                Button(role: .destructive) { confirmingDelete = true } label: {
                    Label {
                        entry.group == .local ? Text("Remove", bundle: .module) : Text("Delete", bundle: .module)
                    } icon: { Image(systemName: "trash") }
                }
                .appAction(.destructive)
                .help(entry.group == .local ? Text("Its folder stays where it is.", bundle: .module) : Text("Delete model from disk", bundle: .module))
            }
        }
    }

    @ViewBuilder private var lifecycleAction: some View {
        if let profile {
            switch state {
            case .loaded:
                if controller.isProfileLoaded(profile) {
                    Button { controller.unloadModel() } label: {
                        Label { Text("Unload", bundle: .module) } icon: { Image(systemName: "eject.fill") }
                    }
                    .appAction(.secondary)
                    .disabled(!controller.canUnloadModel)
                    .help(Text("Eject model and release unified memory", bundle: .module))
                }
            case .installed:
                if SpeechToTextJob.live.accepts(profile) {
                    Text("Opens when a take starts.", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                } else {
                    Button { controller.load(profile) } label: {
                        Label { Text("Load Model", bundle: .module) } icon: { Image(systemName: "bolt.fill") }
                    }
                    .appAction(.secondary)
                    .disabled(!controller.canLoad(profile))
                    .help(entry.runtimeUnsupported
                        ? Text("Not usable in this app version", bundle: .module)
                        : Text("Load model into memory slot cache for inference", bundle: .module))
                }
            case .needsAdoption:
                if let legacy = entry.legacy {
                    Button { controller.adopt(legacy) } label: {
                        Label { Text("Adopt existing install", bundle: .module) } icon: { Image(systemName: "checkmark.seal") }
                    }
                    .appAction(.primary)
                    .disabled(controller.isManagingInstall || controller.isBusy)
                }
            case .notInstalled, .downloading:
                AudioModelDownloadControls(model: model, controller: controller, profile: profile, size: .regular)
            case .needsAttention:
                EmptyView()
            }
        }
    }

    @ViewBuilder private var notice: some View {
        switch entry.group {
        case .needsAdoption:
            noticeCard(title: Text("Found on disk without a receipt", bundle: .module),
                       body: Text("Adopting checks every file against the pinned revision before any mode can use it.", bundle: .module),
                       symbol: "exclamationmark.triangle.fill")
        case .needsAttention:
            noticeCard(title: Text("Needs attention", bundle: .module), body: Text(verbatim: entry.error ?? ""),
                       symbol: "xmark.octagon.fill")
        case .otherAudio:
            noticeCard(title: Text("Not in the Catalog", bundle: .module),
                       body: Text("The catalog no longer lists this install, so no mode can use it. Remove its folder in Finder if you no longer need it.", bundle: .module),
                       symbol: "questionmark.folder")
        case .installed where entry.runtimeUnsupported:
            noticeCard(title: Text("Not usable in this app version", bundle: .module),
                       body: Text("This app version has no runtime for this model, so no mode can use it. You can delete it to free disk space.", bundle: .module),
                       symbol: "exclamationmark.triangle.fill")
        case .installed, .local, .available:
            EmptyView()
        }
    }

    private func noticeCard(title: Text, body: Text, symbol: String) -> some View {
        Label {
            VStack(alignment: .leading, spacing: 4) {
                title.themedFont(.small, weight: .semibold)
                body.themedFont(.small).foregroundStyle(.appSecondary)
                    .textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            }
        } icon: { Image(systemName: symbol).accessibilityHidden(true) }
        .frame(maxWidth: .infinity, alignment: .leading)
        .appCardSurface(padding: 12)
        .accessibilityElement(children: .combine)
    }

    private var facts: some View {
        VStack(alignment: .leading, spacing: 10) {
            AppSectionHeader(title: Text("Model details", bundle: .module))
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 16, verticalSpacing: 8) {
                if let profile {
                    row(Text("Engine", bundle: .module), Text(LocalizedStringKey(AudioEngineLabel(profile).title), bundle: .module))
                    if !profile.capabilities.languages.isEmpty {
                        row(Text("Languages", bundle: .module), Text(verbatim: profile.capabilities.languages.joined(separator: ", ")))
                    }
                    if let limit = profile.capabilities.maxInputSeconds {
                        row(Text("Input limit (seconds)", bundle: .module), Text(verbatim: String(limit)))
                    }
                    if profile.residentMemoryBytes > 0 {
                        row(Text("Memory", bundle: .module),
                            Text("Up to \(MetricFormat.memory(profile.residentMemoryBytes)), estimate", bundle: .module))
                    }
                    if !profile.identity.revision.isEmpty, entry.group != .local {
                        row(Text("Revision", bundle: .module), Text(verbatim: profile.identity.revision), code: true)
                    }
                }
                if let size = entry.installBytes ?? profile?.downloadBytes, size > 0 {
                    row(Text("Size", bundle: .module), Text(verbatim: MetricFormat.storage(size)))
                }
                if let location {
                    row(Text("Location", bundle: .module), Text(verbatim: location), code: true)
                }
            }
            .themedFont(.small)
        }
        .appCardSurface(padding: 14)
    }

    private func row(_ title: Text, _ value: Text, code: Bool = false) -> some View {
        GridRow {
            title.foregroundStyle(.appSecondary)
            Group {
                if code { value.themedCode(.tiny) } else { value }
            }
            .textSelection(.enabled).lineLimit(code ? 1 : nil).truncationMode(.middle)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var canDelete: Bool {
        switch entry.group {
        case .local: return !controller.isBusy && !controller.hasRecordingActivity
        case .installed, .needsAdoption: return profile.map(controller.canDelete) ?? false
        case .needsAttention, .otherAudio, .available: return false
        }
    }

    private var deleteTitle: Text {
        entry.group == .local ? Text("Remove \(entry.name) from the list", bundle: .module) : Text("Delete \(entry.name)?", bundle: .module)
    }

    private func delete() {
        if entry.group == .local { controller.removeLocalModel(entry.alias) }
        else if let profile { controller.deleteModel(profile) }
    }
}

/// The Speech-to-Text modes that could use a model, which one each uses
/// now, and a Use button to switch. Only a pinned catalog model is called
/// pinned; a mode's choice is "In use".
@MainActor
struct AudioInstalledUsedByCard: View {
    @ObservedObject var controller: AudioWorkspaceController
    let profile: AudioProfile

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            AppSectionHeader(title: Text("Used By", bundle: .module),
                             subtitle: Text("Speech-to-Text modes that can use this model", bundle: .module))
            ForEach(slots, id: \.key) { slot in
                HStack(spacing: 10) {
                    Image(systemName: slot.mode.symbol).foregroundStyle(.appSecondary).frame(width: 20).accessibilityHidden(true)
                    VStack(alignment: .leading, spacing: 1) {
                        Text(LocalizedStringKey(slot.mode.titleKey), bundle: .module).themedFont(.small, weight: .semibold)
                        Text(LocalizedStringKey(slot.mode.titleKey(for: slot.job)), bundle: .module)
                            .themedFont(.tiny).foregroundStyle(.appSecondary)
                        if !slot.isOn {
                            Text("Speaker labels are off", bundle: .module).themedFont(.tiny).foregroundStyle(.appSecondary)
                        } else if !slot.inUse, let current = slot.current {
                            Text("Uses \(current)", bundle: .module).themedFont(.tiny).foregroundStyle(.appSecondary)
                        }
                    }
                    Spacer(minLength: 8)
                    if slot.inUse {
                        Text("In use", bundle: .module).themedFont(.tiny, weight: .semibold).foregroundStyle(.appAccent)
                    } else {
                        // Using a model for a mode's speakers also turns its labels on.
                        Button {
                            controller.setSpeechToTextModel(profile.identity.alias, mode: slot.mode, job: slot.job)
                            if !slot.isOn { controller.setLabelsSpeakers(true, in: slot.mode) }
                        } label: {
                            Text("Use", bundle: .module)
                        }
                        .appAction(.secondary, size: .small)
                        .disabled(!controller.canChangeSpeechToTextModels)
                        .accessibilityLabel(Text("Use", bundle: .module) + Text(verbatim: ": ")
                            + Text(LocalizedStringKey(slot.mode.titleKey), bundle: .module))
                    }
                }
                .accessibilityElement(children: .combine)
            }
            Text("A mode that uses a model you delete moves to another installed model.", bundle: .module)
                .themedFont(.tiny).foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
        .appCardSurface(padding: 14)
    }

    private struct Slot {
        var mode: SpeechToTextMode
        var job: SpeechToTextJob
        var current: String?
        var inUse: Bool
        /// False for a speakers slot whose mode has labels off.
        var isOn: Bool
        var key: String { SpeechToTextModels.key(mode, job) }
    }

    private var slots: [Slot] {
        SpeechToTextMode.allCases.flatMap { mode in
            mode.jobs.filter { $0.accepts(profile) }.map { job in
                let current = controller.speechToTextProfile(mode, job)
                let isOn = job != .speakers || controller.labelsSpeakers(in: mode)
                return Slot(mode: mode, job: job, current: current?.displayName,
                            inUse: isOn && current?.identity.alias == profile.identity.alias, isOn: isOn)
            }
        }
    }
}
