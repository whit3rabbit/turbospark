import SwiftUI
import TurboSpark

/// The Speech-to-Text header button's popover: one slot per job the mode
/// runs (live text, the final transcript, and a meeting's speakers), each
/// with its model's state, the one action that state allows, and the pinned
/// catalog models that can do the job.
@MainActor
struct SpeechToTextModelPanel: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let mode: SpeechToTextMode
    @State private var expanded: SpeechToTextJob?
    @Environment(\.appTheme) private var theme

    var body: some View {
        let scale = AppMetrics.textScale(theme)
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                header
                if controller.isRefreshingModels && controller.profiles.isEmpty {
                    ProgressView().controlSize(.small)
                } else if let error = controller.modelCatalogError {
                    Text(error).themedFont(.small).foregroundStyle(.appSecondary).textSelection(.enabled)
                    Button { controller.refreshModels() } label: { Text("Retry", bundle: .module) }
                        .appAction(.secondary, size: .small)
                }
                ForEach(mode.jobs) { job in
                    SpeechToTextModelSlot(model: model, controller: controller, mode: mode, job: job, expanded: $expanded)
                }
                if !controller.canChangeSpeechToTextModels {
                    Text("Models can change once the recording or job finishes.", bundle: .module)
                        .themedFont(.small).foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }
                AudioLanguagePicker(controller: controller)
                footer
            }
            .padding(16)
        }
        // A popover sizes to its content, and a scroll view has none.
        .frame(width: 384 * scale)
        .frame(minHeight: 240 * scale, idealHeight: 560 * scale, maxHeight: 680 * scale)
    }

    private var header: some View {
        VStack(alignment: .leading, spacing: 2) {
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("Models", bundle: .module).themedFont(.callout, weight: .semibold)
                    .accessibilityAddTraits(.isHeader)
                Text(LocalizedStringKey(mode.titleKey), bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
            }
            Text("Each mode uses its own model for each job. A change applies to the next take.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    private var footer: some View {
        VStack(alignment: .leading, spacing: 8) {
            // The workspace holds one model at a time (the controller's
            // session). Live text opens its own stream beside it per take.
            Label {
                Text("One audio model stays loaded at a time. Loading another unloads it.", bundle: .module)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: { Image(systemName: "memorychip").accessibilityHidden(true) }
            .themedFont(.small).foregroundStyle(.appSecondary)
            Button {
                model.installedModelKind = .audio
                model.activeSection = .modelManager
            } label: {
                Label { Text("Manage in Installed", bundle: .module) } icon: { Image(systemName: "internaldrive") }
            }
            .appAction(.quiet, size: .small)
        }
    }
}

/// One job's slot: the model in use, its state and action, and its Change list.
@MainActor
struct SpeechToTextModelSlot: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let mode: SpeechToTextMode
    let job: SpeechToTextJob
    @Binding var expanded: SpeechToTextJob?

    private var profile: AudioProfile? { controller.speechToTextProfile(mode, job) }
    private var isExpanded: Bool { expanded == job }
    /// Speaker labels are optional per mode; the other jobs always run.
    private var isOn: Bool { job != .speakers || controller.labelsSpeakers(in: mode) }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                Text(LocalizedStringKey(mode.titleKey(for: job)), bundle: .module)
                    .themedFont(.small, weight: .semibold).foregroundStyle(.appSecondary)
                    .accessibilityAddTraits(.isHeader)
                Spacer(minLength: 4)
                if isOn, let profile { AudioModelStateLabel(state: state(profile)) }
            }
            if job == .speakers {
                Toggle(isOn: Binding(get: { controller.labelsSpeakers(in: mode) },
                                     set: { on in TSMotion.animate(TSMotion.select) { controller.setLabelsSpeakers(on, in: mode) } })) {
                    Text("Label speakers", bundle: .module).themedFont(.small)
                }
                .toggleStyle(.switch).controlSize(.small)
                .disabled(!controller.canChangeSpeechToTextModels)
            }
            if isOn { details }
            Text(LocalizedStringKey(job.hintKey), bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            if isOn { controls }
        }
        .appCardSurface(padding: 12)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text(LocalizedStringKey(mode.titleKey(for: job)), bundle: .module))
    }

    @ViewBuilder private var details: some View {
        if let profile {
            FlowLayout(spacing: 6, lineSpacing: 4) {
                Text(verbatim: profile.displayName).themedFont(.callout, weight: .semibold)
                AudioPinnedBadge(profile: profile, isLocalFolder: controller.preferences.localModels[profile.identity.alias] != nil)
            }
            AudioModelFacts(profile: profile)
        } else {
            Text("No catalog model can do this yet.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
        }
    }

    @ViewBuilder private var controls: some View {
        HStack(alignment: .center, spacing: AppMetrics.actionSpacing) {
            if let profile { action(profile) }
            Spacer(minLength: 0)
            changeButton
        }
        if isExpanded {
            SpeechToTextModelOptions(model: model, controller: controller, mode: mode, job: job)
                .transition(.opacity.combined(with: .offset(y: -4)))
        }
    }

    private func state(_ profile: AudioProfile) -> AudioModelState {
        controller.modelState(profile, download: model.audioDownload(for: profile))
    }

    @ViewBuilder private func action(_ profile: AudioProfile) -> some View {
        switch state(profile) {
        case .loaded where job.staysLoaded:
            Button { controller.unloadModel() } label: {
                Label { Text("Unload", bundle: .module) } icon: { Image(systemName: "eject.fill") }
            }
            .appAction(.secondary, size: .small)
            .disabled(!controller.canUnloadModel)
            .help(Text("Eject model and release unified memory", bundle: .module))
        case .installed where job.staysLoaded:
            Button { controller.load(profile) } label: {
                Label { Text("Load Model", bundle: .module) } icon: { Image(systemName: "bolt.fill") }
            }
            .appAction(.secondary, size: .small)
            .disabled(!controller.canLoad(profile))
            .help(Text("Load model into memory slot cache for inference", bundle: .module))
        case .installed:
            Text("Opens when a take starts.", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
        case .loaded:
            EmptyView()
        case .needsAdoption:
            if let legacy = controller.adoptable(profile.identity.alias) {
                Button { controller.adopt(legacy) } label: {
                    Label { Text("Adopt existing install", bundle: .module) } icon: { Image(systemName: "checkmark.seal") }
                }
                .appAction(.secondary, size: .small)
                .disabled(controller.isManagingInstall || controller.isBusy)
            }
        case .needsAttention:
            Text("Open Installed to see what is wrong.", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
        case .notInstalled, .downloading:
            AudioModelDownloadControls(model: model, controller: controller, profile: profile)
        }
    }

    private var changeButton: some View {
        Button {
            TSMotion.animate(TSMotion.select) { expanded = isExpanded ? nil : job }
        } label: {
            Label { Text("Change", bundle: .module) } icon: {
                Image(systemName: "chevron.down").rotationEffect(.degrees(isExpanded ? 180 : 0))
            }
            .labelStyle(.titleAndIcon)
        }
        .appAction(.quiet, size: .small)
        .accessibilityValue(isExpanded ? Text("Expanded", bundle: .module) : Text("Collapsed", bundle: .module))
        .help(Text("Choose another model for this job", bundle: .module))
    }
}

/// The pinned catalog models that can do a job, as a single-choice list.
/// The one in use is marked "In use"; choosing another applies to the next take.
@MainActor
struct SpeechToTextModelOptions: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    let mode: SpeechToTextMode
    let job: SpeechToTextJob
    @Environment(\.appTheme) private var theme

    var body: some View {
        let current = controller.speechToTextProfile(mode, job)?.identity.alias
        VStack(alignment: .leading, spacing: 2) {
            ForEach(controller.speechToTextCandidates(job)) { option in
                row(option, inUse: option.identity.alias == current)
            }
        }
        .padding(.top, 4)
        .accessibilityElement(children: .contain)
    }

    private func row(_ option: AudioProfile, inUse: Bool) -> some View {
        Button {
            controller.setSpeechToTextModel(option.identity.alias, mode: mode, job: job)
        } label: {
            HStack(alignment: .firstTextBaseline, spacing: 8) {
                Image(systemName: inUse ? "largecircle.fill.circle" : "circle")
                    .foregroundStyle(inUse ? AnyShapeStyle(AppThemeColor.appAccent) : AnyShapeStyle(AppThemeColor.appSecondary))
                    .accessibilityHidden(true)
                VStack(alignment: .leading, spacing: 2) {
                    FlowLayout(spacing: 6, lineSpacing: 2) {
                        Text(verbatim: option.displayName).themedFont(.small, weight: .semibold)
                        AudioPinnedBadge(profile: option, isLocalFolder: controller.preferences.localModels[option.identity.alias] != nil)
                    }
                    AudioModelFacts(profile: option)
                }
                Spacer(minLength: 8)
                if inUse {
                    Text("In use", bundle: .module).themedFont(.tiny, weight: .semibold).foregroundStyle(.appAccent)
                } else {
                    AudioModelStateLabel(state: controller.modelState(option, download: model.audioDownload(for: option)))
                }
            }
            .padding(6)
            .contentShape(Rectangle())
        }
        .buttonStyle(.plain)
        .background(inUse ? theme.selection : Color.clear,
                    in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
        .disabled(!controller.canChangeSpeechToTextModels)
        .accessibilityAddTraits(inUse ? [.isButton, .isSelected] : .isButton)
        .accessibilityHint(inUse ? Text("In use", bundle: .module) : Text("Use", bundle: .module))
    }
}
