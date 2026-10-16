import SwiftUI

// One treatment for the actions every Audio route shares (Library, Record,
// Transcribe, Voiceover, Voice Clone, Voice Design, Music, Cleanup and
// Advanced all draw their saved items through these components):
// playback, Save As and Export are `.secondary`, Delete is `.destructive`,
// all `.small` because they sit in cards and rows. Rows pass
// `compact: true` for icon-only Save As and Delete with the same roles.
// The route's run action and the model panel's Download/Retry/Adopt are
// the only `.primary` actions.

@MainActor
struct AudioTransportView: View {
    @ObservedObject var controller: AudioWorkspaceController
    @State private var seekTime: Double = 0

    private var duration: Double { max(0, controller.playbackDuration) }
    private var position: Double { max(0, min(duration, controller.playbackTime)) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 14) {
                AppIconButton("gobackward.5", label: Text("Back five seconds", bundle: .module)) {
                    controller.seek(to: max(0, position - 5))
                }
                .appAction(.quiet, size: .small)
                Button { controller.playPause() } label: {
                    Label {
                        controller.isPlaying ? Text("Pause", bundle: .module) : Text("Play", bundle: .module)
                    } icon: { Image(systemName: controller.isPlaying ? "pause.fill" : "play.fill") }
                }
                .appAction(.secondary, size: .small)
                .keyboardShortcut(.space, modifiers: .option)
                AppIconButton("goforward.5", label: Text("Forward five seconds", bundle: .module)) {
                    controller.seek(to: min(duration, position + 5))
                }
                .appAction(.quiet, size: .small)
                Spacer(minLength: 0)
                Text(verbatim: audioTime(position) + " / " + audioTime(duration))
                    .themedFont(.small).monospacedDigit()
                    .accessibilityLabel(Text("Playback position", bundle: .module))
            }
            Slider(value: Binding(get: { position }, set: { controller.seek(to: $0) }), in: 0...max(0.01, duration)) {
                Text("Playback position", bundle: .module)
            }
            .accessibilityValue(Text(verbatim: audioTime(position)))
            .accessibilityAdjustableAction { direction in
                switch direction {
                case .increment: controller.seek(to: min(duration, position + 5))
                case .decrement: controller.seek(to: max(0, position - 5))
                @unknown default: break
                }
            }
            DisclosureGroup {
                HStack {
                    TextField(value: $seekTime, format: .number.precision(.fractionLength(0...2))) {
                        Text("Position in seconds", bundle: .module)
                    }
                    .textFieldStyle(.roundedBorder).frame(maxWidth: 140)
                    .onSubmit { seek() }
                    Button { seek() } label: { Text("Go to position", bundle: .module) }
                        .appAction(.secondary, size: .small)
                }.padding(.top, 8)
            } label: { Text("Go to position", bundle: .module) }
            .themedFont(.small)
        }
        .disabled(duration <= 0 || controller.hasRecordingActivity)
        .padding(16).background(.appSurface, in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
    }

    private func seek() {
        guard seekTime.isFinite else { return }
        controller.seek(to: max(0, min(duration, seekTime)))
    }
}

@MainActor
struct AudioExportMenu: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem

    var body: some View {
        Menu {
            if item.preferredTranscript != nil {
                Button { controller.exportTranscript(item, format: .text) } label: { Text("Text", bundle: .module) }
                Button { controller.exportTranscript(item, format: .srt) } label: { Text("SRT subtitles", bundle: .module) }
                Button { controller.exportTranscript(item, format: .vtt) } label: { Text("WebVTT subtitles", bundle: .module) }
                Button { controller.exportTranscript(item, format: .json) } label: { Text("Structured result (JSON)", bundle: .module) }
            }
        } label: {
            Label { Text("Export", bundle: .module) } icon: { Image(systemName: "square.and.arrow.up") }
        }
        .appMenuAction(size: .small)
        .disabled(item.preferredTranscript == nil || controller.hasRecordingActivity || controller.isExporting)
    }
}

@MainActor
struct AudioSaveAsButton: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem
    /// Icon only, for rows; the title stays the tooltip and VoiceOver name.
    var compact = false

    var body: some View {
        Button { controller.exportAudio(item) } label: {
            Label { Text("Save As...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                .labelStyle(AudioActionLabelStyle(compact: compact))
        }
        .appAction(.secondary, size: .small)
        .fixedSize()
        .help(Text("Save As...", bundle: .module))
        .disabled(!item.hasPlayableAudio || controller.hasRecordingActivity || controller.isExporting)
    }
}

/// Title and icon, or icon only for a row, without a conditional modifier
/// that would change the button's identity.
struct AudioActionLabelStyle: LabelStyle {
    let compact: Bool

    func makeBody(configuration: Configuration) -> some View {
        if compact {
            configuration.icon
        } else {
            Label(title: { configuration.title }, icon: { configuration.icon })
        }
    }
}

@MainActor
struct AudioDeleteButton: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem
    /// Icon only, for rows; the title stays the tooltip and VoiceOver name.
    var compact = false
    @State private var deletingItem: AudioLibraryItem?

    var body: some View {
        Button(role: .destructive) { deletingItem = item } label: {
            Label { Text("Delete", bundle: .module) } icon: { Image(systemName: "trash") }
                .labelStyle(AudioActionLabelStyle(compact: compact))
        }
        .appAction(.destructive, size: .small)
        .fixedSize()
        .help(Text("Delete", bundle: .module))
        .disabled(!controller.canDelete(item))
        .modifier(AudioDeleteConfirmation(controller: controller, item: $deletingItem))
    }
}

@MainActor
struct AudioDeleteConfirmation: ViewModifier {
    @ObservedObject var controller: AudioWorkspaceController
    @Binding var item: AudioLibraryItem?

    func body(content: Content) -> some View {
        content.alert(Text("Delete saved audio?", bundle: .module), isPresented: Binding(
            get: { item != nil }, set: { if !$0 { item = nil } }
        ), presenting: item) { candidate in
            Button(role: .destructive) { controller.delete(candidate); item = nil } label: { Text("Delete", bundle: .module) }
            Button(role: .cancel) { item = nil } label: { Text("Cancel", bundle: .module) }
        } message: { candidate in
            Text(verbatim: candidate.title + "\n\n") + Text("This removes the item and its edits from your audio library. Files saved to other folders are kept.", bundle: .module)
        }
    }
}

/// This badge also lives in the app chrome, so changing tasks never hides capture.
@MainActor
struct AudioRecordingBadge: View {
    @ObservedObject var controller: AudioWorkspaceController
    var openWorkspace: () -> Void = {}

    var body: some View {
        if controller.hasRecordingActivity {
            // A voice reference records from the Voice Cloning page, where its
            // inline recorder is; a meeting records from Record.
            Button { openWorkspace(); controller.selectPage(controller.isRecordingVoiceReference ? .voiceClone : .record) } label: {
                HStack(spacing: 6) {
                    if controller.recordingSetupStatus != nil {
                        Image(systemName: "hourglass")
                        Text("Preparing recording...", bundle: .module)
                    } else {
                        Image(systemName: controller.isPaused ? "pause.circle.fill" : "record.circle")
                        Text(controller.isPaused ? "Paused" : "Recording", bundle: .module)
                        Text(verbatim: audioTime(controller.recordingSeconds)).monospacedDigit()
                    }
                }.themedFont(.small, weight: .semibold)
            }
            .appAction(.secondary)
            .accessibilityHint(Text("Open recording controls", bundle: .module))
        }
    }
}

@MainActor
struct AudioWorkspaceStatusView: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            if let error = controller.error {
                HStack(alignment: .top) {
                    Image(systemName: "exclamationmark.triangle")
                    Text(verbatim: error).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 8)
                    AppIconButton("xmark", label: Text("Dismiss error", bundle: .module)) { controller.error = nil }
                        .appAction(.quiet, size: .small)
                }
                Text("Your input and completed takes are kept. Adjust the settings and try again.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
            if controller.error == nil || controller.isBusy || controller.isExporting || !controller.pendingSaves.isEmpty || !controller.pendingTranscriptEdits.isEmpty {
                HStack(spacing: 12) {
                    if controller.isBusy || controller.isInstalling || controller.recordingSetupStatus != nil {
                        if let progress = controller.progress {
                            AppProgressBar(value: progress, label: Text(verbatim: controller.status),
                                           valueText: controller.generationProgress?.counter.map { Text(verbatim: $0) },
                                           size: .compact)
                                .frame(minWidth: 120, idealWidth: 180, maxWidth: 240)
                        } else {
                            ProgressView().controlSize(.small)
                        }
                    } else if controller.isExporting {
                        ProgressView().controlSize(.small)
                    } else {
                        Image(systemName: controller.error == nil ? "checkmark.circle" : "exclamationmark.triangle")
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text(controller.isExporting && !controller.isBusy
                             ? AppStrings.localized("Exporting audio") : controller.status).lineLimit(2)
                        if let generation = controller.generationProgress {
                            HStack(spacing: 12) {
                                if let counter = generation.counter { Text(verbatim: counter).monospacedDigit() }
                                if let detail = generation.detail { Text(verbatim: detail).monospacedDigit() }
                            }.foregroundStyle(.appSecondary)
                        }
                    }
                    Spacer(minLength: 12)
                    if controller.transcriptBacklog > 0 {
                        HStack(spacing: 5) {
                            Text("Transcription backlog", bundle: .module)
                            Text(verbatim: String(controller.transcriptBacklog)).monospacedDigit()
                        }
                    }
                    if controller.recordingSetupStatus != nil {
                        Button { controller.cancelRecordingSetup() } label: { Text("Cancel", bundle: .module) }
                            .appAction(.secondary, size: .small)
                            .keyboardShortcut(".", modifiers: .command)
                    } else if controller.isBusy {
                        Button { controller.cancel() } label: { Text("Cancel", bundle: .module) }
                            .appAction(.secondary, size: .small)
                            .keyboardShortcut(".", modifiers: .command)
                    }
                    if !controller.pendingSaves.isEmpty || !controller.pendingTranscriptEdits.isEmpty {
                        Button { controller.retryAutosave() } label: { Text("Retry", bundle: .module) }
                            .appAction(.secondary, size: .small)
                    }
                }
            }
            if let generation = controller.generationProgress {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    HStack(spacing: 16) {
                        HStack(spacing: 6) {
                            Text("Elapsed time", bundle: .module)
                            Text(verbatim: audioTime(context.date.timeIntervalSince(generation.startedAt))).monospacedDigit()
                        }
                        if generation.phase != .preparing {
                            HStack(spacing: 6) {
                                Text("Last update", bundle: .module)
                                Text(verbatim: audioTime(context.date.timeIntervalSince(generation.updatedAt))).monospacedDigit()
                            }
                        }
                    }.foregroundStyle(.appSecondary)
                }
            }
        }
        .themedFont(.small)
        .padding(.horizontal, 20).padding(.vertical, 12)
        .background(.appSurface)
        .accessibilityElement(children: .contain)
        .accessibilityLabel(Text("Audio status", bundle: .module))
    }
}

@MainActor
struct AudioNameSheet: View {
    let title: String
    @Binding var value: String
    var allowEmpty = false
    let save: () -> Void
    @Environment(\.dismiss) private var dismiss
    @FocusState private var focused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            Text(LocalizedStringKey(title), bundle: .module).themedFont(.title2, weight: .semibold)
            TextField(text: $value) { Text(LocalizedStringKey(title), bundle: .module) }
                .textFieldStyle(.roundedBorder).focused($focused)
                .onSubmit { if valid { save() } }
            HStack {
                Spacer()
                Button { dismiss() } label: { Text("Cancel", bundle: .module) }
                    .appAction(.secondary)
                    .keyboardShortcut(.cancelAction)
                Button { save() } label: { Text("Save", bundle: .module) }
                    .appAction(.primary)
                    .keyboardShortcut(.defaultAction).disabled(!valid)
            }
        }
        .padding(24).frame(minWidth: 340).themedFont(.base)
        .onAppear { focused = true }
    }

    private var valid: Bool { allowEmpty || !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty }
}

func audioTime(_ seconds: Double) -> String {
    guard seconds.isFinite else { return "0:00" }
    let value = Int(min(Double(Int32.max), max(0, seconds)))
    if value >= 3600 { return String(format: "%d:%02d:%02d", value / 3600, value / 60 % 60, value % 60) }
    return String(format: "%d:%02d", value / 60, value % 60)
}
