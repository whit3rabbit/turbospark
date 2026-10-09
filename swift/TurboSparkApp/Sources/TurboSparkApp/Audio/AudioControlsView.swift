import SwiftUI

@MainActor
struct AudioTransportView: View {
    @ObservedObject var controller: AudioWorkspaceController
    @State private var seekTime: Double = 0

    private var duration: Double { max(0, controller.playbackDuration) }
    private var position: Double { max(0, min(duration, controller.playbackTime)) }

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 14) {
                Button { controller.seek(to: max(0, position - 5)) } label: { Image(systemName: "gobackward.5") }
                    .accessibilityLabel(Text("Back five seconds", bundle: .module))
                Button { controller.playPause() } label: {
                    Label { Text(controller.isPlaying ? "Pause" : "Play", bundle: .module) }
                        icon: { Image(systemName: controller.isPlaying ? "pause.fill" : "play.fill") }
                }
                .keyboardShortcut(.space, modifiers: .option)
                Button { controller.seek(to: min(duration, position + 5)) } label: { Image(systemName: "goforward.5") }
                    .accessibilityLabel(Text("Forward five seconds", bundle: .module))
                Spacer(minLength: 0)
                Text(verbatim: audioTime(position) + " / " + audioTime(duration))
                    .themedFont(.small).monospacedDigit()
                    .accessibilityLabel(Text("Playback position", bundle: .module))
            }
            Slider(value: Binding(get: { position }, set: { controller.seek(to: $0) }), in: 0...max(0.01, duration)) {
                Text("Playback position", bundle: .module)
            }
            .accessibilityValue(Text(audioTime(position)))
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
                }.padding(.top, 8)
            } label: { Text("Go to position", bundle: .module) }
            .themedFont(.small)
        }
        .disabled(duration <= 0 || controller.hasRecordingActivity)
        .padding(16).background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
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
        .menuStyle(.borderlessButton).fixedSize()
        .disabled(item.preferredTranscript == nil || controller.hasRecordingActivity || controller.isExporting)
    }
}

@MainActor
struct AudioSaveAsButton: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem

    var body: some View {
        Button { controller.exportAudio(item) } label: {
            Label { Text("Save As...", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
        }
        .fixedSize()
        .disabled(!item.hasPlayableAudio || controller.hasRecordingActivity || controller.isExporting)
    }
}

@MainActor
struct AudioDeleteButton: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem
    @State private var deletingItem: AudioLibraryItem?

    var body: some View {
        Button(role: .destructive) { deletingItem = item } label: {
            Label { Text("Delete", bundle: .module) } icon: { Image(systemName: "trash") }
        }
        .fixedSize()
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
            Button { openWorkspace(); controller.selectPage(.record) } label: {
                HStack(spacing: 6) {
                    if controller.recordingSetupStatus != nil {
                        Image(systemName: "hourglass")
                        Text("Preparing recording...", bundle: .module)
                    } else {
                        Image(systemName: controller.isPaused ? "pause.circle.fill" : "record.circle")
                        Text(controller.isPaused ? "Paused" : "Recording", bundle: .module)
                        Text(audioTime(controller.recordingSeconds)).monospacedDigit()
                    }
                }.themedFont(.small, weight: .semibold)
            }
            .buttonStyle(.bordered)
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
                    Text(error).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 8)
                    Button { controller.error = nil } label: { Image(systemName: "xmark") }
                        .accessibilityLabel(Text("Dismiss error", bundle: .module))
                }
                Text("Your input and completed takes are kept. Adjust the settings and try again.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
            if controller.error == nil || controller.isBusy || controller.isExporting || !controller.pendingSaves.isEmpty || !controller.pendingTranscriptEdits.isEmpty {
                HStack(spacing: 12) {
                    if controller.isBusy || controller.isInstalling || controller.recordingSetupStatus != nil {
                        if let progress = controller.progress {
                            ProgressView(value: max(0, min(1, progress))).frame(width: 180)
                                .accessibilityLabel(Text(controller.status))
                                .accessibilityValue(Text(controller.generationProgress?.counter ?? "\(Int(progress * 100))%"))
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
                             ? String(localized: "Exporting audio", bundle: .module) : controller.status).lineLimit(2)
                        if let generation = controller.generationProgress {
                            HStack(spacing: 12) {
                                if let counter = generation.counter { Text(counter).monospacedDigit() }
                                if let detail = generation.detail { Text(detail).monospacedDigit() }
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
                            .keyboardShortcut(".", modifiers: .command)
                    } else if controller.isBusy {
                        Button { controller.cancel() } label: { Text("Cancel", bundle: .module) }
                            .keyboardShortcut(".", modifiers: .command)
                    }
                    if !controller.pendingSaves.isEmpty || !controller.pendingTranscriptEdits.isEmpty {
                        Button { controller.retryAutosave() } label: { Text("Retry", bundle: .module) }
                    }
                }
            }
            if let generation = controller.generationProgress {
                TimelineView(.periodic(from: .now, by: 1)) { context in
                    HStack(spacing: 16) {
                        HStack(spacing: 6) {
                            Text("Elapsed time", bundle: .module)
                            Text(audioTime(context.date.timeIntervalSince(generation.startedAt))).monospacedDigit()
                        }
                        if generation.phase != .preparing {
                            HStack(spacing: 6) {
                                Text("Last update", bundle: .module)
                                Text(audioTime(context.date.timeIntervalSince(generation.updatedAt))).monospacedDigit()
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
                    .keyboardShortcut(.cancelAction)
                Button { save() } label: { Text("Save", bundle: .module) }
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
