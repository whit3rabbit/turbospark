import SwiftUI

@MainActor
struct AudioTransportView: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem
    @State private var seekTime: Double = 0
    @State private var waveform: (clips: [AudioLibraryClip], peaks: [Float])?

    private var playingItem: AudioLibraryItem {
        guard let id = controller.playbackItemID, id != item.id else { return item }
        return controller.items.first { $0.id == id } ?? item
    }
    private var duration: Double {
        let value = controller.playbackItemID == nil ? item.duration : controller.playbackDuration
        return value.isFinite ? max(0, value) : 0
    }
    private var position: Double { max(0, min(duration, controller.playbackTime)) }

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            if playingItem.id != item.id {
                Text(verbatim: playingItem.title).themedFont(.small).foregroundStyle(.appSecondary)
            }
            AppAdaptiveHeader {
                transportButtons
            } actions: {
                Text(verbatim: audioTime(position) + " / " + audioTime(duration))
                    .themedFont(.small).monospacedDigit().foregroundStyle(.appSecondary)
                    .accessibilityHidden(true)
            }
            if let waveform, waveform.clips == playingItem.clips, !waveform.peaks.isEmpty,
               controller.playbackItemID == nil || controller.playbackItemID == playingItem.id {
                AudioPlaybackWaveform(peaks: waveform.peaks, progress: position / max(0.01, duration)) {
                    seek(to: $0 * duration)
                }
            }
            positionSlider
            DisclosureGroup {
                ViewThatFits(in: .horizontal) {
                    HStack { positionField; jumpButton }
                    VStack(alignment: .leading) { positionField; jumpButton }
                }
                .padding(.top, 6)
            } label: { Text("Go to position", bundle: .module) }
            .themedFont(.small)
        }
        .disabled(duration <= 0 || controller.hasRecordingActivity)
        .appCardSurface(padding: 12)
        .task(id: playingItem.clips) {
            let current = playingItem
            let peaks = try? await controller.waveformPeaks(for: current, count: 64)
            guard !Task.isCancelled else { return }
            waveform = (current.clips, peaks ?? [])
        }
    }

    private var transportButtons: some View {
        HStack(spacing: 6) {
            Button { controller.playPause(item: item) } label: {
                Label {
                    controller.isPlaying ? Text("Pause", bundle: .module) : Text("Play", bundle: .module)
                } icon: { Image(systemName: controller.isPlaying ? "pause.fill" : "play.fill") }
            }
            .appAction(.primary, size: .small)
            .keyboardShortcut(.space, modifiers: .option)
            AppIconButton("gobackward.5", label: Text("Back five seconds", bundle: .module)) {
                seek(to: position - 5)
            }
            .appAction(.quiet, size: .small)
            AppIconButton("goforward.5", label: Text("Forward five seconds", bundle: .module)) {
                seek(to: position + 5)
            }
            .appAction(.quiet, size: .small)
        }
    }

    private var positionSlider: some View {
        Slider(value: Binding(get: { position }, set: { seek(to: $0) }), in: 0...max(0.01, duration)) {
            Text("Playback position", bundle: .module)
        }
        .labelsHidden()
        .controlSize(.small)
        .accessibilityValue(Text(verbatim: audioTime(position) + " / " + audioTime(duration)))
        .accessibilityAdjustableAction { direction in
            switch direction {
            case .increment: seek(to: position + 5)
            case .decrement: seek(to: position - 5)
            @unknown default: break
            }
        }
    }

    private var positionField: some View {
        TextField(value: $seekTime, format: .number.precision(.fractionLength(0...2))) {
            Text("Position in seconds", bundle: .module)
        }
        .textFieldStyle(.roundedBorder)
        .frame(maxWidth: 140)
        .onSubmit { seek(to: seekTime) }
    }

    private var jumpButton: some View {
        Button { seek(to: seekTime) } label: { Text("Go to position", bundle: .module) }
            .appAction(.quiet, size: .small)
    }

    private func seek(to seconds: Double) {
        guard seconds.isFinite else { return }
        controller.seek(to: min(duration, max(0, seconds)), item: controller.playbackItemID == nil ? item : nil)
    }
}

@MainActor
struct AudioExportMenu: View {
    @ObservedObject var controller: AudioWorkspaceController
    let item: AudioLibraryItem

    var body: some View {
        Menu {
            if item.hasPlayableAudio {
                Button { controller.exportAudio(item) } label: { Text("Save As...", bundle: .module) }
                if item.preferredTranscript != nil { Divider() }
            }
            if item.preferredTranscript != nil {
                Button { controller.exportTranscript(item, format: .text) } label: { Text("Text", bundle: .module) }
                Button { controller.exportTranscript(item, format: .srt) } label: { Text("SRT subtitles", bundle: .module) }
                Button { controller.exportTranscript(item, format: .vtt) } label: { Text("WebVTT subtitles", bundle: .module) }
                Button { controller.exportTranscript(item, format: .json) } label: { Text("Structured result (JSON)", bundle: .module) }
            }
        } label: {
            Label { Text("Export", bundle: .module) } icon: { Image(systemName: "square.and.arrow.up") }
        }
        .appMenuAction(size: .small, quiet: true)
        .disabled(controller.hasRecordingActivity || controller.isExporting)
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
            // Opens the page that started the capture: Voice Cloning for a
            // voice reference, the Speech-to-Text mode, or Record.
            Button { openWorkspace(); controller.openActiveCapture() } label: {
                AudioRecordingReadout(controller: controller)
            }
            .appAction(.secondary)
            .accessibilityHint(Text("Open recording controls", bundle: .module))
        }
    }
}

@MainActor
struct AudioRecordingReadout: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        HStack(spacing: 6) {
            if controller.recordingSetupStatus != nil {
                Image(systemName: "hourglass").accessibilityHidden(true)
                Text("Preparing recording...", bundle: .module)
            } else {
                Image(systemName: controller.isPaused ? "pause.circle.fill" : "record.circle.fill")
                    .foregroundStyle(controller.isPaused ? AnyShapeStyle(.appSecondary) : AnyShapeStyle(.appDanger))
                    .accessibilityHidden(true)
                if controller.isPaused {
                    Text("Paused", bundle: .module)
                } else {
                    Text("Recording", bundle: .module)
                }
                Text(verbatim: audioTime(controller.recordingSeconds)).monospacedDigit()
            }
        }
        .themedFont(.small, weight: .semibold)
        .accessibilityElement(children: .combine)
    }
}

@MainActor
struct AudioWorkspaceStatusView: View {
    @ObservedObject var controller: AudioWorkspaceController

    private var showsStatus: Bool {
        controller.error != nil || controller.isBusy || controller.isInstalling || controller.isExporting
            || controller.recordingSetupStatus != nil || controller.transcriptBacklog > 0
            || !controller.pendingSaves.isEmpty || !controller.pendingTranscriptEdits.isEmpty
            || controller.status != AppStrings.localized("Saved in this profile")
    }

    var body: some View {
        if showsStatus {
            statusContent
        }
    }

    private var statusContent: some View {
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
                    if !controller.pendingSaves.isEmpty || !controller.pendingTranscriptEdits.isEmpty
                        || (controller.error != nil && !controller.pendingUserNotes.isEmpty) {
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
        .padding(.horizontal, 24).padding(.vertical, 8)
        .background(.appPage)
        .overlay(alignment: .top) { Divider() }
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
