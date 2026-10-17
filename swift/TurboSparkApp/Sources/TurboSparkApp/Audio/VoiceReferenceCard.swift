import SwiftUI
import UniformTypeIdentifiers

/// The Voice Cloning reference: a drop zone with Import, Record and Choose
/// from library while empty; an inline recorder while a reference is being
/// recorded; and the chosen clip with its waveform, trim and preview.
@MainActor
struct VoiceReferenceCard: View {
    @ObservedObject var controller: AudioWorkspaceController
    @ObservedObject var attention: VoiceCloneAttention
    let missing: [VoiceCloneStep]
    @ObservedObject private var permissions = RecordingPermissions.shared
    @State private var showingPicker = false
    @State private var dropTargeted = false
    @State private var accessRequest: RecordingAccessRequest?

    @Environment(\.appTheme) private var theme

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            AppSectionHeader(title: Text("Reference voice", bundle: .module),
                             subtitle: Text("Use a clear recording of one person speaking for 1 to 30 seconds.", bundle: .module))
            if controller.isRecordingVoiceReference {
                recorder.transition(.opacity)
            } else if let reference = controller.speechSettings.reference {
                VoiceReferenceDetails(controller: controller, reference: reference,
                                      choose: { showingPicker = true }, record: recordReference)
                    .transition(.opacity)
            } else {
                dropZone.transition(.opacity)
            }
            if needsAttention {
                AppAttentionNote(message: Text("Choose 1 to 30 seconds of reference speech.", bundle: .module))
            } else if let reason = unavailableReason {
                Label { reason.fixedSize(horizontal: false, vertical: true) } icon: { Image(systemName: "info.circle") }
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
        }
        .tsAnimation(TSMotion.pane, value: controller.speechSettings.reference == nil)
        .tsAnimation(TSMotion.pane, value: controller.isRecordingVoiceReference)
        .tsAnimation(TSMotion.select, value: needsAttention)
        .appCardSurface()
        .appAttention(needsAttention, ping: attention.ping(.voice))
        .sheet(isPresented: $showingPicker) { VoiceReferencePickerSheet(controller: controller) }
        .recordingAccessGate($accessRequest, permissions: permissions)
    }

    private var needsAttention: Bool { attention.isOn(.voice, missing: missing) }

    /// Recording starts only once macOS allows the microphone: the capture
    /// layer checks access and never prompts, so a missing grant goes through
    /// the same access sheet as the Record page. A reference has no other
    /// source to fall back to, so declining the microphone records nothing.
    private func recordReference() {
        permissions.refresh()
        guard !permissions.missing([.microphone]).isEmpty else {
            controller.startVoiceReferenceRecording()
            return
        }
        accessRequest = RecordingAccessRequest(purpose: .start, needed: [.microphone]) { [controller] declined in
            guard !declined.contains(.microphone) else { return }
            controller.startVoiceReferenceRecording()
        }
    }

    /// Why the reference actions are disabled, when nothing else on the
    /// page already says so (the progress card covers a running job).
    private var unavailableReason: Text? {
        if controller.speechSettings.mode != .clone { return Text("Choose a voice cloning model first.", bundle: .module) }
        if controller.hasRecordingActivity && !controller.isRecordingVoiceReference {
            return Text("A recording is already active. Stop it before starting another.", bundle: .module)
        }
        return nil
    }

    private var dropZone: some View {
        VStack(spacing: 12) {
            Image(systemName: "waveform.badge.plus")
                .themedFont(.title)
                .foregroundStyle(.appAccent)
                .scaleEffect(dropTargeted ? 1.12 : 1)
                .accessibilityHidden(true)
            Text("Drop an audio file here, import one, record your voice, or choose saved audio.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
                .multilineTextAlignment(.center)
                .fixedSize(horizontal: false, vertical: true)
            if controller.importTask != nil {
                AppProgressBar(value: nil, label: Text("Importing audio", bundle: .module), size: .compact)
                    .frame(minWidth: 120, idealWidth: 180, maxWidth: 240)
            }
            ViewThatFits(in: .horizontal) {
                HStack(spacing: AppMetrics.actionSpacing) { sourceActions }
                VStack(spacing: AppMetrics.actionSpacing) { sourceActions }
            }
        }
        .padding(18)
        .frame(maxWidth: .infinity)
        .background(dropTargeted ? AnyShapeStyle(.appAccent.opacity(0.08)) : AnyShapeStyle(Color.clear),
                    in: RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous))
        .overlay {
            // The halo rides on the dashed edge itself, so the text and
            // buttons inside the zone do not glow with it.
            RoundedRectangle(cornerRadius: AppMetrics.cardRadius, style: .continuous)
                .stroke(dropTargeted ? AnyShapeStyle(.appAccent) : AnyShapeStyle(.appBorder),
                        style: StrokeStyle(lineWidth: dropTargeted ? 2 : 1, dash: [6, 4]))
                .shadow(color: dropTargeted && !theme.isHighContrast ? theme.accent.opacity(0.45) : .clear, radius: 8)
        }
        .tsAnimation(TSMotion.hover, value: dropTargeted)
        .dropDestination(for: URL.self) { urls, _ in
            // A disabled view still receives drops, so the guard lives here.
            guard controller.canEditVoiceReference, controller.importTask == nil,
                  let url = urls.first(where: AudioWorkspaceController.acceptsAudioFile) else { return false }
            return controller.importAudio(from: url, asVoiceReference: true)
        } isTargeted: { dropTargeted = $0 && controller.canEditVoiceReference }
    }

    @ViewBuilder private var sourceActions: some View {
        Button { controller.importAudio(asVoiceReference: true) } label: {
            Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
        }
        .appAction(.secondary, size: .small)
        .disabled(!controller.canEditVoiceReference || controller.importTask != nil)
        Button(action: recordReference) {
            Label { Text("Record", bundle: .module) } icon: { Image(systemName: "mic") }
        }
        .appAction(.secondary, size: .small)
        .disabled(!controller.canEditVoiceReference)
        Button { showingPicker = true } label: {
            Label { Text("Choose from library", bundle: .module) } icon: { Image(systemName: "books.vertical") }
        }
        .appAction(.secondary, size: .small)
        .disabled(!controller.canEditVoiceReference)
    }

    private var recorder: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Circle().fill(Color.red).frame(width: 9, height: 9)
                    .tsPulse(active: controller.isRecording && !controller.isPaused)
                    .accessibilityHidden(true)
                Group {
                    if controller.isRecording { Text("Recording", bundle: .module) }
                    else { Text("Preparing recording...", bundle: .module) }
                }
                .themedFont(.base, weight: .semibold)
                Spacer(minLength: 8)
                Text(verbatim: audioTime(controller.recordingSeconds)).monospacedDigit()
                    .foregroundStyle(.appSecondary)
            }
            AppProgressBar(
                value: controller.isRecording
                    ? controller.recordingSeconds / AudioWorkspaceController.voiceReferenceRecordingLimit : nil,
                label: Text("Recording", bundle: .module),
                valueText: Text(verbatim: audioTime(controller.recordingSeconds)))
            AudioLiveLevels(controller: controller)
                .padding(.horizontal, 10).padding(.vertical, 8)
                .background(.appPage.opacity(0.6), in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                .overlay(RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous).stroke(.appBorder, lineWidth: 1))
            HStack(spacing: 8) {
                Image(systemName: "mic.fill").themedFont(.tiny).foregroundStyle(.appSecondary).accessibilityHidden(true)
                AppProgressBar(value: Double(max(0, min(1, controller.inputLevel))), label: Text("Input level", bundle: .module),
                               size: .compact)
            }
            Text("Recording stops at 30 seconds and becomes the reference.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            HStack {
                Spacer()
                if controller.isRecording {
                    Button { controller.stopRecording() } label: {
                        Label { Text("Stop recording", bundle: .module) } icon: { Image(systemName: "stop.fill") }
                    }
                    .appAction(.secondary, size: .small)
                } else {
                    Button { controller.cancelRecordingSetup() } label: { Text("Cancel", bundle: .module) }
                        .appAction(.secondary, size: .small)
                }
            }
        }
    }
}

/// The chosen reference: source, waveform, trim and preview.
@MainActor
private struct VoiceReferenceDetails: View {
    @ObservedObject var controller: AudioWorkspaceController
    let reference: AudioVoiceReference
    let choose: () -> Void
    let record: () -> Void

    private var editable: Bool { controller.canEditVoiceReference }

    private var sourceKind: AudioLibraryItem.Kind? {
        controller.items.first { $0.id == reference.sourceID }?.kind
    }

    private func trim(_ path: WritableKeyPath<AudioVoiceReference, Double>) -> Binding<Double> {
        Binding(get: { controller.speechSettings.reference?[keyPath: path] ?? 0 },
                set: { controller.setVoiceReferenceTrim(path, to: $0) })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            source
            VoiceReferenceWaveformView(controller: controller, reference: reference)
            trimControls
            if !reference.isValid {
                Label { Text("Choose 1 to 30 seconds of reference speech.", bundle: .module) } icon: {
                    Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(.orange)
                }
                .themedFont(.small)
            }
            actions
        }
    }

    private var source: some View {
        HStack(spacing: 12) {
            Image(systemName: sourceKind?.voiceReferenceSymbol ?? "waveform")
                .themedFont(.base).foregroundStyle(.appAccent)
                .frame(width: 32, height: 32)
                .background(.appAccent.opacity(0.12), in: Circle())
                .accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: reference.title).themedFont(.base, weight: .medium)
                    .lineLimit(2).truncationMode(.middle)
                    .help(Text(verbatim: reference.title))
                HStack(spacing: 8) {
                    if let sourceKind { sourceKind.voiceReferenceTitle }
                    Text("\(audioTime(reference.duration)) of \(audioTime(reference.totalDuration))", bundle: .module)
                        .monospacedDigit()
                }
                .themedFont(.tiny).foregroundStyle(.appSecondary)
            }
            Spacer(minLength: 8)
            Menu {
                Button { controller.importAudio(asVoiceReference: true) } label: {
                    Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                }
                Button(action: record) {
                    Label { Text("Record", bundle: .module) } icon: { Image(systemName: "mic") }
                }
                Button(action: choose) {
                    Label { Text("Choose from library", bundle: .module) } icon: { Image(systemName: "books.vertical") }
                }
            } label: { Text("Replace", bundle: .module) }
            .appMenuAction(size: .small)
            .disabled(!editable || controller.importTask != nil)
        }
        .accessibilityElement(children: .contain)
    }

    @ViewBuilder private var trimControls: some View {
        let maxStart = max(0, reference.totalDuration - reference.duration)
        VStack(alignment: .leading, spacing: 10) {
            // A zero-width slider range is a fault, and a source that is all
            // reference has nothing to slide over.
            if maxStart >= 0.1 {
                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text("Start (seconds)", bundle: .module).themedFont(.small)
                        Spacer()
                        Text(verbatim: audioTime(reference.start)).themedFont(.small).monospacedDigit()
                            .foregroundStyle(.appSecondary)
                    }
                    Slider(value: trim(\.start), in: 0...maxStart) { Text("Start (seconds)", bundle: .module) }
                        .labelsHidden()
                        .accessibilityValue(Text(verbatim: audioTime(reference.start)))
                }
            }
            HStack(spacing: 12) {
                Text("Duration (seconds)", bundle: .module).themedFont(.small)
                TextField(value: trim(\.duration), format: .number) { Text("Duration (seconds)", bundle: .module) }
                    .textFieldStyle(.roundedBorder)
                    .frame(minWidth: 60, idealWidth: 80, maxWidth: 120)
                    .accessibilityLabel(Text("Duration (seconds)", bundle: .module))
                Spacer(minLength: 0)
            }
        }
        .disabled(!editable)
    }

    private var actions: some View {
        HStack(spacing: AppMetrics.actionSpacing) {
            // Playback looks the same here as in the transport and the API
            // test panel: secondary, with a play/stop symbol.
            Button { controller.previewVoiceReference() } label: {
                Label {
                    if controller.referencePlaying { Text("Stop preview", bundle: .module) }
                    else { Text("Preview reference audio", bundle: .module) }
                } icon: {
                    Image(systemName: controller.referencePlaying ? "stop.fill" : "play.fill")
                        .contentTransition(.symbolEffect(.replace))
                }
            }
            .appAction(.secondary, size: .small)
            .tsAnimation(TSMotion.hover, value: controller.referencePlaying)
            .disabled(!reference.isValid || controller.hasRecordingActivity)
            Spacer(minLength: 8)
            Button { controller.clearVoiceReference() } label: {
                Label { Text("Remove", bundle: .module) } icon: { Image(systemName: "xmark") }
            }
            .appAction(.quiet, size: .small)
            .disabled(!editable)
        }
    }
}
