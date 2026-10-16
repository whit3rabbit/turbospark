import SwiftUI

/// The Record route's capture panel: sources, start/pause/stop, markers and
/// the live transcription model. Split from AudioWorkspaceView to keep that
/// file readable.
@MainActor
struct AudioRecordingPanel: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @ObservedObject private var permissions = RecordingPermissions.shared
    @State private var accessRequest: RecordingAccessRequest?

    private var neededPermissions: [RecordingPermission] {
        RecordingPermission.needed(microphone: controller.captureMicrophone, systemAudio: controller.captureSystemAudio)
    }

    /// Start goes through the access sheet when a selected source lacks its
    /// macOS grant, so the user decides before a draft take is created. A
    /// source the user records without is switched off here, where they see it.
    private func beginRecording() {
        let needed = neededPermissions
        permissions.refresh()
        guard !permissions.missing(needed).isEmpty else { controller.startRecording(); return }
        accessRequest = RecordingAccessRequest(purpose: .start, needed: needed) { [controller] declined in
            if declined.contains(.microphone) { controller.captureMicrophone = false }
            if declined.contains(.systemAudio) { controller.captureSystemAudio = false }
            controller.startRecording()
        }
    }

    private func reviewAccess() {
        accessRequest = RecordingAccessRequest(purpose: .review, needed: neededPermissions) { [controller] _ in
            controller.prepareRecording()
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            AudioRecordingPresetPicker(controller: controller)
            captureSources
            if let setupStatus = controller.recordingSetupStatus {
                HStack(alignment: .top, spacing: 12) {
                    ProgressView().controlSize(.small)
                    Text(setupStatus).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 0)
                    Button { controller.cancelRecordingSetup() } label: { Text("Cancel", bundle: .module) }
                        .appAction(.secondary, size: .small)
                }
            } else if controller.isRecording {
                HStack {
                    AudioRecordingBadge(controller: controller)
                    Spacer()
                    Button { controller.addMarker() } label: {
                        Label { Text("Add marker", bundle: .module) } icon: { Image(systemName: "bookmark") }
                    }
                    .appAction(.secondary, size: .small)
                }
                ProgressView(value: Double(max(0, min(1, controller.inputLevel)))) {
                    Text("Input level", bundle: .module)
                }
                .accessibilityValue(Text(verbatim: "\(Int(max(0, min(1, controller.inputLevel)) * 100))%"))
                HStack {
                    Button {
                        if controller.isPaused { controller.resumeRecording() } else { controller.pauseRecording() }
                    } label: {
                        Label {
                            controller.isPaused ? Text("Resume", bundle: .module) : Text("Pause", bundle: .module)
                        } icon: { Image(systemName: controller.isPaused ? "play.fill" : "pause.fill") }
                    }
                    .appAction(.secondary)
                    // Finishing the take is this panel's one primary action
                    // while recording, as Start is before it.
                    Button { controller.stopRecording() } label: {
                        Label { Text("Stop recording", bundle: .module) } icon: { Image(systemName: "stop.fill") }
                    }
                    .appAction(.primary)
                }
            } else {
                Button { beginRecording() } label: {
                    Label { Text("Start recording", bundle: .module) } icon: { Image(systemName: "record.circle") }
                }
                .appAction(.primary)
                .disabled(controller.isBusy || (!controller.captureMicrophone && !controller.captureSystemAudio))
                .keyboardShortcut("r", modifiers: [.command, .shift])
            }
            AudioLiveTextView(live: controller.live)
            DisclosureGroup {
                Toggle(isOn: $controller.liveTranscription) { Text("Show a live draft transcript", bundle: .module) }
                if controller.liveTranscription { AudioLiveTextModelRow(model: model, controller: controller).padding(.top, 4) }
                AudioModelPickerView(model: model, controller: controller).padding(.top, 8)
                Text("Live text is a draft. Saved audio remains available when transcription falls behind.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            } label: { Text("Transcription model", bundle: .module) }
        }
        .padding(20).background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).stroke(.appBorder, lineWidth: 1) }
        .recordingAccessGate($accessRequest, permissions: permissions)
    }

    /// Warns before Start rather than after a failed setup. Glyph plus text,
    /// so the state does not depend on color.
    @ViewBuilder private var accessWarning: some View {
        if !permissions.missing(neededPermissions).isEmpty {
            Label {
                Text("A selected source needs macOS access before you can record.", bundle: .module)
                    .fixedSize(horizontal: false, vertical: true)
            } icon: {
                Image(systemName: "exclamationmark.triangle.fill").accessibilityHidden(true)
            }
            .themedFont(.small)
            .foregroundStyle(.appSecondary)
        }
    }

    private var captureSources: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: $controller.captureMicrophone) { Text("Microphone", bundle: .module) }
            if controller.captureMicrophone {
                Picker(selection: $controller.microphoneID) {
                    Text("Default microphone", bundle: .module).tag(Optional<String>.none)
                    ForEach(controller.microphones, id: \.id) { device in Text(device.name).tag(Optional(device.id)) }
                } label: { Text("Microphone", bundle: .module) }
            }
            Toggle(isOn: $controller.captureSystemAudio) { Text("App or system audio", bundle: .module) }
            if controller.captureSystemAudio {
                Picker(selection: $controller.selectedApplicationID) {
                    Text("All system audio", bundle: .module).tag(Optional<Int32>.none)
                    ForEach(controller.audioApplications, id: \.id) { app in Text(app.name).tag(Optional(app.id)) }
                } label: { Text("Audio source", bundle: .module) }
                Text("TurboSpark playback is excluded. Only audio is saved.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
            accessWarning
            Button { reviewAccess() } label: {
                Label { Text("Check devices and permissions", bundle: .module) } icon: { Image(systemName: "checkmark.shield") }
            }
            .appAction(.secondary, size: .small)
        }.disabled(controller.hasRecordingActivity)
    }
}
