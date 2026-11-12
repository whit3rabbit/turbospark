import SwiftUI

/// Start, pause, marker and stop for the one capture that can run. The Record
/// page and every Speech-to-Text mode mount this, so the start shortcut, the
/// access gate and the disabled reasons live in one place.
@MainActor
struct AudioRecordBar<Accessory: View>: View {
    enum Layout { case panel, capsule }

    @ObservedObject var controller: AudioWorkspaceController
    let layout: Layout
    let startTitle: Text
    /// What sits before Start while idle in the capsule, such as a source picker.
    @ViewBuilder let accessory: () -> Accessory
    @ObservedObject private var permissions = RecordingPermissions.shared
    @State private var accessRequest: RecordingAccessRequest?

    init(controller: AudioWorkspaceController, layout: Layout = .panel,
         startTitle: Text = Text("Start recording", bundle: .module),
         @ViewBuilder accessory: @escaping () -> Accessory) {
        self.controller = controller
        self.layout = layout
        self.startTitle = startTitle
        self.accessory = accessory
    }

    private var neededPermissions: [RecordingPermission] {
        RecordingPermission.needed(microphone: controller.captureMicrophone, systemAudio: controller.captureSystemAudio)
    }

    /// Why Start cannot run now, or nil. Speech-to-Text sets its sources at
    /// Start, so only the Record page can be left with none.
    private var startDisabledReason: Text? {
        if controller.isBusy { return Text("Recording can start when the current audio job finishes.", bundle: .module) }
        if controller.page != .transcribe && !controller.captureMicrophone && !controller.captureSystemAudio {
            return Text("Select a microphone or system audio source before recording.", bundle: .module)
        }
        return nil
    }

    /// Start goes through the access sheet when a selected source lacks its
    /// macOS grant, so the user decides before a draft take is created. A
    /// source the user records without is switched off here, where they see it.
    private func beginRecording() {
        controller.prepareRecordingSources()
        let needed = neededPermissions
        permissions.refresh()
        guard !permissions.missing(needed).isEmpty else { controller.startRecording(); return }
        accessRequest = RecordingAccessRequest(purpose: .start, needed: needed) { [controller] declined in
            if declined.contains(.microphone) { controller.captureMicrophone = false }
            if declined.contains(.systemAudio) { controller.captureSystemAudio = false }
            controller.startRecording()
        }
    }

    var body: some View {
        Group {
            switch layout {
            case .panel: panel
            case .capsule: capsule
            }
        }
        .recordingAccessGate($accessRequest, permissions: permissions)
    }

    private var panel: some View {
        VStack(alignment: .leading, spacing: 18) {
            if let setupStatus = controller.recordingSetupStatus {
                HStack(alignment: .top, spacing: 12) {
                    ProgressView().controlSize(.small)
                    Text(setupStatus).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 0)
                    cancelButton
                }
            } else if controller.isRecording {
                HStack {
                    AudioRecordingBadge(controller: controller)
                    Spacer()
                    markerButton
                }
                AppProgressBar(
                    value: Double(level),
                    label: Text("Input level", bundle: .module),
                    valueText: Text(verbatim: "\(Int(level * 100))%"),
                    size: .compact
                )
                HStack { pauseButton; stopButton }
            } else {
                startButton
                if let reason = startDisabledReason {
                    reason.themedFont(.small).foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
                }
            }
        }
    }

    private var capsule: some View {
        HStack(spacing: 10) {
            if let setupStatus = controller.recordingSetupStatus {
                ProgressView().controlSize(.small)
                Text(setupStatus).lineLimit(2).fixedSize(horizontal: false, vertical: true)
                Spacer(minLength: 8)
                cancelButton
            } else if controller.isRecording {
                pauseButton
                AudioLiveLevels(controller: controller, barCount: 40, height: 26)
                    .frame(minWidth: 80, maxWidth: .infinity)
                Text(verbatim: audioTime(controller.recordingSeconds)).monospacedDigit().themedFont(.small, weight: .semibold)
                    .accessibilityLabel(Text("Recording", bundle: .module))
                    .accessibilityValue(Text(verbatim: audioTime(controller.recordingSeconds)))
                markerButton
                stopButton
            } else {
                accessory()
                if let reason = startDisabledReason {
                    reason.themedFont(.small).foregroundStyle(.appSecondary).lineLimit(2)
                }
                Spacer(minLength: 8)
                startButton.appPrimaryGlow(startDisabledReason == nil)
            }
        }
        .padding(.horizontal, 14).padding(.vertical, 10)
        .appComposerSurface(focused: controller.hasRecordingActivity)
        .tsAnimation(TSMotion.pane, value: controller.isRecording)
    }

    private var level: Double { Double(max(0, min(1, controller.inputLevel))) }

    private var startButton: some View {
        Button { beginRecording() } label: {
            Label { startTitle } icon: { Image(systemName: "record.circle") }
        }
        .appAction(.primary)
        .disabled(startDisabledReason != nil)
        .help(startDisabledReason ?? startTitle)
        .keyboardShortcut("r", modifiers: [.command, .shift])
    }

    private var cancelButton: some View {
        Button { controller.cancelRecordingSetup() } label: { Text("Cancel", bundle: .module) }
            .appAction(.secondary, size: .small)
    }

    private var markerButton: some View {
        Button { controller.addMarker() } label: {
            Label { Text("Add marker", bundle: .module) } icon: { Image(systemName: "bookmark") }
        }
        .appAction(.secondary, size: .small)
    }

    private var pauseButton: some View {
        Button {
            if controller.isPaused { controller.resumeRecording() } else { controller.pauseRecording() }
        } label: {
            Label {
                controller.isPaused ? Text("Resume", bundle: .module) : Text("Pause", bundle: .module)
            } icon: { Image(systemName: controller.isPaused ? "play.fill" : "pause.fill") }
        }
        .appAction(.secondary)
    }

    // Finishing the take is the bar's one primary action while recording, as
    // Start is before it.
    private var stopButton: some View {
        Button { controller.stopRecording() } label: {
            Label { Text("Stop recording", bundle: .module) } icon: { Image(systemName: "stop.fill") }
        }
        .appAction(.primary)
    }
}

extension AudioRecordBar where Accessory == EmptyView {
    init(controller: AudioWorkspaceController, layout: Layout = .panel,
         startTitle: Text = Text("Start recording", bundle: .module)) {
        self.init(controller: controller, layout: layout, startTitle: startTitle) { EmptyView() }
    }
}

/// The microphone and app pickers, with the access check. The Record page
/// shows a form with a switch per source. A Speech-to-Text mode decides its
/// sources, so its record bar shows compact menus for the ones it records.
@MainActor
struct AudioCaptureSourcePickers: View {
    enum Style { case form, menus }

    @ObservedObject var controller: AudioWorkspaceController
    var style: Style = .form
    @ObservedObject private var permissions = RecordingPermissions.shared
    @State private var accessRequest: RecordingAccessRequest?

    private var neededPermissions: [RecordingPermission] {
        RecordingPermission.needed(microphone: controller.captureMicrophone, systemAudio: controller.captureSystemAudio)
    }

    private func reviewAccess() {
        controller.prepareRecordingSources()
        accessRequest = RecordingAccessRequest(purpose: .review, needed: neededPermissions) { [controller] _ in
            controller.prepareRecording()
        }
    }

    var body: some View {
        Group {
            switch style {
            case .form: form
            case .menus: menus
            }
        }
        .disabled(controller.hasRecordingActivity)
        .recordingAccessGate($accessRequest, permissions: permissions)
    }

    private var form: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: $controller.captureMicrophone) { Text("Microphone", bundle: .module) }
            if controller.captureMicrophone { microphonePicker }
            Toggle(isOn: $controller.captureSystemAudio) { Text("App or system audio", bundle: .module) }
            if controller.captureSystemAudio {
                applicationPicker
                Text("TurboSpark playback is excluded. Only audio is saved.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
            accessWarning
            Button { reviewAccess() } label: {
                Label { Text("Check devices and permissions", bundle: .module) } icon: { Image(systemName: "checkmark.shield") }
            }
            .appAction(.secondary, size: .small)
        }
    }

    private var menus: some View {
        HStack(spacing: 8) {
            if controller.captureMicrophone {
                microphonePicker.pickerStyle(.menu).labelsHidden().fixedSize()
                    .help(Text("Microphone", bundle: .module))
            }
            if controller.captureSystemAudio {
                applicationPicker.pickerStyle(.menu).labelsHidden().fixedSize()
                    .help(Text("TurboSpark playback is excluded. Only audio is saved.", bundle: .module))
            }
            AppIconButton(permissions.missing(neededPermissions).isEmpty ? "checkmark.shield" : "exclamationmark.triangle.fill",
                          label: Text("Check devices and permissions", bundle: .module)) { reviewAccess() }
                .appAction(.quiet, size: .small)
        }
    }

    private var microphonePicker: some View {
        Picker(selection: $controller.microphoneID) {
            Text("Default microphone", bundle: .module).tag(Optional<String>.none)
            ForEach(controller.microphones, id: \.id) { device in Text(device.name).tag(Optional(device.id)) }
        } label: { Text("Microphone", bundle: .module) }
    }

    private var applicationPicker: some View {
        Picker(selection: $controller.selectedApplicationID) {
            Text("All system audio", bundle: .module).tag(Optional<Int32>.none)
            // ScreenCaptureKit also reports processes with no name; a blank row helps no one.
            ForEach(controller.audioApplications.filter { !$0.name.isEmpty }, id: \.id) { app in Text(app.name).tag(Optional(app.id)) }
        } label: { Text("Audio source", bundle: .module) }
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
}

/// The live picture of what the capture hears: the last couple of seconds
/// of input level, scrolling toward the trailing edge. Decorative for
/// VoiceOver; a meter or timer beside it carries the value.
@MainActor
struct AudioLiveLevels: View {
    @ObservedObject var controller: AudioWorkspaceController
    var barCount = 56
    var height: CGFloat = 56

    @ObservedObject private var appearanceManager = AppearanceManager.shared
    @Environment(\.accessibilityReduceMotion) private var systemReduceMotion
    @State private var levels: [Float] = []

    private var reduceMotion: Bool {
        appearanceManager.shouldReduceMotion(systemReduceMotion: systemReduceMotion)
    }

    private var shown: [Float] { levels.isEmpty ? [Float](repeating: 0, count: barCount) : levels }

    var body: some View {
        AppLevelWaveform(levels: shown, height: height)
            // Sampling runs only while this view exists, which is only while
            // a capture runs. Reduce Motion samples slowly so the bars step
            // instead of streaming.
            .onReceive(Timer.publish(every: reduceMotion ? 0.25 : 0.06, on: .main, in: .common).autoconnect()) { _ in
                guard controller.isRecording else { return }
                levels = AppLevelWaveform.pushed(controller.inputLevel, onto: shown, count: barCount)
            }
    }
}
