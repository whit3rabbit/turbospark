import SwiftUI

/// OC-08: mounts the call panel over the conversation pane and owns the one
/// path that starts a call, so the Call button and the menu item share it.
///
/// Starting goes through `.recordingAccessGate`: when the microphone is
/// already allowed the call starts at once, otherwise the access sheet
/// explains what the access is for and the macOS prompt appears only after
/// the user chooses Allow there. Not now leaves the call idle and every
/// other feature (typing, dictation) untouched.
@MainActor
struct VoiceCallHostView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var call: VoiceCallController
    @ObservedObject private var permissions = RecordingPermissions.shared
    @State private var accessRequest: RecordingAccessRequest?

    private var showsPanel: Bool { call.state.isActive || call.refusal != nil }

    var body: some View {
        ZStack(alignment: .bottomTrailing) {
            if showsPanel {
                VoiceCallPanelView(
                    call: call, onDownloadModel: { model.downloadVoiceCallSpeechModel() },
                    showsTalk: model.voiceCallEnabled && model.voiceCallAutoSendEnabled)
                    // Clear of the composer by default; the user can drag it.
                    .padding(.trailing, 20)
                    .padding(.bottom, 190)
                    .transition(.opacity)
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .bottomTrailing)
        .allowsHitTesting(showsPanel)
        .recordingAccessGate($accessRequest, permissions: permissions)
        .onReceive(NotificationCenter.default.publisher(for: .toggleVoiceCall)) { _ in toggle() }
        .onReceive(NotificationCenter.default.publisher(for: .toggleVoiceTalk)) { _ in call.talk.toggle() }
        .onChange(of: permissions.state(.microphone)) { _, state in
            call.microphoneAccessChanged(allowed: state == .granted)
        }
    }

    private func toggle() {
        if call.state.isActive { call.end(); return }
        call.dismissRefusal()
        permissions.refresh()
        let needed: [RecordingPermission] = [.microphone]
        guard !permissions.missing(needed).isEmpty else {
            Task { await call.start() }
            return
        }
        accessRequest = RecordingAccessRequest(purpose: .call, needed: needed) { [call] _ in
            Task { await call.start() }
        }
    }
}

/// The Call control beside the composer's dictation button. Shown only while
/// voice call mode is on. It posts the same request the menu item does.
@MainActor
struct VoiceCallButton: View {
    @ObservedObject var model: AppModel
    @Environment(\.appTheme) private var theme
    @State private var isHovered = false

    var body: some View {
        let active = model.voiceCallActive
        let size = (28 * AppMetrics.textScale(theme)).rounded()
        Button {
            NotificationCenter.default.post(name: .toggleVoiceCall, object: nil)
        } label: {
            Image(systemName: active ? "phone.down.fill" : "phone")
                .themedFont(.callout, weight: .medium)
                .foregroundStyle(active ? .appAccentText : .appSecondary)
                .frame(width: size, height: size)
                .background(isHovered ? AnyShapeStyle(.appElevated) : AnyShapeStyle(Color.clear), in: Circle())
                .contentShape(Circle())
        }
        .buttonStyle(.plain)
        .onHover { isHovered = $0 }
        .help(active ? Text("End Voice Call", bundle: .module) : Text("Start Voice Call", bundle: .module))
        .accessibilityLabel(Text("Voice call", bundle: .module))
        .accessibilityValue(active ? Text("End Voice Call", bundle: .module) : Text("Start Voice Call", bundle: .module))
    }
}
