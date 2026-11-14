import SwiftUI

/// OC-08: the one setting for voice call mode. Off by default. Turning it on
/// adds the Call button beside the composer and a menu item; it asks for
/// nothing from macOS. The microphone prompt appears only when the user
/// presses Call and chooses Allow.
@MainActor
struct VoiceCallSettingsSection: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel

    var body: some View {
        Section(header: Text("Voice call", bundle: .module)) {
            Toggle(isOn: Binding(
                get: { model.voiceCallEnabled },
                set: { model.setVoiceCallEnabled($0) })
            ) {
                Text("Voice call mode", bundle: .module)
            }
            .settingsControl("Voice call mode", pane: .general, timing: .immediate)
            Text(
                "Adds a Call button beside the message box. Speak, review the text it writes, press Send, and the reply is read aloud with the system voice. Everything runs on this Mac and no audio is saved. The microphone is used only while a call is open.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)

            Toggle(isOn: Binding(
                get: { model.voiceCallAutoSendEnabled },
                set: { model.setVoiceCallAutoSendEnabled($0) })
            ) {
                Text("Auto-send voice utterances", bundle: .module)
            }
            .disabled(!model.voiceCallEnabled)
            .settingsControl("Auto-send voice utterances", pane: .general, timing: .immediate)
            Text(
                "Off by default. When on, the loaded model on this Mac checks each finished sentence and sends it only if it is clearly a complete request to the assistant. Anything that mentions running, deleting, sending, paying or changing settings, and anything unclear, still waits in the message box for you to press Send. Nothing leaves this Mac.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appSecondary)
            .fixedSize(horizontal: false, vertical: true)
            if model.voiceCallAutoSendEnabled, model.session == nil {
                Text(
                    "No model is loaded, so spoken text stays in the message box until you load one.",
                    bundle: .module
                )
                .font(theme.ui(.small))
                .foregroundStyle(.appWarning)
                .fixedSize(horizontal: false, vertical: true)
            }
        }
    }
}
