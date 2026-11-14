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
        }
    }
}
