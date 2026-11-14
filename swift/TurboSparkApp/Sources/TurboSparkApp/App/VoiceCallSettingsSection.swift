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
            deciderModelPicker
            if model.voiceCallAutoSendEnabled, model.session == nil,
               model.voiceCallDeciderResolution.dedicatedCandidate == nil {
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

    /// The model that answers the auto-send check. Automatic is the loaded
    /// chat model; a chosen model loads only while a call with auto-send on is
    /// open, and never downloads anything.
    @ViewBuilder
    private var deciderModelPicker: some View {
        let candidates = model.voiceCallDeciderCandidates
        let resolution = model.voiceCallDeciderResolution
        Picker(selection: Binding<String?>(
            get: { resolution.dedicatedCandidate?.installId },
            set: { model.setVoiceCallDeciderModelID($0) })
        ) {
            Text("Automatic (the loaded chat model)", bundle: .module).tag(String?.none)
            ForEach(candidates) { candidate in
                Text(verbatim: "\(candidate.alias) (\(Self.sizeText(candidate.bytes)))")
                    .tag(Optional(candidate.installId))
            }
        } label: {
            Text("Decider model", bundle: .module)
        }
        .disabled(!model.voiceCallEnabled || !model.voiceCallAutoSendEnabled)
        .settingsControl("Decider model", pane: .general, timing: .immediate)
        Text(
            "Automatic uses the loaded chat model. Or pick a small installed model: it loads only while a call with auto-send is open, unloads when the call ends or after a few idle minutes, and until it is loaded spoken text waits in the message box. Nothing is downloaded.",
            bundle: .module
        )
        .font(theme.ui(.small))
        .foregroundStyle(.appSecondary)
        .fixedSize(horizontal: false, vertical: true)
        if resolution.note == .savedModelUnavailable {
            Text(
                "The model you chose is no longer available, so the loaded chat model is used instead.",
                bundle: .module
            )
            .font(theme.ui(.small))
            .foregroundStyle(.appWarning)
            .fixedSize(horizontal: false, vertical: true)
        }
        if resolution.dedicatedCandidate != nil {
            VoiceCallDeciderStatusNote(host: model.voiceCallDeciderHost)
        }
    }

    private static func sizeText(_ bytes: UInt64) -> String {
        ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file)
    }
}

/// Observes the host so a failed load shows without reopening Settings.
@MainActor
private struct VoiceCallDeciderStatusNote: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var host: VoiceCallDeciderHost

    var body: some View {
        switch host.status {
        case .failed(.tooLarge):
            note("This model is too large to keep beside the chat model, so spoken text stays in the message box.")
        case .failed:
            note("The decider model did not load, so spoken text stays in the message box.")
        case .unloaded, .loading, .ready:
            EmptyView()
        }
    }

    private func note(_ key: LocalizedStringKey) -> some View {
        Text(key, bundle: .module)
            .font(theme.ui(.small))
            .foregroundStyle(.appWarning)
            .fixedSize(horizontal: false, vertical: true)
    }
}
