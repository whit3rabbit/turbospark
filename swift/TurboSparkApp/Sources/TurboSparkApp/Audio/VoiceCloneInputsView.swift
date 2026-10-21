import SwiftUI

/// Inputs of an engine that clones from reference audio in a native session
/// (MegaTTS3, OmniVoice, Drama 3): the reference, the user's consent for that
/// exact region, and the diffusion settings, or for Drama 3 the reference's
/// transcript instead. Speed, language and the script come from the shared
/// speech editor. The Voice Cloning page lays the same parts
/// (`SpeechInputParts.swift`) out as cards; this composite serves Advanced.
@MainActor
struct VoiceCloneInputsView: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VoiceReferenceControls(controller: controller)
            VoiceCloneConsentControls(controller: controller)
            if controller.selectedProfileNeedsReferenceTranscript {
                Qwen3ReferenceTranscriptEditor(controller: controller)
            } else {
                VoiceCloneQualityControls(controller: controller)
            }
            AudioSpeechValidationLabel(controller: controller)
        }
        .disabled(controller.isBusy || controller.hasRecordingActivity)
    }
}
