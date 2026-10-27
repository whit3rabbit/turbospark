import SwiftUI

@MainActor
struct Qwen3TTSInputsView: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if controller.speechSettings.mode == .clone { referenceControls }
            if controller.speechSupportsStyle {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Voice description or style", bundle: .module)
                        .themedFont(.small, weight: .semibold)
                        .accessibilityAddTraits(.isHeader)
                    TextEditor(text: controller.speechBinding(\.instruction))
                        .themedFont(.base)
                        .appEditorSurface(minHeight: 80)
                        .accessibilityLabel(Text("Voice description or style", bundle: .module))
                    Text("Describe age, accent, pitch, pace, and emotion.", bundle: .module)
                        .themedFont(.small).foregroundStyle(.appSecondary)
                }
            }
            Qwen3SpeechGenerationSettings(controller: controller)
            AudioSpeechValidationLabel(controller: controller)
        }
        .disabled(controller.isBusy || controller.hasRecordingActivity)
    }

    private var referenceControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            VoiceReferenceControls(controller: controller)
            Qwen3EmbeddingOnlyToggle(controller: controller)
            if !controller.speechSettings.embeddingOnly {
                Qwen3ReferenceTranscriptEditor(controller: controller)
            }
        }
    }
}
