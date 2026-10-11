import SwiftUI

@MainActor
struct Qwen3TTSInputsView: View {
    @ObservedObject var controller: AudioWorkspaceController

    private func setting<T>(_ path: WritableKeyPath<AudioSpeechSettings, T>) -> Binding<T> {
        Binding(get: { controller.speechSettings[keyPath: path] }, set: {
            var settings = controller.speechSettings; settings[keyPath: path] = $0; controller.speechSettings = settings
        })
    }
    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if controller.speechSettings.mode == .clone { referenceControls }
            if controller.speechSupportsStyle {
                VStack(alignment: .leading, spacing: 8) {
                    Text("Voice description or style", bundle: .module).themedFont(.small, weight: .semibold)
                    TextEditor(text: setting(\.instruction))
                        .scrollContentBackground(.hidden).frame(height: 80).padding(8)
                        .overlay { RoundedRectangle(cornerRadius: 8).stroke(.appBorder) }
                        .accessibilityLabel(Text("Voice description or style", bundle: .module))
                    Text("Describe age, accent, pitch, pace, and emotion.", bundle: .module)
                        .themedFont(.small).foregroundStyle(.appSecondary)
                }
            }
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 12) {
                    HStack {
                        Text("Temperature", bundle: .module)
                        Slider(value: setting(\.temperature), in: 0...2, step: 0.05)
                            .accessibilityLabel(Text("Temperature", bundle: .module))
                        Text(controller.speechSettings.temperature, format: .number.precision(.fractionLength(2))).monospacedDigit()
                    }
                    Stepper(value: setting(\.maxTokens), in: 64...4096, step: 64) {
                        HStack { Text("Speech token limit", bundle: .module); Text(controller.speechSettings.maxTokens, format: .number).monospacedDigit() }
                    }
                    Text("Generated audio that reaches the token limit is refused. Increase the limit or shorten the speech text.", bundle: .module)
                        .themedFont(.small).foregroundStyle(.appSecondary)
                }.padding(.top, 12)
            } label: { Text("Speech generation settings", bundle: .module) }
            if let message = controller.speechValidationMessage {
                Label { Text(message).fixedSize(horizontal: false, vertical: true) } icon: { Image(systemName: "info.circle") }
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
        }
        .disabled(controller.isBusy || controller.hasRecordingActivity)
    }

    private var referenceControls: some View {
        VStack(alignment: .leading, spacing: 12) {
            VoiceReferenceControls(controller: controller)
            Toggle(isOn: setting(\.embeddingOnly)) { Text("Use voice embedding only", bundle: .module) }
            Text("Voice embedding mode does not require a transcript. A matching transcript usually improves voice similarity.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
            if !controller.speechSettings.embeddingOnly {
                Text("Transcript of reference speech", bundle: .module).themedFont(.small, weight: .semibold)
                TextEditor(text: setting(\.referenceText))
                    .scrollContentBackground(.hidden).frame(height: 90).padding(8)
                    .overlay { RoundedRectangle(cornerRadius: 8).stroke(.appBorder) }
                    .accessibilityLabel(Text("Transcript of reference speech", bundle: .module))
            }
        }
    }
}
