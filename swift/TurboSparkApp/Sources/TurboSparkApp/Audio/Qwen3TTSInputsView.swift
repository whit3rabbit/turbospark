import SwiftUI

@MainActor
struct Qwen3TTSInputsView: View {
    @ObservedObject var controller: AudioWorkspaceController

    private func setting<T>(_ path: WritableKeyPath<AudioSpeechSettings, T>) -> Binding<T> {
        Binding(get: { controller.speechSettings[keyPath: path] }, set: {
            var settings = controller.speechSettings; settings[keyPath: path] = $0; controller.speechSettings = settings
        })
    }
    private func trim(_ path: WritableKeyPath<AudioVoiceReference, Double>) -> Binding<Double> {
        Binding(get: { controller.speechSettings.reference?[keyPath: path] ?? 0 }, set: {
            controller.stopReferencePreview()
            controller.speechSettings.reference?[keyPath: path] = $0
            // A transcript of a different region is invalid conditioning.
            controller.speechSettings.referenceText = ""
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
            Text("Reference voice", bundle: .module).themedFont(.base, weight: .semibold)
            HStack {
                Button { controller.importAudio(asVoiceReference: true) } label: {
                    Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
                }.disabled(controller.importTask != nil)
                Menu {
                    ForEach(controller.items.filter { $0.hasPlayableAudio && $0.status == .completed }) { item in
                        Button { controller.setVoiceReference(item) } label: { Text(item.title) }
                    }
                } label: { Text("Saved audio", bundle: .module) }
                Button { controller.selectPage(.record) } label: { Text("Record", bundle: .module) }
            }
            Text("Import MP3, WAV, M4A, FLAC, AIFF, or other audio formats supported by macOS.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
            if let reference = controller.speechSettings.reference {
                HStack {
                    Image(systemName: "waveform").accessibilityHidden(true)
                    Text(reference.title).lineLimit(2)
                    Spacer()
                    Button { controller.previewVoiceReference() } label: {
                        if controller.referencePlaying { Text("Stop preview", bundle: .module) }
                        else { Text("Preview reference audio", bundle: .module) }
                    }.disabled(!reference.isValid)
                    Button { controller.clearVoiceReference() } label: { Text("Remove", bundle: .module) }
                }
                HStack {
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Start (seconds)", bundle: .module).themedFont(.small)
                        TextField(value: trim(\.start), format: .number) { Text("Start (seconds)", bundle: .module) }
                            .accessibilityLabel(Text("Start (seconds)", bundle: .module))
                    }
                    VStack(alignment: .leading, spacing: 6) {
                        Text("Duration (seconds)", bundle: .module).themedFont(.small)
                        TextField(value: trim(\.duration), format: .number) { Text("Duration (seconds)", bundle: .module) }
                            .accessibilityLabel(Text("Duration (seconds)", bundle: .module))
                    }
                }.textFieldStyle(.roundedBorder)
                Text("Use a clear recording of one person speaking for 1 to 30 seconds.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
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
