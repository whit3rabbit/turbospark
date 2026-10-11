import SwiftUI

/// Inputs of an engine that clones from reference audio in a native session
/// (MegaTTS3): the reference, the user's consent for that exact region, and
/// the diffusion settings. Speed, language and the script come from the
/// shared speech editor.
@MainActor
struct VoiceCloneInputsView: View {
    @ObservedObject var controller: AudioWorkspaceController

    private func option<T>(_ path: WritableKeyPath<AudioCloneOptions, T>) -> Binding<T> {
        Binding(get: { controller.cloneOptions[keyPath: path] }, set: {
            var options = controller.cloneOptions; options[keyPath: path] = $0; controller.cloneOptions = options
        })
    }

    /// A preset, or nil while the stepper holds a custom count.
    private var quality: Binding<CloneQuality?> {
        Binding(get: { CloneQuality(steps: controller.cloneOptions.steps) }, set: {
            guard let preset = $0 else { return }
            var options = controller.cloneOptions; options.steps = preset.steps; controller.cloneOptions = options
        })
    }

    private func title(_ preset: CloneQuality) -> Text {
        switch preset {
        case .fast: Text("Fast", bundle: .module)
        case .balanced: Text("Balanced", bundle: .module)
        case .best: Text("Best", bundle: .module)
        }
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            VoiceReferenceControls(controller: controller)
            Toggle(isOn: Binding(get: { controller.cloneConsentGiven }, set: { controller.setCloneConsent($0) })) {
                Text("I have the right to use this voice", bundle: .module)
            }
            .disabled(controller.speechSettings.reference?.isValid != true)
            Text("Only clone your own voice or the voice of someone who agreed. Changing the reference clears this confirmation.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
            Picker(selection: quality) {
                ForEach(CloneQuality.allCases, id: \.self) { preset in title(preset).tag(Optional(preset)) }
                if quality.wrappedValue == nil { Text("Custom", bundle: .module).tag(CloneQuality?.none) }
            } label: { Text("Quality", bundle: .module) }
                .pickerStyle(.segmented)
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 12) {
                    Stepper(value: option(\.steps), in: 1...100) {
                        HStack { Text("Steps", bundle: .module); Text(controller.cloneOptions.steps, format: .number).monospacedDigit() }
                    }
                    HStack {
                        Text("Intelligibility", bundle: .module)
                        Slider(value: option(\.intelligibility), in: 0.1...5, step: 0.1)
                            .accessibilityLabel(Text("Intelligibility", bundle: .module))
                        Text(controller.cloneOptions.intelligibility, format: .number.precision(.fractionLength(1))).monospacedDigit()
                    }
                    HStack {
                        Text("Similarity to the reference", bundle: .module)
                        Slider(value: option(\.similarity), in: 0.1...10, step: 0.1)
                            .accessibilityLabel(Text("Similarity to the reference", bundle: .module))
                        Text(controller.cloneOptions.similarity, format: .number.precision(.fractionLength(1))).monospacedDigit()
                    }
                    HStack {
                        Text("Seed", bundle: .module)
                        TextField(value: $controller.recipe.seed, format: .number) { Text("Seed", bundle: .module) }
                            .textFieldStyle(.roundedBorder).frame(maxWidth: 160)
                            .accessibilityLabel(Text("Seed", bundle: .module))
                    }
                    Text("Fewer steps are faster. Shorter references are faster too: at most the first 28 seconds are used.", bundle: .module)
                        .themedFont(.small).foregroundStyle(.appSecondary)
                        .fixedSize(horizontal: false, vertical: true)
                }.padding(.top, 12)
            } label: { Text("Cloning settings", bundle: .module) }
            if let message = controller.speechValidationMessage {
                Label { Text(message).fixedSize(horizontal: false, vertical: true) } icon: { Image(systemName: "info.circle") }
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
        }
        .disabled(controller.isBusy || controller.hasRecordingActivity)
    }
}
