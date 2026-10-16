import SwiftUI

/// The pieces of the speech engines' input views, split so the Voice Cloning
/// page can place each in its own card while the Advanced page keeps the
/// composite views (`VoiceCloneInputsView`, `Qwen3TTSInputsView`) unchanged.
extension AudioWorkspaceController {
    func speechBinding<T>(_ path: WritableKeyPath<AudioSpeechSettings, T>) -> Binding<T> {
        Binding(get: { self.speechSettings[keyPath: path] }, set: {
            var settings = self.speechSettings; settings[keyPath: path] = $0; self.speechSettings = settings
        })
    }

    func cloneOptionBinding<T>(_ path: WritableKeyPath<AudioCloneOptions, T>) -> Binding<T> {
        Binding(get: { self.cloneOptions[keyPath: path] }, set: {
            var options = self.cloneOptions; options[keyPath: path] = $0; self.cloneOptions = options
        })
    }
}

extension CloneQuality {
    var titleText: Text {
        switch self {
        case .fast: Text("Fast", bundle: .module)
        case .balanced: Text("Balanced", bundle: .module)
        case .best: Text("Best", bundle: .module)
        }
    }
}

/// The consent toggle for one exact reference region, and why it exists.
@MainActor
struct VoiceCloneConsentControls: View {
    @ObservedObject var controller: AudioWorkspaceController
    /// Keyboard focus for the checkbox, so the page can move it here.
    var focus: FocusState<Bool>.Binding?
    /// Called when the box is ticked with no valid reference. The Voice
    /// Cloning page uses it to point at the reference card, so the box stays
    /// pressable instead of sitting disabled with no explanation. Without it
    /// the box is disabled until there is a reference, as in Advanced.
    var onNeedsReference: (() -> Void)?

    private var hasReference: Bool { controller.speechSettings.reference?.isValid == true }

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            toggle
            Text("Only clone your own voice or the voice of someone who agreed. Changing the reference clears this confirmation.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }

    @ViewBuilder private var toggle: some View {
        let base = Toggle(isOn: Binding(get: { controller.cloneConsentGiven }, set: { granted in
            // Consent is for one exact clip, so there is nothing to grant yet.
            if granted && !hasReference, let onNeedsReference { onNeedsReference(); return }
            controller.setCloneConsent(granted)
        })) {
            Text("I have the right to use this voice", bundle: .module)
        }
        .disabled(!hasReference && onNeedsReference == nil)
        if let focus {
            base.focused(focus)
        } else {
            base
        }
    }
}

/// The diffusion quality preset and the detailed cloning settings.
@MainActor
struct VoiceCloneQualityControls: View {
    @ObservedObject var controller: AudioWorkspaceController
    /// False under a card that already titles the section; the picker keeps
    /// its label for VoiceOver.
    var showsPickerLabel = true

    /// OmniVoice has no guidance weights: the intelligibility and
    /// similarity sliders are MegaTTS3's, and its reference uses the
    /// full 30 seconds rather than the first 28.
    private var showsGuidanceWeights: Bool {
        controller.selectedProfile?.family != "omnivoice"
    }

    /// A preset, or nil while the stepper holds a custom count.
    private var quality: Binding<CloneQuality?> {
        Binding(get: { CloneQuality(steps: controller.cloneOptions.steps) }, set: {
            guard let preset = $0 else { return }
            var options = controller.cloneOptions; options.steps = preset.steps; controller.cloneOptions = options
        })
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            if showsPickerLabel { picker } else { picker.labelsHidden() }
            DisclosureGroup {
                VStack(alignment: .leading, spacing: 12) {
                    Stepper(value: controller.cloneOptionBinding(\.steps), in: 1...100) {
                        HStack { Text("Steps", bundle: .module); Text(controller.cloneOptions.steps, format: .number).monospacedDigit() }
                    }
                    if showsGuidanceWeights {
                        HStack {
                            Text("Intelligibility", bundle: .module)
                            Slider(value: controller.cloneOptionBinding(\.intelligibility), in: 0.1...5, step: 0.1)
                                .accessibilityLabel(Text("Intelligibility", bundle: .module))
                            Text(controller.cloneOptions.intelligibility, format: .number.precision(.fractionLength(1))).monospacedDigit()
                        }
                        HStack {
                            Text("Similarity to the reference", bundle: .module)
                            Slider(value: controller.cloneOptionBinding(\.similarity), in: 0.1...10, step: 0.1)
                                .accessibilityLabel(Text("Similarity to the reference", bundle: .module))
                            Text(controller.cloneOptions.similarity, format: .number.precision(.fractionLength(1))).monospacedDigit()
                        }
                    }
                    HStack {
                        Text("Seed", bundle: .module)
                        TextField(value: $controller.recipe.seed, format: .number) { Text("Seed", bundle: .module) }
                            .textFieldStyle(.roundedBorder).frame(maxWidth: 160)
                            .accessibilityLabel(Text("Seed", bundle: .module))
                    }
                    if showsGuidanceWeights {
                        Text("Fewer steps are faster. Shorter references are faster too: at most the first 28 seconds are used.", bundle: .module)
                            .themedFont(.small).foregroundStyle(.appSecondary)
                            .fixedSize(horizontal: false, vertical: true)
                    }
                }.padding(.top, 12)
            } label: { Text("Cloning settings", bundle: .module) }
        }
    }

    private var picker: some View {
        Picker(selection: quality) {
            ForEach(CloneQuality.allCases, id: \.self) { preset in preset.titleText.tag(Optional(preset)) }
            if quality.wrappedValue == nil { Text("Custom", bundle: .module).tag(CloneQuality?.none) }
        } label: { Text("Quality", bundle: .module) }
            .pickerStyle(.segmented)
    }
}

/// Qwen3's choice between a transcript-conditioned clone and the speaker
/// embedding alone.
@MainActor
struct Qwen3EmbeddingOnlyToggle: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        VStack(alignment: .leading, spacing: 8) {
            Toggle(isOn: controller.speechBinding(\.embeddingOnly)) { Text("Use voice embedding only", bundle: .module) }
            Text("Voice embedding mode does not require a transcript. A matching transcript usually improves voice similarity.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
                .fixedSize(horizontal: false, vertical: true)
        }
    }
}

/// The transcript editor of the Advanced page's Qwen3 and Drama 3 clone
/// inputs.
@MainActor
struct Qwen3ReferenceTranscriptEditor: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        Text("Transcript of reference speech", bundle: .module).themedFont(.small, weight: .semibold)
        TextEditor(text: controller.speechBinding(\.referenceText))
            .scrollContentBackground(.hidden).frame(height: 90).padding(8)
            .overlay { RoundedRectangle(cornerRadius: 8).stroke(.appBorder) }
            .accessibilityLabel(Text("Transcript of reference speech", bundle: .module))
    }
}

/// Qwen3's sampling temperature and token limit.
@MainActor
struct Qwen3SpeechGenerationSettings: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text("Temperature", bundle: .module)
                    Slider(value: controller.speechBinding(\.temperature), in: 0...2, step: 0.05)
                        .accessibilityLabel(Text("Temperature", bundle: .module))
                    Text(controller.speechSettings.temperature, format: .number.precision(.fractionLength(2))).monospacedDigit()
                }
                Stepper(value: controller.speechBinding(\.maxTokens), in: 64...4096, step: 64) {
                    HStack { Text("Speech token limit", bundle: .module); Text(controller.speechSettings.maxTokens, format: .number).monospacedDigit() }
                }
                Text("Generated audio that reaches the token limit is refused. Increase the limit or shorten the speech text.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }.padding(.top, 12)
        } label: { Text("Speech generation settings", bundle: .module) }
    }
}

/// The engine's reason the inputs cannot run yet. The message is already a
/// localized `String`, so it renders as-is.
@MainActor
struct AudioSpeechValidationLabel: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        if let message = controller.speechValidationMessage {
            Label { Text(message).fixedSize(horizontal: false, vertical: true) } icon: { Image(systemName: "info.circle") }
                .themedFont(.small).foregroundStyle(.appSecondary)
        }
    }
}
