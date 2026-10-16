import SwiftUI

/// Speech controls shared by the Audio workspace's generic speech editor and
/// the Voice Cloning composer. Moved out of `AudioWorkspaceView` unchanged.
@MainActor
struct AudioLanguagePicker: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        if let profile = controller.selectedProfile, !profile.capabilities.languages.isEmpty {
            Picker(selection: $controller.recipe.language) {
                if controller.task == .speechToText || profile.capabilities.languages.contains("auto") {
                    Text("Detect automatically", bundle: .module).tag("auto")
                }
                ForEach(profile.capabilities.languages.filter { $0 != "auto" }, id: \.self) { language in
                    Text(Locale.current.localizedString(forLanguageCode: language) ?? language).tag(language)
                }
            } label: { Text("Language", bundle: .module) }
        }
    }
}

@MainActor
struct AudioVoicePicker: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        if let profile = controller.selectedProfile, !profile.capabilities.voices.isEmpty {
            Picker(selection: $controller.recipe.voice) {
                ForEach(profile.capabilities.voices, id: \.self) { voice in Text(voice).tag(voice) }
            } label: { Text("Voice", bundle: .module) }
        }
    }
}

/// Speech speed. Qwen3-TTS has no speed control.
@MainActor
struct AudioSpeedControl: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        if controller.selectedProfile != nil, controller.selectedProfile?.family != "qwen3_tts" {
            HStack {
                Text("Speed", bundle: .module)
                Slider(value: $controller.recipe.speed, in: 0.5...2, step: 0.05)
                    .accessibilityLabel(Text("Speech speed", bundle: .module))
                Text(controller.recipe.speed, format: .number.precision(.fractionLength(2)))
                    .monospacedDigit()
            }
        }
    }
}

/// Sections of the script, split on blank lines, each generated on its own.
@MainActor
struct AudioScriptSections: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        let sections = controller.recipe.text.components(separatedBy: "\n\n")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        if !sections.isEmpty {
            DisclosureGroup {
                ForEach(Array(sections.enumerated()), id: \.offset) { index, text in
                    HStack(alignment: .top, spacing: 12) {
                        Text(verbatim: String(index + 1)).monospacedDigit().foregroundStyle(.appSecondary)
                        Text(text).lineLimit(4).frame(maxWidth: .infinity, alignment: .leading)
                        Button { controller.runSection(text) } label: {
                            Text("Generate section", bundle: .module)
                        }
                        .disabled(!controller.canRun)
                    }
                    .padding(.vertical, 8)
                }
            } label: { Text("Script sections", bundle: .module) }
        }
    }
}
