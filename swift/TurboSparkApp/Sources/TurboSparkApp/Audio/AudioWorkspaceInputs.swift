import SwiftUI
import TurboSpark

// Inputs moved out of AudioWorkspaceView unchanged, to keep that file near
// the 400-line guideline.

/// The Clean Up and Advanced pages' choice of operation.
@MainActor
struct AudioOperationPicker: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        Picker(selection: Binding(get: { controller.recipe.task }, set: { controller.selectTask($0) })) {
            if controller.page == .cleanup {
                if AudioTask.enhancement.isRunnable {
                    Text("Enhance speech", bundle: .module).tag(AudioTask.enhancement.rawValue)
                }
                if AudioTask.separation.isRunnable {
                    Text("Separate stems", bundle: .module).tag(AudioTask.separation.rawValue)
                }
            } else {
                // Only tasks the runtime can open. Offering the rest produced
                // a picker whose every choice failed at session open.
                if AudioTask.speechToText.isRunnable {
                    Text("Transcribe", bundle: .module).tag(AudioTask.speechToText.rawValue)
                }
                if AudioTask.textToSpeech.isRunnable {
                    Text("Voiceover", bundle: .module).tag(AudioTask.textToSpeech.rawValue)
                }
                if AudioTask.music.isRunnable {
                    Text("Music", bundle: .module).tag(AudioTask.music.rawValue)
                }
                if AudioTask.speechDetection.isRunnable {
                    Text("Speech detection", bundle: .module).tag(AudioTask.speechDetection.rawValue)
                }
                if AudioTask.alignment.isRunnable {
                    Text("Alignment", bundle: .module).tag(AudioTask.alignment.rawValue)
                }
                if AudioTask.diarization.isRunnable {
                    Text("Speaker diarization", bundle: .module).tag(AudioTask.diarization.rawValue)
                }
                if AudioTask.codec.isRunnable {
                    Text("Codec", bundle: .module).tag(AudioTask.codec.rawValue)
                }
                if AudioTask.languageIdentification.isRunnable {
                    Text("Language identification", bundle: .module).tag(AudioTask.languageIdentification.rawValue)
                }
            }
        } label: { Text("Operation", bundle: .module) }
        .disabled(controller.isBusy)
    }
}

/// The Music page's description, lyrics, duration and generation settings.
@MainActor
struct AudioMusicInputs: View {
    @ObservedObject var controller: AudioWorkspaceController
    let showGenerator: () -> Void

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Music description", bundle: .module).themedFont(.small, weight: .semibold)
                    Spacer()
                    Button { showGenerator() } label: {
                        Label { Text("Prompt generator", bundle: .module) } icon: { Image(systemName: "wand.and.stars") }
                    }
                    .appAction(.secondary, size: .small)
                }
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $controller.recipe.caption)
                        .scrollContentBackground(.hidden)
                        .themedFont(.base)
                        .accessibilityLabel(Text("Music description", bundle: .module))
                    if controller.recipe.caption.isEmpty {
                        Text("Describe the mood, instruments, and style", bundle: .module)
                            .foregroundStyle(.appSecondary)
                            .padding(.horizontal, 5).padding(.vertical, 8)
                            .allowsHitTesting(false).accessibilityHidden(true)
                    }
                }
                .frame(height: 110).padding(8)
                .overlay { RoundedRectangle(cornerRadius: 8).stroke(.appBorder) }
            }
            if controller.selectedProfile?.capabilities.supportsLyrics == true {
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text("Lyrics", bundle: .module).themedFont(.small, weight: .semibold)
                        Spacer()
                        if AudioWorkspaceController.effectiveLyrics(controller.recipe.lyrics) == AudioWorkspaceController.instrumentalLyrics {
                            Text("Instrumental", bundle: .module).themedFont(.small).foregroundStyle(.appSecondary)
                        }
                    }
                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $controller.recipe.lyrics)
                            .themedFont(.base)
                            .scrollContentBackground(.hidden)
                            .accessibilityLabel(Text("Lyrics", bundle: .module))
                        if controller.recipe.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text(verbatim: AudioWorkspaceController.instrumentalLyrics)
                                .foregroundStyle(.appSecondary)
                                .padding(.horizontal, 5).padding(.vertical, 8)
                                .allowsHitTesting(false).accessibilityHidden(true)
                        }
                    }
                    .frame(height: 100).padding(8)
                    .overlay { RoundedRectangle(cornerRadius: 8).stroke(.appBorder) }
                }
            }
            if controller.selectedProfile != nil {
                HStack {
                    Text("Duration (seconds)", bundle: .module)
                    TextField(value: $controller.recipe.durationSeconds, format: .number) {
                        Text("Duration (seconds)", bundle: .module)
                    }.textFieldStyle(.roundedBorder).frame(maxWidth: 120)
                }
                DisclosureGroup {
                    VStack(alignment: .leading, spacing: 10) {
                        HStack {
                            Text("Seed", bundle: .module)
                            TextField(value: $controller.recipe.seed, format: .number) { Text("Seed", bundle: .module) }
                                .textFieldStyle(.roundedBorder).frame(maxWidth: 200)
                        }
                        Stepper(value: $controller.recipe.steps, in: 1...30) {
                            HStack { Text("Steps", bundle: .module); Text(verbatim: String(controller.recipe.steps)) }
                        }
                    }.padding(.top, 8)
                } label: { Text("Generation settings", bundle: .module) }
            }
        }
    }
}
