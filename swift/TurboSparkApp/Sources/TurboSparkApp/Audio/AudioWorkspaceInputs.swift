import SwiftUI
import TurboSpark

// Inputs moved out of AudioWorkspaceView unchanged, to keep that file near
// the 400-line guideline.

/// The Clean Up and Advanced pages' choice of operation.
@MainActor
struct AudioOperationPicker: View {
    @ObservedObject var controller: AudioWorkspaceController

    var body: some View {
        Picker(selection: Binding(get: { controller.task.rawValue }, set: { controller.selectTask($0) })) {
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
    @Environment(\.appTheme) private var theme
    @ObservedObject var controller: AudioWorkspaceController
    let showGenerator: () -> Void
    @State private var showingLyrics = false
    @FocusState private var descriptionFocused: Bool
    @FocusState private var lyricsFocused: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                AppAdaptiveHeader(
                    leading: {
                        Text("Music description", bundle: .module)
                            .themedFont(.small, weight: .semibold)
                            .accessibilityAddTraits(.isHeader)
                    },
                    actions: {
                        Button { showGenerator() } label: {
                            Label { Text("Prompt generator", bundle: .module) } icon: { Image(systemName: "wand.and.stars") }
                        }
                        .appAction(.quiet, size: .small)
                    }
                )
                ZStack(alignment: .topLeading) {
                    TextEditor(text: $controller.recipe.caption)
                        .themedFont(.base)
                        .appEditorSurface(minHeight: 110 * AppMetrics.textScale(theme))
                        .focused($descriptionFocused)
                        .appFocusGlow(descriptionFocused)
                        .accessibilityLabel(Text("Music description", bundle: .module))
                    if controller.recipe.caption.isEmpty {
                        Text("Describe the mood, instruments, and style", bundle: .module)
                            .foregroundStyle(.appSecondary)
                            .padding(.horizontal, 13).padding(.vertical, 16)
                            .allowsHitTesting(false).accessibilityHidden(true)
                    }
                }
            }
            if controller.selectedProfile?.capabilities.supportsLyrics == true {
                DisclosureGroup(isExpanded: $showingLyrics) {
                    ZStack(alignment: .topLeading) {
                        TextEditor(text: $controller.recipe.lyrics)
                            .themedFont(.base)
                            .appEditorSurface(minHeight: 100 * AppMetrics.textScale(theme))
                            .focused($lyricsFocused)
                            .appFocusGlow(lyricsFocused)
                            .accessibilityLabel(Text("Lyrics", bundle: .module))
                        if controller.recipe.lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                            Text(verbatim: AudioWorkspaceController.instrumentalLyrics)
                                .foregroundStyle(.appSecondary)
                                .padding(.horizontal, 13).padding(.vertical, 16)
                                .allowsHitTesting(false).accessibilityHidden(true)
                        }
                    }
                    .padding(.top, 8)
                } label: {
                    HStack(spacing: 8) {
                        Text("Lyrics", bundle: .module)
                        if !hasLyrics {
                            Text("Instrumental", bundle: .module)
                                .foregroundStyle(.appSecondary)
                        }
                    }
                    .themedFont(.small)
                }
            }
            let advice = controller.musicMemoryAdvice
            if controller.page == .music {
                durations(upTo: advice.maxSupportedSeconds)
            } else if controller.selectedProfile != nil {
                durationField
            }

            if controller.selectedProfile != nil {
                DisclosureGroup {
                    VStack(alignment: .leading, spacing: 10) {
                        Label {
                            Text(verbatim: advice.tierSummary).fixedSize(horizontal: false, vertical: true)
                        } icon: {
                            Image(systemName: "memorychip").foregroundStyle(.appAccentText)
                        }
                        .themedFont(.small).foregroundStyle(.appSecondary)
                        if advice.canSupportMaxContext {
                            Button {
                                controller.setMusicContextToMaxIfSupported()
                            } label: {
                                Text("Set to max (360s)", bundle: .module)
                            }
                            .appAction(.quiet, size: .small)
                            .help(Text("Sets duration to maximum context limit (9,000 frames / 360s) supported by your hardware.", bundle: .module))
                        }
                        framesField(frameRate: advice.frameRate)
                        HStack {
                            Text("Context size (tokens)", bundle: .module)
                            TextField(
                                value: Binding(
                                    get: { controller.recipe.music?.contextSize ?? advice.maxModelContextTokens },
                                    set: { val in
                                        if controller.recipe.music == nil {
                                            controller.recipe.music = AudioMusicSettings(contextSize: val)
                                        } else {
                                            controller.recipe.music?.contextSize = val
                                        }
                                    }
                                ),
                                format: .number
                            ) {
                                Text("Context size (tokens)", bundle: .module)
                            }
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 120)
                            Text(verbatim: "/ \(advice.maxModelContextTokens)")
                                .themedFont(.small)
                                .foregroundStyle(.appSecondary)
                        }
                        HStack {
                            Text("Max frames cap", bundle: .module)
                            TextField(
                                value: Binding(
                                    get: { controller.recipe.music?.maxFrames ?? advice.maxSupportedFrames },
                                    set: { val in
                                        if controller.recipe.music == nil {
                                            controller.recipe.music = AudioMusicSettings(maxFrames: val)
                                        } else {
                                            controller.recipe.music?.maxFrames = val
                                        }
                                    }
                                ),
                                format: .number
                            ) {
                                Text("Max frames cap", bundle: .module)
                            }
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 120)
                            Text(verbatim: "/ \(advice.maxModelFrames)")
                                .themedFont(.small)
                                .foregroundStyle(.appSecondary)
                        }
                        HStack {
                            Text("Seed", bundle: .module)
                            TextField(value: $controller.recipe.seed, format: .number) { Text("Seed", bundle: .module) }
                                .textFieldStyle(.roundedBorder).frame(maxWidth: 200)
                        }
                        let recommendation = controller.musicSamplingRecommendation
                        Stepper(value: Binding(
                            get: { controller.recipe.effectiveMusicStepsOverride ?? recommendation.steps },
                            set: { controller.recipe.musicStepsOverride = $0 }
                        ), in: 1...30) {
                            HStack {
                                Text(verbatim: recommendation.profile)
                                Text("Steps", bundle: .module)
                                Text(verbatim: String(controller.recipe.effectiveMusicStepsOverride ?? recommendation.steps))
                            }
                        }
                        if controller.recipe.effectiveMusicStepsOverride != nil {
                            Button {
                                controller.recipe.musicStepsOverride = nil
                                controller.recipe.steps = 30
                            } label: {
                                Text("Use Default", bundle: .module)
                            }
                            .appAction(.secondary, size: .small)
                        }
                    }.padding(.top, 8)
                } label: { Text("Generation settings", bundle: .module) }
            }
        }
        .onAppear { showingLyrics = hasLyrics }
        .onChange(of: controller.recipe.lyrics) { _, _ in
            if hasLyrics { showingLyrics = true }
        }
    }

    private var hasLyrics: Bool {
        AudioWorkspaceController.effectiveLyrics(controller.recipe.lyrics) != AudioWorkspaceController.instrumentalLyrics
    }

    private func durations(upTo limit: Double) -> some View {
        let choices = [Double(15), 30, 60].filter { $0 <= limit }
        return FlowLayout(spacing: 12, lineSpacing: 8) {
            Text("Duration (seconds)", bundle: .module)
                .themedFont(.small, weight: .medium)
            durationEntry
            if !choices.isEmpty {
                AppPillPicker(
                    selection: $controller.recipe.durationSeconds,
                    items: choices.map { seconds in
                        AppPillItem(
                            value: seconds,
                            label: Text(verbatim: Duration.seconds(seconds).formatted(
                                .units(allowed: [.seconds], width: .abbreviated)
                            ))
                        )
                    },
                    label: Text("Duration (seconds)", bundle: .module)
                )
            }
        }
    }

    private var durationField: some View {
        HStack {
            Text("Duration (seconds)", bundle: .module)
            durationEntry
        }
    }

    private var durationEntry: some View {
        TextField(value: $controller.recipe.durationSeconds, format: .number) {
            Text("Duration (seconds)", bundle: .module)
        }
        .textFieldStyle(.roundedBorder)
        .frame(width: 76 * AppMetrics.textScale(theme))
    }

    private func framesField(frameRate: Double) -> some View {
        HStack {
            Text("Frames", bundle: .module)
            TextField(
                value: Binding(
                    get: { controller.recipe.musicFrames },
                    set: { newFrames in
                        let rate = controller.recipe.music?.frameRate ?? frameRate
                        controller.recipe.durationSeconds = Double(newFrames) / rate
                    }
                ),
                format: .number
            ) {
                Text("Frames", bundle: .module)
            }
            .textFieldStyle(.roundedBorder)
            .frame(maxWidth: 100)
            Text(verbatim: "@ \(Int(frameRate)) fps")
                .themedFont(.small)
                .foregroundStyle(.appSecondary)
        }
    }
}
