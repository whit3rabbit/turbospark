import SwiftUI
import TurboSpark

@MainActor
struct AudioWorkspaceView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @State private var presetName = ""
    @State private var showingPreset = false
    @State private var showingPrompts = false
    @State private var savingPrompt = false
    @State private var showingGenerator = false
    // Held here, not in the sheet, so reopening the generator keeps the choices.
    @State private var promptSelection = MusicPromptSelection()

    var body: some View {
        VStack(spacing: 0) {
            navigation
            Divider()
            if controller.page == .library {
                AudioLibraryView(controller: controller)
            } else if controller.page == .voiceClone {
                VoiceCloneWorkspaceView(
                    model: model, controller: controller,
                    description: Text(LocalizedStringKey(pageDescription), bundle: .module),
                    showPromptLibrary: { showingPrompts = true }, savePrompt: beginSavingPrompt)
            } else {
                GeometryReader { geometry in
                    ScrollView {
                        if geometry.size.width >= 940 && controller.page != .record {
                            HStack(alignment: .top, spacing: 24) {
                                workflow.frame(maxWidth: .infinity)
                                modelPanel.frame(width: 300)
                            }
                            .frame(maxWidth: 1440)
                            .frame(maxWidth: .infinity)
                            .padding(24)
                        } else {
                            VStack(alignment: .leading, spacing: 20) {
                                if controller.page != .record { compactModelPanel }
                                workflow
                            }
                            .padding(24)
                        }
                    }
                }
            }
            AudioWorkspaceStatusView(controller: controller)
        }
        .background(.appPage)
        .themedFont(.base)
        .sheet(isPresented: $showingPreset) {
            AudioNameSheet(title: savingPrompt ? "Save Prompt" : "Save preset", value: $presetName) {
                if savingPrompt {
                    if controller.savePrompt(name: presetName) { showingPreset = false }
                } else {
                    controller.savePreset(name: presetName)
                    showingPreset = false
                }
            }
        }
        .sheet(isPresented: $showingPrompts) { AudioPromptLibraryView(controller: controller) }
        .sheet(isPresented: $showingGenerator) {
            MusicPromptGeneratorView(controller: controller, selection: $promptSelection)
        }
    }

    private var navigation: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 16) { workspaceTitle; Spacer(minLength: 12); workspaceActions }
            VStack(alignment: .leading, spacing: 12) { workspaceTitle; workspaceActions }
        }
        .padding(.horizontal, 24).padding(.vertical, 18)
    }

    private var workspaceTitle: some View {
        HStack(spacing: 12) {
            Image(systemName: controller.page.symbol)
                .themedFont(.title2).foregroundStyle(.appAccent)
                .accessibilityHidden(true)
            Text(LocalizedStringKey(controller.page.title), bundle: .module)
                .themedFont(.title2, weight: .semibold).accessibilityAddTraits(.isHeader)
        }
    }

    private var workspaceActions: some View {
        HStack(spacing: 12) {
            if controller.page != .library {
                Button { controller.selectPage(.library) } label: {
                    Label { Text("Audio library", bundle: .module) } icon: { Image(systemName: "books.vertical") }
                }
                .appAction(.secondary)
            }
            Menu { AudioDestinationMenu(model: model) } label: {
                Label { Text("Audio tasks", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
            }
            .appMenuAction()
        }
        .themedFont(.small)
    }

    private var workflow: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text(LocalizedStringKey(pageDescription), bundle: .module)
                .foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
            if controller.page == .record { AudioRecordingPanel(model: model, controller: controller) }
            else { modelAndInputs }
            if let item = controller.workspaceResult, item.status != .processing {
                AudioItemDetailView(controller: controller, item: item)
            } else if !controller.isBusy && AudioWorkspacePage.destinations.contains(controller.page) {
                VStack(spacing: 10) {
                    Image(systemName: controller.page == .transcribe ? "text.alignleft" : "waveform")
                        .themedFont(.title).foregroundStyle(.appSecondary)
                    Text("No outputs yet", bundle: .module).themedFont(.base, weight: .medium)
                    Text("Audio and edits are saved in your encrypted profile.", bundle: .module)
                        .themedFont(.small).foregroundStyle(.appSecondary).multilineTextAlignment(.center)
                }
                .frame(maxWidth: .infinity).padding(28)
                .overlay { RoundedRectangle(cornerRadius: 12).stroke(.appBorder, style: StrokeStyle(lineWidth: 1, dash: [5])) }
            }
            if !controller.workspaceHistory.isEmpty {
                VStack(alignment: .leading, spacing: 12) {
                    Text("Saved audio", bundle: .module).themedFont(.base, weight: .semibold)
                    ForEach(controller.workspaceHistory.prefix(5)) { item in
                        HStack(spacing: 12) {
                            Button { controller.select(item) } label: {
                                HStack(spacing: 12) {
                                    Image(systemName: controller.page.symbol).foregroundStyle(.appAccent)
                                    VStack(alignment: .leading, spacing: 4) {
                                        Text(item.title).lineLimit(1)
                                        Text(item.createdAt, format: .dateTime.month(.abbreviated).day().hour().minute())
                                            .themedFont(.tiny).foregroundStyle(.appSecondary)
                                    }
                                    Spacer(minLength: 8)
                                    Text(audioTime(item.duration)).monospacedDigit().foregroundStyle(.appSecondary)
                                }
                                .contentShape(Rectangle())
                            }.buttonStyle(.plain)
                            AudioSaveAsButton(controller: controller, item: item, compact: true)
                            AudioDeleteButton(controller: controller, item: item, compact: true)
                        }
                        .padding(12).background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
                    }
                }
            }
        }
    }

    private var modelPanel: some View {
        AudioModelPickerView(model: model, controller: controller)
            .padding(18)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
            .overlay { RoundedRectangle(cornerRadius: 12).stroke(.appBorder, lineWidth: 1) }
    }

    private var compactModelPanel: some View {
        DisclosureGroup {
            AudioModelPickerView(model: model, controller: controller).padding(.top, 12)
        } label: {
            HStack(spacing: 12) {
                Label { Text("Audio model", bundle: .module) } icon: { Image(systemName: "cpu") }
                Spacer(minLength: 8)
                if let profile = controller.selectedProfile {
                    Text(profile.displayName).lineLimit(1).foregroundStyle(.appSecondary)
                } else {
                    Text("Choose a model", bundle: .module).foregroundStyle(.appSecondary)
                }
            }.themedFont(.small)
        }
        .padding(16)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).stroke(.appBorder, lineWidth: 1) }
    }

    private var pageDescription: String {
        switch controller.page {
        case .library: return "Reopen recordings and takes, or start with an audio file."
        case .record: return "Record a meeting, add markers, and keep the original audio."
        case .transcribe: return "Turn a recording into editable text and subtitles."
        case .voiceover: return "Write a script in sections. Preview or regenerate each section as a new take."
        case .voiceClone: return "Choose a reference voice, review its transcript, and generate new speech."
        case .voiceDesign: return "Describe a voice and generate an audio clip. Reuse the clip as a cloning reference for a consistent voice."
        case .music: return "Describe a soundtrack and keep each take for comparison."
        case .cleanup: return "Create a cleaned recording or separate stems while keeping the original."
        case .advanced: return "Explore supported audio operations and inspect model details."
        }
    }

    private var modelAndInputs: some View {
        VStack(alignment: .leading, spacing: 18) {
            if controller.page == .cleanup || controller.page == .advanced { operationPicker }
            if controller.page == .advanced {
                Toggle(isOn: $controller.allowPortableExperiment) {
                    Text("Allow portable model experiments", bundle: .module)
                }
                Text("Portable experiments run on the CPU and have not passed production quality or performance gates.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
                if controller.selectedProfile?.family == "qwen3_asr" {
                    Toggle(isOn: $controller.allowExperimentalMetal) {
                        Text("Use experimental Metal decoder", bundle: .module)
                    }.disabled(controller.isBusy)
                }
            }
            if controller.page == .advanced { familyDetails }
            if controller.page == .transcribe || controller.page == .cleanup
                || (controller.page == .advanced && controller.task != .textToSpeech && controller.task != .music) {
                sourcePicker
            }
            switch controller.page {
            case .voiceover, .voiceClone, .voiceDesign: voiceoverInputs
            case .music: musicInputs
            case .transcribe: languagePicker
            case .advanced:
                if controller.task == .speechToText { languagePicker }
                if controller.task == .alignment { scriptEditor }
                if controller.task == .textToSpeech { voiceoverInputs }
                if controller.task == .music { musicInputs }
            default: EmptyView()
            }
            experimentActions
        }
        .padding(20)
        .background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).stroke(.appBorder, lineWidth: 1) }
    }

    private var operationPicker: some View {
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

    private var sourcePicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            ViewThatFits(in: .horizontal) {
                HStack { importButton; recordButton; savedSourceMenu }
                VStack(alignment: .leading) { importButton; recordButton; savedSourceMenu }
            }
            if let item = controller.selectedItem, !item.clips.isEmpty {
                HStack {
                    Image(systemName: "waveform")
                    Text(item.title).lineLimit(2)
                    Spacer()
                    Text(audioTime(item.duration)).monospacedDigit()
                }
                .foregroundStyle(.appSecondary)
            } else {
                Text("Choose a saved recording or import audio to begin.", bundle: .module)
                    .foregroundStyle(.appSecondary)
            }
        }
    }

    private var recordButton: some View {
        Button { controller.selectPage(.record) } label: {
            Label { Text("Record", bundle: .module) } icon: { Image(systemName: "mic") }
        }
        .appAction(.secondary, size: .small)
        .disabled(controller.isBusy)
    }

    private var savedSourceMenu: some View {
        Menu {
            ForEach(controller.items.filter { !$0.clips.isEmpty }) { item in
                Button { controller.select(item) } label: { Text(item.title) }
            }
        } label: {
            Label { Text("Saved audio", bundle: .module) } icon: { Image(systemName: "books.vertical") }
        }
        .appMenuAction(size: .small)
        .disabled(controller.isBusy || controller.hasRecordingActivity || !controller.items.contains { !$0.clips.isEmpty })
    }

    private var importButton: some View {
        Button { controller.importAudio() } label: {
            Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
        }
        .appAction(.secondary, size: .small)
        .disabled(controller.isBusy || controller.hasRecordingActivity)
    }

    private var voiceoverInputs: some View {
        VStack(alignment: .leading, spacing: 14) {
            scriptEditor
            Text("Separate sections with a blank line.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
            if controller.selectedProfile?.family == "qwen3_tts" {
                Qwen3TTSInputsView(controller: controller)
            } else if controller.selectedProfileClonesNatively {
                VoiceCloneInputsView(controller: controller)
            }
            AudioVoicePicker(controller: controller)
            languagePicker
            AudioSpeedControl(controller: controller)
            AudioScriptSections(controller: controller)
        }
    }

    private var scriptEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Script", bundle: .module).themedFont(.small, weight: .semibold)
            TextEditor(text: $controller.recipe.text)
                .themedFont(.base).frame(minHeight: 180)
                .scrollContentBackground(.hidden)
                .accessibilityLabel(Text("Script", bundle: .module))
                .padding(8).overlay { RoundedRectangle(cornerRadius: 8).stroke(.appBorder) }
        }
    }

    private var languagePicker: some View { AudioLanguagePicker(controller: controller) }

    private var musicInputs: some View {
        VStack(alignment: .leading, spacing: 16) {
            VStack(alignment: .leading, spacing: 8) {
                HStack {
                    Text("Music description", bundle: .module).themedFont(.small, weight: .semibold)
                    Spacer()
                    Button { showingGenerator = true } label: {
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

    private var experimentActions: some View {
        ViewThatFits(in: .horizontal) {
            HStack(spacing: 12) { libraryActions; Spacer(minLength: 16); runButton }
            VStack(alignment: .leading, spacing: 12) {
                libraryActions
                runButton
            }
        }
        .padding(.top, 4)
    }

    @ViewBuilder private var libraryActions: some View {
        if controller.task == .music || controller.task == .textToSpeech {
            HStack(spacing: 12) {
                Button { showingPrompts = true } label: {
                    Label { Text("Prompt library", bundle: .module) } icon: { Image(systemName: "text.book.closed") }
                }
                .appAction(.secondary, size: .small)
                Button(action: beginSavingPrompt) { Text("Save Prompt", bundle: .module) }
                    .appAction(.secondary, size: .small)
                    .disabled(!controller.canSavePrompt)
            }.fixedSize()
        } else {
            HStack(spacing: 12) { presetMenu; saveDraftButton }
        }
    }

    private func beginSavingPrompt() {
        savingPrompt = true
        presetName = String((controller.task == .music ? controller.recipe.caption : controller.recipe.text).prefix(60))
        showingPreset = true
    }

    private var familyDetails: some View {
        DisclosureGroup {
            VStack(alignment: .leading, spacing: 14) {
                ForEach(controller.familyCapabilities.filter { $0.task == controller.task }) { family in
                    VStack(alignment: .leading, spacing: 6) {
                        Label {
                            Text(family.family).themedFont(.small, weight: .semibold)
                        } icon: { Image(systemName: family.component ? "puzzlepiece" : "waveform") }
                        Text(family.operations.joined(separator: ", ")).themedCode(.small)
                        Text(LocalizedStringKey(AudioEngineLabel(backend: family.backend, compute: nil).title), bundle: .module)
                        if let reason = family.unavailableReason { Text(reason) }
                        Text(family.evidence).foregroundStyle(.appSecondary)
                    }
                }
            }.themedFont(.small).padding(.top, 8)
        } label: { Text("Model details", bundle: .module) }
    }

    private var presetMenu: some View {
        Menu {
            Button { savingPrompt = false; presetName = ""; showingPreset = true } label: { Text("Save preset", bundle: .module) }
            if !controller.presets.isEmpty {
                Divider()
                ForEach(controller.presets) { preset in
                    Button { controller.applyPreset(preset) } label: { Text(preset.name) }
                }
            }
        } label: { Text("Presets", bundle: .module) }
        .appMenuAction(size: .small)
        .disabled(controller.isBusy)
    }

    private var saveDraftButton: some View {
        Button { controller.saveDraft() } label: { Text("Save draft", bundle: .module) }
            .appAction(.secondary, size: .small)
            .disabled(controller.isBusy)
    }

    private var runButton: some View {
        Button { controller.run() } label: {
            Label { Text(LocalizedStringKey(controller.runTitle), bundle: .module) }
                icon: { Image(systemName: controller.page.symbol) }
        }
        .appAction(.primary, size: .large).disabled(!controller.canRun)
        .keyboardShortcut(.return, modifiers: .command)
    }
}
