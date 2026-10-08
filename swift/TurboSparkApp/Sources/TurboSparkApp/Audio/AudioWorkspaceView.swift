import SwiftUI
import TurboSpark

@MainActor
struct AudioWorkspaceView: View {
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @State private var presetName = ""
    @State private var showingPreset = false

    var body: some View {
        VStack(spacing: 0) {
            navigation
            Divider()
            if controller.page == .library {
                AudioLibraryView(controller: controller)
            } else {
                GeometryReader { geometry in
                    ScrollView {
                        if geometry.size.width >= 940 && controller.page != .record {
                            HStack(alignment: .top, spacing: 24) {
                                workflow.frame(maxWidth: .infinity)
                                modelPanel.frame(width: 300)
                            }
                            .frame(maxWidth: 1280)
                            .frame(maxWidth: .infinity)
                            .padding(24)
                        } else {
                            VStack(alignment: .leading, spacing: 20) {
                                if controller.page != .record { modelPanel }
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
            AudioNameSheet(title: "Save preset", value: $presetName) {
                controller.savePreset(name: presetName)
                showingPreset = false
            }
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
                .buttonStyle(.borderless)
            }
            Menu { AudioDestinationMenu(model: model) } label: {
                Label { Text("Audio tasks", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
            }
            .menuStyle(.borderlessButton).fixedSize()
        }
        .themedFont(.small)
    }

    private var workflow: some View {
        VStack(alignment: .leading, spacing: 24) {
            Text(LocalizedStringKey(pageDescription), bundle: .module)
                .foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
            if controller.page == .record { recording }
            else { modelAndInputs }
            if let item = controller.workspaceResult {
                AudioItemDetailView(controller: controller, item: item)
            } else if AudioWorkspacePage.destinations.contains(controller.page) {
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
                        Button { controller.select(item) } label: {
                            HStack(spacing: 12) {
                                Image(systemName: controller.page.symbol).foregroundStyle(.appAccent)
                                VStack(alignment: .leading, spacing: 4) {
                                    Text(item.title).lineLimit(1)
                                    Text(item.createdAt, format: .dateTime.month(.abbreviated).day().hour().minute())
                                        .themedFont(.tiny).foregroundStyle(.appSecondary)
                                }
                                Spacer()
                                Text(audioTime(item.duration)).monospacedDigit().foregroundStyle(.appSecondary)
                                Image(systemName: "chevron.right").foregroundStyle(.appSecondary)
                            }
                            .padding(12).background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
                            .contentShape(Rectangle())
                        }.buttonStyle(.plain)
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

    private var pageDescription: String {
        switch controller.page {
        case .library: return "Reopen recordings and takes, or start with an audio file."
        case .record: return "Record a meeting, add markers, and keep the original audio."
        case .transcribe: return "Turn a recording into editable text and subtitles."
        case .voiceover: return "Write a script in sections. Preview or regenerate each section as a new take."
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
            case .voiceover: voiceoverInputs
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
        Picker(selection: Binding(get: { controller.recipe.task }, set: controller.selectTask)) {
            if controller.page == .cleanup {
                Text("Enhance speech", bundle: .module).tag(AudioTask.enhancement.rawValue)
                Text("Separate stems", bundle: .module).tag(AudioTask.separation.rawValue)
            } else {
                Text("Transcribe", bundle: .module).tag(AudioTask.speechToText.rawValue)
                Text("Voiceover", bundle: .module).tag(AudioTask.textToSpeech.rawValue)
                Text("Music", bundle: .module).tag(AudioTask.music.rawValue)
                Text("Speech detection", bundle: .module).tag(AudioTask.speechDetection.rawValue)
                Text("Alignment", bundle: .module).tag(AudioTask.alignment.rawValue)
                Text("Speaker diarization", bundle: .module).tag(AudioTask.diarization.rawValue)
                Text("Codec", bundle: .module).tag(AudioTask.codec.rawValue)
                Text("Language identification", bundle: .module).tag(AudioTask.languageIdentification.rawValue)
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
        .menuStyle(.borderlessButton).fixedSize()
        .disabled(controller.isBusy || controller.hasRecordingActivity || !controller.items.contains { !$0.clips.isEmpty })
    }

    private var importButton: some View {
        Button { controller.importAudio() } label: {
            Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
        }
        .disabled(controller.isBusy || controller.hasRecordingActivity)
    }

    private var voiceoverInputs: some View {
        VStack(alignment: .leading, spacing: 14) {
            scriptEditor
            Text("Separate sections with a blank line.", bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary)
            if let profile = controller.selectedProfile {
                if !profile.capabilities.voices.isEmpty {
                    Picker(selection: $controller.recipe.voice) {
                        ForEach(profile.capabilities.voices, id: \.self) { voice in Text(voice).tag(voice) }
                    } label: { Text("Voice", bundle: .module) }
                }
                languagePicker
                if controller.selectedProfile != nil {
                    HStack {
                        Text("Speed", bundle: .module)
                        Slider(value: $controller.recipe.speed, in: 0.5...2, step: 0.05)
                            .accessibilityLabel(Text("Speech speed", bundle: .module))
                        Text(controller.recipe.speed, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                    }
                }
            }
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

    @ViewBuilder private var languagePicker: some View {
        if let profile = controller.selectedProfile, !profile.capabilities.languages.isEmpty {
            Picker(selection: $controller.recipe.language) {
                if controller.task == .speechToText {
                    Text("Detect automatically", bundle: .module).tag("auto")
                }
                ForEach(profile.capabilities.languages.filter { $0 != "auto" }, id: \.self) { language in
                    Text(Locale.current.localizedString(forLanguageCode: language) ?? language).tag(language)
                }
            } label: { Text("Language", bundle: .module) }
        }
    }

    private var musicInputs: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Music description", bundle: .module).themedFont(.small, weight: .semibold)
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
            if controller.selectedProfile?.capabilities.supportsLyrics == true {
                Text("Lyrics", bundle: .module).themedFont(.small, weight: .semibold)
                TextEditor(text: $controller.recipe.lyrics)
                    .frame(minHeight: 100).themedFont(.base)
                    .scrollContentBackground(.hidden).padding(8)
                    .overlay { RoundedRectangle(cornerRadius: 8).stroke(.appBorder) }
                    .accessibilityLabel(Text("Lyrics", bundle: .module))
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
                                .textFieldStyle(.roundedBorder)
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
            HStack { presetMenu; saveDraftButton; Spacer(); runButton }
            VStack(alignment: .leading, spacing: 12) {
                HStack { presetMenu; saveDraftButton }
                runButton
            }
        }
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
                        Text(family.backend).themedCode(.small)
                        if let reason = family.unavailableReason { Text(reason) }
                        Text(family.evidence).foregroundStyle(.appSecondary)
                    }
                }
            }.themedFont(.small).padding(.top, 8)
        } label: { Text("Model details", bundle: .module) }
    }

    private var presetMenu: some View {
        Menu {
            Button { presetName = ""; showingPreset = true } label: { Text("Save preset", bundle: .module) }
            if !controller.presets.isEmpty {
                Divider()
                ForEach(controller.presets) { preset in
                    Button { controller.applyPreset(preset) } label: { Text(preset.name) }
                }
            }
        } label: { Text("Presets", bundle: .module) }
        .menuStyle(.borderlessButton).fixedSize()
        .disabled(controller.isBusy)
    }

    private var saveDraftButton: some View {
        Button { controller.saveDraft() } label: { Text("Save draft", bundle: .module) }
            .disabled(controller.isBusy)
    }

    private var runButton: some View {
        Button { controller.run() } label: {
            Label { Text(LocalizedStringKey(controller.runTitle), bundle: .module) }
                icon: { Image(systemName: controller.page.symbol) }
        }
        .buttonStyle(.borderedProminent).controlSize(.large).disabled(!controller.canRun)
        .keyboardShortcut(.return, modifiers: .command)
    }

    private var recording: some View {
        VStack(alignment: .leading, spacing: 18) {
            captureSources
            if let setupStatus = controller.recordingSetupStatus {
                HStack(alignment: .top, spacing: 12) {
                    ProgressView().controlSize(.small)
                    Text(setupStatus).fixedSize(horizontal: false, vertical: true)
                    Spacer(minLength: 0)
                    Button { controller.cancelRecordingSetup() } label: { Text("Cancel", bundle: .module) }
                }
            } else if controller.isRecording {
                HStack {
                    AudioRecordingBadge(controller: controller)
                    Spacer()
                    Button { controller.addMarker() } label: {
                        Label { Text("Add marker", bundle: .module) } icon: { Image(systemName: "bookmark") }
                    }
                }
                ProgressView(value: Double(max(0, min(1, controller.inputLevel)))) {
                    Text("Input level", bundle: .module)
                }
                .accessibilityValue(Text(verbatim: "\(Int(max(0, min(1, controller.inputLevel)) * 100))%"))
                HStack {
                    Button {
                        if controller.isPaused { controller.resumeRecording() } else { controller.pauseRecording() }
                    } label: {
                        Label {
                            Text(controller.isPaused ? "Resume" : "Pause", bundle: .module)
                        } icon: { Image(systemName: controller.isPaused ? "play.fill" : "pause.fill") }
                    }
                    Button { controller.stopRecording() } label: {
                        Label { Text("Stop recording", bundle: .module) } icon: { Image(systemName: "stop.fill") }
                    }.buttonStyle(.borderedProminent)
                }
            } else {
                Button { controller.startRecording() } label: {
                    Label { Text("Start recording", bundle: .module) } icon: { Image(systemName: "record.circle") }
                }
                .buttonStyle(.borderedProminent)
                .disabled(controller.isBusy || (!controller.captureMicrophone && !controller.captureSystemAudio))
                .keyboardShortcut("r", modifiers: [.command, .shift])
            }
            DisclosureGroup {
                Toggle(isOn: $controller.liveTranscription) { Text("Show a live draft transcript", bundle: .module) }
                AudioModelPickerView(model: model, controller: controller).padding(.top, 8)
                Text("Live text is a draft. Saved audio remains available when transcription falls behind.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            } label: { Text("Transcription model", bundle: .module) }
        }
        .padding(20).background(.appSurface, in: RoundedRectangle(cornerRadius: 12))
        .overlay { RoundedRectangle(cornerRadius: 12).stroke(.appBorder, lineWidth: 1) }
    }

    private var captureSources: some View {
        VStack(alignment: .leading, spacing: 12) {
            Toggle(isOn: $controller.captureMicrophone) { Text("Microphone", bundle: .module) }
            if controller.captureMicrophone {
                Picker(selection: $controller.microphoneID) {
                    Text("Default microphone", bundle: .module).tag(Optional<String>.none)
                    ForEach(controller.microphones, id: \.id) { device in Text(device.name).tag(Optional(device.id)) }
                } label: { Text("Microphone", bundle: .module) }
            }
            Toggle(isOn: $controller.captureSystemAudio) { Text("App or system audio", bundle: .module) }
            if controller.captureSystemAudio {
                Picker(selection: $controller.selectedApplicationID) {
                    Text("All system audio", bundle: .module).tag(Optional<Int32>.none)
                    ForEach(controller.audioApplications, id: \.id) { app in Text(app.name).tag(Optional(app.id)) }
                } label: { Text("Audio source", bundle: .module) }
                Text("TurboSpark playback is excluded. Only audio is saved.", bundle: .module)
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
            Button { controller.prepareRecording() } label: {
                Label { Text("Check devices and permissions", bundle: .module) } icon: { Image(systemName: "checkmark.shield") }
            }
        }.disabled(controller.hasRecordingActivity)
    }
}
