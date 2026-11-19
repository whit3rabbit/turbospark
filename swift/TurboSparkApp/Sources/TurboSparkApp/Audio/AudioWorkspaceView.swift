import SwiftUI
import TurboSpark

@MainActor
struct AudioWorkspaceView: View {
    @Environment(\.appTheme) private var theme
    @ObservedObject var model: AppModel
    @ObservedObject var controller: AudioWorkspaceController
    @State private var presetName = ""
    @State private var showingPreset = false
    @State private var showingPrompts = false
    @State private var savingPrompt = false
    @State private var showingGenerator = false
    // Held here, not in the sheet, so reopening the generator keeps the choices.
    @State private var promptSelection = MusicPromptSelection()

    private var scale: CGFloat { AppMetrics.textScale(theme) }

    var body: some View {
        VStack(spacing: 0) {
            navigation
            Divider()
            pageArea
            AudioWorkspaceStatusView(controller: controller)
        }
        .background(.appPage)
        .themedFont(.base)
        .sheet(isPresented: $showingPreset) {
            AudioNameSheet(title: savingPrompt ? "Save Prompt" : "Save Preset", value: $presetName) {
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
        .task {
            if controller.installedPaths.isEmpty || controller.profiles.isEmpty {
                controller.refreshModels()
            }
        }
    }

    /// Music keeps its composer and listening controls available during setup.
    @ViewBuilder
    private var pageArea: some View {
        if controller.page == .music {
            AudioMusicWorkspaceView(model: model, controller: controller, showGenerator: { showingGenerator = true }) {
                libraryActions
            }
        } else {
            let gate = controller.modelGate
            ZStack {
                if let gate {
                    AudioModelGateView(model: model, controller: controller, gate: gate)
                        .transition(.opacity)
                } else {
                    pageContent
                }
            }
            .tsAnimation(TSMotion.pane, value: gate != nil)
        }
    }

    @ViewBuilder private var pageContent: some View {
        if controller.page == .library {
            AudioLibraryView(controller: controller)
        } else if controller.page == .transcribe {
            SpeechToTextWorkspaceView(model: model, controller: controller)
        } else if controller.page == .voiceClone {
            VoiceCloneWorkspaceView(
                model: model, controller: controller,
                description: Text(LocalizedStringKey(pageDescription), bundle: .module),
                showPromptLibrary: { showingPrompts = true }, savePrompt: beginSavingPrompt)
        } else {
            GeometryReader { geometry in
                ScrollView {
                    if geometry.size.width >= (940 * scale).rounded() && controller.page != .record {
                        HStack(alignment: .top, spacing: 24) {
                            workflow.frame(maxWidth: .infinity)
                            modelPanel.frame(width: (280 * scale).rounded())
                        }
                        .frame(maxWidth: (1440 * scale).rounded())
                        .frame(maxWidth: .infinity)
                        .padding(24)
                    } else {
                        VStack(alignment: .leading, spacing: 20) {
                            if controller.page != .record { modelPanel }
                            workflow
                        }
                        .padding(24)
                        // Record is the only page that always takes this branch,
                        // so this cap is its reading measure; a narrow render of
                        // the other pages is already below it, because the branch
                        // itself opens under 940 scaled points.
                        .frame(maxWidth: (980 * scale).rounded())
                        .frame(maxWidth: .infinity)
                    }
                }
            }
        }
    }

    private var navigation: some View {
        AppAdaptiveHeader(leading: {
            workspaceTitle
        }, actions: {
            workspaceActions
        })
        .padding(.horizontal, 24).padding(.vertical, 12)
    }

    private var workspaceTitle: some View {
        HStack(spacing: 12) {
            Image(systemName: controller.page.symbol)
                .themedFont(.title3).foregroundStyle(.appAccentText)
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
                .appAction(.quiet)
            }
            Menu { AudioDestinationMenu(model: model) } label: {
                Label { Text("Audio tasks", bundle: .module) } icon: { Image(systemName: "ellipsis.circle") }
            }
            .appMenuAction(quiet: true)
        }
        .themedFont(.small)
    }

    private var workflow: some View {
        let recentTakes = Array(controller.workspaceHistory.prefix(5))
        return VStack(alignment: .leading, spacing: 16) {
            Text(LocalizedStringKey(pageDescription), bundle: .module)
                .themedFont(.small).foregroundStyle(.appSecondary).fixedSize(horizontal: false, vertical: true)
            if controller.page == .record { AudioRecordingPanel(model: model, controller: controller) }
            else { modelAndInputs }
            if let item = controller.workspaceResult, item.status != .processing {
                AudioItemDetailView(controller: controller, item: item)
            } else if recentTakes.isEmpty && !controller.isBusy && AudioWorkspacePage.destinations.contains(controller.page) {
                Label { Text("No outputs yet", bundle: .module) } icon: { Image(systemName: "waveform") }
                    .themedFont(.small).foregroundStyle(.appSecondary)
            }
            if !recentTakes.isEmpty {
                VStack(alignment: .leading, spacing: 8) {
                    AppSectionHeader(title: Text("Saved audio", bundle: .module))
                    AudioLibraryList(controller: controller, items: recentTakes)
                }
            }
        }
    }

    private var modelPanel: some View {
        AudioWorkspaceModelPanel(model: model, controller: controller)
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
        VStack(alignment: .leading, spacing: 14) {
            if controller.page == .cleanup || controller.page == .advanced { AudioOperationPicker(controller: controller) }
            if controller.page == .cleanup
                || (controller.page == .advanced && controller.task != .textToSpeech && controller.task != .music) {
                sourcePicker
            }
            switch controller.page {
            case .voiceover, .voiceClone, .voiceDesign: voiceoverInputs
            case .advanced:
                if controller.task == .speechToText { languagePicker }
                if controller.task == .alignment { scriptEditor }
                if controller.task == .textToSpeech { voiceoverInputs }
                if controller.task == .music { AudioMusicInputs(controller: controller) { showingGenerator = true } }
            default: EmptyView()
            }
            experimentActions
            if controller.page == .advanced { familyDetails }
        }
        .appCardSurface(padding: 16)
    }

    private var sourcePicker: some View {
        VStack(alignment: .leading, spacing: 8) {
            ViewThatFits(in: .horizontal) {
                HStack { importButton; recordButton; savedSourceMenu }
                VStack(alignment: .leading) { importButton; recordButton; savedSourceMenu }
            }
            if let item = controller.selectedItem, !item.clips.isEmpty {
                // The source under the card's own surface: an inset, not a
                // second card, so the picked recording reads as one object.
                HStack {
                    Image(systemName: "waveform")
                    Text(item.title).lineLimit(2)
                    Spacer()
                    Text(audioTime(item.duration)).monospacedDigit()
                }
                .foregroundStyle(.appSecondary)
                .padding(.horizontal, 12).padding(.vertical, 8)
                .background(.appPage, in: RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: AppMetrics.controlRadius, style: .continuous)
                        .stroke(.appBorder, lineWidth: 1)
                }
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
        .appAction(.quiet, size: .small)
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
        .appMenuAction(size: .small, quiet: true)
        .disabled(controller.isBusy || controller.hasRecordingActivity || !controller.items.contains { !$0.clips.isEmpty })
    }

    private var importButton: some View {
        Button { controller.importAudio() } label: {
            Label { Text("Import audio", bundle: .module) } icon: { Image(systemName: "square.and.arrow.down") }
        }
        .appAction(.quiet, size: .small)
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
            AudioSpeechOptions(controller: controller)
            AudioScriptSections(controller: controller)
        }
    }

    private var scriptEditor: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("Script", bundle: .module).themedFont(.small, weight: .semibold)
                .accessibilityAddTraits(.isHeader)
            TextEditor(text: $controller.recipe.text)
                .themedFont(.base)
                .accessibilityLabel(Text("Script", bundle: .module))
                .appEditorSurface(minHeight: 112 * scale)
        }
    }

    private var languagePicker: some View { AudioLanguagePicker(controller: controller) }

    private var experimentActions: some View {
        AudioRunControls(controller: controller) { libraryActions }
    }

    @ViewBuilder private var libraryActions: some View {
        if controller.task == .music || controller.task == .textToSpeech {
            FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: 6) {
                Button { showingPrompts = true } label: {
                    Label { Text("Prompt library", bundle: .module) } icon: { Image(systemName: "text.book.closed") }
                }
                .appAction(.quiet, size: .small)
                Button(action: beginSavingPrompt) { Text("Save Prompt", bundle: .module) }
                    .appAction(.quiet, size: .small)
                    .disabled(!controller.canSavePrompt)
            }
        } else {
            FlowLayout(spacing: AppMetrics.actionSpacing, lineSpacing: 6) { presetMenu; saveDraftButton }
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
            Button { savingPrompt = false; presetName = ""; showingPreset = true } label: { Text("Save Preset", bundle: .module) }
            if !controller.presets.isEmpty {
                Divider()
                ForEach(controller.presets) { preset in
                    Button { controller.applyPreset(preset) } label: { Text(preset.name) }
                }
            }
        } label: { Text("Presets", bundle: .module) }
        .appMenuAction(size: .small, quiet: true)
        .disabled(controller.isBusy)
    }

    private var saveDraftButton: some View {
        Button { controller.saveDraft() } label: { Text("Save draft", bundle: .module) }
            .appAction(.quiet, size: .small)
            .disabled(controller.isBusy)
    }

}
