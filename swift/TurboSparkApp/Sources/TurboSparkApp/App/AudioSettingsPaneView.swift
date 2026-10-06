import AVFoundation
import SwiftUI
import TurboSpark

/// Settings > Audio & Voice (docs/AUDIO_UI.md).
///
/// Each section is its own computed property: a Section inline in a larger
/// Form expression is the shape that blew the macOS 14 SDK type-checker
/// (see `GeneralSettingsPaneView`).
// Isolated explicitly: only `body` is isolated by the protocol on the
// macOS 14 SDK (swift/CLAUDE.md Gotcha 45).
@MainActor
struct AudioSettingsPaneView: View {
    @ObservedObject var model: AppModel
    @ObservedObject private var capabilities = AudioCapabilities.shared
    @StateObject private var micTest = AudioCaptureService()
    @Environment(\.appTheme) private var theme

    @AppStorage(AudioPreferences.experimentalEnabledKey) private var audioEnabled = false
    @AppStorage(AudioPreferences.speechEngineKey) private var speechEngine = SpeechEngineChoice.automatic.rawValue
    @AppStorage(AudioPreferences.autoSendAfterDictationKey) private var autoSend = false
    @AppStorage(AudioPreferences.autoTranscribeAttachmentsKey) private var autoTranscribe = true
    @AppStorage(AudioPreferences.maxClipSecondsKey) private var maxClipSeconds = AudioPreferences.defaultMaxClipSeconds
    @AppStorage(AudioPreferences.readAloudVoiceKey) private var voiceIdentifier = ""
    @AppStorage(AudioPreferences.readAloudRateKey) private var readAloudRate = Double(AVSpeechUtteranceDefaultSpeechRate)

    var body: some View {
        Form {
            experimentalSection
            engineSection
            inputSection
            dictationSection
            readAloudSection
            appAudioSection
            attachmentsSection
        }
        .formStyle(.grouped)
        .padding(16)
        .onAppear {
            capabilities.refresh()
            // The test is a meter, not a recording: stop at the length cap.
            micTest.onLimitReached = { [weak micTest] in micTest?.cancel() }
        }
        .onDisappear { micTest.cancel() }
        .onChange(of: audioEnabled) { _, _ in capabilities.refresh() }
        .onChange(of: speechEngine) { _, _ in capabilities.refresh() }
    }

    private var experimentalSection: some View {
        Section(header: Text("Experimental", bundle: .module)) {
            Toggle(isOn: $audioEnabled) {
                Text("Enable in-app audio", bundle: .module)
            }
            .settingsControl("Enable in-app audio", pane: .audio, timing: .immediate)
            Text("Recording, voice notes, audio attachments and app audio capture. With this off, the microphone button uses macOS Dictation.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

    /// What the Rust engine reports, verbatim: the user should see that the
    /// engine is asked first and why it declines today.
    private var engineSection: some View {
        Section(header: Text("TurboSpark audio engine", bundle: .module)) {
            engineRow(Text("Speech to text", bundle: .module), AudioCapabilities.engine?.speechToText)
            engineRow(Text("Text to speech", bundle: .module), AudioCapabilities.engine?.textToSpeech)
            engineRow(Text("Music", bundle: .module), AudioCapabilities.engine?.music)
            if let formats = AudioCapabilities.engine?.decodeExtensions {
                LabeledContent {
                    Text(verbatim: formats.sorted().joined(separator: ", "))
                        .themedCode(.tiny)
                        .foregroundStyle(.appSecondary)
                        .multilineTextAlignment(.trailing)
                } label: {
                    Text("Decodes", bundle: .module)
                }
            }
        }
    }

    private func engineRow(_ title: Text, _ status: AudioTaskStatus?) -> some View {
        LabeledContent {
            VStack(alignment: .trailing, spacing: 2) {
                if status?.active == true {
                    Text("Available", bundle: .module).foregroundStyle(.green)
                } else {
                    Text("Not available", bundle: .module).foregroundStyle(.appSecondary)
                }
                if let reason = status?.reason {
                    Text(verbatim: reason)
                        .themedFont(.micro)
                        .foregroundStyle(.tertiary)
                        .multilineTextAlignment(.trailing)
                }
            }
        } label: {
            title
        }
    }

    private var inputSection: some View {
        Section(header: Text("Input", bundle: .module)) {
            LabeledContent {
                HStack {
                    Text(verbatim: AVCaptureDevice.default(for: .audio)?.localizedName ?? "-")
                        .foregroundStyle(.appSecondary)
                    Button {
                        AudioCapabilities.openSoundSettings()
                    } label: {
                        Text("Sound Settings...", bundle: .module)
                    }
                }
            } label: {
                Text("Microphone", bundle: .module)
            }
            .settingsControl("Microphone", pane: .audio, timing: .action)

            LabeledContent {
                HStack(spacing: 8) {
                    AudioLevelMeter(level: micTest.levels.last ?? 0)
                        .frame(width: 140)
                    Button {
                        toggleMicTest()
                    } label: {
                        if micTest.isRecording {
                            Text("Stop", bundle: .module)
                        } else {
                            Text("Test", bundle: .module)
                        }
                    }
                    .disabled(!audioEnabled)
                }
            } label: {
                Text("Test microphone", bundle: .module)
            }
            .settingsControl("Test microphone", pane: .audio, timing: .action)

            permissionRow(
                Text("Microphone access", bundle: .module),
                availability: capabilities.snapshot.micCapture, kind: .microphone)
        }
    }

    private var dictationSection: some View {
        Section(header: Text("Speech engine", bundle: .module)) {
            Picker(selection: $speechEngine) {
                ForEach(SpeechEngineChoice.allCases) { choice in
                    Text(LocalizedStringKey(choice.titleKey), bundle: .module).tag(choice.rawValue)
                }
            } label: {
                Text("Transcription and read aloud", bundle: .module)
            }
            .settingsControl("Transcription and read aloud", pane: .audio, timing: .immediate)
            .disabled(!audioEnabled)

            LabeledContent {
                if let provider = capabilities.snapshot.transcription.provider {
                    Text(verbatim: provider.label)
                } else {
                    Text(verbatim: capabilities.snapshot.transcription.reason ?? "")
                        .foregroundStyle(.appSecondary)
                        .multilineTextAlignment(.trailing)
                }
            } label: {
                Text("Transcribes with", bundle: .module)
            }

            permissionRow(
                Text("Speech recognition access", bundle: .module),
                availability: capabilities.snapshot.transcription, kind: .speechRecognition)

            Toggle(isOn: $autoSend) {
                Text("Send automatically after dictation", bundle: .module)
            }
            .settingsControl("Send automatically after dictation", pane: .audio, timing: .immediate)
            .disabled(!audioEnabled)
        }
    }

    private var readAloudSection: some View {
        Section(header: Text("Read aloud", bundle: .module)) {
            Picker(selection: $voiceIdentifier) {
                Text("Language default", bundle: .module).tag("")
                ForEach(AVSpeechSynthesisVoice.speechVoices().sorted { $0.name < $1.name }, id: \.identifier) { voice in
                    Text(verbatim: "\(voice.name) (\(voice.language))").tag(voice.identifier)
                }
            } label: {
                Text("Voice", bundle: .module)
            }
            .settingsControl("Voice", pane: .audio, timing: .nextTurn)

            Slider(
                value: $readAloudRate,
                in: Double(AVSpeechUtteranceMinimumSpeechRate)...Double(AVSpeechUtteranceMaximumSpeechRate)
            ) {
                Text("Speaking rate", bundle: .module)
            }
            .settingsControl("Speaking rate", pane: .audio, timing: .nextTurn)

            Text("Apple voices are used until the TurboSpark engine has a voice model.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

    private var appAudioSection: some View {
        Section(header: Text("App audio capture", bundle: .module)) {
            LabeledContent {
                if capabilities.snapshot.systemCapture.isUsable {
                    Text("Available", bundle: .module)
                } else {
                    Text(verbatim: capabilities.snapshot.systemCapture.reason ?? "")
                        .foregroundStyle(.appSecondary)
                }
            } label: {
                Text("Status", bundle: .module)
            }
            Text("macOS asks for permission the first time TurboSpark records another app. Only record audio you have the right to use.", bundle: .module)
                .themedFont(.tiny)
                .foregroundStyle(.appSecondary)
        }
    }

    private var attachmentsSection: some View {
        Section(header: Text("Audio attachments", bundle: .module)) {
            Toggle(isOn: $autoTranscribe) {
                Text("Transcribe audio attachments automatically", bundle: .module)
            }
            .settingsControl("Transcribe audio attachments automatically", pane: .audio, timing: .immediate)
            .disabled(!audioEnabled)

            Picker(selection: $maxClipSeconds) {
                ForEach(AudioPreferences.maxClipSecondsChoices, id: \.self) { seconds in
                    Text(verbatim: WaveformMath.formatDuration(TimeInterval(seconds))).tag(seconds)
                }
            } label: {
                Text("Maximum recording length", bundle: .module)
            }
            .settingsControl("Maximum recording length", pane: .audio, timing: .immediate)
            .disabled(!audioEnabled)
        }
    }

    @ViewBuilder
    private func permissionRow(
        _ title: Text, availability: AudioAvailability, kind: AudioPermissionKind
    ) -> some View {
        LabeledContent {
            switch availability {
            case .denied(kind):
                Button {
                    AudioCapabilities.openPrivacySettings(for: kind)
                } label: {
                    Text("Open System Settings", bundle: .module)
                }
            case .needsPermission(kind):
                Text("Not yet requested", bundle: .module).foregroundStyle(.appSecondary)
            case .disabled:
                Text("Off", bundle: .module).foregroundStyle(.appSecondary)
            default:
                Text("Allowed", bundle: .module).foregroundStyle(.appSecondary)
            }
        } label: {
            title
        }
    }

    private func toggleMicTest() {
        if micTest.isRecording {
            micTest.cancel()
            return
        }
        Task {
            guard await capabilities.requestMicrophoneAccess() else { return }
            do {
                try micTest.start()
            } catch {
                model.showToast(error.localizedDescription, style: .error)
            }
        }
    }
}
