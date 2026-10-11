import AVFoundation
import AppKit
import SwiftUI
import TurboSpark
import UniformTypeIdentifiers

@MainActor
struct AudioAPIPaneView: View {
    @ObservedObject var model: AppModel
    @State private var selectedAlias = ""
    @State private var text = "Hello! This voice clone is running locally on TurboSpark."
    @State private var voice = "clone"
    @State private var language = "en"
    @State private var instruction = ""
    @State private var referenceText = ""
    @State private var testing = false
    @State private var resultWav: Data?
    @State private var errorMessage: String?
    @State private var elapsed: Double?
    @State private var player: AVAudioPlayer?
    @State private var isPlaying = false
    @State private var timer: Timer?
    @State private var referenceURL: URL?
    @State private var consent = false

    @State private var installedAudioModels: [AudioInstalledRecord] = []

    private var supported: [AudioInstalledRecord] {
        installedAudioModels.filter { ["qwen3_tts", "kokoro", "megatts3"].contains($0.family) }
    }

    private var selected: AudioInstalledRecord? {
        supported.first { $0.alias == selectedAlias } ?? supported.first
    }

    /// Only Qwen3-TTS runs in the Swift speech bridge. Every other family is
    /// served by the Rust audio provider, keyed by its alias.
    private var providerAttached: AudioInstalledRecord? {
        supported.first { $0.family != "qwen3_tts" && model.serverInfo?.audioModels.contains($0.alias) == true }
    }

    private var attached: AudioInstalledRecord? { model.serverSpeechAttachedModel ?? providerAttached }

    private var attachedClonesNatively: Bool { attached?.family == "megatts3" }

    private func attach(_ record: AudioInstalledRecord) {
        if record.family == "qwen3_tts" { model.attachAudioSpeechModelToServer(record) }
        else { model.attachAudioModelToServer(path: record.path, task: AudioTask.textToSpeech.rawValue) }
    }

    private func detach() {
        if model.serverSpeechAttachedModel != nil { model.detachAudioSpeechModelFromServer() }
        else if let providerAttached { model.detachModelFromServer(id: providerAttached.alias) }
    }

    private func chooseReference() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.allowedContentTypes = [.wav]
        guard panel.runModal() == .OK, let url = panel.url else { return }
        referenceURL = url
        consent = false
    }

    private var exampleBody: String {
        let value: [String: Any] = [
            "model": attached?.alias ?? "<attach a speech model>",
            "input": "Speech synthesis with voice cloning",
            "voice": "clone",
            "language": "en",
            "reference_text": "Reference text snippet",
            "instruction": "Speak warmly and clearly",
            "response_format": "wav"
        ]
        guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.prettyPrinted, .sortedKeys])
        else { return "" }
        return String(decoding: data, as: UTF8.self)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 18) {
            ServerHeaderBandView(model: model)
            VStack(alignment: .leading, spacing: 12) {
                Text("Audio Speech & Voice Cloning API", bundle: .module).themedFont(.title3, weight: .semibold)
                Text("The same address and API key as Text. Serves OpenAI-compatible /v1/audio/speech with voice cloning extensions.", bundle: .module)
                    .foregroundStyle(.secondary)
                HStack {
                    Picker("Audio model", selection: $selectedAlias) {
                        ForEach(supported) { item in
                            Text(item.alias).tag(item.alias)
                        }
                    }
                    .frame(maxWidth: 360)
                    Button {
                        if let selected { attach(selected) }
                    } label: { Text("Attach", bundle: .module) }
                    .disabled(model.server == nil || selected == nil || attached != nil || model.serverBusy)
                    Button { detach() } label: {
                        Text("Detach", bundle: .module)
                    }
                    .disabled(attached == nil)
                }
                if supported.isEmpty {
                    Text("Install a supported Qwen3-TTS, Kokoro, or MegaTTS3 model in Audio Studio first.", bundle: .module)
                        .foregroundStyle(.secondary)
                }
                if let info = model.serverInfo?.baseURL {
                    Text(verbatim: "POST " + info.absoluteString + "/v1/audio/speech")
                        .themedCode(.callout)
                        .textSelection(.enabled)
                    Text(exampleBody)
                        .themedCode(.small)
                        .textSelection(.enabled)
                }
                Text(verbatim: "Attached: \(attached?.alias ?? "none")")
                Text(verbatim: "Progress: \(model.serverSpeechProgress)")
                    .accessibilityLabel(Text(verbatim: "Audio progress: \(model.serverSpeechProgress)"))
            }
            .padding(16)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))

            VStack(alignment: .leading, spacing: 12) {
                Text("Test speech request", bundle: .module).themedFont(.title3, weight: .semibold)
                TextEditor(text: $text)
                    .frame(height: 74)
                    .border(.appBorder)
                    .accessibilityLabel(Text("Speech input text", bundle: .module))

                if attachedClonesNatively {
                    VStack(alignment: .leading, spacing: 6) {
                        HStack {
                            Text("Reference voice", bundle: .module)
                            Button { chooseReference() } label: { Text("Choose WAV file", bundle: .module) }
                            if let referenceURL { Text(referenceURL.lastPathComponent).lineLimit(1).truncationMode(.middle) }
                        }
                        Toggle(isOn: $consent) { Text("I have the right to use this voice", bundle: .module) }
                            .disabled(referenceURL == nil)
                    }
                } else {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Voice", bundle: .module).themedFont(.small).foregroundStyle(.secondary)
                        TextField("Voice", text: $voice)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 160)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Language", bundle: .module).themedFont(.small).foregroundStyle(.secondary)
                        TextField("Language (en, zh, auto)", text: $language)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 160)
                    }
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Style instruction", bundle: .module).themedFont(.small).foregroundStyle(.secondary)
                        TextField("e.g. Whisper, Cheerful", text: $instruction)
                            .textFieldStyle(.roundedBorder)
                            .frame(maxWidth: 220)
                    }
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text("Reference transcript (optional)", bundle: .module).themedFont(.small).foregroundStyle(.secondary)
                    TextField("Transcript of reference voice if cloning", text: $referenceText)
                        .textFieldStyle(.roundedBorder)
                }
                }

                HStack {
                    Button {
                        Task { await runTest() }
                    } label: { Text("Synthesize", bundle: .module) }
                    .buttonStyle(.borderedProminent)
                    .disabled(testing || attached == nil || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                        || (attachedClonesNatively && (referenceURL == nil || !consent)))

                    if testing { ProgressView().controlSize(.small) }
                    if let elapsed { Text(String(format: "%.1f s", elapsed)) }

                    if let resultWav {
                        Button {
                            togglePlayback(wav: resultWav)
                        } label: {
                            Image(systemName: isPlaying ? "stop.fill" : "play.fill")
                            Text(isPlaying ? "Stop" : "Play", bundle: .module)
                        }
                    }
                }
                if let errorMessage { Text(errorMessage).foregroundStyle(.red).textSelection(.enabled) }
            }
            .padding(16)
            .background(.appSurface, in: RoundedRectangle(cornerRadius: 10))
        }
        .onAppear {
            refreshInstalled()
        }
        .onDisappear {
            stopPlayback()
        }
    }

    private func refreshInstalled() {
        Task {
            if let report = try? AudioCatalog.installed() {
                self.installedAudioModels = report.installed
                if selectedAlias.isEmpty {
                    selectedAlias = supported.first?.alias ?? ""
                }
            }
        }
    }

    private func togglePlayback(wav: Data) {
        if isPlaying {
            stopPlayback()
            return
        }
        do {
            player = try AVAudioPlayer(data: wav)
            player?.prepareToPlay()
            player?.play()
            isPlaying = true
            timer = Timer.scheduledTimer(withTimeInterval: 0.1, repeats: true) { _ in
                Task { @MainActor in
                    if self.player?.isPlaying != true {
                        self.stopPlayback()
                    }
                }
            }
        } catch {
            errorMessage = error.localizedDescription
        }
    }

    private func stopPlayback() {
        player?.stop()
        player = nil
        isPlaying = false
        timer?.invalidate()
        timer = nil
    }

    private func runTest() async {
        guard let modelID = attached?.alias,
              let baseURL = model.serverInfo?.baseURL else { return }
        let cloning = attachedClonesNatively
        testing = true
        resultWav = nil
        errorMessage = nil
        elapsed = nil
        stopPlayback()
        let started = Date()
        let submittedText = text
        do {
            var request = URLRequest(url: baseURL.appendingPathComponent("v1/audio/speech"))
            request.httpMethod = "POST"
            request.timeoutInterval = 1_200
            request.setValue("application/json", forHTTPHeaderField: "content-type")
            if let key = AppModel.serverAPIKey(from: model.serverAPIKeyInput) {
                request.setValue("Bearer \(key)", forHTTPHeaderField: "authorization")
            }
            var payload: [String: Any] = [
                "model": modelID,
                "input": submittedText,
                "voice": voice,
                "response_format": "wav"
            ]
            if cloning {
                // MegaTTS3 takes its voice from the reference and refuses the Qwen-only fields.
                guard let referenceURL else { throw CocoaError(.fileNoSuchFile) }
                payload["reference_audio"] = try Data(contentsOf: referenceURL).base64EncodedString()
                payload["consent"] = consent
            } else {
                if !language.isEmpty { payload["language"] = language }
                if !instruction.isEmpty { payload["instruction"] = instruction }
                if !referenceText.isEmpty { payload["reference_text"] = referenceText }
            }

            request.httpBody = try JSONSerialization.data(withJSONObject: payload)
            let (data, response) = try await URLSession.shared.data(for: request)
            guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                throw NSError(domain: "Audio API", code: 1, userInfo: [
                    NSLocalizedDescriptionKey: String(data: data.prefix(2048), encoding: .utf8) ?? "Invalid speech response"
                ])
            }
            resultWav = data
            elapsed = Date().timeIntervalSince(started)
        } catch {
            errorMessage = error.localizedDescription
        }
        testing = false
    }
}
