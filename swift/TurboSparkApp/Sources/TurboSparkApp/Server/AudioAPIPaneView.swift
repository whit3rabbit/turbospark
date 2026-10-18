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
    /// Speech installs on a drive that is away (task 6.10): listed so the pane
    /// does not read "install a model first", never offered for attaching.
    @State private var offlineSpeechRows: [AudioOfflineInstalls.Row] = []

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
            servingPanel
            testPanel
        }
        .onAppear {
            refreshInstalled()
        }
        .onDisappear {
            stopPlayback()
        }
    }

    private var servingPanel: some View {
        APIPanel(
            title: Text("Audio Speech & Voice Cloning API", bundle: .module),
            subtitle: Text("The same address and API key as Text. Serves OpenAI-compatible /v1/audio/speech with voice cloning extensions.", bundle: .module)
        ) {
            Picker(selection: $selectedAlias) {
                ForEach(supported) { item in
                    Text(verbatim: item.alias).tag(item.alias)
                }
            } label: { Text("Audio model", bundle: .module) }
            .frame(minWidth: 180, idealWidth: 280, maxWidth: 360)
            Button {
                if let selected { attach(selected) }
            } label: { Text("Attach", bundle: .module) }
            .appAction(.primary, size: APIControlSize.panel)
            .disabled(model.server == nil || selected == nil || attached != nil || model.serverBusy)
        } content: {
            if supported.isEmpty && offlineSpeechRows.isEmpty {
                Text("Install a supported Qwen3-TTS, Kokoro, or MegaTTS3 model in Audio Studio first.", bundle: .module)
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
            }
            ForEach(offlineSpeechRows, id: \.alias) { row in
                offlineRow(row)
            }
            APIRowList {
                if let attached {
                    APIServedModelRow(
                        name: attached.alias,
                        badge: Text("audio", bundle: .module),
                        badgeTint: .purple,
                        detachHelp: Text("Stop serving this and release it from memory.", bundle: .module),
                        detach: { detach() }
                    ) {
                        Text("Speech model", bundle: .module)
                    }
                } else {
                    APINothingServedRow()
                }
            }
            APIStatusLine(label: Text("Progress", bundle: .module), value: APIStatusText.text(model.serverSpeechProgress))
            if let info = model.serverInfo?.baseURL {
                APIRequestPreview(method: "POST", url: info.absoluteString + "/v1/audio/speech", example: exampleBody)
            }
        }
    }

    /// An install on a drive that is away: its name, a glyph and the statement.
    /// Nothing here attaches it.
    private func offlineRow(_ row: AudioOfflineInstalls.Row) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 6) {
            Image(systemName: row.glyph).foregroundStyle(.appSecondary).accessibilityHidden(true)
            VStack(alignment: .leading, spacing: 2) {
                Text(verbatim: UntrustedText.display(row.alias))
                    .themedFont(.small, weight: .medium)
                Text(verbatim: row.statement())
                    .themedFont(.small)
                    .foregroundStyle(.appSecondary)
                    .fixedSize(horizontal: false, vertical: true)
            }
        }
        .accessibilityElement(children: .combine)
    }

    private var testPanel: some View {
        APIPanel(title: Text("Test speech request", bundle: .module)) {
            TextEditor(text: $text)
                .themedFont(.base)
                .frame(height: 74)
                .appEditorSurface()
                .accessibilityLabel(Text("Speech input text", bundle: .module))

            if attachedClonesNatively {
                VStack(alignment: .leading, spacing: 6) {
                    HStack(spacing: AppMetrics.actionSpacing) {
                        Text("Reference voice", bundle: .module)
                        Button { chooseReference() } label: { Text("Choose WAV file", bundle: .module) }
                            .appAction(.secondary, size: APIControlSize.panel)
                        if let referenceURL {
                            Text(verbatim: referenceURL.lastPathComponent).lineLimit(1).truncationMode(.middle)
                        }
                    }
                    Toggle(isOn: $consent) { Text("I have the right to use this voice", bundle: .module) }
                        .disabled(referenceURL == nil)
                }
            } else {
                ViewThatFits(in: .horizontal) {
                    HStack(alignment: .top, spacing: 12) { speechFields }
                    VStack(alignment: .leading, spacing: 8) { speechFields }
                }
                fieldColumn(Text("Reference transcript (optional)", bundle: .module)) {
                    TextField(text: $referenceText, prompt: Text("Transcript of reference voice if cloning", bundle: .module)) {
                        Text("Reference transcript (optional)", bundle: .module)
                    }
                    .textFieldStyle(.roundedBorder)
                }
            }

            HStack(spacing: AppMetrics.actionSpacing) {
                Button {
                    Task { await runTest() }
                } label: { Text("Synthesize", bundle: .module) }
                .appAction(.primary, size: APIControlSize.panel)
                .disabled(testing || attached == nil || text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                    || (attachedClonesNatively && (referenceURL == nil || !consent)))

                APIRunIndicator(running: testing, elapsed: elapsed)

                if let resultWav {
                    Button {
                        togglePlayback(wav: resultWav)
                    } label: {
                        Label {
                            isPlaying ? Text("Stop", bundle: .module) : Text("Play", bundle: .module)
                        } icon: {
                            Image(systemName: isPlaying ? "stop.fill" : "play.fill")
                        }
                    }
                    .appAction(.secondary, size: APIControlSize.panel)
                }
            }
            if let errorMessage {
                Text(verbatim: errorMessage).foregroundStyle(.red).textSelection(.enabled)
            }
        }
    }

    @ViewBuilder
    private var speechFields: some View {
        fieldColumn(Text("Voice", bundle: .module)) {
            TextField(text: $voice, prompt: Text("Voice", bundle: .module)) { Text("Voice", bundle: .module) }
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 120, idealWidth: 160, maxWidth: 200)
        }
        fieldColumn(Text("Language", bundle: .module)) {
            TextField(text: $language, prompt: Text("Language (en, zh, auto)", bundle: .module)) { Text("Language", bundle: .module) }
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 120, idealWidth: 160, maxWidth: 200)
        }
        fieldColumn(Text("Style instruction", bundle: .module)) {
            TextField(text: $instruction, prompt: Text("e.g. Whisper, Cheerful", bundle: .module)) { Text("Style instruction", bundle: .module) }
                .textFieldStyle(.roundedBorder)
                .frame(minWidth: 140, idealWidth: 220, maxWidth: 280)
        }
    }

    private func fieldColumn<Field: View>(_ label: Text, @ViewBuilder field: () -> Field) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            label.themedFont(.small).foregroundStyle(.appSecondary)
            field()
        }
    }

    private func refreshInstalled() {
        Task {
            if let report = try? AudioCatalog.installed() {
                self.installedAudioModels = report.installed
                self.offlineSpeechRows = AudioOfflineInstalls.offlineSpeechRows(in: report)
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
                    NSLocalizedDescriptionKey: String(data: data.prefix(2048), encoding: .utf8) ?? AppStrings.localized("Invalid speech response")
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
