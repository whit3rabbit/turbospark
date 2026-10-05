import AVFoundation
import SwiftUI
import TurboSpark

/// Read aloud, engine first.
///
/// When the engine has a text-to-speech model (`ts_audio_synthesize_json`)
/// it renders the reply to a WAV that `AudioPlaybackController` plays. Until
/// an engine voice family exists, Apple's `AVSpeechSynthesizer` is the
/// native helper, with the voice and rate chosen in Settings > Audio & Voice.
/// Speech and clip playback exclude each other: starting one stops the other.
@MainActor
public final class AppSpeechSynthesizer: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    public static let shared = AppSpeechSynthesizer()

    private let synthesizer = AVSpeechSynthesizer()
    /// A second synthesizer for "Save as audio": rendering to buffers on the
    /// speaking instance would cut off whatever it is reading.
    private var renderer: AVSpeechSynthesizer?
    private var engineTask: Task<Void, Never>?

    /// The ID of the chat message currently being read out loud, if any.
    @Published public private(set) var speakingMessageID: UUID?
    /// Whether speech synthesis is currently active.
    @Published public private(set) var isSpeaking: Bool = false

    public override init() {
        super.init()
        synthesizer.delegate = self
    }

    /// Toggles speech playback for the specified message turn.
    /// If currently speaking the same message, it stops. Otherwise, it starts reading the given text.
    public func toggleSpeech(text: String, messageID: UUID) {
        if isSpeaking && speakingMessageID == messageID {
            stop()
        } else {
            speak(text: text, messageID: messageID)
        }
    }

    /// Speaks the provided text and associates the session with the message ID.
    public func speak(text: String, messageID: UUID) {
        stop()
        AudioPlaybackController.shared.stop()

        let cleanText = Self.sanitizeForSpeech(text)
        guard !cleanText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }
        speakingMessageID = messageID
        isSpeaking = true

        let availability = AudioCapabilities.shared.snapshot.readAloud
        if availability.provider == .engine {
            speakWithEngine(cleanText, messageID: messageID)
        } else if availability.isUsable {
            synthesizer.speak(Self.utterance(for: cleanText))
        } else {
            speakingMessageID = nil
            isSpeaking = false
        }
    }

    /// Immediately stops any ongoing speech output.
    public func stop() {
        engineTask?.cancel()
        engineTask = nil
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        if speakingMessageID != nil, AudioPlaybackController.shared.activeKey == Self.engineKey {
            AudioPlaybackController.shared.stop()
        }
        speakingMessageID = nil
        isSpeaking = false
    }

    /// Renders `text` to a WAV file for "Save as audio", by the same
    /// provider read aloud would use. The caller owns the returned file.
    func renderToFile(text: String) async throws -> URL {
        let cleanText = Self.sanitizeForSpeech(text)
        let destination = AudioEngineBridge.temporaryURL(extension: "wav")
        if AudioCapabilities.shared.snapshot.readAloud.provider == .engine {
            let session = try await TurboSparkAudioSession(
                modelPath: AudioPreferences.engineVoiceModelPath)
            _ = try await session.synthesize(cleanText, to: destination)
            return destination
        }
        let renderer = AVSpeechSynthesizer()
        self.renderer = renderer
        defer { self.renderer = nil }
        let sink = SpeechFileSink(url: destination)
        let utterance = Self.utterance(for: cleanText)
        try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<Void, Error>) in
            renderer.write(utterance) { buffer in
                guard let pcm = buffer as? AVAudioPCMBuffer else { return }
                if pcm.frameLength == 0 {
                    // A zero-length buffer is the end-of-utterance marker;
                    // `finish` answers only once, so a repeat cannot resume
                    // the continuation twice.
                    if let result = sink.finish() { continuation.resume(with: result) }
                } else {
                    sink.append(pcm)
                }
            }
        }
        return destination
    }

    private static var engineKey: String { AudioPlaybackController.readAloudKey }

    private func speakWithEngine(_ text: String, messageID: UUID) {
        engineTask = Task { [weak self] in
            do {
                let url = try await self?.renderToFile(text: text)
                guard let self, let url, !Task.isCancelled, self.speakingMessageID == messageID else { return }
                AudioPlaybackController.shared.onFinish = { [weak self] key in
                    guard key == Self.engineKey, self?.speakingMessageID == messageID else { return }
                    self?.speakingMessageID = nil
                    self?.isSpeaking = false
                    try? FileManager.default.removeItem(at: url)
                }
                AudioPlaybackController.shared.play(url: url, key: Self.engineKey)
            } catch {
                self?.speakingMessageID = nil
                self?.isSpeaking = false
            }
        }
    }

    /// One utterance with the user's voice and rate.
    private static func utterance(for text: String) -> AVSpeechUtterance {
        let utterance = AVSpeechUtterance(string: text)
        let identifier = AudioPreferences.readAloudVoiceIdentifier
        if !identifier.isEmpty, let voice = AVSpeechSynthesisVoice(identifier: identifier) {
            utterance.voice = voice
        } else if let preferredVoice = AVSpeechSynthesisVoice(
            language: AVSpeechSynthesisVoice.currentLanguageCode())
        {
            utterance.voice = preferredVoice
        }
        utterance.rate = min(
            AVSpeechUtteranceMaximumSpeechRate,
            max(AVSpeechUtteranceMinimumSpeechRate, AudioPreferences.readAloudRate))
        return utterance
    }

    /// Strips markdown code blocks and symbols to provide a smoother reading experience.
    static func sanitizeForSpeech(_ input: String) -> String {
        var text = input
        // Remove code block markers
        text = text.replacingOccurrences(of: "```[a-zA-Z0-9_-]*\\n", with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: "```", with: "")
        // Remove markdown headers
        text = text.replacingOccurrences(of: "(?m)^#{1,6}\\s+", with: "", options: .regularExpression)
        // Remove markdown bold/italic asterisks
        text = text.replacingOccurrences(of: "\\*\\*", with: "")
        text = text.replacingOccurrences(of: "__", with: "")
        return text
    }

    // MARK: - AVSpeechSynthesizerDelegate

    public nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didFinish utterance: AVSpeechUtterance
    ) {
        Task { @MainActor in
            self.speakingMessageID = nil
            self.isSpeaking = false
        }
    }

    public nonisolated func speechSynthesizer(
        _ synthesizer: AVSpeechSynthesizer,
        didCancel utterance: AVSpeechUtterance
    ) {
        Task { @MainActor in
            self.speakingMessageID = nil
            self.isSpeaking = false
        }
    }
}

/// Accumulates synthesizer buffers into a WAV. Called from the
/// synthesizer's callback queue, so it locks.
private final class SpeechFileSink: @unchecked Sendable {
    private let url: URL
    private let lock = NSLock()
    private var file: AVAudioFile?
    private var error: Error?
    private var finished = false

    init(url: URL) {
        self.url = url
    }

    func append(_ buffer: AVAudioPCMBuffer) {
        lock.lock()
        defer { lock.unlock() }
        guard error == nil, !finished else { return }
        do {
            if file == nil {
                file = try AVAudioFile(
                    forWriting: url, settings: AudioCaptureService.wavSettings(for: buffer.format),
                    commonFormat: buffer.format.commonFormat, interleaved: buffer.format.isInterleaved)
            }
            try file?.write(from: buffer)
        } catch {
            self.error = error
        }
    }

    /// Closes the file and reports the first write error, if any. Returns
    /// nil on every call after the first, so a continuation resumes once.
    func finish() -> Result<Void, Error>? {
        lock.lock()
        defer { lock.unlock() }
        guard !finished else { return nil }
        finished = true
        file = nil
        if let error { return .failure(error) }
        return .success(())
    }
}
