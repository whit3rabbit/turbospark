import AVFoundation
import SwiftUI

/// Service providing native macOS Text-to-Speech (TTS) using Apple's built-in `AVSpeechSynthesizer`.
@MainActor
public final class AppSpeechSynthesizer: NSObject, ObservableObject, AVSpeechSynthesizerDelegate {
    public static let shared = AppSpeechSynthesizer()

    private let synthesizer = AVSpeechSynthesizer()

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

        let cleanText = sanitizeForSpeech(text)
        guard !cleanText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            return
        }

        let utterance = AVSpeechUtterance(string: cleanText)
        if let preferredVoice = AVSpeechSynthesisVoice(language: AVSpeechSynthesisVoice.currentLanguageCode()) {
            utterance.voice = preferredVoice
        }
        utterance.rate = AVSpeechUtteranceDefaultSpeechRate

        speakingMessageID = messageID
        isSpeaking = true
        synthesizer.speak(utterance)
    }

    /// Immediately stops any ongoing speech output.
    public func stop() {
        if synthesizer.isSpeaking {
            synthesizer.stopSpeaking(at: .immediate)
        }
        speakingMessageID = nil
        isSpeaking = false
    }

    /// Strips markdown code blocks and symbols to provide a smoother reading experience.
    private func sanitizeForSpeech(_ input: String) -> String {
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
