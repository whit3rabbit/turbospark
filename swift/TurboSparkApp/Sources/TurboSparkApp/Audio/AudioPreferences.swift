import AVFoundation
import Foundation

/// The persisted audio settings, under one namespace.
///
/// `@AppStorage` keys are strings, and two views spelling the same key by
/// hand drift exactly the way two content-type lists did
/// (`swift/CLAUDE.md` Gotcha 22). Every reader and writer goes through these
/// constants; the defaults live here too, so a view and a service can never
/// disagree about what an unset key means.
enum AudioPreferences {
    /// Master switch for the in-app audio surfaces (docs/AUDIO_UI.md).
    /// Off by default: with it off the composer mic keeps forwarding to
    /// macOS system dictation, exactly as before the audio work landed.
    static let experimentalEnabledKey = "TurboSpark.audio.experimentalEnabled"
    static let speechEngineKey = "TurboSpark.audio.speechEngine"
    static let autoSendAfterDictationKey = "TurboSpark.audio.autoSendAfterDictation"
    static let autoTranscribeAttachmentsKey = "TurboSpark.audio.autoTranscribeAttachments"
    static let maxClipSecondsKey = "TurboSpark.audio.maxClipSeconds"
    static let readAloudVoiceKey = "TurboSpark.audio.readAloudVoice"
    static let readAloudRateKey = "TurboSpark.audio.readAloudRate"
    /// Install path of the engine speech-to-text model. Empty until an audio
    /// model family exists to install (`ts_audio_session_open` refuses all).
    static let engineSpeechModelPathKey = "TurboSpark.audio.engineSpeechModelPath"
    /// Install path of the engine text-to-speech model. Same caveat.
    static let engineVoiceModelPathKey = "TurboSpark.audio.engineVoiceModelPath"

    /// Ten minutes. Long enough for a voice note or a meeting excerpt, short
    /// enough that a forgotten recording does not fill the disk.
    static let defaultMaxClipSeconds = 600
    static let maxClipSecondsChoices = [60, 300, 600, 1800]

    private static var defaults: UserDefaults { .standard }

    static var experimentalEnabled: Bool {
        defaults.bool(forKey: experimentalEnabledKey)
    }

    static var speechEngine: SpeechEngineChoice {
        defaults.string(forKey: speechEngineKey)
            .flatMap(SpeechEngineChoice.init(rawValue:)) ?? .automatic
    }

    static var autoSendAfterDictation: Bool {
        defaults.bool(forKey: autoSendAfterDictationKey)
    }

    /// Defaults to ON, so it cannot use `bool(forKey:)`, which reads an unset
    /// key as false.
    static var autoTranscribeAttachments: Bool {
        defaults.object(forKey: autoTranscribeAttachmentsKey) as? Bool ?? true
    }

    static var maxClipSeconds: Int {
        let stored = defaults.integer(forKey: maxClipSecondsKey)
        return stored > 0 ? stored : defaultMaxClipSeconds
    }

    /// Empty means "the voice for the current language".
    static var readAloudVoiceIdentifier: String {
        defaults.string(forKey: readAloudVoiceKey) ?? ""
    }

    static var readAloudRate: Float {
        let stored = defaults.object(forKey: readAloudRateKey) as? Double
        return Float(stored ?? Double(AVSpeechUtteranceDefaultSpeechRate))
    }

    static var engineSpeechModelPath: String {
        defaults.string(forKey: engineSpeechModelPathKey) ?? ""
    }

    static var engineVoiceModelPath: String {
        defaults.string(forKey: engineVoiceModelPathKey) ?? ""
    }
}

/// Who turns speech into text (and text into speech).
///
/// The engine is always asked first: Rust owns STT/TTS
/// (`crates/audio`), and Swift's Apple frameworks are the native HELPER
/// that covers the gap until an engine audio family exists. The choice is
/// only whether that helper may step in.
enum SpeechEngineChoice: String, CaseIterable, Identifiable, Sendable {
    /// TurboSpark engine first; Apple Speech on device and
    /// `AVSpeechSynthesizer` when the engine has no model.
    case automatic
    /// TurboSpark engine only. Shows the engine's refusal until a model
    /// family lands.
    case engineOnly
    /// The pre-existing `startDictation:` forward for the composer. No
    /// in-app recording; attachments cannot be transcribed.
    case systemDictation

    var id: String { rawValue }

    /// Catalog key for the picker row.
    var titleKey: String {
        switch self {
        case .automatic: return "Automatic (engine, then Apple on device)"
        case .engineOnly: return "TurboSpark engine only"
        case .systemDictation: return "macOS Dictation"
        }
    }
}
