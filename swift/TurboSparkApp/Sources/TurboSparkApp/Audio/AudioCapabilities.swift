import AppKit
import AVFoundation
import Foundation
import Speech
import TurboSpark

/// Which system permission an audio capability is waiting on.
enum AudioPermissionKind: String, Sendable {
    case microphone
    case speechRecognition
}

/// Who would serve a speech task right now. Shown next to every transcript
/// and on the read-aloud control, so a user always knows whether the engine
/// or the Apple helper produced it.
enum SpeechProvider: Equatable, Sendable {
    case engine
    case apple

    var label: String {
        switch self {
        case .engine: return String(localized: "TurboSpark engine", bundle: .module)
        case .apple: return String(localized: "Apple (on device)", bundle: .module)
        }
    }
}

/// Whether one audio capability can be used right now, and if not, why.
///
/// Modelled on the vision contract (`sessionInfo.vision.{active, reason}`):
/// a control the user cannot use stays VISIBLE and disabled, and its help
/// text is `reason`. A capability that silently disappears teaches the user
/// nothing; one that says "needs macOS 14.2" tells them what to do.
enum AudioAvailability: Equatable, Sendable {
    case available(SpeechProvider?)
    /// Never asked. Using the control triggers the system prompt.
    case needsPermission(AudioPermissionKind)
    /// Asked and refused. Only System Settings can change it.
    case denied(AudioPermissionKind)
    case unsupportedOS(minimum: String)
    /// The engine has no model for this; carries the engine's own reason.
    case needsModel(String)
    /// Off in Settings > Audio & Voice.
    case disabled
    /// Present on this OS but not right now (recognizer unavailable for the
    /// language, system dictation chosen).
    case unavailable(String)

    /// True when the control should be enabled. A capability that only needs
    /// a first-time prompt is usable: clicking it is how the prompt appears.
    var isUsable: Bool {
        switch self {
        case .available, .needsPermission: return true
        default: return false
        }
    }

    var provider: SpeechProvider? {
        if case .available(let provider) = self { return provider }
        return nil
    }

    /// The sentence a disabled control shows as help, or nil when usable.
    var reason: String? {
        switch self {
        case .available, .needsPermission:
            return nil
        case .denied(.microphone):
            return String(
                localized: "Microphone access is off for TurboSpark. Turn it on in System Settings.",
                bundle: .module)
        case .denied(.speechRecognition):
            return String(
                localized: "Speech recognition is off for TurboSpark. Turn it on in System Settings.",
                bundle: .module)
        case .unsupportedOS(let minimum):
            return String(localized: "Requires macOS \(minimum) or later.", bundle: .module)
        case .needsModel(let engineReason):
            return engineReason
        case .disabled:
            return String(
                localized: "In-app audio is off. Turn it on in Settings > Audio & Voice.",
                bundle: .module)
        case .unavailable(let detail):
            return detail
        }
    }
}

/// Everything `AudioCapabilities.resolve` reads, as plain values, so the
/// decision table is a pure function a test can drive without a microphone.
struct AudioCapabilityInputs: Equatable, Sendable {
    var experimentalEnabled: Bool
    var microphone: AVAuthorizationStatus
    var speech: SFSpeechRecognizerAuthorizationStatus
    /// `SFSpeechRecognizer.supportsOnDeviceRecognition` for the app locale.
    var onDeviceSpeechSupported: Bool
    var choice: SpeechEngineChoice
    /// `ts_audio_capabilities_json` speechToText / textToSpeech, plus whether
    /// a model path is configured to open.
    var engineSpeechToText: AudioTaskStatus
    var engineTextToSpeech: AudioTaskStatus
    var engineSpeechModelConfigured: Bool
    var engineVoiceModelConfigured: Bool
    /// macOS 14.2 introduced Core Audio process taps.
    var supportsProcessTaps: Bool
}

/// One availability per capability in docs/AUDIO_UI.md.
struct AudioCapabilitySnapshot: Equatable, Sendable {
    var micCapture: AudioAvailability
    var transcription: AudioAvailability
    var readAloud: AudioAvailability
    var systemCapture: AudioAvailability
    var attachments: AudioAvailability

    static let allDisabled = AudioCapabilitySnapshot(
        micCapture: .disabled, transcription: .disabled, readAloud: .available(.apple),
        systemCapture: .disabled, attachments: .disabled)
}

/// The single place views ask "can I offer this audio control?".
///
/// Views never probe AVFoundation or the engine themselves: two probes of the
/// same permission at different times disagree, and the control that drifted
/// would offer a recording the service then refuses.
@MainActor
final class AudioCapabilities: ObservableObject {
    static let shared = AudioCapabilities()

    @Published private(set) var snapshot: AudioCapabilitySnapshot = .allDisabled

    /// The engine's table, read once: it describes the linked library, which
    /// cannot change while the app runs. Nil only if the call itself failed,
    /// which the snapshot reports as the engine having no model.
    nonisolated static let engine: AudioEngineCapabilities? = try? TurboSparkAudio.capabilities()

    private static let engineMissing = AudioTaskStatus(
        active: false, reason: "the audio engine did not report its capabilities")

    init() {
        refresh()
    }

    /// Re-reads permissions and preferences. Cheap; call on appear and after
    /// any permission prompt or settings change.
    func refresh() {
        snapshot = Self.resolve(Self.currentInputs())
    }

    static func currentInputs() -> AudioCapabilityInputs {
        let recognizer = SFSpeechRecognizer(locale: Locale.current) ?? SFSpeechRecognizer()
        var processTaps = false
        if #available(macOS 14.2, *) { processTaps = true }
        return AudioCapabilityInputs(
            experimentalEnabled: AudioPreferences.experimentalEnabled,
            microphone: AVCaptureDevice.authorizationStatus(for: .audio),
            speech: SFSpeechRecognizer.authorizationStatus(),
            onDeviceSpeechSupported: recognizer?.supportsOnDeviceRecognition ?? false,
            choice: AudioPreferences.speechEngine,
            engineSpeechToText: engine?.speechToText ?? engineMissing,
            engineTextToSpeech: engine?.textToSpeech ?? engineMissing,
            engineSpeechModelConfigured: !AudioPreferences.engineSpeechModelPath.isEmpty,
            engineVoiceModelConfigured: !AudioPreferences.engineVoiceModelPath.isEmpty,
            supportsProcessTaps: processTaps)
    }

    /// The decision table. Pure: `AudioCapabilitiesTests` covers its branches
    /// without touching hardware.
    nonisolated static func resolve(_ inputs: AudioCapabilityInputs) -> AudioCapabilitySnapshot {
        let readAloud = resolveReadAloud(inputs)
        guard inputs.experimentalEnabled else {
            var snapshot = AudioCapabilitySnapshot.allDisabled
            snapshot.readAloud = readAloud
            return snapshot
        }

        let mic: AudioAvailability
        switch inputs.microphone {
        case .authorized: mic = .available(nil)
        case .notDetermined: mic = .needsPermission(.microphone)
        default: mic = .denied(.microphone)
        }

        let system: AudioAvailability = inputs.supportsProcessTaps
            ? .available(nil) : .unsupportedOS(minimum: "14.2")

        return AudioCapabilitySnapshot(
            micCapture: mic,
            transcription: resolveTranscription(inputs),
            readAloud: readAloud,
            systemCapture: system,
            attachments: .available(nil))
    }

    /// Engine first; the Apple helper only under `.automatic`.
    nonisolated static func resolveTranscription(_ inputs: AudioCapabilityInputs) -> AudioAvailability {
        let engineReason = inputs.engineSpeechToText.reason
            ?? String(localized: "No engine speech model is installed.", bundle: .module)
        if inputs.engineSpeechToText.active && inputs.engineSpeechModelConfigured {
            return .available(.engine)
        }
        switch inputs.choice {
        case .engineOnly:
            return .needsModel(engineReason)
        case .systemDictation:
            // System dictation types into the editor itself; it cannot
            // transcribe a file, so attachments get no transcript.
            return .unavailable(String(
                localized: "macOS Dictation cannot transcribe recordings. Choose Automatic in Settings > Audio & Voice.",
                bundle: .module))
        case .automatic:
            switch inputs.speech {
            case .authorized:
                return inputs.onDeviceSpeechSupported
                    ? .available(.apple)
                    : .unavailable(String(
                        localized: "On-device speech recognition is not available for this language.",
                        bundle: .module))
            case .notDetermined:
                return .needsPermission(.speechRecognition)
            default:
                return .denied(.speechRecognition)
            }
        }
    }

    /// Read aloud predates the audio flag and keeps working with it off: the
    /// engine when it has a voice model, `AVSpeechSynthesizer` otherwise
    /// (except under `.engineOnly`, which refuses with the engine's reason).
    nonisolated static func resolveReadAloud(_ inputs: AudioCapabilityInputs) -> AudioAvailability {
        if inputs.engineTextToSpeech.active && inputs.engineVoiceModelConfigured {
            return .available(.engine)
        }
        if inputs.experimentalEnabled && inputs.choice == .engineOnly {
            return .needsModel(inputs.engineTextToSpeech.reason
                ?? String(localized: "No engine voice model is installed.", bundle: .module))
        }
        return .available(.apple)
    }

    /// Shows the system microphone prompt when it has never been answered.
    /// Returns whether access is granted afterwards.
    func requestMicrophoneAccess() async -> Bool {
        let granted: Bool
        switch AVCaptureDevice.authorizationStatus(for: .audio) {
        case .authorized:
            granted = true
        case .notDetermined:
            granted = await AVCaptureDevice.requestAccess(for: .audio)
        default:
            granted = false
        }
        refresh()
        return granted
    }

    /// Shows the speech recognition prompt when it has never been answered.
    func requestSpeechAccess() async -> Bool {
        var status = SFSpeechRecognizer.authorizationStatus()
        if status == .notDetermined {
            status = await withCheckedContinuation { continuation in
                SFSpeechRecognizer.requestAuthorization { continuation.resume(returning: $0) }
            }
        }
        refresh()
        return status == .authorized
    }

    /// Deep links into the matching Privacy & Security page.
    static func openPrivacySettings(for kind: AudioPermissionKind) {
        let anchor: String
        switch kind {
        case .microphone: anchor = "Privacy_Microphone"
        case .speechRecognition: anchor = "Privacy_SpeechRecognition"
        }
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.security?\(anchor)") {
            NSWorkspace.shared.open(url)
        }
    }

    /// Opens System Settings > Sound, where the input device is chosen.
    static func openSoundSettings() {
        if let url = URL(string: "x-apple.systempreferences:com.apple.preference.sound") {
            NSWorkspace.shared.open(url)
        }
    }
}
