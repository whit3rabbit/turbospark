import Foundation

/// The FluidAudio families the audio catalog can pin (backend
/// `swift_coreml`). Raw values are the persisted catalog family strings, so a
/// row reaches FluidAudio only through a case here. This file does not import
/// FluidAudio: routing and tests read it without the package.
/// docs/FLUIDAUDIO.md has the pins, layouts and placements behind each case.
enum FluidAudioFamily: String, CaseIterable, Sendable {
    case sileroVAD = "fluidaudio_silero_vad"
    case diarizer = "fluidaudio_diarizer"
    case kokoroANE = "fluidaudio_kokoro_ane"
    case pocketTTS = "fluidaudio_pocket_tts"
    case senseVoice = "fluidaudio_sensevoice"
    case paraformerZh = "fluidaudio_paraformer_zh"
    case parakeetJa = "fluidaudio_parakeet_ja"

    /// The folder under the install root that FluidAudio's loader reads. The
    /// catalog row declares the same value as `install_subdir`.
    var expectedInstallSubdir: String? {
        switch self {
        case .kokoroANE: return "kokoro-82m-coreml"
        case .pocketTTS: return "Models/pocket-tts"
        case .sileroVAD, .diarizer, .senseVoice, .paraformerZh, .parakeetJa: return nil
        }
    }

    /// Oldest macOS whose Core ML runs the pinned export. Every export here
    /// targets iOS 15 to 17 programs, so the app's floor holds.
    var minimumMacOSMajor: Int { 14 }

    /// Whether the workspace can run the family end to end: an adapter, its
    /// jobs, and residency. A family flips only when all three ship, so a
    /// catalog row never routes to FluidAudio before the code that runs it.
    var hasAdapter: Bool {
        switch self {
        case .paraformerZh, .parakeetJa:
            return true
        case .sileroVAD, .diarizer, .kokoroANE, .pocketTTS, .senseVoice:
            return false
        }
    }

    var isSupportedOnThisMac: Bool {
        ProcessInfo.processInfo.operatingSystemVersion.majorVersion >= minimumMacOSMajor
    }
}
