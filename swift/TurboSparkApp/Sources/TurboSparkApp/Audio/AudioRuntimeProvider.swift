import TurboSpark

/// Which runtime executes an audio profile, read from its catalog backend.
/// Dispatch sites ask this instead of comparing family strings, so a new
/// Swift-hosted engine adds a case here rather than a branch at every site.
enum AudioRuntimeProvider: Equatable, Sendable {
    /// The Rust native worker (`AudioSession`): Metal, Rust Core ML, CPU.
    case rust
    /// Qwen3-TTS through the vendored Swift MLX provider.
    case qwenMLX
    /// Breeze TTS 2 through the same vendored package (`BreezeTTSModel`).
    case breezeMLX
    /// FluidAudio (Core ML) inside the app.
    case fluidAudio(FluidAudioFamily)
    /// The macOS built-in recognizer (`AppleSpeech`). It has no download and
    /// runs speech-to-text only.
    case appleSpeech
    /// A Swift-hosted backend this build cannot run: an unknown family, or a
    /// FluidAudio family with no job for the row's task. Pickers hide these.
    case unsupported

    init(_ profile: AudioProfile) {
        switch profile.capabilities.backend {
        case "swift_mlx":
            switch profile.family {
            case "qwen3_tts": self = .qwenMLX
            case "breeze_tts": self = .breezeMLX
            default: self = .unsupported
            }
        case "swift_coreml":
            // The task gate keeps a row whose job is not wired out of every
            // picker, so it cannot reach the run-time refusal in `beginRun`.
            if let family = FluidAudioFamily(rawValue: profile.family),
               family.supportedTasks.contains(profile.identity.task) {
                self = .fluidAudio(family)
            } else {
                self = .unsupported
            }
        case AppleSpeech.backend:
            self = profile.family == AppleSpeech.family && profile.identity.task == .speechToText ? .appleSpeech : .unsupported
        default:
            self = .rust
        }
    }

    /// Whether this app can run the profile on this Mac before any opt-in
    /// experiment. Rust rows carry the Rust verdict. A Swift-hosted row's
    /// `canRun` speaks only for a Rust session, so its own provider decides.
    func canRun(_ profile: AudioProfile) -> Bool {
        switch self {
        case .rust: return profile.capabilities.canRun
        case .qwenMLX, .breezeMLX: return true
        case .fluidAudio(let family): return family.isSupportedOnThisMac
        case .appleSpeech: return AppleSpeech.isSupportedOnThisMac
        case .unsupported: return false
        }
    }

    /// FluidAudio loads only the catalog's verified installs. A folder the
    /// user picks never passed the per-file check, and the signature check
    /// alone stops a trap, not wrong weights. The built-in recognizer has no
    /// files to pick.
    var acceptsLocalFolder: Bool {
        switch self {
        case .fluidAudio, .appleSpeech: return false
        case .rust, .qwenMLX, .breezeMLX, .unsupported: return true
        }
    }
}
