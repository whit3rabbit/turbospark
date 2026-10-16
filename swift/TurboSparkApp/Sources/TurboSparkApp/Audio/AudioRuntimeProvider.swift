import TurboSpark

/// Which runtime executes an audio profile, read from its catalog backend.
/// Dispatch sites ask this instead of comparing family strings, so a new
/// Swift-hosted engine adds a case here rather than a branch at every site.
enum AudioRuntimeProvider: Equatable, Sendable {
    /// The Rust native worker (`AudioSession`): Metal, Rust Core ML, CPU.
    case rust
    /// Qwen3-TTS through the vendored Swift MLX provider.
    case qwenMLX
    /// FluidAudio (Core ML) inside the app.
    case fluidAudio(FluidAudioFamily)
    /// A Swift-hosted backend this build cannot run: an unknown family, or a
    /// FluidAudio family whose adapter has not shipped. Pickers hide these.
    case unsupported

    init(_ profile: AudioProfile) {
        switch profile.capabilities.backend {
        case "swift_mlx":
            self = profile.family == "qwen3_tts" ? .qwenMLX : .unsupported
        case "swift_coreml":
            if let family = FluidAudioFamily(rawValue: profile.family), family.hasAdapter {
                self = .fluidAudio(family)
            } else {
                self = .unsupported
            }
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
        case .qwenMLX: return true
        case .fluidAudio(let family): return family.isSupportedOnThisMac
        case .unsupported: return false
        }
    }

    /// FluidAudio loads only the catalog's verified installs. A folder the
    /// user picks never passed the per-file check, and the signature check
    /// alone stops a trap, not wrong weights.
    var acceptsLocalFolder: Bool {
        if case .fluidAudio = self { return false }
        return true
    }
}
