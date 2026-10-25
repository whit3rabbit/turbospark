import TurboSpark

/// What runs an audio model, named next to it: the engine and, for Core ML,
/// where Core ML places the run. Placement comes from the catalog's measured
/// `compute`, never from the units a loader requests (Silero VAD requests the
/// Neural Engine and runs on the CPU). Every surface that names an engine
/// reads this one mapping. Titles and tooltips are catalog keys, listed in
/// `LocalizationParityTests`.
enum AudioEngineLabel: CaseIterable, Equatable {
    case coreMLNeuralEngine, coreMLGPU, coreMLCPU, coreMLMixed, coreML
    case mlxGPU, metalGPU, cpu, experimental, unavailable

    init(backend: String, compute: String?) {
        switch backend {
        case "coreml", "swift_coreml":
            switch compute {
            case "ane": self = .coreMLNeuralEngine
            case "gpu": self = .coreMLGPU
            case "cpu": self = .coreMLCPU
            case "mixed": self = .coreMLMixed
            default: self = .coreML
            }
        case "swift_mlx": self = .mlxGPU
        case "metal": self = .metalGPU
        case "portable_cpu": self = .cpu
        case "experimental_metal_or_portable", "experimental_metal_or_portable_cpu": self = .experimental
        default: self = .unavailable
        }
    }

    init(_ profile: AudioProfile) {
        self.init(backend: profile.capabilities.backend, compute: profile.capabilities.compute)
    }

    var title: String {
        switch self {
        case .coreMLNeuralEngine: return "Core ML (Neural Engine)"
        case .coreMLGPU: return "Core ML (GPU)"
        case .coreMLCPU: return "Core ML (CPU)"
        case .coreMLMixed: return "Core ML (mixed)"
        case .coreML: return "Core ML"
        case .mlxGPU: return "MLX (GPU)"
        case .metalGPU: return "Metal (GPU)"
        case .cpu: return "CPU"
        case .experimental: return "Metal or CPU (experimental)"
        case .unavailable: return "No runtime"
        }
    }

    var tooltip: String {
        switch self {
        case .coreMLNeuralEngine: return "Runs through Core ML, mostly on the Neural Engine."
        case .coreMLGPU: return "Runs through Core ML, mostly on the GPU."
        case .coreMLCPU: return "Runs through Core ML on the CPU."
        case .coreMLMixed: return "Runs through Core ML across several processors."
        case .coreML: return "Runs through Core ML. Its placement was not measured."
        case .mlxGPU: return "Runs on the GPU through MLX."
        case .metalGPU: return "Runs on the GPU through Metal."
        case .cpu: return "Runs on the CPU only."
        case .experimental: return "Experimental. Runs partly on the GPU, or only on the CPU."
        case .unavailable: return "No runtime in this app runs this model yet."
        }
    }

    var systemImage: String {
        switch self {
        case .coreMLNeuralEngine: return "brain"
        case .coreMLGPU, .coreMLCPU, .mlxGPU, .metalGPU, .cpu: return "cpu"
        case .coreMLMixed, .coreML: return "square.stack.3d.up"
        case .experimental: return "flask"
        case .unavailable: return "wrench.and.screwdriver"
        }
    }

    /// Who ships the code that runs the profile. Product names, shown as
    /// written in every language.
    static func provider(for profile: AudioProfile) -> String {
        profile.capabilities.backend == "swift_coreml" ? "FluidAudio" : "TurboSpark"
    }

    /// True when only this app can run the profile: the API server refuses it.
    /// Payloads from before the flag existed leave it nil, never app-only.
    static func isAppOnly(_ profile: AudioProfile) -> Bool {
        profile.capabilities.httpApi == false
    }
}
