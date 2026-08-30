import Foundation

/// Configuration mode for Forge tool-call guardrails.
public enum AppGuardrailsMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Always active for all tool generations.
    case alwaysOn = "alwaysOn"
    /// Disabled across all tool generations.
    case alwaysOff = "alwaysOff"
    /// Selectable per model / project (defaults to on for tool-calling capable LLMs).
    case select = "select"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .alwaysOn: return "Always On"
        case .alwaysOff: return "Always Off"
        case .select: return "Select (Per Model / Project)"
        }
    }

    public var shortLabel: String {
        switch self {
        case .alwaysOn: return "Always On"
        case .alwaysOff: return "Always Off"
        case .select: return "Select"
        }
    }

    public var descriptionText: String {
        switch self {
        case .alwaysOn:
            return "Forge Guardrails are always active. Tool calls in any format are rescued, argument schemas are strictly checked, and malformed calls are automatically retried."
        case .alwaysOff:
            return "Forge Guardrails are turned off. Model tool calls pass directly without dialect rescue or argument schema validation."
        case .select:
            return "Automatically enabled for LLMs that support tool calling. Can be toggled per project or underneath the prompt composer."
        }
    }
}

/// Persistent model generation parameters, speculation settings, steering vectors, and directory paths for the macOS app.
public struct MacAppSettings: Codable, Equatable, Sendable {
    /// Context window limit in tokens (0 = auto / checkpoint trained context).
    public var contextTokens: Int
    /// Expert cache slot capacity override (0 = auto).
    public var expertCacheSlots: Int
    /// Sampling temperature (0.0 = deterministic greedy).
    public var temperature: Double
    /// Whether top-k sampling is active.
    public var topKEnabled: Bool
    /// Top-k token candidate count.
    public var topK: Int
    /// Whether nucleus top-p filtering is active.
    public var topPEnabled: Bool
    /// Nucleus top-p cumulative probability threshold.
    public var topP: Double
    /// Whether batched/chunked prefill optimization is enabled.
    public var prefillEnabled: Bool
    /// Thinking / reasoning effort level ("off", "low", "medium", "high").
    public var reasoning: String
    /// Maximum number of output tokens generated per turn.
    public var maxNewTokens: Int
    /// Whether repetition penalty is applied.
    public var repetitionPenaltyEnabled: Bool
    /// Repetition penalty factor.
    public var repetitionPenalty: Double
    /// Whether a fixed random seed is specified.
    public var seedEnabled: Bool
    /// Explicit RNG seed for reproducible generation.
    public var seed: UInt64
    /// Comma-separated custom stop sequence tokens/strings.
    public var stopSequences: String
    /// GPU power profile setting ("auto", "low", "high").
    public var powerProfile: String
    /// Speculative decoding mode ("off", "auto", or integer token budget).
    public var speculation: String
    /// Speculative drafter engine ("auto", "mtp", "dflash").
    public var speculativeDrafter: String
    /// Output generation rate cap in tokens/second (0 = uncapped).
    public var maxTokensPerSec: Double
    /// File path to steering vector tensor file.
    public var steeringPath: String
    /// Steering vector application mode ("ablate", "add", "project").
    public var steeringMode: String
    /// Scaling multiplier for directional steering.
    public var steeringScale: Double
    /// Layer index range for activation steering (e.g. "10..20").
    public var steeringLayers: String
    /// Target projection magnitude for steering.
    public var steeringTarget: Double
    /// Activation gate threshold for steering.
    public var steeringGate: Double
    /// Default root directory path for local model installs.
    public var modelsDirectory: String
    /// Whether auto-discovery of LM Studio model repositories is enabled.
    public var enableLMStudioDetection: Bool
    /// Custom path to LM Studio models folder if not in default location.
    public var lmStudioDirectory: String
    /// User-configured custom model storage folder paths.
    public var customModelDirectories: [String]
    /// Forge Tool-Call Guardrails global mode ("alwaysOn", "alwaysOff", "select").
    public var guardrailsMode: String
    /// Memory load guardrail tier ("off", "relaxed", "balanced", "strict",
    /// "custom"). Unrelated to `guardrailsMode` above, which is about TOOL
    /// CALLS; see `docs/LOAD_GUARD.md`.
    public var loadGuard: String
    /// The ceiling the "custom" tier applies, in bytes. 0 falls back to
    /// "relaxed" rather than to a ceiling of nothing.
    public var loadGuardCustomBytes: UInt64
    /// Minimum tokens an automatically-sized context window must reach, or 0
    /// for no floor. Does not constrain an explicit context length.
    public var minAutoContextTokens: UInt32
    /// Remembered reasoning effort setting per model alias or path.
    public var modelReasoningDefaults: [String: String]

    public init(
        contextTokens: Int = 0,
        expertCacheSlots: Int = 0,
        temperature: Double = 0.2,
        topKEnabled: Bool = true,
        topK: Int = 64,
        topPEnabled: Bool = true,
        topP: Double = 0.95,
        prefillEnabled: Bool = true,
        reasoning: String = "off",
        maxNewTokens: Int = 2048,
        repetitionPenaltyEnabled: Bool = false,
        repetitionPenalty: Double = 1.0,
        seedEnabled: Bool = false,
        seed: UInt64 = 0,
        stopSequences: String = "",
        powerProfile: String = "auto",
        loadGuard: String = "relaxed",
        loadGuardCustomBytes: UInt64 = 0,
        minAutoContextTokens: UInt32 = 0,
        speculation: String = "auto",
        speculativeDrafter: String = "auto",
        maxTokensPerSec: Double = 0,
        steeringPath: String = "",
        steeringMode: String = "ablate",
        steeringScale: Double = 1.0,
        steeringLayers: String = "",
        steeringTarget: Double = 0.0,
        steeringGate: Double = 0.0,
        modelsDirectory: String = "",
        enableLMStudioDetection: Bool = true,
        lmStudioDirectory: String = "",
        customModelDirectories: [String] = [],
        guardrailsMode: String = "select",
        modelReasoningDefaults: [String: String] = [:]
    ) {
        self.contextTokens = contextTokens
        self.expertCacheSlots = expertCacheSlots
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.prefillEnabled = prefillEnabled
        self.reasoning = reasoning
        self.maxNewTokens = maxNewTokens
        self.repetitionPenaltyEnabled = repetitionPenaltyEnabled
        self.repetitionPenalty = repetitionPenalty
        self.seedEnabled = seedEnabled
        self.seed = seed
        self.stopSequences = stopSequences
        self.powerProfile = powerProfile
        self.speculation = speculation
        self.speculativeDrafter = speculativeDrafter
        self.maxTokensPerSec = maxTokensPerSec
        self.steeringPath = steeringPath
        self.steeringMode = steeringMode
        self.steeringScale = steeringScale
        self.steeringLayers = steeringLayers
        self.steeringTarget = steeringTarget
        self.steeringGate = steeringGate
        self.modelsDirectory = modelsDirectory
        self.enableLMStudioDetection = enableLMStudioDetection
        self.lmStudioDirectory = lmStudioDirectory
        self.customModelDirectories = customModelDirectories
        self.guardrailsMode = guardrailsMode
        self.loadGuard = loadGuard
        self.loadGuardCustomBytes = loadGuardCustomBytes
        self.minAutoContextTokens = minAutoContextTokens
        self.modelReasoningDefaults = modelReasoningDefaults
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.contextTokens = try c.decodeIfPresent(Int.self, forKey: .contextTokens) ?? 0
        self.expertCacheSlots = try c.decodeIfPresent(Int.self, forKey: .expertCacheSlots) ?? 0
        self.temperature = try c.decodeIfPresent(Double.self, forKey: .temperature) ?? 0.2
        self.topKEnabled = try c.decodeIfPresent(Bool.self, forKey: .topKEnabled) ?? true
        self.topK = try c.decodeIfPresent(Int.self, forKey: .topK) ?? 64
        self.topPEnabled = try c.decodeIfPresent(Bool.self, forKey: .topPEnabled) ?? true
        self.topP = try c.decodeIfPresent(Double.self, forKey: .topP) ?? 0.95
        self.prefillEnabled = try c.decodeIfPresent(Bool.self, forKey: .prefillEnabled) ?? true
        self.reasoning = try c.decodeIfPresent(String.self, forKey: .reasoning) ?? "off"
        self.maxNewTokens = try c.decodeIfPresent(Int.self, forKey: .maxNewTokens) ?? 2048
        self.repetitionPenaltyEnabled = try c.decodeIfPresent(Bool.self, forKey: .repetitionPenaltyEnabled) ?? false
        self.repetitionPenalty = try c.decodeIfPresent(Double.self, forKey: .repetitionPenalty) ?? 1.0
        self.seedEnabled = try c.decodeIfPresent(Bool.self, forKey: .seedEnabled) ?? false
        self.seed = try c.decodeIfPresent(UInt64.self, forKey: .seed) ?? 0
        self.stopSequences = try c.decodeIfPresent(String.self, forKey: .stopSequences) ?? ""
        self.powerProfile = try c.decodeIfPresent(String.self, forKey: .powerProfile) ?? "auto"
        self.speculation = try c.decodeIfPresent(String.self, forKey: .speculation) ?? "auto"
        self.speculativeDrafter = try c.decodeIfPresent(String.self, forKey: .speculativeDrafter) ?? "auto"
        self.maxTokensPerSec = try c.decodeIfPresent(Double.self, forKey: .maxTokensPerSec) ?? 0
        self.steeringPath = try c.decodeIfPresent(String.self, forKey: .steeringPath) ?? ""
        self.steeringMode = try c.decodeIfPresent(String.self, forKey: .steeringMode) ?? "ablate"
        self.steeringScale = try c.decodeIfPresent(Double.self, forKey: .steeringScale) ?? 1.0
        self.steeringLayers = try c.decodeIfPresent(String.self, forKey: .steeringLayers) ?? ""
        self.steeringTarget = try c.decodeIfPresent(Double.self, forKey: .steeringTarget) ?? 0.0
        self.steeringGate = try c.decodeIfPresent(Double.self, forKey: .steeringGate) ?? 0.0
        self.modelsDirectory = try c.decodeIfPresent(String.self, forKey: .modelsDirectory) ?? ""
        self.enableLMStudioDetection = try c.decodeIfPresent(Bool.self, forKey: .enableLMStudioDetection) ?? true
        self.lmStudioDirectory = try c.decodeIfPresent(String.self, forKey: .lmStudioDirectory) ?? ""
        self.customModelDirectories = try c.decodeIfPresent([String].self, forKey: .customModelDirectories) ?? []
        self.guardrailsMode = try c.decodeIfPresent(String.self, forKey: .guardrailsMode) ?? "select"
        self.loadGuard = try c.decodeIfPresent(String.self, forKey: .loadGuard) ?? "relaxed"
        self.loadGuardCustomBytes =
            try c.decodeIfPresent(UInt64.self, forKey: .loadGuardCustomBytes) ?? 0
        self.minAutoContextTokens =
            try c.decodeIfPresent(UInt32.self, forKey: .minAutoContextTokens) ?? 0
        self.modelReasoningDefaults =
            try c.decodeIfPresent([String: String].self, forKey: .modelReasoningDefaults) ?? [:]
    }
}

/// JSON persistence storage provider for `MacAppSettings` under `~/Library/Application Support/TurboSpark/settings.json`.
public enum MacAppSettingsFileStore {
    private static var settingsDirectory: URL {
        let appSupport = FileManager.default.urls(for: .applicationSupportDirectory, in: .userDomainMask).first!
        let directory = appSupport.appendingPathComponent("TurboSpark", isDirectory: true)
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }

    private static var settingsFileURL: URL {
        settingsDirectory.appendingPathComponent("settings.json")
    }

    /// Loads application settings from disk or returns defaults if uninitialized.
    public static func load() -> MacAppSettings {
        guard let data = try? Data(contentsOf: settingsFileURL),
              let settings = try? JSONDecoder().decode(MacAppSettings.self, from: data) else {
            return MacAppSettings()
        }
        return settings
    }

    /// Writes application settings to disk atomically as JSON.
    public static func save(_ settings: MacAppSettings) {
        if let data = try? JSONEncoder().encode(settings) {
            try? data.write(to: settingsFileURL, options: .atomic)
        }
    }
}
