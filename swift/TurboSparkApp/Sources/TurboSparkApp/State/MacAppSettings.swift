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
    /// Named steering directions the operator has registered.
    ///
    /// **NOTHING SHIPS A DIRECTION**, so this is empty until someone supplies
    /// a `.gguf` control vector of their own (`docs/OBLITERATION.md`,
    /// `scripts/extract_direction.py`). The six `steering*` keys above remain
    /// the implicit "Custom" preset the Inspector's raw knobs write to, so a
    /// configuration made before presets existed keeps working.
    public var steeringPresets: [AppSteeringPreset]
    /// `id` of the selected preset, or empty for the Inspector's raw knobs.
    public var activeSteeringPresetID: String
    /// Whether steering is applied at the next model open.
    ///
    /// **OFF BY DEFAULT AND NOT DERIVED FROM "is a path set".** A path is a
    /// configuration; running the edit is a decision, and separating them is
    /// what lets the A/B this feature exists for be one click rather than a
    /// re-typed path.
    public var steeringEnabled: Bool
    /// Default root directory path for local model installs.
    public var modelsDirectory: String
    /// Whether auto-discovery of LM Studio model repositories is enabled.
    public var enableLMStudioDetection: Bool
    /// Custom path to LM Studio models folder if not in default location.
    public var lmStudioDirectory: String
    /// User-configured custom model storage folder paths.
    public var customModelDirectories: [String]
    /// Whether the local command classifier may veto an ALLOWLISTED command,
    /// sending it to the approval sheet.
    ///
    /// Off by default and that is the measurement, not caution: held out by
    /// generator the model scores 0.7010 against 0.9972 in-distribution, and
    /// `python3 -m pytest tests/ -q` scores 1.000, so no threshold stops it
    /// prompting on a command `.auto` promises to run silently. The classifier's
    /// reason strings are shown regardless, since those only decorate a verdict
    /// that was already `.ask`.
    public var commandAdvisoryVeto: Bool
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
    /// Last-used primary interaction mode ("chat" or "projects").
    public var interactionMode: String
    /// Port the in-process server pins to, 0 = first free port. Persisted
    /// because the field had a settings UI that reset every launch; the
    /// server API key deliberately does NOT live here (see
    /// `ServerKeychain`).
    public var serverPinnedPort: UInt16
    /// User-authored system prompt applied to every turn that has no per-chat
    /// prompt of its own.
    ///
    /// It is the FIRST section of the assembled system message, ahead of every
    /// project-derived one, so a project's own rules read as refinements of it
    /// rather than as a competing set of instructions. Empty means no default,
    /// which is what every install written before this field existed decodes
    /// to and therefore leaves their behaviour unchanged.
    public var defaultSystemPrompt: String
    /// User-scope plugin enable state, keyed `<plugin>@<origin>`
    /// (`docs/SWIFT_PLUGINS.md`). Absent means enabled: an installed plugin
    /// that nothing disabled runs. Claude Code's own setting is consulted
    /// only for its interop plugins and only when neither this nor a
    /// project's map answers.
    public var enabledPlugins: [String: Bool]

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
        steeringPresets: [AppSteeringPreset] = [],
        activeSteeringPresetID: String = "",
        steeringEnabled: Bool = false,
        modelsDirectory: String = "",
        enableLMStudioDetection: Bool = true,
        lmStudioDirectory: String = "",
        customModelDirectories: [String] = [],
        commandAdvisoryVeto: Bool = false,
        guardrailsMode: String = "select",
        modelReasoningDefaults: [String: String] = [:],
        interactionMode: String = "chat",
        serverPinnedPort: UInt16 = 0,
        defaultSystemPrompt: String = "",
        enabledPlugins: [String: Bool] = [:]
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
        self.steeringPresets = steeringPresets
        self.activeSteeringPresetID = activeSteeringPresetID
        self.steeringEnabled = steeringEnabled
        self.modelsDirectory = modelsDirectory
        self.enableLMStudioDetection = enableLMStudioDetection
        self.lmStudioDirectory = lmStudioDirectory
        self.customModelDirectories = customModelDirectories
        self.commandAdvisoryVeto = commandAdvisoryVeto
        self.guardrailsMode = guardrailsMode
        self.loadGuard = loadGuard
        self.loadGuardCustomBytes = loadGuardCustomBytes
        self.minAutoContextTokens = minAutoContextTokens
        self.modelReasoningDefaults = modelReasoningDefaults
        self.interactionMode = interactionMode
        self.serverPinnedPort = serverPinnedPort
        self.defaultSystemPrompt = defaultSystemPrompt
        self.enabledPlugins = enabledPlugins
    }

    /// Tolerant of a wrong TYPE as well as an absent key (state#59).
    ///
    /// The doc on `MacAppSettingsFileStore.load` used to say a failure here
    /// "means genuinely corrupt JSON rather than an added field". That was
    /// true of an added field and false of an edited value: `decodeIfPresent`
    /// throws `typeMismatch` on `"seed": -1` or `"topK": "64"`, so a single
    /// hand-edited character quarantined the file and reset every preference
    /// the user had.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self) 
        self.contextTokens = c.decodeLenient(Int.self, forKey: .contextTokens, fallback: 0)
        self.expertCacheSlots = c.decodeLenient(Int.self, forKey: .expertCacheSlots, fallback: 0)
        self.temperature = c.decodeLenient(Double.self, forKey: .temperature, fallback: 0.2)
        self.topKEnabled = c.decodeLenient(Bool.self, forKey: .topKEnabled, fallback: true)
        self.topK = c.decodeLenient(Int.self, forKey: .topK, fallback: 64)
        self.topPEnabled = c.decodeLenient(Bool.self, forKey: .topPEnabled, fallback: true)
        self.topP = c.decodeLenient(Double.self, forKey: .topP, fallback: 0.95)
        self.prefillEnabled = c.decodeLenient(Bool.self, forKey: .prefillEnabled, fallback: true)
        self.reasoning = c.decodeLenient(String.self, forKey: .reasoning, fallback: "off")
        self.maxNewTokens = c.decodeLenient(Int.self, forKey: .maxNewTokens, fallback: 2048)
        self.repetitionPenaltyEnabled = c.decodeLenient(Bool.self, forKey: .repetitionPenaltyEnabled, fallback: false)
        self.repetitionPenalty = c.decodeLenient(Double.self, forKey: .repetitionPenalty, fallback: 1.0)
        self.seedEnabled = c.decodeLenient(Bool.self, forKey: .seedEnabled, fallback: false)
        self.seed = c.decodeLenient(UInt64.self, forKey: .seed, fallback: 0)
        self.stopSequences = c.decodeLenient(String.self, forKey: .stopSequences, fallback: "")
        self.powerProfile = c.decodeLenient(String.self, forKey: .powerProfile, fallback: "auto")
        self.speculation = c.decodeLenient(String.self, forKey: .speculation, fallback: "auto")
        self.speculativeDrafter = c.decodeLenient(String.self, forKey: .speculativeDrafter, fallback: "auto")
        self.maxTokensPerSec = c.decodeLenient(Double.self, forKey: .maxTokensPerSec, fallback: 0)
        self.steeringPath = c.decodeLenient(String.self, forKey: .steeringPath, fallback: "")
        self.steeringMode = c.decodeLenient(String.self, forKey: .steeringMode, fallback: "ablate")
        self.steeringScale = c.decodeLenient(Double.self, forKey: .steeringScale, fallback: 1.0)
        self.steeringLayers = c.decodeLenient(String.self, forKey: .steeringLayers, fallback: "")
        self.steeringTarget = c.decodeLenient(Double.self, forKey: .steeringTarget, fallback: 0.0)
        self.steeringGate = c.decodeLenient(Double.self, forKey: .steeringGate, fallback: 0.0)
        // ELEMENT-LEVEL tolerance, not just key-level (state#45). Decoding
        // the array whole would throw on ONE bad preset and `decodeLenient`
        // would then fall back to `[]`, discarding every good one -- the same
        // shape as the archive loss `swift/CLAUDE.md` Gotcha 13 records.
        self.steeringPresets = c.decodeLenientElements(AppSteeringPreset.self, forKey: .steeringPresets)
        self.activeSteeringPresetID = c.decodeLenient(
            String.self, forKey: .activeSteeringPresetID, fallback: "")
        self.steeringEnabled = c.decodeLenient(
            Bool.self, forKey: .steeringEnabled, fallback: false)
        self.modelsDirectory = c.decodeLenient(String.self, forKey: .modelsDirectory, fallback: "")
        self.enableLMStudioDetection = c.decodeLenient(Bool.self, forKey: .enableLMStudioDetection, fallback: true)
        self.lmStudioDirectory = c.decodeLenient(String.self, forKey: .lmStudioDirectory, fallback: "")
        self.customModelDirectories = c.decodeLenient([String].self, forKey: .customModelDirectories, fallback: [])
        self.commandAdvisoryVeto = c.decodeLenient(Bool.self, forKey: .commandAdvisoryVeto, fallback: false)
        self.guardrailsMode = c.decodeLenient(String.self, forKey: .guardrailsMode, fallback: "select")
        self.loadGuard = c.decodeLenient(String.self, forKey: .loadGuard, fallback: "relaxed")
        self.loadGuardCustomBytes = c.decodeLenient(UInt64.self, forKey: .loadGuardCustomBytes, fallback: 0)
        self.minAutoContextTokens = c.decodeLenient(UInt32.self, forKey: .minAutoContextTokens, fallback: 0)
        self.modelReasoningDefaults = c.decodeLenient([String: String].self, forKey: .modelReasoningDefaults, fallback: [:])
        self.interactionMode = c.decodeLenient(String.self, forKey: .interactionMode, fallback: "chat")
        self.serverPinnedPort = c.decodeLenient(UInt16.self, forKey: .serverPinnedPort, fallback: 0)
        self.defaultSystemPrompt = c.decodeLenient(
            String.self, forKey: .defaultSystemPrompt, fallback: "")
        self.enabledPlugins = c.decodeLenient(
            [String: Bool].self, forKey: .enabledPlugins, fallback: [:])
    }
}

/// JSON persistence storage provider for `MacAppSettings` under `~/Library/Application Support/TurboSpark/settings.json`.
public enum MacAppSettingsFileStore {
    private static var settingsDirectory: URL {
        AppStorageRoot.directory
    }

    private static var settingsFileURL: URL {
        settingsDirectory.appendingPathComponent("settings.json")
    }

    /// Loads application settings from disk or returns defaults if uninitialized.
    ///
    /// `MacAppSettings` decodes every field leniently (state#59), so a
    /// failure here means the JSON itself will not parse -- not an added
    /// field and not one edited to the wrong type. The file is quarantined
    /// rather than overwritten, since every preference would otherwise be
    /// silently reset by the next write.
    public static func load() -> MacAppSettings {
        AppJSONStore.load(MacAppSettings.self, from: settingsFileURL, label: "settings")
            ?? MacAppSettings()
    }

    /// Writes application settings to disk atomically as JSON.
    public static func save(_ settings: MacAppSettings) {
        AppJSONStore.save(settings, to: settingsFileURL, label: "Settings")
    }
}
