import Foundation

public struct MacAppSettings: Codable, Equatable, Sendable {
    public var contextTokens: Int
    public var expertCacheSlots: Int
    public var temperature: Double
    public var topKEnabled: Bool
    public var topK: Int
    public var topPEnabled: Bool
    public var topP: Double
    public var prefillEnabled: Bool
    public var reasoning: String
    public var maxNewTokens: Int
    public var repetitionPenaltyEnabled: Bool
    public var repetitionPenalty: Double
    public var seedEnabled: Bool
    public var seed: UInt64
    public var stopSequences: String
    public var powerProfile: String
    public var speculation: String
    public var speculativeDrafter: String
    public var maxTokensPerSec: Double
    public var steeringPath: String
    public var steeringMode: String
    public var steeringScale: Double
    public var steeringLayers: String
    public var steeringTarget: Double
    public var steeringGate: Double

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
        speculation: String = "auto",
        speculativeDrafter: String = "auto",
        maxTokensPerSec: Double = 0,
        steeringPath: String = "",
        steeringMode: String = "ablate",
        steeringScale: Double = 1.0,
        steeringLayers: String = "",
        steeringTarget: Double = 0.0,
        steeringGate: Double = 0.0
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
    }
}

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

    public static func load() -> MacAppSettings {
        guard let data = try? Data(contentsOf: settingsFileURL),
              let settings = try? JSONDecoder().decode(MacAppSettings.self, from: data) else {
            return MacAppSettings()
        }
        return settings
    }

    public static func save(_ settings: MacAppSettings) {
        if let data = try? JSONEncoder().encode(settings) {
            try? data.write(to: settingsFileURL, options: .atomic)
        }
    }
}
