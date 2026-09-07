import Foundation

/// A complete snapshot of the generation sampling knobs, as a chat override
/// or as a saveable preset.
///
/// Ported from Unsloth Studio's thread-scoped settings: a chat may carry its
/// OWN sampling snapshot, applied to every turn that runs in it, while `nil`
/// keeps the app-wide defaults the Inspector has always edited. The fields
/// mirror `AppModel`'s published sampling properties one for one.
///
/// `reasoning` deliberately does NOT travel here (Studio's presets exclude
/// it too): reasoning already has its own per-model memory
/// (`modelReasoningDefaults`) and its own scope, so a sampling preset that
/// also moved the thinking level would surprise in both directions.
public struct AppSamplingSettings: Codable, Equatable, Sendable {
    public var temperature: Double
    public var topKEnabled: Bool
    public var topK: Int
    public var topPEnabled: Bool
    public var topP: Double
    public var maxNewTokens: Int
    public var repetitionPenaltyEnabled: Bool
    public var repetitionPenalty: Double
    public var seedEnabled: Bool
    public var seed: UInt64
    public var stopSequences: String

    /// Defaults are `AppModel`'s own declared defaults, so a preset or an
    /// override written before a field existed decodes to the value that
    /// field's controls already show.
    public init(
        temperature: Double = 0.2,
        topKEnabled: Bool = true,
        topK: Int = 64,
        topPEnabled: Bool = true,
        topP: Double = 0.95,
        maxNewTokens: Int = 2048,
        repetitionPenaltyEnabled: Bool = false,
        repetitionPenalty: Double = 1.0,
        seedEnabled: Bool = false,
        seed: UInt64 = 0,
        stopSequences: String = ""
    ) {
        self.temperature = temperature
        self.topKEnabled = topKEnabled
        self.topK = topK
        self.topPEnabled = topPEnabled
        self.topP = topP
        self.maxNewTokens = maxNewTokens
        self.repetitionPenaltyEnabled = repetitionPenaltyEnabled
        self.repetitionPenalty = repetitionPenalty
        self.seedEnabled = seedEnabled
        self.seed = seed
        self.stopSequences = stopSequences
    }

    /// Every ranged field forced into the range its control offers.
    ///
    /// **THIS STRUCT IS READ STRAIGHT OUT OF A JSON FILE, AND TWO OF ITS
    /// FIELDS BECOME `UInt32` AT THE USE SITE** (state#35's trap, one layer
    /// out): `UInt32(topK)` and `UInt32(maxNewTokens)` both crash the
    /// process on a value above `UInt32.max`, and nothing else on the
    /// per-chat path clamps -- the app-wide path has `clampedSetting` at
    /// load and `UInt32(clamping:)` at the turn, and this struct reaches the
    /// turn without passing either. `topK` keeps 0 as a legal floor because
    /// the engine reads 0 as "off", which is what a hand-edited `topK: 0`
    /// with the toggle on means today; the ceilings are what the UI ranges
    /// promise, and `maxNewTokens` is floored at its stepper's own minimum
    /// (64) so no decoded value can sit below the control showing it.
    public func clamped() -> AppSamplingSettings {
        var clamped = self
        clamped.temperature = min(max(0.0, temperature), 2.0)
        clamped.topK = min(max(0, topK), 256)
        clamped.topP = min(max(0.01, topP), 1.0)
        clamped.maxNewTokens = min(max(64, maxNewTokens), 16384)
        clamped.repetitionPenalty = min(max(1.0, repetitionPenalty), 2.0)
        return clamped
    }

    /// Tolerant of absent keys AND wrong types, like every other persisted
    /// struct in this app (`TolerantDecoding.swift`, state#59): this
    /// snapshot can arrive inside a hand-edited `chats_archive.json` or
    /// `settings.json`, and one bad field must cost one field, not the
    /// archive.
    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self = AppSamplingSettings(
            temperature: c.decodeLenient(Double.self, forKey: .temperature, fallback: 0.2),
            topKEnabled: c.decodeLenient(Bool.self, forKey: .topKEnabled, fallback: true),
            topK: c.decodeLenient(Int.self, forKey: .topK, fallback: 64),
            topPEnabled: c.decodeLenient(Bool.self, forKey: .topPEnabled, fallback: true),
            topP: c.decodeLenient(Double.self, forKey: .topP, fallback: 0.95),
            maxNewTokens: c.decodeLenient(Int.self, forKey: .maxNewTokens, fallback: 2048),
            repetitionPenaltyEnabled: c.decodeLenient(
                Bool.self, forKey: .repetitionPenaltyEnabled, fallback: false),
            repetitionPenalty: c.decodeLenient(Double.self, forKey: .repetitionPenalty, fallback: 1.0),
            seedEnabled: c.decodeLenient(Bool.self, forKey: .seedEnabled, fallback: false),
            seed: c.decodeLenient(UInt64.self, forKey: .seed, fallback: 0),
            stopSequences: c.decodeLenient(String.self, forKey: .stopSequences, fallback: "")
        )
        .clamped()
    }
}

/// A named, saved `AppSamplingSettings` snapshot, listed under the
/// Inspector's Generation Sampling section.
///
/// Nothing ships a preset: the list is empty until the user saves one, and
/// the implicit "default" is simply no override at all (Studio needs a
/// builtin row because its panel edits one flat `params`; here the scope
/// picker's "app defaults" IS the default).
public struct AppSamplingPreset: Identifiable, Codable, Equatable, Sendable {
    public var id: UUID
    public var name: String
    public var settings: AppSamplingSettings

    public init(id: UUID = UUID(), name: String, settings: AppSamplingSettings) {
        self.id = id
        self.name = name
        self.settings = settings
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = c.decodeLenient(UUID.self, forKey: .id, fallback: UUID())
        name = c.decodeLenient(String.self, forKey: .name, fallback: "Preset")
        // Lenient at the KEY level; element-level tolerance for the array
        // holding these is `decodeLenientElements` at the settings-file
        // call site (state#45's shape: one bad preset must not discard the
        // rest).
        settings = ((try? c.decodeIfPresent(AppSamplingSettings.self, forKey: .settings)) ?? nil)
            ?? AppSamplingSettings()
    }
}
