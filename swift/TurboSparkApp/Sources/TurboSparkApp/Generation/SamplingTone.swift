import Foundation

/// The composer sampling pill's built-in tones.
///
/// The app reads no per-model recommended sampling (generation_config.json
/// is installed but never parsed for temperature), so Balanced cannot be
/// "the model's default". It is the app's own shipped default instead
/// (`AppSamplingSettings()`), which is what every chat already runs with, so
/// a fresh install reads Balanced. Precise and Creative are conservative
/// steps either side. None of the three is measured per family: a wrong
/// temperature on a small model can loop or ramble, which is why the
/// popover prints the numbers and says so.
enum SamplingTone: String, CaseIterable, Identifiable, Sendable {
    case precise
    case balanced
    case creative

    var id: String { rawValue }

    /// The knobs a tone sets. Everything else (length, repetition penalty,
    /// seed, stops) stays the user's.
    struct Values: Equatable, Sendable {
        let temperature: Double
        let topK: Int
        let topP: Double
    }

    /// Data, kept in one table so a measured revision changes one place.
    /// No tone uses temperature 0: the sampler's deterministic path skips
    /// repetition penalty and truncation, so greedy can loop where 0.1 does not.
    var values: Values {
        switch self {
        case .precise:
            return Values(temperature: 0.1, topK: 20, topP: 0.8)
        case .balanced:
            let d = AppSamplingSettings()
            return Values(temperature: d.temperature, topK: d.topK, topP: d.topP)
        case .creative:
            return Values(temperature: 0.8, topK: 64, topP: 0.95)
        }
    }
}

/// What the current sampling snapshot reads as in the pill.
enum SamplingPresetMatch: Equatable, Sendable {
    case tone(SamplingTone)
    case saved(UUID)
    case custom
}

/// Where a pill pick writes. Mirrors the Inspector's storage: the chat's
/// override when it has one, the app-wide settings otherwise.
enum SamplingPickScope: Equatable, Sendable {
    case thisChat
    case allChats
}

enum SamplingToneMatcher {
    /// Well under half the finest slider step (0.01 for top-p, 0.05 for
    /// temperature), so JSON or arithmetic drift still matches while one
    /// real step reads as Custom.
    static let tolerance = 0.001

    static func scope(hasChatOverride: Bool) -> SamplingPickScope {
        hasChatOverride ? .thisChat : .allChats
    }

    /// `settings` with the tone's knobs set and both truncation toggles on
    /// (a tone's top-k/top-p would otherwise be inert).
    static func applying(_ tone: SamplingTone, to settings: AppSamplingSettings) -> AppSamplingSettings {
        var out = settings
        let v = tone.values
        out.temperature = v.temperature
        out.topKEnabled = true
        out.topK = v.topK
        out.topPEnabled = true
        out.topP = v.topP
        return out
    }

    /// Built-in tones first, so the name does not depend on what is saved,
    /// then the first saved preset equal on every field, else Custom.
    static func match(_ settings: AppSamplingSettings, savedPresets: [AppSamplingPreset]) -> SamplingPresetMatch {
        for tone in SamplingTone.allCases where matches(settings, tone: tone) {
            return .tone(tone)
        }
        if let preset = savedPresets.first(where: { equivalent(settings, $0.settings) }) {
            return .saved(preset.id)
        }
        return .custom
    }

    private static func close(_ a: Double, _ b: Double) -> Bool {
        abs(a - b) <= tolerance
    }

    private static func matches(_ s: AppSamplingSettings, tone: SamplingTone) -> Bool {
        let v = tone.values
        return close(s.temperature, v.temperature)
            && s.topKEnabled && s.topK == v.topK
            && s.topPEnabled && close(s.topP, v.topP)
    }

    /// Every field that reaches the engine. A value behind a disabled toggle
    /// never does (`makeOptions`), so it is ignored.
    private static func equivalent(_ a: AppSamplingSettings, _ b: AppSamplingSettings) -> Bool {
        guard close(a.temperature, b.temperature),
            a.maxNewTokens == b.maxNewTokens,
            a.topKEnabled == b.topKEnabled,
            a.topPEnabled == b.topPEnabled,
            a.repetitionPenaltyEnabled == b.repetitionPenaltyEnabled,
            a.seedEnabled == b.seedEnabled
        else { return false }
        if a.topKEnabled && a.topK != b.topK { return false }
        if a.topPEnabled && !close(a.topP, b.topP) { return false }
        if a.repetitionPenaltyEnabled && !close(a.repetitionPenalty, b.repetitionPenalty) { return false }
        if a.seedEnabled && a.seed != b.seed { return false }
        return stops(a.stopSequences) == stops(b.stopSequences)
    }

    /// Same split as `makeOptions`, so "a, b" and "a,b" are one stop list.
    private static func stops(_ raw: String) -> [String] {
        raw.split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }
}
