import Foundation
import TurboSpark

/// A named, reusable directional-steering configuration: one control vector
/// plus the four knobs that decide what the edit does with it.
///
/// **NOTHING SHIPS A DIRECTION, AND THE UI HAS TO SAY SO.** This engine is
/// agnostic to what a vector encodes -- the same code path serves concept
/// steering, style vectors, interpretability probes and refusal-direction work
/// -- so provenance is the operator's. A preset with a path nobody supplied
/// steers nothing, and a control labelled as though it shipped a behaviour
/// would be claiming work that does not exist. `scripts/extract_direction.py`
/// and `docs/OBLITERATION.md` are the route to a real one.
///
/// Presets exist because the alternative shipped for months and was unusable:
/// six raw numeric fields in the Inspector, a hand-typed `.gguf` path, no
/// compatibility check, and no way to turn the whole thing off.
public struct AppSteeringPreset: Codable, Equatable, Identifiable, Sendable {
    public var id: UUID
    /// What the operator calls this direction. Free text on purpose: only
    /// they know what their vector encodes.
    public var name: String
    /// Absolute path to a `.gguf` control vector in llama.cpp layout.
    public var vectorPath: String
    public var mode: AppSteeringModeOption
    public var scale: Double
    /// `START:END`, 0-based and inclusive, or empty for every covered layer.
    public var layers: String
    /// The coefficient `clamp` pins to. Ignored by every other mode.
    public var target: Double
    /// Only steer where the direction's own coefficient reaches this
    /// magnitude. 0 always fires.
    public var gate: Double
    public var notes: String

    /// The hidden size the vector file itself declares, recorded when the
    /// preset was registered.
    ///
    /// **`nil` MEANS NOT READ, NEVER ZERO-WIDTH.** A preset restored from a
    /// settings file written before this field existed, or one whose file
    /// could not be read, has no width -- and rendering that as `0` would
    /// show a width mismatch against every model instead of "unknown"
    /// (`swift/CLAUDE.md` Gotcha 23).
    public var vectorHidden: Int?
    /// The span the vector's directions cover, which is what a model's layer
    /// count is compared against.
    ///
    /// **THE SPAN, NOT THE COUNT.** A vector carrying 63 directions across
    /// blocks 1...63 spans 64, and it is 64 that has to fit. Comparing the
    /// count would call it compatible with a 63-layer model it overruns.
    public var vectorSpannedLayers: Int?

    public init(
        id: UUID = UUID(),
        name: String = "",
        vectorPath: String = "",
        mode: AppSteeringModeOption = .ablate,
        scale: Double = 0.3,
        layers: String = "",
        target: Double = 0.0,
        gate: Double = 0.0,
        notes: String = "",
        vectorHidden: Int? = nil,
        vectorSpannedLayers: Int? = nil
    ) {
        self.id = id
        self.name = name
        self.vectorPath = vectorPath
        self.mode = mode
        self.scale = scale
        self.layers = layers
        self.target = target
        self.gate = gate
        self.notes = notes
        self.vectorHidden = vectorHidden
        self.vectorSpannedLayers = vectorSpannedLayers
    }

    /// **THE DEFAULT SCALE IS 0.3 AND NOT 1.0, and that is a measurement
    /// rather than a preference.** `docs/OBLITERATION.md` records ablation at
    /// alpha 1.0 across every layer COLLAPSING the turn on a real 27B install
    /// -- the model emitted its end-of-turn token immediately and generated
    /// nothing. The probe in that repo defaulted to 1.0 once, read a huge
    /// divergence, and passed; a default that lands on a documented failure
    /// mode is worse than no default.
    public static let defaultScale: Double = 0.3

    /// Hand-written for `swift/CLAUDE.md` Gotcha 13 and state#45: one element
    /// that will not decode must cost that element, never the whole settings
    /// file. `decodeIfPresent` tolerates an ABSENT key and throws on a wrong
    /// TYPE or an unknown enum case, which is exactly how a hand-edited
    /// `settings.json` has already quarantined every preference here once.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        self.id = (try? c.decodeIfPresent(UUID.self, forKey: .id)) .flatMap { $0 } ?? UUID()
        self.name = (try? c.decodeIfPresent(String.self, forKey: .name)).flatMap { $0 } ?? ""
        self.vectorPath =
            (try? c.decodeIfPresent(String.self, forKey: .vectorPath)).flatMap { $0 } ?? ""
        self.mode =
            (try? c.decodeIfPresent(String.self, forKey: .mode))
            .flatMap { $0 }
            .flatMap(AppSteeringModeOption.init(rawValue:)) ?? .ablate
        self.scale =
            (try? c.decodeIfPresent(Double.self, forKey: .scale)).flatMap { $0 }
            ?? Self.defaultScale
        self.layers = (try? c.decodeIfPresent(String.self, forKey: .layers)).flatMap { $0 } ?? ""
        self.target = (try? c.decodeIfPresent(Double.self, forKey: .target)).flatMap { $0 } ?? 0.0
        self.gate = (try? c.decodeIfPresent(Double.self, forKey: .gate)).flatMap { $0 } ?? 0.0
        self.notes = (try? c.decodeIfPresent(String.self, forKey: .notes)).flatMap { $0 } ?? ""
        self.vectorHidden = (try? c.decodeIfPresent(Int.self, forKey: .vectorHidden)).flatMap { $0 }
        self.vectorSpannedLayers =
            (try? c.decodeIfPresent(Int.self, forKey: .vectorSpannedLayers)).flatMap { $0 }
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, vectorPath, mode, scale, layers, target, gate, notes
        case vectorHidden, vectorSpannedLayers
    }

    /// `mode` is stored as its raw string so a future case cannot make an old
    /// settings file undecodable.
    public func encode(to encoder: Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(id, forKey: .id)
        try c.encode(name, forKey: .name)
        try c.encode(vectorPath, forKey: .vectorPath)
        try c.encode(mode.rawValue, forKey: .mode)
        try c.encode(scale, forKey: .scale)
        try c.encode(layers, forKey: .layers)
        try c.encode(target, forKey: .target)
        try c.encode(gate, forKey: .gate)
        try c.encode(notes, forKey: .notes)
        try c.encodeIfPresent(vectorHidden, forKey: .vectorHidden)
        try c.encodeIfPresent(vectorSpannedLayers, forKey: .vectorSpannedLayers)
    }

    /// A display name that is never empty, so a row cannot render as a blank.
    public var displayName: String {
        let trimmed = name.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmed.isEmpty { return trimmed }
        let file = (vectorPath as NSString).lastPathComponent
        return file.isEmpty ? "Untitled direction" : file
    }
}
