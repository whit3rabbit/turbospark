import Foundation

/// Options for `TurboSparkSession.humanize`.
///
/// Only the budget and the seed exist, on purpose. The humanizer model's
/// sampling recipe (temperature 1.0, top_p 0.95, top_k off, EOS-only stop) is
/// its checkpoint's contract and is applied inside the library; the engine
/// refuses any other key by name, so this type simply has no place for one.
public struct HumanizeOptions: Encodable, Sendable, Equatable {
    /// The output budget in tokens. Nil takes the publisher's rule (about 2.5x
    /// the draft's tokens, within 256...2048), clamped to the model's window.
    public var maxNewTokens: UInt32?
    /// Fixes the sampler's stream. Nil samples fresh, so a repeat call is a
    /// new rewrite, which is what a "resample" button wants.
    public var seed: UInt64?

    public init(maxNewTokens: UInt32? = nil, seed: UInt64? = nil) {
        self.maxNewTokens = maxNewTokens
        self.seed = seed
    }
}

/// What the library checked after a rewrite.
///
/// Every field is a WARNING for the reader, never a correction: the model's
/// publisher reports that fact slips still happen, so the audit's job is to
/// point a proofreader at the right sentences.
public struct HumanizeAudit: Decodable, Sendable, Equatable {
    /// Digit runs in the draft that the rewrite dropped, one entry per
    /// missing occurrence.
    public let droppedNumbers: [String]
    /// Digit runs in the rewrite that the draft does not contain.
    public let inventedNumbers: [String]
    /// The rewrite's dominant script (Latin or CJK) differs from the draft's.
    public let languageDrift: Bool
    /// Share of the rewrite's word-level 5-grams copied verbatim from the draft.
    public let reuseRatio: Double
    /// `reuseRatio` is above the level the model was trained against.
    public let reuseFlagged: Bool
    /// The same findings pre-rendered in English; empty when clean. Not
    /// localized: build localized text from the structured fields above.
    public let warnings: [String]

    /// True when nothing needs a second look.
    public var isClean: Bool {
        droppedNumbers.isEmpty && inventedNumbers.isEmpty && !languageDrift && !reuseFlagged
    }
}
