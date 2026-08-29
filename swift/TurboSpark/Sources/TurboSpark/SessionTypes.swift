import Foundation

/// Everything resolved when the model was opened.
public struct SessionInfo: Decodable, Sendable, Equatable {
    public enum ReasoningSupport: String, Decodable, Sendable {
        /// The template takes a level.
        case level
        /// The template can only turn thinking ON; a level is dropped. Grey
        /// out the levels rather than hiding the toggle.
        case toggleOnly
        /// No template at all. Asking for a level throws.
        case none
    }

    public let modelPath: String
    public let family: String
    /// The RESOLVED window, not what was asked for.
    public let maxContext: UInt32
    public let trainedContext: UInt32?
    /// Above the trained context the model still runs; quality degrades.
    public let pastTrainedContext: Bool
    /// The RESOLVED slot count. No throughput or footprint figure is
    /// readable without it.
    public let expertCacheSlots: Int
    public let vocabSize: Int
    public let dialect: String
    public let reasoningSupport: ReasoningSupport
    /// What directional steering resolved to for this session.
    public let steering: Steering
    /// What speculative decoding resolved to for this session.
    public let speculation: Speculation
    /// Special token identifiers for tokenizer inspection.
    public let specialTokens: SpecialTokens

    /// Special token identifiers for tokenizer introspection.
    public struct SpecialTokens: Decodable, Sendable, Equatable {
        public let bosId: Int32?
        public let eosId: Int32?
        public let padId: Int32?
        public let endOfTurnId: Int32?
        public let stopTokenIds: [Int32]
        public let thinkStartId: Int32?
        public let thinkEndId: Int32?
    }

    /// The session's resolved directional steering, reported once.
    public struct Steering: Decodable, Sendable, Equatable {
        /// True when a control vector is active on this session.
        public let active: Bool
        /// `ablate` | `add` | `clamp` | `renorm`, present only when active.
        public let mode: String?
        /// Active scale multiplier, present only when active.
        public let scale: Double?
        /// Human-readable one-line description, or nil when inactive.
        public let summary: String?
    }

    /// The session's resolved speculative decoding, reported once.
    public struct Speculation: Decodable, Sendable, Equatable {
        public enum Drafter: String, Decodable, Sendable {
            /// The checkpoint's own multi-token-prediction head, a token at
            /// a time.
            case mtp
            /// The DFlash2 block-diffusion drafter, a whole block per pass.
            case dflash
        }

        /// How many tokens a round proposes, or nil when this session does
        /// not draft ahead. THIS is the "is it on" test; there is no
        /// separate flag that could disagree with it.
        ///
        /// **Non-nil is a statement about the SESSION, not about the next
        /// turn.** Acceptance is exact only at temperature 0, so a sampled
        /// turn decodes sequentially whatever this says. Send
        /// `GenerateOptions.temperature = 0` to speculate.
        public let block: Int?
        /// Non-nil exactly when `block` is. Two drafters serve one family
        /// with different shapes and different measured optima, so a
        /// throughput figure is unreadable without knowing which ran.
        public let drafter: Drafter?
        /// Why speculation is off, when the caller might have expected it
        /// on. Nil both when `.off` was asked for and when it is on. Worth
        /// surfacing: an install carrying a drafter and decoding one token
        /// at a time with nothing said is the failure this feature exists
        /// to end.
        public let reason: String?
    }
}
