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
    /// What KIND of reasoning control is meaningful. See `reasoningLevels`
    /// for what to put in it.
    public let reasoningSupport: ReasoningSupport
    /// The spellings this checkpoint's own template can express, ascending,
    /// always opening at `"off"`. Prefer the typed `reasoningEfforts`.
    ///
    /// Stored as strings rather than as `[GenerateOptions.Reasoning]` on
    /// purpose. `SessionInfo` is decoded inside `TurboSparkSession.init`,
    /// whose failure path has to close the C handle by hand (Gotcha 31), so
    /// a sixth level added to the engine one day must not take down every
    /// session open on this side. A field-NAME drift still fails the decode,
    /// which is the guard Gotcha 5 wants; an unknown VALUE is dropped.
    public let reasoningLevels: [String]
    /// What directional steering resolved to for this session.
    public let steering: Steering
    /// Whether this checkpoint's own markup frames tool calls.
    public let toolCalling: ToolCalling
    /// What speculative decoding resolved to for this session.
    public let speculation: Speculation
    /// Whether this session would accept an image.
    public let vision: Vision
    /// Special token identifiers for tokenizer inspection.
    public let specialTokens: SpecialTokens
    /// The RESOLVED `kvBits` selection: `"off"`, `"2"`, `"3"`, `"4"`, or
    /// `"3.5 (K3/V4)"` when the two widths differ. Unlike `speculation`
    /// there is no auto-detect here: a named width either opens this
    /// session or `TurboSparkSession.init` throws, so what was asked for
    /// and what this session runs at are always the same value.
    public let kvBits: String

    /// The reasoning levels this checkpoint accepts. **BUILD A PICKER FROM
    /// THIS AND FROM NOTHING ELSE.**
    ///
    /// The set belongs to the checkpoint and cannot be derived from the
    /// family: Qwen 3.8 answers `[.off, .low, .medium, .xhigh]` and RAISES
    /// on `.high`, where gpt-oss and Muse Glimmer answer
    /// `[.off, .low, .medium, .high]`. Sending a level absent from here
    /// fails the turn with the template's own error, so a menu offering all
    /// five is a menu with a broken entry in it.
    ///
    /// Levels rendering the same prompt are already collapsed by the engine,
    /// so a `.toggleOnly` checkpoint answers exactly two. Its second entry is
    /// `.low` BY POSITION and is not a label: read `reasoningSupport` and
    /// present that case as an on/off switch.
    public var reasoningEfforts: [GenerateOptions.Reasoning] {
        reasoningLevels.compactMap(GenerateOptions.Reasoning.init(rawValue:))
    }

    /// Whether an image sent to this session would actually be SERVED.
    ///
    /// **`active` is not "does this family have a tower".** An install can
    /// carry one and still refuse every image: the pixel budget is read from
    /// the checkpoint's own `preprocessor_config.json` and has no default
    /// worth falling back to, so an install streamed without that sidecar
    /// reports `active == false` with the reason. Gate an attach control on
    /// THIS, or the control promises work the engine then declines.
    public struct Vision: Decodable, Sendable, Equatable {
        /// True when this session would encode and inject an image.
        public let active: Bool
        /// The `<|image_pad|>` id, or nil when inactive.
        public let imageTokenId: Int32?
        /// Why images are refused, non-nil exactly when the install has a
        /// tower and `active` is false. That is the only case a caller can
        /// act on, so it is the only case that says anything.
        public let reason: String?
    }

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
        /// **A DIFFERENT QUESTION FROM `active`, and the one to gate a
        /// control on.**
        ///
        /// `active` says a vector is running. This says one COULD be:
        /// `TurboSparkSession.init` REFUSES a control vector on a family
        /// whose decode flow does not dispatch the edit, so a UI offering the
        /// knob against such an install offers one whose only outcome is a
        /// failed load. An unsteered session on a family that steers answers
        /// `active: false, supported: true`.
        public let supported: Bool
        /// Why `supported` is false, in the words the open would refuse with.
        /// `nil` when it is true.
        public let reason: String?
        /// `ablate` | `add` | `clamp` | `renorm`, present only when active.
        public let mode: String?
        /// Active scale multiplier, present only when active.
        public let scale: Double?
        /// Human-readable one-line description, or nil when inactive.
        public let summary: String?
    }

    /// Whether this checkpoint's OWN markup carries tool calls the engine
    /// parses.
    ///
    /// **`native == false` IS NOT "TOOLS DO NOT WORK", and hiding a tool
    /// control on it is the wrong reading.** A model on a dialect with no
    /// tool markup can still be prompted into emitting a call as ordinary
    /// prose, and recovering exactly that is what a guardrail rescue is for,
    /// so `false` marks the case a rescue helps MOST rather than a case to
    /// refuse. Report it; do not gate on it.
    public struct ToolCalling: Decodable, Sendable, Equatable {
        /// True when a call can arrive already framed, i.e. when the engine's
        /// structured decoder can hand one over.
        public let native: Bool
        /// Why `native` is false, naming the dialect. One checkpoint family
        /// frames calls in markup this engine has no parser for and reports
        /// them as reasoning instead; its reason says so. `nil` when `native`
        /// is true.
        public let reason: String?
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
