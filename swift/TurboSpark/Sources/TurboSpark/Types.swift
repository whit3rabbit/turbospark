import Foundation

// The Rust side emits camelCase, so every type here decodes with no
// `CodingKeys` and the two definitions cannot drift over a spelling.

/// A single message in a conversation.
public struct ChatMessage: Codable, Sendable, Equatable {
    /// The sender role of the message.
    public enum Role: String, Codable, Sendable {
        case system, developer, user, assistant, tool
    }

    /// The role of the message sender.
    public var role: Role
    /// The message text content.
    public var content: String

    /// Creates a new chat message with the given role and content.
    public init(role: Role, content: String) {
        self.role = role
        self.content = content
    }
}

/// How a session is opened. `nil` everywhere means fully automatic, which is
/// what the CLI and the server default to.
public struct OpenOptions: Encodable, Sendable {
    public enum Sizing: Encodable, Sendable {
        case auto
        case fixed(UInt32)

        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .auto: try c.encode("auto")
            case .fixed(let n): try c.encode(n)
            }
        }
    }

    public enum PowerProfile: String, Encodable, Sendable {
        case performance, balanced, efficiency
    }

    /// Whether and how far this session drafts ahead.
    ///
    /// Settled when the model is OPENED, because that is where the
    /// drafter's state is allocated, and not per turn. `.block` throws from
    /// `TurboSparkSession.init` when the install cannot serve it, while
    /// `.auto` opens and explains itself in
    /// `SessionInfo.speculation.reason`.
    public enum Speculation: Encodable, Sendable {
        case auto
        case off
        /// A named block size. The engine accepts 1 through 15; anything
        /// else throws, naming the range.
        case block(UInt32)

        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .auto: try c.encode("auto")
            case .off: try c.encode("off")
            case .block(let n): try c.encode(n)
            }
        }
    }

    /// Which drafter `speculation` drives. The two are ALTERNATIVES rather
    /// than a spectrum: the checkpoint's own MTP head drafts a token at a
    /// time, DFlash2 proposes a whole block in one pass.
    public enum SpeculativeDrafter: String, Encodable, Sendable {
        /// Whichever the install carries -- but `auto` ENABLES an MTP head
        /// and only REPORTS a DFlash2 one, which is measured rather than
        /// stylistic: DFlash2 reads 1.43-1.50x on code and math and 0.96x
        /// throughput at +17.4% J/token on PROSE.
        case auto
        case mtp
        case dflash
    }

    public var maxContext: Sizing?
    public var expertCacheSlots: Sizing?
    /// `nil` ASKS THE OS, so Low Power Mode selects efficiency. Name one
    /// explicitly when measuring anything.
    public var powerProfile: PowerProfile?
    public var maxTokensPerSec: Double?
    /// `nil` means `.auto`, which is what the CLI and the server default to.
    public var speculation: Speculation?
    /// `nil` means `.auto`.
    public var speculativeDrafter: SpeculativeDrafter?

    /// Creates options for opening a model session.
    public init(
        maxContext: Sizing? = nil,
        expertCacheSlots: Sizing? = nil,
        powerProfile: PowerProfile? = nil,
        maxTokensPerSec: Double? = nil,
        speculation: Speculation? = nil,
        speculativeDrafter: SpeculativeDrafter? = nil
    ) {
        self.maxContext = maxContext
        self.expertCacheSlots = expertCacheSlots
        self.powerProfile = powerProfile
        self.maxTokensPerSec = maxTokensPerSec
        self.speculation = speculation
        self.speculativeDrafter = speculativeDrafter
    }
}

/// How one turn is generated. The defaults are the CLI's.
public struct GenerateOptions: Encodable, Sendable {
    /// The accepted set is the CHECKPOINT'S, not this library's. Qwen 3.8
    /// rejects `.high` and its top setting is `.xhigh`; Harmony and Muse
    /// Glimmer accept `.high`. A level a template rejects throws, naming it.
    public enum Reasoning: String, Encodable, Sendable, CaseIterable {
        case off, low, medium, high, xhigh
    }

    public var maxNewTokens: UInt32 = 512
    public var temperature: Double = 0.2
    public var topK: UInt32 = 64
    public var topP: Double = 0.95
    public var repetitionPenalty: Double = 1.0
    public var seed: UInt64?
    public var stop: [String] = []
    public var reasoning: Reasoning = .off

    public init() {}
}

/// What a finished turn reports.
public struct GenerationResult: Decodable, Sendable, Equatable {
    /// The condition that terminated text generation.
    public enum StopReason: String, Decodable, Sendable {
        case endOfTurn, toolCalls, eos, stopString, maxTokens
        /// The caller pressed Stop. The partial turn in `content` is valid
        /// and the conversation can continue from it.
        case cancelled
    }

    public let promptTokens: Int
    public let newTokens: Int
    public let prefillSeconds: Double
    public let decodeSeconds: Double
    public let stopReason: StopReason
    /// Nil when no decoding happened, so a caller cannot plot a rate that
    /// was never measured.
    public let tokensPerSecond: Double?
    /// The reply. THIS is the assistant turn to append to history.
    public let content: String
    /// The model's thinking. Do NOT append it to history: Harmony's own
    /// convention drops prior-turn analysis and Qwen's template drops
    /// prior-turn `<think>` blocks, so feeding it back sends the model
    /// something it was never trained to read.
    public let reasoning: String
}

/// One streamed event.
public enum GenerationEvent: Sendable, Equatable {
    case prefill(done: Int, total: Int)
    case content(String)
    case reasoning(String)
    case finished(GenerationResult)
}

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
    /// What speculative decoding resolved to for this session.
    public let speculation: Speculation

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

/// The decode phase breakdown. Cumulative over every forward pass this
/// session has served, prefill included.
public struct PhaseReport: Decodable, Sendable, Equatable {
    public let calls: UInt64
    public let totalMsPerCall: Double
    public let gpuWaitMs: Double
    public let finalWaitMs: Double
    public let routerMs: Double
    public let expertIoMs: Double
    public let bindMs: Double
    public let pipelineWaitMs: Double
    public let cb1GpuMs: Double
    public let routedCbGpuMs: Double
    public let finalCbGpuMs: Double
    public let expertRequests: UInt64
    public let expertHits: UInt64
    /// Nil before anything has been requested, rather than a 0% rate on no
    /// data.
    public let expertHitRate: Double?
}
