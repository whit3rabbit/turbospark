import Foundation

/// How a session is opened. `nil` everywhere means fully automatic, which is
/// what the CLI and the server default to.
public struct OpenOptions: Encodable, Sendable {
    /// Sizing mode for context window or expert slot caching.
    public enum Sizing: Encodable, Sendable {
        /// Automatic sizing resolved by the engine.
        case auto
        /// Fixed capacity in slots or tokens.
        case fixed(UInt32)

        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .auto: try c.encode("auto")
            case .fixed(let n): try c.encode(n)
            }
        }
    }

    /// Power and thermal management profile.
    public enum PowerProfile: String, Encodable, Sendable {
        /// Maximum GPU performance and frequency.
        case performance
        /// Balanced power and thermal profile.
        case balanced
        /// Energy-saving efficiency profile.
        case efficiency
    }

    /// Whether and how far this session drafts ahead.
    ///
    /// Settled when the model is OPENED, because that is where the
    /// drafter's state is allocated, and not per turn. `.block` throws from
    /// `TurboSparkSession.init` when the install cannot serve it, while
    /// `.auto` opens and explains itself in
    /// `SessionInfo.speculation.reason`.
    public enum Speculation: Encodable, Sendable {
        /// Automatic block size resolution.
        case auto
        /// Speculative decoding disabled.
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
        /// Multi-token prediction head drafter.
        case mtp
        /// DFlash2 diffusion block drafter.
        case dflash
    }

    /// How much of the machine may be committed to loading a model.
    ///
    /// `relaxed` is the default and is what this binding did before the
    /// option existed -- every published footprint figure for this engine was
    /// measured under it. See `docs/LOAD_GUARD.md`.
    ///
    /// **If you also call `TurboSparkCatalog.recommend`, pass it the SAME
    /// value.** The ranking and the loader's refusal share one memory budget
    /// by construction; recommending under one tier while opening under
    /// another promises a fit the loader then refuses.
    public enum LoadGuard: Encodable, Sendable {
        /// No memory precautions: reserves nothing and, more importantly,
        /// declines to REFUSE. A window too large for the machine becomes the
        /// Metal allocation failure you asked for rather than an error.
        case off
        /// The shipped behaviour: a 4 GiB reserve and a quarter of the rest.
        case relaxed
        /// A larger reserve and a smaller share, so a model and a browser can
        /// share the machine.
        case balanced
        /// Larger still, for a machine running work that must not be
        /// interrupted.
        case strict
        /// `relaxed`'s shares plus an absolute ceiling, in BYTES, on what the
        /// engine ALLOCATES (slot cache plus KV).
        ///
        /// Not on the install's size: a large model streaming its experts
        /// from disk is what this engine is for, and a cap read against the
        /// install would refuse a 13 GB model on a 16 GB machine that runs it
        /// fine.
        case custom(UInt64)

        public func encode(to encoder: Encoder) throws {
            var c = encoder.singleValueContainer()
            switch self {
            case .off: try c.encode("off")
            case .relaxed: try c.encode("relaxed")
            case .balanced: try c.encode("balanced")
            case .strict: try c.encode("strict")
            case .custom(let bytes): try c.encode(bytes)
            }
        }
    }

    /// The edit applied along a control vector.
    public enum SteeringMode: String, Encodable, Sendable {
        /// Suppress activations along the control vector.
        case ablate
        /// Additive vector injection.
        case add
        /// Clamp activations along the control vector.
        case clamp
        /// Renormalize activations following modification.
        case renorm
    }

    /// Maximum context window sizing.
    public var maxContext: Sizing?
    /// Expert cache slot sizing for MoE models.
    public var expertCacheSlots: Sizing?
    /// `nil` ASKS THE OS, so Low Power Mode selects efficiency. Name one
    /// explicitly when measuring anything.
    public var powerProfile: PowerProfile?
    /// Maximum throughput generation rate cap in tokens per second.
    public var maxTokensPerSec: Double?
    /// `nil` means `.relaxed`, which is what shipped before this existed.
    public var loadGuard: LoadGuard?
    /// Refuse to open when `maxContext` is `.auto` and resolves below this
    /// many tokens. `nil` and 0 both mean no floor.
    ///
    /// **Constrains AUTOMATIC sizing only.** It says nothing about an
    /// explicit `.fixed(2048)`: a caller naming a number has decided how to
    /// spend their own machine.
    public var minAutoContext: UInt32?
    /// `nil` means `.auto`, which is what the CLI and the server default to.
    public var speculation: Speculation?
    /// `nil` means `.auto`.
    public var speculativeDrafter: SpeculativeDrafter?
    /// Path to a .gguf control vector (llama.cpp layout). `nil` disables steering.
    public var steering: String?
    /// `nil` uses the vector's declared mode or `.ablate`.
    public var steeringMode: SteeringMode?
    /// Multiplier on edit strength (default 1.0; 0.0 is identity).
    public var steeringScale: Double?
    /// Layer range to steer, "START:END" inclusive and 0-based (default all).
    public var steeringLayers: String?
    /// Target coefficient for `.clamp` mode (default 0.0).
    public var steeringTarget: Double?
    /// Activation magnitude threshold to trigger the edit (default 0.0).
    public var steeringGate: Double?

    /// Creates options for opening a model session.
    public init(
        maxContext: Sizing? = nil,
        expertCacheSlots: Sizing? = nil,
        powerProfile: PowerProfile? = nil,
        maxTokensPerSec: Double? = nil,
        loadGuard: LoadGuard? = nil,
        minAutoContext: UInt32? = nil,
        speculation: Speculation? = nil,
        speculativeDrafter: SpeculativeDrafter? = nil,
        steering: String? = nil,
        steeringMode: SteeringMode? = nil,
        steeringScale: Double? = nil,
        steeringLayers: String? = nil,
        steeringTarget: Double? = nil,
        steeringGate: Double? = nil
    ) {
        self.maxContext = maxContext
        self.expertCacheSlots = expertCacheSlots
        self.powerProfile = powerProfile
        self.maxTokensPerSec = maxTokensPerSec
        self.loadGuard = loadGuard
        self.minAutoContext = minAutoContext
        self.speculation = speculation
        self.speculativeDrafter = speculativeDrafter
        self.steering = steering
        self.steeringMode = steeringMode
        self.steeringScale = steeringScale
        self.steeringLayers = steeringLayers
        self.steeringTarget = steeringTarget
        self.steeringGate = steeringGate
    }
}

/// How one turn is generated. The defaults are the CLI's.
public struct GenerateOptions: Encodable, Sendable {
    /// The accepted set is the CHECKPOINT'S, not this library's. Qwen 3.8
    /// rejects `.high` and its top setting is `.xhigh`; Harmony and Muse
    /// Glimmer accept `.high`. A level a template rejects throws, naming it.
    public enum Reasoning: String, Encodable, Sendable, CaseIterable {
        /// Reasoning turned off.
        case off
        /// Low reasoning effort.
        case low
        /// Medium reasoning effort.
        case medium
        /// High reasoning effort.
        case high
        /// Extra high reasoning effort.
        case xhigh
    }

    /// Maximum new tokens to emit.
    public var maxNewTokens: UInt32 = 512
    /// Sampling temperature (0 for greedy).
    public var temperature: Double = 0.2
    /// Top-K sampling cutoff.
    public var topK: UInt32 = 64
    /// Nucleus top-P probability cutoff.
    public var topP: Double = 0.95
    /// Multiplicative repetition penalty.
    public var repetitionPenalty: Double = 1.0
    /// Deterministic RNG seed.
    public var seed: UInt64?
    /// Custom stop sequence strings.
    public var stop: [String] = []
    /// Explicit stop token IDs.
    public var stopTokens: [UInt32] = []
    /// Reasoning effort level.
    public var reasoning: Reasoning = .off

    /// Creates default generation options.
    public init() {}
}
