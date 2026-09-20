import Foundation

/// Where a counted footprint came from.
///
/// **`unknown` IS NOT ZERO.** A row whose checkpoint header nobody has read
/// reports zeros in every sizing field, and a zero rendered as a figure reads
/// as "fits easily", which is the exact inverse of what it means. Branch on
/// this before showing `countedBytes`, `slotCacheSlots` or `largestContext`.
public enum CountedSource: String, Codable, Sendable {
    /// A frozen row in the catalog, taken on this chip at this context and
    /// this slot count.
    case measured
    /// Computed from the checkpoint's own shape.
    case estimated
    /// Nothing has read this checkpoint's header. Probe it.
    case unknown
}

/// How a candidate lands on this machine, at one context and one slot count.
///
/// The two byte figures answer different questions and fail differently.
/// `countedBytes` is what the engine ALLOCATES (expert slot cache plus KV),
/// so exceeding memory there is a failed open. `mappedBytes` is what it
/// READS, and exceeding memory there is the streaming this engine is built
/// around: it costs throughput, not correctness.
public struct ProbeFit: Decodable, Sendable, Equatable {
    public let verdict: ModelRecommendation.FitVerdict
    public let verdictSummary: String
    public let runs: Bool
    /// Slot cache plus KV: what the engine allocates.
    public let countedBytes: UInt64
    public let countedSource: CountedSource
    /// What the engine reads. See `mappedSource`.
    public let mappedBytes: UInt64
    /// `"download"` means `mappedBytes` is the PUBLISHED CHECKPOINT and not
    /// the install this port would write. The two differ. Label it rather
    /// than presenting it as an install size.
    public let mappedSource: String
    public let slotCacheSlots: Int
    public let slotCacheBytes: UInt64
    public let kvBytes: UInt64
    public let residentBytes: UInt64
    public let largestContext: UInt32
    /// The window this fit was computed at. A footprint is only meaningful
    /// beside its context and its slot count.
    public let context: UInt32
    /// What other windows would cost. Empty when nothing could be sized.
    public let contextLadder: [ContextLadderRung]
    public let trainedContext: UInt32?
}

/// One context window and what it costs.
///
/// **DO NOT EXTRAPOLATE FROM ONE RUNG.** KV is not linear in the window: a
/// sliding-window layer is a ring capped at `sliding_window + 128` and stops
/// growing past it, while a fully-attentive layer grows forever. Measured
/// 4,096 to 131,072 on the shipped baselines, Mistral 7B grows 32x (512 MiB
/// to 16,384) and Gemma 4 grows 9x (305 to 2,785). Multiplying a 4,096 figure
/// is 3.5x high on Gemma, in the direction that refuses a window that runs.
public struct ContextLadderRung: Decodable, Sendable, Equatable, Identifiable {
    public var id: UInt32 { context }
    public let context: UInt32
    public let kvBytes: UInt64
    /// Slot cache plus this window's KV. The slot cache term does not move
    /// with the window, so on a streamed MoE a long-context question is
    /// really a question about the KV alone.
    public let counted: UInt64
    public let verdict: ModelRecommendation.FitVerdict
    public let runs: Bool
    /// Past the checkpoint's trained window. A WARNING and not a refusal:
    /// RoPE extrapolates rather than failing, and some checkpoints ship YaRN
    /// scaling meant to exceed it.
    public let pastTrained: Bool
    public let isTrainedMax: Bool
    public let isLargestFitting: Bool
}

/// The ladder for an installed model.
public struct ContextLadder: Decodable, Sendable, Equatable {
    public let path: String
    /// The window the checkpoint was trained at, or `nil` when it declares
    /// none. Never 0.
    public let trainedContext: UInt32?
    /// **Empty when the install's shape could not be read.** That is a
    /// question nothing answered, not a model with no memory cost.
    public let rungs: [ContextLadderRung]
}

/// A measured decode band and the silicon it was taken on.
///
/// **THE CHIP IS PART OF THE VALUE, NOT PROVENANCE.** tok/s does not transfer
/// across silicon. When `measuredOnThisChip` is false a host must NAME the
/// chip, or it is presenting another machine's number as this one's answer.
public struct ThroughputBand: Codable, Sendable, Equatable {
    public let minTokensPerSecond: Double
    public let maxTokensPerSecond: Double
    public let chip: String
    public let measuredOnThisChip: Bool
}

/// One block type a checkpoint carries, and whether this port can run it.
public struct ProbeTypeShare: Decodable, Sendable, Equatable {
    public let name: String
    public let tensors: Int
    /// `nil` when this port cannot size the type. **Render that as UNSIZED,
    /// never as 0 bytes**: an unsized type is usually the one carrying the
    /// model, and a zero sorts to the bottom of a share column, which is the
    /// inverse of its real rank.
    public let bytes: UInt64?
    /// Whether a kernel exists for it here.
    public let executable: Bool
}

/// The MLX affine quantization a checkpoint declares.
public struct ProbeAffine: Decodable, Sendable, Equatable {
    public let bits: UInt32
    public let groupSize: UInt32
}

/// What one expert-cache slot count would pin.
public struct ProbeSlotCache: Decodable, Sendable, Equatable {
    public let slots: UInt64
    public let bytes: UInt64
}

/// What this engine makes of a Hugging Face repository, read from its header
/// alone: kilobytes and seconds, no download.
///
/// **READ `slotCacheBytes` BEFORE `downloadBytes`.** What decides whether a
/// model runs here is `slots x layers x expert stride`, not the model's size.
public struct ProbeReport: Decodable, Sendable, Equatable {
    public let repo: String
    public let revision: String
    public let file: String?
    public let downloadBytes: UInt64?
    /// The architecture string the checkpoint declares.
    public let architecture: String?
    /// The family this port resolved it to, or `nil` when it resolved none.
    public let family: String?
    public let runnable: Bool
    /// Why not, in the engine's own words, including the bring-up clause for
    /// an architecture that is recognized but has no decode flow here.
    public let refusedBecause: String?
    public let types: [ProbeTypeShare]
    public let affine: ProbeAffine?
    /// Bytes of ONE routed expert in ONE layer, or `nil` for a dense model.
    public let expertStride: UInt64?
    /// The window this checkpoint was trained at, or `nil` when the file
    /// declares none. Never 0.
    public let trainedContext: UInt32?
    public let slotCacheBytes: [ProbeSlotCache]
    public let sidecarsPresent: [String]
    public let sidecarsMissing: [String]
    public let chatTemplate: String?
    public let warnings: [String]
    /// This machine's answer, or `nil` when the header yielded no shape.
    /// Never a zeroed object.
    public let fit: ProbeFit?
}

/// One publishable `.gguf` in a repository.
public struct RepoVariant: Decodable, Sendable, Equatable, Identifiable {
    public var id: String { file }
    public let file: String
    /// `nil` when Hugging Face reported no length. Not 0.
    public let bytes: UInt64?
    /// The quantization the FILE NAME declares, or `nil` when it names none
    /// this port recognizes. A `Q4_K_M` suffix is a file-name label rather
    /// than a ggml type, so this is what the name says and not what the
    /// header contains.
    public let quantLabel: String?
    /// Position on the quality ladder, best first. `nil` sorts last.
    public let ladderRank: Int?
    /// Whether the type this name declares has kernels here. A `false` row is
    /// LISTED rather than hidden, because a picker showing three of a
    /// repository's eight files reads as the repository having three.
    public let executable: Bool
}

/// Every `.gguf` a repository publishes, best quality first.
public struct RepoVariants: Decodable, Sendable, Equatable {
    public let repo: String
    public let revision: String
    public let variants: [RepoVariant]
    /// How many `.gguf` files were shard parts, which this port cannot walk.
    /// Nonzero is why a picker may be short or empty, and saying so is the
    /// difference between "this port cannot walk a shard set" and the reading
    /// a user would otherwise take, "this repository publishes no GGUF".
    public let shardedSkipped: Int
}
