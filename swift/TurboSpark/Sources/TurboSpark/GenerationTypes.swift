import Foundation

/// What a finished turn reports.
public struct GenerationResult: Decodable, Sendable, Equatable {
    /// The condition that terminated text generation.
    public enum StopReason: String, Decodable, Sendable {
        /// Reached model end of turn token.
        case endOfTurn
        /// Generated a tool invocation block.
        case toolCalls
        /// Reached end of sequence.
        case eos
        /// Matched a stop string sequence.
        case stopString
        /// Reached maximum token generation budget.
        case maxTokens
        /// The caller pressed Stop. The partial turn in `content` is valid
        /// and the conversation can continue from it.
        case cancelled
    }

    /// Number of tokens in the prompt prefix.
    public let promptTokens: Int
    /// Number of new tokens generated.
    public let newTokens: Int
    /// Time spent in the prompt prefill phase in seconds.
    public let prefillSeconds: Double
    /// Time spent in the token decode phase in seconds.
    public let decodeSeconds: Double
    /// Termination reason for this turn.
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

/// One streamed event during generation.
public enum GenerationEvent: Sendable, Equatable {
    /// Prefill progress update with completed and total prompt tokens.
    case prefill(done: Int, total: Int)
    /// Incremental generated assistant text content.
    case content(String)
    /// Incremental reasoning or thinking content.
    case reasoning(String)
    /// Generation completion event with final result.
    case finished(GenerationResult)
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
