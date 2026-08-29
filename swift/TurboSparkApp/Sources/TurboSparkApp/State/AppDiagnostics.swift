import Foundation
import TurboSpark

/// Post-generation performance metrics, token counts, throughput, and memory stats.
public struct AppDiagnostics: Sendable, Equatable {
    /// Number of tokens processed during the prefill / prompt evaluation phase.
    public let promptTokens: Int
    /// Number of new tokens generated during the sequential decode phase.
    public let generatedTokens: Int
    /// Wall-clock time in seconds spent in prompt prefill.
    public let prefillSeconds: Double
    /// Wall-clock time in seconds spent in sequential token decoding.
    public let decodeSeconds: Double
    /// Generation throughput in tokens per second.
    public let tokensPerSecond: Double
    /// Peak physical memory footprint in bytes recorded during generation.
    public let peakMemoryBytes: UInt64?
    /// Reason why the generation ended (stop token, max tokens, user cancelled, etc.).
    public let stopReason: GenerationResult.StopReason
    /// Detailed phase profiling counters if enabled.
    public let phases: PhaseReport?

    public init(
        result: GenerationResult,
        peakMemory: UInt64? = nil,
        phases: PhaseReport? = nil
    ) {
        self.promptTokens = result.promptTokens
        self.generatedTokens = result.newTokens
        self.prefillSeconds = result.prefillSeconds
        self.decodeSeconds = result.decodeSeconds
        self.tokensPerSecond = result.tokensPerSecond ?? (result.decodeSeconds > 0 ? Double(result.newTokens) / result.decodeSeconds : 0)
        self.peakMemoryBytes = peakMemory
        self.stopReason = result.stopReason
        self.phases = phases
    }
}
