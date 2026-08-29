import Foundation
import TurboSpark

public struct AppDiagnostics: Sendable, Equatable {
    public let promptTokens: Int
    public let generatedTokens: Int
    public let prefillSeconds: Double
    public let decodeSeconds: Double
    public let tokensPerSecond: Double
    public let peakMemoryBytes: UInt64?
    public let stopReason: GenerationResult.StopReason
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
