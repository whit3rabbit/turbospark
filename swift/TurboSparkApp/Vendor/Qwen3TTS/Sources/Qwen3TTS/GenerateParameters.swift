import Foundation

public struct GenerateParameters: Sendable {
    public var maxTokens: Int?
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var minP: Float
    public var repetitionPenalty: Float?

    public init(maxTokens: Int = 4096, temperature: Float = 0.9, topP: Float = 1,
                topK: Int = 50, minP: Float = 0, repetitionPenalty: Float = 1.05) {
        self.maxTokens = maxTokens; self.temperature = temperature; self.topP = topP
        self.topK = topK; self.minP = minP; self.repetitionPenalty = repetitionPenalty
    }
}
