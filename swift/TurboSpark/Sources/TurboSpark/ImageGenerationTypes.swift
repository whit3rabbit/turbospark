import Foundation

/// The fixed IG4 image-generation envelope.
public struct ImageGenerateOptions: Codable, Sendable, Equatable {
    public var prompt: String
    public var seed: UInt64
    public var width: UInt32
    public var height: UInt32
    public var steps: UInt32

    public init(
        prompt: String,
        seed: UInt64,
        width: UInt32 = 1024,
        height: UInt32 = 1024,
        steps: UInt32 = 9
    ) {
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.steps = steps
    }
}

/// Metadata embedded in a completed PNG and returned beside it.
public struct GeneratedImageSchedulerMetadata: Decodable, Sendable, Equatable {
    public let numTrainTimesteps: Float
    public let shift: Float
    public let timesteps: [Float]
    public let sigmas: [Float]
    public let evaluationCount: UInt32
    public let guidancePolicy: String
}

public struct GeneratedImageMetadata: Decodable, Sendable, Equatable {
    public let prompt: String
    public let seed: UInt64
    public let width: UInt32
    public let height: UInt32
    public let batch: UInt32
    public let schedulerSteps: UInt32
    public let transformerForwards: UInt32
    public let guidanceScale: Float
    public let modelID: String
    public let modelRevision: String
    public let componentRevisions: [String: String]
    public let quantization: String
    public let scheduler: GeneratedImageSchedulerMetadata
    public let noiseProvenance: String
    public let engineRevision: String
}

public struct ImageGenerationResult: Sendable, Equatable {
    public let png: Data
    public let metadata: GeneratedImageMetadata

    public init(png: Data, metadata: GeneratedImageMetadata) {
        self.png = png
        self.metadata = metadata
    }
}

public enum ImageGenerationEvent: Sendable, Equatable {
    case stage(name: String, completed: Int, total: Int)
    case finished(ImageGenerationResult)
    case cancelled
}
