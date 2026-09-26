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

    public init(
        numTrainTimesteps: Float,
        shift: Float,
        timesteps: [Float],
        sigmas: [Float],
        evaluationCount: UInt32,
        guidancePolicy: String
    ) {
        self.numTrainTimesteps = numTrainTimesteps
        self.shift = shift
        self.timesteps = timesteps
        self.sigmas = sigmas
        self.evaluationCount = evaluationCount
        self.guidancePolicy = guidancePolicy
    }
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

    public init(
        prompt: String,
        seed: UInt64,
        width: UInt32,
        height: UInt32,
        batch: UInt32,
        schedulerSteps: UInt32,
        transformerForwards: UInt32,
        guidanceScale: Float,
        modelID: String,
        modelRevision: String,
        componentRevisions: [String: String],
        quantization: String,
        scheduler: GeneratedImageSchedulerMetadata,
        noiseProvenance: String,
        engineRevision: String
    ) {
        self.prompt = prompt
        self.seed = seed
        self.width = width
        self.height = height
        self.batch = batch
        self.schedulerSteps = schedulerSteps
        self.transformerForwards = transformerForwards
        self.guidanceScale = guidanceScale
        self.modelID = modelID
        self.modelRevision = modelRevision
        self.componentRevisions = componentRevisions
        self.quantization = quantization
        self.scheduler = scheduler
        self.noiseProvenance = noiseProvenance
        self.engineRevision = engineRevision
    }
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
