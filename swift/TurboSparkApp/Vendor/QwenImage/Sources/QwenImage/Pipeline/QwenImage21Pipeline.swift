import Foundation
import Logging
import MLX
import MLXNN
import MLXRandom

public struct QwenImageGenerationRequest: Sendable {
  public var prompt: String
  public var width: Int
  public var height: Int
  public var steps: Int
  public var trueCfgScale: Float
  public var seed: UInt64?
  public var outputPath: URL
  public var model: String
  public var promptMaxTokens: Int

  public init(
    prompt: String,
    width: Int = QwenImageModelMetadata.recommendedWidth,
    height: Int = QwenImageModelMetadata.recommendedHeight,
    steps: Int = QwenImageModelMetadata.recommendedInferenceSteps,
    trueCfgScale: Float = QwenImageModelMetadata.recommendedTrueCfgScale,
    seed: UInt64? = nil,
    outputPath: URL = URL(fileURLWithPath: "qwen-image.png"),
    model: String,
    promptMaxTokens: Int = QwenImageModelMetadata.promptMaxTokens
  ) {
    self.prompt = prompt
    self.width = width
    self.height = height
    self.steps = steps
    self.trueCfgScale = trueCfgScale
    self.seed = seed
    self.outputPath = outputPath
    self.model = model
    self.promptMaxTokens = promptMaxTokens
  }
}

public final class QwenImage21Pipeline: @unchecked Sendable {
  public struct GenerationProgress: Sendable {
    public let stage: Stage
    public let stepIndex: Int
    public let totalSteps: Int

    /// Stage names mirror the Z-Image pipeline's raw values so the app and
    /// benchmark mappings stay family-neutral.
    public enum Stage: String, Sendable {
      case loadingModel = "Loading model"
      case encodingText = "Encoding text"
      case loadingTransformer = "Loading transformer"
      case loadingVAE = "Loading VAE"
      case denoising = "Denoising"
      case decoding = "Decoding"
      case saving = "Saving"
    }

    public var fractionCompleted: Double {
      guard totalSteps > 0 else { return 0 }
      return Double(stepIndex) / Double(totalSteps)
    }
  }

  public typealias ProgressHandler = (GenerationProgress) -> Void

  private let logger: Logger
  private var tokenizer: QwenTokenizer?
  private var textEncoder: Qwen3VLTextEncoder?
  private var transformer: QwenImage21Transformer2DModel?
  private var vae: AutoencoderKLQwenImage21?
  private var configs: QwenImageWeightsMapper.ComponentConfigs?
  private var snapshot: URL?
  private var isModelLoaded = false
  private var loadedModelPath: String?
  private var systemDropIndex = 0

  public init(logger: Logger = Logger(label: "qwen-image.pipeline")) {
    self.logger = logger
  }

  public var isLoaded: Bool { isModelLoaded }

  public func unloadModel() {
    tokenizer = nil
    textEncoder = nil
    transformer = nil
    vae = nil
    configs = nil
    snapshot = nil
    isModelLoaded = false
    loadedModelPath = nil
    Memory.clearCache()
  }

  // MARK: - Model loading

  public func loadModel(
    modelSpec: String,
    progressHandler: ProgressHandler? = nil
  ) async throws {
    if isModelLoaded, loadedModelPath == modelSpec {
      logger.info("Model already loaded, skipping load")
      return
    }
    if isModelLoaded {
      unloadModel()
    }

    let resolved = URL(fileURLWithPath: (modelSpec as NSString).expandingTildeInPath)
    let required = [
      "transformer/config.json",
      "text_encoder/config.json",
      "vae/config.json",
      "scheduler/scheduler_config.json",
    ]
    let missing = required.filter {
      !FileManager.default.fileExists(atPath: resolved.appending(path: $0).path)
    }
    guard missing.isEmpty else {
      throw QwenImagePipelineError.invalidModelPath(
        "\(resolved.path) is not a Qwen-Image-2.1 snapshot (missing \(missing.joined(separator: ", ")))")
    }

    progressHandler?(GenerationProgress(stage: .loadingModel, stepIndex: 0, totalSteps: 1))
    logger.info("Loading Qwen-Image-2.1 model from \(resolved.path)")

    let componentConfigs = try QwenImageWeightsMapper.ComponentConfigs.load(from: resolved)
    try Task.checkCancellation()

    // Tokenizer plus the system-segment length the pipeline drops.
    let loadedTokenizer = try QwenTokenizer.load(from: resolved)
    systemDropIndex = loadedTokenizer.tokenCount(of: QwenImageModelMetadata.systemSegmentT2I)

    progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: 1))
    logger.info("Loading text encoder...")
    let encoder = Qwen3VLTextEncoder(configuration: componentConfigs.textEncoder)
    // The vision tower and language-model head never run in text-to-image
    // generation; skipping them keeps ~1 GiB of the snapshot unloaded.
    let encoderWeights = try QwenImageWeightsMapper.loadComponent(
      resolved.appending(path: "text_encoder", directoryHint: .isDirectory),
      skipPrefixes: ["vision_tower.", "language_model.lm_head"]
    )
    QwenImageWeightsMapper.apply(encoderWeights, to: encoder, logger: logger)
    textEncoder = encoder
    Memory.clearCache()
    try Task.checkCancellation()

    progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
    logger.info("Loading transformer...")
    let loadedTransformer = QwenImage21Transformer2DModel(configuration: componentConfigs.transformer)
    let transformerWeights = try QwenImageWeightsMapper.loadComponent(
      resolved.appending(path: "transformer", directoryHint: .isDirectory))
    QwenImageWeightsMapper.apply(transformerWeights, to: loadedTransformer, logger: logger)
    transformer = loadedTransformer
    Memory.clearCache()
    try Task.checkCancellation()

    progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
    logger.info("Loading VAE decoder...")
    let loadedVAE = AutoencoderKLQwenImage21(configuration: componentConfigs.vae)
    let vaeWeights = try QwenImageWeightsMapper.loadComponent(
      resolved.appending(path: "vae", directoryHint: .isDirectory))
    QwenImageWeightsMapper.applyDense(vaeWeights, to: loadedVAE, logger: logger)
    vae = loadedVAE
    Memory.clearCache()

    tokenizer = loadedTokenizer
    configs = componentConfigs
    snapshot = resolved
    isModelLoaded = true
    loadedModelPath = modelSpec
    logger.info("Model loaded")
  }

  // MARK: - Generation

  public func generate(
    _ request: QwenImageGenerationRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> URL {
    let data = try await generateToMemory(request, progressHandler: progressHandler)
    try data.write(to: request.outputPath)
    return request.outputPath
  }

  public func generateToMemory(
    _ request: QwenImageGenerationRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> Data {
    let decoded = try await generateCore(request, progressHandler: progressHandler)
    progressHandler?(GenerationProgress(stage: .saving, stepIndex: request.steps, totalSteps: request.steps))
    let data = try QwenImageIO.pngData(from: decoded)
    logger.info("Generated image data (\(data.count) bytes)")
    return data
  }

  private func generateCore(
    _ request: QwenImageGenerationRequest,
    progressHandler: ProgressHandler? = nil
  ) async throws -> MLXArray {
    let multiple = QwenImageModelMetadata.dimensionMultiple
    let width = request.width / multiple * multiple
    let height = request.height / multiple * multiple
    guard width > 0, height > 0 else {
      throw QwenImagePipelineError.invalidDimensions(
        "\(request.width)x\(request.height) collapses to zero at multiple \(multiple)")
    }

    try await loadModel(modelSpec: request.model, progressHandler: progressHandler)

    guard let tokenizer, let textEncoder, let transformer, let vae, let configs else {
      throw QwenImagePipelineError.modelNotLoaded
    }
    try Task.checkCancellation()

    // 1. Prompt encoding: raw template, no padding (batch of one).
    progressHandler?(GenerationProgress(stage: .encodingText, stepIndex: 0, totalSteps: request.steps))
    logger.info("Encoding prompt...")
    let framed = String(
      format: QwenImageModelMetadata.promptTemplateT2I,
      request.prompt.isEmpty ? " " : request.prompt
    )
    let tokenIds = tokenizer.encodeRaw(framed, maxLength: request.promptMaxTokens)
    guard tokenIds.count > systemDropIndex else {
      throw QwenImagePipelineError.invalidModelPath("the framed prompt produced no user tokens")
    }
    let inputIds = MLXArray(tokenIds.map { Int32($0) }).reshaped(1, tokenIds.count)
    var hiddenStates = textEncoder.encodeHiddenStates(inputIds: inputIds, attentionMask: nil)
    if systemDropIndex > 0 {
      hiddenStates = hiddenStates[0..., systemDropIndex..., 0...]
    }
    let textEmbeddings = hiddenStates.asType(.bfloat16)
    MLX.eval(textEmbeddings)
    Memory.clearCache()
    let textLength = textEmbeddings.dim(1)
    logger.info("Prompt encoded: \(textLength) tokens")

    // 2. Latents: [1, C, 1, h, w] noise, packed to [1, h*w, C].
    let scale = QwenImageModelMetadata.vaeScaleFactor
    let latentHeight = height / scale
    let latentWidth = width / scale
    let latentShape: [Int] = [
      1, QwenImageModelMetadata.latentChannels, 1, latentHeight, latentWidth,
    ]
    let randomKey = request.seed.map { MLXRandom.key($0) }
    let noise = MLXRandom.normal(latentShape, loc: 0, scale: 1, key: randomKey).asType(.bfloat16)
    var latents = noise.reshaped(1, QwenImageModelMetadata.latentChannels, latentHeight * latentWidth)
      .transposed(0, 2, 1)
    MLX.eval(latents)

    // 3. Schedule.
    let scheduler = QwenImage21Scheduler(
      numInferenceSteps: request.steps,
      imageSequenceLength: latentHeight * latentWidth,
      config: configs.scheduler
    )
    let timesteps = scheduler.timestepValues

    // 4. Denoising loop with the prefix KV cache.
    let layout = QwenImage21Transformer2DModel.ForwardLayout(
      textLength: textLength,
      imageHeight: latentHeight,
      imageWidth: latentWidth
    )
    let cache = QwenImage21KVCache(numLayers: transformer.configuration.numLayers)
    logger.info("Running \(request.steps) denoising steps at \(width)x\(height)...")
    for stepIndex in 0..<request.steps {
      try Task.checkCancellation()
      let timestep = MLXArray([timesteps[stepIndex] / Float(configs.scheduler.numTrainTimesteps)])
      let noisePrediction: MLXArray
      if stepIndex == 0 {
        noisePrediction = transformer.forwardPrefill(
          latents: latents,
          textEmbeddings: textEmbeddings,
          timestep: timestep,
          layout: layout,
          cache: cache
        )
      } else {
        noisePrediction = transformer.forwardCached(
          latents: latents,
          timestep: timestep,
          layout: layout,
          cache: cache
        )
      }
      MLX.eval(noisePrediction)

      // The reference steps in float32 and casts back to the model dtype.
      let floatSample = latents.asType(.float32)
      let updated = scheduler.step(
        modelOutput: noisePrediction.asType(.float32), stepIndex: stepIndex, sample: floatSample)
      latents = updated.asType(.bfloat16)
      MLX.eval(latents)

      progressHandler?(GenerationProgress(
        stage: .denoising, stepIndex: stepIndex + 1, totalSteps: request.steps))
    }
    logger.info("Denoising complete")

    // 5. VAE decode with per-channel denormalization.
    progressHandler?(GenerationProgress(
      stage: .decoding, stepIndex: request.steps, totalSteps: request.steps))
    let packed = vae.denormalizeLatents(
      latents.transposed(0, 2, 1).reshaped(
        1, QwenImageModelMetadata.latentChannels, 1, latentHeight, latentWidth
      ).asType(.float32))
    let decoded = vae.decode(packed)
    MLX.eval(decoded)

    // [-1, 1] -> [0, 1], dropping the alpha channel at encode time.
    let image = (decoded[0..., 0..., 0, 0..., 0...] + 1.0) * 0.5
    let clamped = MLX.minimum(MLX.maximum(image, 0.0), 1.0)
    MLX.eval(clamped)
    Memory.clearCache()

    return clamped
  }
}

