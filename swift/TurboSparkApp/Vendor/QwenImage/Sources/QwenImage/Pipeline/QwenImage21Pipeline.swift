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

/// How long the pipeline keeps modules in memory.
public enum QwenImageResidency: Sendable {
  /// Keep the text encoder, transformer, and VAE resident between jobs.
  /// Fastest repeat jobs, highest steady-state footprint.
  case warm
  /// Load each stage when it is needed and release it as soon as the next
  /// stage no longer needs it: the encoder after encoding, the transformer
  /// before the VAE decode, the VAE after the job. Peak is the largest
  /// stage instead of the sum, at the cost of reloading weights per job.
  case staged
}

/// Precision of the VAE decode.
public enum QwenImageVAEPrecision: Sendable {
  /// The conservative default: float32 activations.
  case float32
  /// The VAE's stored dtype, which the Diffusers reference decodes in, at
  /// roughly half the activation memory.
  case bfloat16

  var dtype: DType {
    switch self {
    case .float32: return .float32
    case .bfloat16: return .bfloat16
    }
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
      case loadingTextEncoder = "Loading text encoder"
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
  private let residency: QwenImageResidency
  private let vaeDecodeDtype: DType
  private var tokenizer: QwenTokenizer?
  private var textEncoder: Qwen3VLTextEncoder?
  private var transformer: QwenImage21Transformer2DModel?
  private var vae: AutoencoderKLQwenImage21?
  private var configs: QwenImageWeightsMapper.ComponentConfigs?
  private var snapshot: URL?
  private var isModelLoaded = false
  private var loadedModelPath: String?
  private var systemDropIndex = 0

  public init(
    residency: QwenImageResidency = .warm,
    vaePrecision: QwenImageVAEPrecision = .float32,
    logger: Logger = Logger(label: "qwen-image.pipeline")
  ) {
    self.residency = residency
    self.vaeDecodeDtype = vaePrecision.dtype
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

  /// Validates the snapshot and loads the configs and tokenizer. Under
  /// `.warm` it also loads every module; under `.staged` the modules load
  /// lazily inside `generate`, one stage at a time.
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

    // Set before the module loads so a failure midway leaves a pipeline the
    // idempotent `load*IfNeeded` calls can finish, not a half-set one.
    tokenizer = loadedTokenizer
    configs = componentConfigs
    snapshot = resolved
    loadedModelPath = modelSpec
    isModelLoaded = true

    if residency == .warm {
      _ = try loadTextEncoderIfNeeded(progressHandler)
      try Task.checkCancellation()
      try loadTransformerIfNeeded(progressHandler)
      try Task.checkCancellation()
      try loadVAEIfNeeded(progressHandler)
      Memory.clearCache()
    }
    logger.info("Model loaded")
  }

  private func loadTextEncoderIfNeeded(
    _ progressHandler: ProgressHandler?
  ) throws -> Qwen3VLTextEncoder {
    if let textEncoder { return textEncoder }
    guard let snapshot, let configs else { throw QwenImagePipelineError.modelNotLoaded }
    progressHandler?(GenerationProgress(stage: .loadingTextEncoder, stepIndex: 0, totalSteps: 1))
    logger.info("Loading text encoder...")
    let encoder = Qwen3VLTextEncoder(configuration: configs.textEncoder)
    // The vision tower and language-model head never run in text-to-image
    // generation; the canonicalizer drops them and maps the transformers-5
    // spelling onto the module tree.
    let encoderWeights = try QwenImageWeightsMapper.loadComponent(
      snapshot.appending(path: "text_encoder", directoryHint: .isDirectory),
      rename: QwenImageWeightsMapper.canonicalTextEncoderName
    )
    try QwenImageWeightsMapper.apply(encoderWeights, to: encoder, logger: logger)
    textEncoder = encoder
    return encoder
  }

  private func loadTransformerIfNeeded(_ progressHandler: ProgressHandler?) throws {
    if transformer != nil { return }
    guard let snapshot, let configs else { throw QwenImagePipelineError.modelNotLoaded }
    progressHandler?(GenerationProgress(stage: .loadingTransformer, stepIndex: 0, totalSteps: 1))
    logger.info("Loading transformer...")
    let loaded = QwenImage21Transformer2DModel(configuration: configs.transformer)
    let weights = try QwenImageWeightsMapper.loadComponent(
      snapshot.appending(path: "transformer", directoryHint: .isDirectory),
      rename: QwenImageWeightsMapper.canonicalTransformerName)
    try QwenImageWeightsMapper.apply(weights, to: loaded, logger: logger)
    transformer = loaded
  }

  private func loadVAEIfNeeded(_ progressHandler: ProgressHandler?) throws {
    if vae != nil { return }
    guard let snapshot, let configs else { throw QwenImagePipelineError.modelNotLoaded }
    progressHandler?(GenerationProgress(stage: .loadingVAE, stepIndex: 0, totalSteps: 1))
    logger.info("Loading VAE decoder...")
    let loaded = AutoencoderKLQwenImage21(configuration: configs.vae)
    let weights = try QwenImageWeightsMapper.loadComponent(
      snapshot.appending(path: "vae", directoryHint: .isDirectory))
    try QwenImageWeightsMapper.applyDense(weights, to: loaded, logger: logger)
    vae = loaded
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

    // The scheduler preconditions steps > 0; surface a throw instead of a trap.
    guard request.steps >= 1 else {
      throw QwenImagePipelineError.invalidSteps(request.steps)
    }

    try await loadModel(modelSpec: request.model, progressHandler: progressHandler)

    guard let configs else { throw QwenImagePipelineError.modelNotLoaded }
    // A published sigma ladder fixes the step count; running another count
    // would silently skip or invent sigmas.
    if let ladder = configs.scheduler.staticSigmaLadder, ladder.count != request.steps {
      throw QwenImagePipelineError.stepsFixedByCheckpoint(
        requested: request.steps, required: ladder.count)
    }
    try Task.checkCancellation()

    // 1. Prompt encoding. Under `.staged` the encoder is released inside
    //    `encodePrompt`, before the transformer and VAE are loaded.
    let textEmbeddings = try encodePrompt(request, progressHandler: progressHandler)
    Memory.clearCache()
    let textLength = textEmbeddings.dim(1)
    logger.info("Prompt encoded: \(textLength) tokens")
    try Task.checkCancellation()

    // 2. Denoise. Under `.staged` the transformer is released inside
    //    `denoise`'s caller, before the VAE decode.
    try loadTransformerIfNeeded(progressHandler)
    let latentHeight = height / QwenImageModelMetadata.vaeScaleFactor
    let latentWidth = width / QwenImageModelMetadata.vaeScaleFactor
    let latents = try await denoise(
      request, textEmbeddings: textEmbeddings, latentHeight: latentHeight,
      latentWidth: latentWidth, progressHandler: progressHandler)
    if residency == .staged {
      transformer = nil
      Memory.clearCache()
    }
    try Task.checkCancellation()

    // 3. VAE decode with per-channel denormalization.
    try loadVAEIfNeeded(progressHandler)
    progressHandler?(GenerationProgress(
      stage: .decoding, stepIndex: request.steps, totalSteps: request.steps))
    let clamped = try decode(latents, latentHeight: latentHeight, latentWidth: latentWidth)
    if residency == .staged {
      vae = nil
    }
    Memory.clearCache()
    return clamped
  }

  /// Encodes the framed prompt. Locals die at return, so the caller's
  /// `Memory.clearCache()` sees a released encoder under `.staged`.
  private func encodePrompt(
    _ request: QwenImageGenerationRequest,
    progressHandler: ProgressHandler?
  ) throws -> MLXArray {
    guard let tokenizer else { throw QwenImagePipelineError.tokenizerNotLoaded }
    let encoder = try loadTextEncoderIfNeeded(progressHandler)
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
    var hiddenStates = encoder.encodeHiddenStates(inputIds: inputIds, attentionMask: nil)
    if systemDropIndex > 0 {
      hiddenStates = hiddenStates[0..., systemDropIndex..., 0...]
    }
    let textEmbeddings = hiddenStates.asType(.bfloat16)
    MLX.eval(textEmbeddings)
    if residency == .staged {
      textEncoder = nil
    }
    return textEmbeddings
  }

  /// Runs the schedule and returns the final packed latents `[1, h*w, C]`.
  private func denoise(
    _ request: QwenImageGenerationRequest,
    textEmbeddings: MLXArray,
    latentHeight: Int,
    latentWidth: Int,
    progressHandler: ProgressHandler?
  ) async throws -> MLXArray {
    guard let transformer, let configs else { throw QwenImagePipelineError.modelNotLoaded }
    let textLength = textEmbeddings.dim(1)

    // Latents: [1, C, 1, h, w] noise, packed to [1, h*w, C].
    let latentShape: [Int] = [
      1, QwenImageModelMetadata.latentChannels, 1, latentHeight, latentWidth,
    ]
    let randomKey = request.seed.map { MLXRandom.key($0) }
    let noise = MLXRandom.normal(latentShape, loc: 0, scale: 1, key: randomKey).asType(.bfloat16)
    var latents = noise.reshaped(1, QwenImageModelMetadata.latentChannels, latentHeight * latentWidth)
      .transposed(0, 2, 1)
    MLX.eval(latents)

    let scheduler = QwenImage21Scheduler(
      numInferenceSteps: request.steps,
      imageSequenceLength: latentHeight * latentWidth,
      config: configs.scheduler
    )
    let timesteps = scheduler.timestepValues

    // Denoising loop with the prefix KV cache.
    let layout = QwenImage21Transformer2DModel.ForwardLayout(
      textLength: textLength,
      imageHeight: latentHeight,
      imageWidth: latentWidth
    )
    let cache = QwenImage21KVCache(numLayers: transformer.configuration.numLayers)
    logger.info(
      "Running \(request.steps) denoising steps at \(latentWidth * QwenImageModelMetadata.vaeScaleFactor)x\(latentHeight * QwenImageModelMetadata.vaeScaleFactor)...")
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
    return latents
  }

  private func decode(
    _ latents: MLXArray,
    latentHeight: Int,
    latentWidth: Int
  ) throws -> MLXArray {
    guard let vae else { throw QwenImagePipelineError.vaeNotLoaded }
    let packed = vae.denormalizeLatents(
      latents.transposed(0, 2, 1).reshaped(
        1, QwenImageModelMetadata.latentChannels, 1, latentHeight, latentWidth
      ).asType(vaeDecodeDtype))
    let decoded = vae.decode(packed)
    MLX.eval(decoded)

    // [-1, 1] -> [0, 1], dropping the alpha channel at encode time. The
    // arithmetic runs in float32 so a bfloat16 decode adds no further
    // rounding of the pixel values.
    let image = (decoded[0..., 0..., 0, 0..., 0...].asType(.float32) + 1.0) * 0.5
    let clamped = MLX.minimum(MLX.maximum(image, 0.0), 1.0)
    MLX.eval(clamped)
    return clamped
  }
}
