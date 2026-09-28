import Foundation

/// Family constants for the Qwen-Image-2.1 MLX pipeline.
public enum QwenImageModelMetadata {
  public static let recommendedWidth = 1024
  public static let recommendedHeight = 1024
  public static let recommendedInferenceSteps = 40
  /// The model samples without classifier-free guidance by default.
  public static let recommendedTrueCfgScale: Float = 1.0
  public static let latentChannels = 64
  public static let vaeScaleFactor = 16
  /// The pipeline floors dimensions to multiples of 32 pixels, which keeps
  /// every latent side even as the checkpoint expects.
  public static let dimensionMultiple = 32
  /// Right-truncation bound for the framed prompt, mirroring the install
  /// envelope's prompt_max_tokens.
  public static let promptMaxTokens = 1024

  public static let systemPrompt = "Comprehend and analyze the provided prompt."

  /// The raw template the checkpoint was trained on. The diffusers pipeline
  /// passes this string straight to the processor instead of calling
  /// `apply_chat_template`: the two tokenize differently and the checkpoint
  /// expects this one.
  public static let promptTemplateT2I =
    "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
    + "<|im_start|>user\n%@<|im_end|>\n"
    + "<|im_start|>assistant\n"

  /// The leading system-role tokens dropped from the hidden states before
  /// they condition the transformer.
  public static let systemSegmentT2I =
    "<|im_start|>system\n\(systemPrompt)<|im_end|>\n"
}

/// Errors surfaced by the Qwen-Image-2.1 pipeline.
public enum QwenImagePipelineError: Error, Sendable, LocalizedError {
  case invalidDimensions(String)
  case invalidModelPath(String)
  case missingSnapshotComponent(String)
  case weightsMissing(String)
  case tokenizerNotLoaded
  case textEncoderNotLoaded
  case transformerNotLoaded
  case vaeNotLoaded
  case modelNotLoaded
  case unsupportedBatch(String)

  public var errorDescription: String? {
    switch self {
    case .invalidDimensions(let reason):
      return "Invalid image dimensions: \(reason)"
    case .invalidModelPath(let path):
      return "Invalid Qwen-Image model path: \(path)"
    case .missingSnapshotComponent(let component):
      return "The Qwen-Image snapshot is missing \(component)."
    case .weightsMissing(let reason):
      return "Qwen-Image weights are incomplete: \(reason)"
    case .tokenizerNotLoaded:
      return "The Qwen-Image tokenizer is not loaded."
    case .textEncoderNotLoaded:
      return "The Qwen-Image text encoder is not loaded."
    case .transformerNotLoaded:
      return "The Qwen-Image transformer is not loaded."
    case .vaeNotLoaded:
      return "The Qwen-Image VAE is not loaded."
    case .modelNotLoaded:
      return "The Qwen-Image model is not loaded."
    case .unsupportedBatch(let reason):
      return "Unsupported batch: \(reason)"
    }
  }
}
