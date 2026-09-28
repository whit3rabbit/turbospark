import Foundation
import MLX
import MLXFast
import MLXNN

/// Text-encoder configuration for the Qwen3-VL language model that
/// conditions Qwen-Image-2.1. The vision tower is deliberately absent: the
/// text-to-image pipeline never feeds pixel inputs, so those tensors are
/// skipped at load time.
public struct Qwen3TextEncoderConfiguration: Sendable {
  public var vocabSize: Int
  public var hiddenSize: Int
  public var numHiddenLayers: Int
  public var numAttentionHeads: Int
  public var numKeyValueHeads: Int
  public var headDim: Int
  public var intermediateSize: Int
  public var ropeTheta: Float
  public var rmsNormEps: Float

  public init(
    vocabSize: Int = 151_936,
    hiddenSize: Int = 4_096,
    numHiddenLayers: Int = 36,
    numAttentionHeads: Int = 32,
    numKeyValueHeads: Int = 8,
    headDim: Int = 128,
    intermediateSize: Int = 12_288,
    ropeTheta: Float = 5_000_000.0,
    rmsNormEps: Float = 1e-6
  ) {
    self.vocabSize = vocabSize
    self.hiddenSize = hiddenSize
    self.numHiddenLayers = numHiddenLayers
    self.numAttentionHeads = numAttentionHeads
    self.numKeyValueHeads = numKeyValueHeads
    self.headDim = headDim
    self.intermediateSize = intermediateSize
    self.ropeTheta = ropeTheta
    self.rmsNormEps = rmsNormEps
  }

  /// Reads the nested `text_config` block of a Qwen3-VL component config.
  public static func fromQwen3VLConfig(_ config: [String: Any]) -> Qwen3TextEncoderConfiguration {
    let text = (config["text_config"] as? [String: Any]) ?? [:]
    func int(_ key: String, _ fallback: Int) -> Int {
      (text[key] as? Int) ?? (config[key] as? Int) ?? fallback
    }
    func float(_ key: String, _ fallback: Float) -> Float {
      if let v = text[key] as? Double { return Float(v) }
      if let v = text[key] as? Int { return Float(v) }
      if let v = config[key] as? Double { return Float(v) }
      if let v = config[key] as? Int { return Float(v) }
      return fallback
    }
    return Qwen3TextEncoderConfiguration(
      vocabSize: int("vocab_size", 151_936),
      hiddenSize: int("hidden_size", 4_096),
      numHiddenLayers: int("num_hidden_layers", 36),
      numAttentionHeads: int("num_attention_heads", 32),
      numKeyValueHeads: int("num_key_value_heads", 8),
      headDim: int("head_dim", 128),
      intermediateSize: int("intermediate_size", 12_288),
      ropeTheta: float("rope_theta", 5_000_000.0),
      rmsNormEps: float("rms_norm_eps", 1e-6)
    )
  }
}

/// The Qwen3-VL language model, exposed with the parameter nesting the
/// MLX checkpoints use (`language_model.model.*`) so weights apply without
/// renaming.
public final class Qwen3VLTextEncoder: Module {
  @ModuleInfo(key: "language_model") var languageModel: Qwen3LanguageModel

  public let configuration: Qwen3TextEncoderConfiguration

  public init(configuration: Qwen3TextEncoderConfiguration = .init()) {
    self.configuration = configuration
    self._languageModel.wrappedValue = Qwen3LanguageModel(configuration: configuration)
  }

  /// Returns the last decoder layer's output before the final RMSNorm.
  ///
  /// The upstream pipeline installs a forward hook that neutralizes the
  /// text model's final norm because that is the representation the
  /// transformer was trained on; running the layers and skipping the final
  /// norm produces the same tensor.
  public func encodeHiddenStates(inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
    languageModel.encodePreNorm(inputIds: inputIds, attentionMask: attentionMask)
  }
}

public final class Qwen3LanguageModel: Module {
  @ModuleInfo(key: "model") var model: Qwen3LMBody

  let configuration: Qwen3TextEncoderConfiguration

  init(configuration: Qwen3TextEncoderConfiguration) {
    self.configuration = configuration
    self._model.wrappedValue = Qwen3LMBody(configuration: configuration)
  }

  func encodePreNorm(inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
    model.encodePreNorm(inputIds: inputIds, attentionMask: attentionMask)
  }
}

public final class Qwen3LMBody: Module {
  @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
  @ModuleInfo(key: "layers") var layers: [Qwen3EncoderLayer]

  let configuration: Qwen3TextEncoderConfiguration

  init(configuration: Qwen3TextEncoderConfiguration) {
    self.configuration = configuration
    self._embedTokens.wrappedValue = Embedding(
      embeddingCount: configuration.vocabSize,
      dimensions: configuration.hiddenSize
    )
    self._layers.wrappedValue = (0..<configuration.numHiddenLayers).map { _ in
      Qwen3EncoderLayer(configuration: configuration)
    }
  }

  func encodePreNorm(inputIds: MLXArray, attentionMask: MLXArray?) -> MLXArray {
    var tokenIds = inputIds
    if tokenIds.dtype != .int32 {
      tokenIds = tokenIds.asType(.int32)
    }
    var hidden = embedTokens(tokenIds)
    let mask = makeMask(hidden: hidden, attentionMask: attentionMask)
    for layer in layers {
      hidden = layer(hidden, mask: mask)
    }
    return hidden
  }

  private func makeMask(
    hidden: MLXArray,
    attentionMask: MLXArray?
  ) -> MLXFast.ScaledDotProductAttentionMaskMode {
    let length = hidden.dim(1)
    if let attentionMask {
      let paddingKeep = attentionMask.asType(.bool).reshaped(attentionMask.dim(0), 1, 1, length)
      let index = MLXArray(0..<length)
      let rows = index.reshaped(length, 1)
      let cols = index.reshaped(1, length)
      let causalKeep = (cols .<= rows).reshaped(1, 1, length, length)
      return .array(causalKeep .&& paddingKeep)
    }
    return .causal
  }
}

final class Qwen3Attention: Module {
  let numAttentionHeads: Int
  let numKeyValueHeads: Int
  let numKeyValueGroups: Int
  let headDim: Int
  let scale: Float

  @ModuleInfo(key: "q_proj") var qProj: Linear
  @ModuleInfo(key: "k_proj") var kProj: Linear
  @ModuleInfo(key: "v_proj") var vProj: Linear
  @ModuleInfo(key: "o_proj") var oProj: Linear
  @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
  @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

  let rope: RoPE

  init(configuration: Qwen3TextEncoderConfiguration) {
    self.numAttentionHeads = configuration.numAttentionHeads
    self.numKeyValueHeads = configuration.numKeyValueHeads
    self.headDim = configuration.headDim
    self.numKeyValueGroups = configuration.numAttentionHeads / configuration.numKeyValueHeads
    self.scale = pow(Float(configuration.headDim), -0.5)

    self._qProj.wrappedValue = Linear(
      configuration.hiddenSize, configuration.numAttentionHeads * configuration.headDim, bias: false)
    self._kProj.wrappedValue = Linear(
      configuration.hiddenSize, configuration.numKeyValueHeads * configuration.headDim, bias: false)
    self._vProj.wrappedValue = Linear(
      configuration.hiddenSize, configuration.numKeyValueHeads * configuration.headDim, bias: false)
    self._oProj.wrappedValue = Linear(
      configuration.numAttentionHeads * configuration.headDim, configuration.hiddenSize, bias: false)
    self._qNorm.wrappedValue = RMSNorm(dimensions: configuration.headDim, eps: configuration.rmsNormEps)
    self._kNorm.wrappedValue = RMSNorm(dimensions: configuration.headDim, eps: configuration.rmsNormEps)
    self.rope = RoPE(
      dimensions: configuration.headDim,
      traditional: false,
      base: configuration.ropeTheta,
      scale: 1.0
    )
  }

  func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
    let batch = x.dim(0)
    let length = x.dim(1)

    var queries = qProj(x)
    var keys = kProj(x)
    let values = vProj(x)

    queries = qNorm(queries.reshaped(batch, length, numAttentionHeads, headDim))
      .transposed(0, 2, 1, 3)
    keys = kNorm(keys.reshaped(batch, length, numKeyValueHeads, headDim)).transposed(0, 2, 1, 3)
    let valuesHeads = values.reshaped(batch, length, numKeyValueHeads, headDim).transposed(0, 2, 1, 3)

    queries = rope(queries, offset: 0)
    keys = rope(keys, offset: 0)

    var expandedKeys = keys
    var expandedValues = valuesHeads
    if numKeyValueHeads != numAttentionHeads {
      expandedKeys = expandHeads(keys, repeats: numKeyValueGroups)
      expandedValues = expandHeads(valuesHeads, repeats: numKeyValueGroups)
    }

    var output = MLXFast.scaledDotProductAttention(
      queries: queries,
      keys: expandedKeys,
      values: expandedValues,
      scale: scale,
      mask: mask
    )
    output = output.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
    return oProj(output)
  }

  private func expandHeads(_ x: MLXArray, repeats: Int) -> MLXArray {
    guard repeats > 1 else { return x }
    var expanded = MLX.expandedDimensions(x, axis: 2)
    expanded = MLX.repeated(expanded, count: repeats, axis: 2)
    let shape = x.shape
    return expanded.reshaped(shape[0], shape[1] * repeats, shape[2], shape[3])
  }
}

final class Qwen3MLP: Module {
  @ModuleInfo(key: "gate_proj") var gateProj: Linear
  @ModuleInfo(key: "down_proj") var downProj: Linear
  @ModuleInfo(key: "up_proj") var upProj: Linear

  init(dimensions: Int, hiddenDimensions: Int) {
    self._gateProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
    self._downProj.wrappedValue = Linear(hiddenDimensions, dimensions, bias: false)
    self._upProj.wrappedValue = Linear(dimensions, hiddenDimensions, bias: false)
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    downProj(silu(gateProj(x)) * upProj(x))
  }
}

final class Qwen3EncoderLayer: Module {
  @ModuleInfo(key: "self_attn") var selfAttention: Qwen3Attention
  @ModuleInfo(key: "mlp") var mlp: Qwen3MLP
  @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
  @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

  init(configuration: Qwen3TextEncoderConfiguration) {
    self._selfAttention.wrappedValue = Qwen3Attention(configuration: configuration)
    self._mlp.wrappedValue = Qwen3MLP(
      dimensions: configuration.hiddenSize,
      hiddenDimensions: configuration.intermediateSize
    )
    self._inputLayerNorm.wrappedValue = RMSNorm(
      dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
    self._postAttentionLayerNorm.wrappedValue = RMSNorm(
      dimensions: configuration.hiddenSize, eps: configuration.rmsNormEps)
  }

  func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
    let attentionOutput = selfAttention(inputLayerNorm(x), mask: mask)
    let hidden = x + attentionOutput
    return hidden + mlp(postAttentionLayerNorm(hidden))
  }
}
