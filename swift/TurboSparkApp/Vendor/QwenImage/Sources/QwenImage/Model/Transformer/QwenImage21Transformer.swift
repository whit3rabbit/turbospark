import Foundation
import MLX
import MLXFast
import MLXNN

/// Configuration of the Qwen-Image-2.1 single-stream DiT.
public struct QwenImage21TransformerConfiguration: Sendable {
  public var inChannels: Int
  public var outChannels: Int
  public var numLayers: Int
  public var attentionHeadDim: Int
  public var numAttentionHeads: Int
  public var contextInDim: Int
  public var mlpRatio: Int
  public var axesDimsRope: [Int]
  public var eps: Float
  public var causalCondition: Bool
  public var groupSize: Int
  public var bits: Int

  public init(
    inChannels: Int = 64,
    outChannels: Int = 64,
    numLayers: Int = 32,
    attentionHeadDim: Int = 128,
    numAttentionHeads: Int = 32,
    contextInDim: Int = 4_096,
    mlpRatio: Int = 3,
    axesDimsRope: [Int] = [16, 56, 56],
    eps: Float = 1e-6,
    causalCondition: Bool = true,
    groupSize: Int = 64,
    bits: Int = 4
  ) {
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.numLayers = numLayers
    self.attentionHeadDim = attentionHeadDim
    self.numAttentionHeads = numAttentionHeads
    self.contextInDim = contextInDim
    self.mlpRatio = mlpRatio
    self.axesDimsRope = axesDimsRope
    self.eps = eps
    self.causalCondition = causalCondition
    self.groupSize = groupSize
    self.bits = bits
  }

  public var innerDim: Int { numAttentionHeads * attentionHeadDim }

  public static func fromConfigJSON(_ config: [String: Any]) -> QwenImage21TransformerConfiguration {
    func int(_ key: String, _ fallback: Int) -> Int {
      (config[key] as? Int) ?? fallback
    }
    let quantization = (config["quantization"] as? [String: Any]) ?? [:]
    let axes = (config["axes_dims_rope"] as? [Any])?.compactMap { $0 as? Int } ?? [16, 56, 56]
    return QwenImage21TransformerConfiguration(
      inChannels: int("in_channels", 64),
      outChannels: int("out_channels", 64),
      numLayers: int("num_layers", 32),
      attentionHeadDim: int("attention_head_dim", 128),
      numAttentionHeads: int("num_attention_heads", 32),
      contextInDim: int("context_in_dim", 4_096),
      mlpRatio: int("mlp_ratio", 3),
      axesDimsRope: axes,
      eps: 1e-6,
      causalCondition: (config["causal_condition"] as? Bool) ?? true,
      groupSize: (quantization["group_size"] as? Int) ?? 64,
      bits: (quantization["bits"] as? Int) ?? 4
    )
  }
}

// MARK: - Rotary embedding

/// 3-axis (frame, height, width) rotary embedding over the joint
/// text/image sequence.
///
/// Text tokens advance a shared position on all three axes. The target
/// image freezes the frame axis at the position reached by the text and
/// lays its tokens on a height/width grid centred on zero, so the block's
/// spatial positions do not depend on where it sits in the sequence.
final class QwenImage21Rope {
  private static let negativeOffset = 1024
  private static let positiveLength = 8192

  /// cos/sin tables over indices [-1024, 8192) per axis: [range, dim/2].
  private let cosTables: [MLXArray]
  private let sinTables: [MLXArray]

  init(theta: Float = 10000, axesDim: [Int]) {
    var cosTables: [MLXArray] = []
    var sinTables: [MLXArray] = []
    let indices = MLXArray(
      ((-Self.negativeOffset)..<Self.positiveLength).map { Float($0) }
    )
    for dim in axesDim {
      let half = dim / 2
      let invFreq = MLXArray((0..<half).map { pow(theta, Float(2 * $0) / Float(dim)) })
      let angles = indices.reshaped(-1, 1) / invFreq.reshaped(1, -1)
      cosTables.append(MLX.cos(angles))
      sinTables.append(MLX.sin(angles))
    }
    self.cosTables = cosTables
    self.sinTables = sinTables
  }

  struct TokenIndices {
    let frame: [Int]
    let height: [Int]
    let width: [Int]
  }

  /// Position indices for the pure T2I layout: `textLength` text tokens
  /// followed by one `height x width` target image block.
  static func t2iTokenIndices(textLength: Int, height: Int, width: Int) -> TokenIndices {
    let textFrame = Array(0..<textLength)
    let frame = textFrame + [Int](repeating: textLength, count: height * width)

    let heightRange = Array(-(height - height / 2)..<(height / 2))
    let widthRange = Array(-(width - width / 2)..<(width / 2))
    var imageHeight: [Int] = []
    var imageWidth: [Int] = []
    imageHeight.reserveCapacity(height * width)
    imageWidth.reserveCapacity(height * width)
    for h in heightRange {
      imageHeight.append(contentsOf: [Int](repeating: h, count: width))
    }
    for _ in 0..<height {
      imageWidth.append(contentsOf: widthRange)
    }

    return TokenIndices(frame: frame, height: textFrame + imageHeight, width: textFrame + imageWidth)
  }

  /// cos/sin rows for the given indices: `[count, sum(axesDim)/2]` each.
  func frequencyTables(for indices: TokenIndices) -> (cos: MLXArray, sin: MLXArray) {
    let axes = [indices.frame, indices.height, indices.width]
    var cosParts: [MLXArray] = []
    var sinParts: [MLXArray] = []
    for (axisIndex, index) in axes.enumerated() {
      let rows = MLXArray(index.map { $0 + Self.negativeOffset })
      cosParts.append(cosTables[axisIndex].take(rows, axis: 0))
      sinParts.append(sinTables[axisIndex].take(rows, axis: 0))
    }
    return (MLX.concatenated(cosParts, axis: -1), MLX.concatenated(sinParts, axis: -1))
  }
}

/// Applies rotary embeddings as an adjacent-pair complex rotation,
/// `(x0 + i*x1) * (cos + i*sin)`, computed in float32 and cast back to the
/// input dtype, matching the reference `use_real=False` path.
enum QwenImage21Rotary {
  static func apply(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    // `x` is `[B, H, S, D]`.
    let shape = x.shape
    let batch = shape[0]
    let heads = shape[1]
    let sequence = shape[2]
    let dim = shape[3]
    let pairCount = dim / 2

    let pairs = x.asType(.float32).reshaped(batch, heads, sequence, pairCount, 2)
    let halves = pairs.split(parts: 2, axis: 4)
    let real = halves[0].reshaped(batch, heads, sequence, pairCount)
    let imag = halves[1].reshaped(batch, heads, sequence, pairCount)

    let broadcastCos = cos.reshaped(1, 1, sequence, pairCount)
    let broadcastSin = sin.reshaped(1, 1, sequence, pairCount)
    let rotatedReal = real * broadcastCos - imag * broadcastSin
    let rotatedImag = real * broadcastSin + imag * broadcastCos

    let stacked = MLX.stacked([rotatedReal, rotatedImag], axis: -1)
    return stacked.reshaped(shape).asType(x.dtype)
  }
}

// MARK: - Building blocks

/// RMSNorm whose learnable weight is stored zero-centered: the effective
/// scale is `weight + 1`, computed in float32.
final class QwenImage21ZeroCenterRMSNorm: Module {
  @ParameterInfo(key: "weight") var weight: MLXArray

  private let eps: Float

  init(dim: Int, eps: Float = 1e-6) {
    self.eps = eps
    self._weight = ParameterInfo(key: "weight")
    self._weight.wrappedValue = MLX.zeros([dim])
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let float = x.asType(.float32)
    let meanSquare = (float * float).mean(axis: -1, keepDims: true)
    let rrms = (meanSquare + eps).pow(-0.5)
    return (float * rrms * (weight.asType(.float32) + 1.0)).asType(x.dtype)
  }
}

/// `ZeroCenterRMSNorm -> Linear -> tanh-GELU -> Linear` text projection.
final class QwenImage21TextProjection: Module {
  @ModuleInfo(key: "text_norm") var textNorm: QwenImage21ZeroCenterRMSNorm
  @ModuleInfo(key: "in_layer") var inLayer: Linear
  @ModuleInfo(key: "out_layer") var outLayer: Linear

  init(contextInDim: Int, hiddenSize: Int, eps: Float) {
    self._textNorm = ModuleInfo(key: "text_norm")
    self._textNorm.wrappedValue = QwenImage21ZeroCenterRMSNorm(dim: contextInDim, eps: eps)
    self._inLayer = ModuleInfo(key: "in_layer")
    self._inLayer.wrappedValue = Linear(contextInDim, hiddenSize, bias: false)
    self._outLayer = ModuleInfo(key: "out_layer")
    self._outLayer.wrappedValue = Linear(hiddenSize, hiddenSize, bias: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    var hidden = textNorm(x)
    hidden = inLayer(hidden)
    hidden = QwenImage21Transformer2DModel.tanhGELU(hidden)
    return outLayer(hidden)
  }
}

/// SwiGLU feed-forward: `out(silu(gate(x)) * proj(x))`.
final class QwenImage21SwiGLU: Module {
  @ModuleInfo(key: "proj") var proj: Linear
  @ModuleInfo(key: "out") var out: Linear
  @ModuleInfo(key: "gate_layer") var gateLayer: Linear

  init(hiddenSize: Int, mlpHiddenSize: Int) {
    self._proj = ModuleInfo(key: "proj")
    self._proj.wrappedValue = Linear(hiddenSize, mlpHiddenSize, bias: false)
    self._out = ModuleInfo(key: "out")
    self._out.wrappedValue = Linear(mlpHiddenSize, hiddenSize, bias: false)
    self._gateLayer = ModuleInfo(key: "gate_layer")
    self._gateLayer.wrappedValue = Linear(hiddenSize, mlpHiddenSize, bias: false)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    out(silu(gateLayer(x)) * proj(x))
  }
}

/// Attention projection layer for the single-stream blocks. The projection
/// layout matches the legacy diffusers `Attention` module so checkpoints
/// load unchanged.
final class QwenImage21Attention: Module {
  let heads: Int
  let headDim: Int

  @ModuleInfo(key: "to_q") var toQ: Linear
  @ModuleInfo(key: "to_k") var toK: Linear
  @ModuleInfo(key: "to_v") var toV: Linear
  @ModuleInfo(key: "to_out") var toOut: Sequential
  @ModuleInfo(key: "norm_q") var normQ: RMSNorm
  @ModuleInfo(key: "norm_k") var normK: RMSNorm

  init(dim: Int, heads: Int, headDim: Int, eps: Float) {
    self.heads = heads
    self.headDim = headDim
    let innerDim = heads * headDim
    self._toQ = ModuleInfo(key: "to_q")
    self._toQ.wrappedValue = Linear(dim, innerDim, bias: false)
    self._toK = ModuleInfo(key: "to_k")
    self._toK.wrappedValue = Linear(dim, innerDim, bias: false)
    self._toV = ModuleInfo(key: "to_v")
    self._toV.wrappedValue = Linear(dim, innerDim, bias: false)
    self._toOut = ModuleInfo(key: "to_out")
    self._toOut.wrappedValue = Sequential(layers: [Linear(innerDim, dim, bias: false)])
    self._normQ = ModuleInfo(key: "norm_q")
    self._normQ.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
    self._normK = ModuleInfo(key: "norm_k")
    self._normK.wrappedValue = RMSNorm(dimensions: headDim, eps: eps)
    super.init()
  }

  /// Projects `x` `[B, S, dim]` to post-norm `[B, H, S, D]` q/k/v.
  func project(_ x: MLXArray) -> (query: MLXArray, key: MLXArray, value: MLXArray) {
    let batch = x.dim(0)
    let length = x.dim(1)
    var query = toQ(x).reshaped(batch, length, heads, headDim)
    var key = toK(x).reshaped(batch, length, heads, headDim)
    let value = toV(x).reshaped(batch, length, heads, headDim)
    query = normQ(query)
    key = normK(key)
    return (
      query.transposed(0, 2, 1, 3),
      key.transposed(0, 2, 1, 3),
      value.transposed(0, 2, 1, 3)
    )
  }

  /// Applies rope to `[B, H, S, D]` tensors.
  func applyRope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    QwenImage21Rotary.apply(x, cos: cos, sin: sin)
  }

  func output(_ attended: MLXArray) -> MLXArray {
    let batch = attended.dim(0)
    let length = attended.dim(2)
    let flattened = attended.transposed(0, 2, 1, 3).reshaped(batch, length, -1)
    return toOut(flattened)
  }
}

/// Single-stream block. Modulation is not learned per block: the parent
/// model computes one shared modulation projection and every block reads
/// the same scales and gates out of it, exactly as the reference does.
final class QwenImage21TransformerBlock: Module {
  let innerDim: Int
  let norm1: LayerNorm
  let norm2: LayerNorm
  @ModuleInfo(key: "attn") var attn: QwenImage21Attention
  @ModuleInfo(key: "img_mlp") var imgMLP: QwenImage21SwiGLU

  init(dim: Int, heads: Int, headDim: Int, mlpRatio: Int, eps: Float) {
    self.innerDim = dim
    self.norm1 = LayerNorm(dimensions: dim, eps: eps, affine: false, bias: false)
    self.norm2 = LayerNorm(dimensions: dim, eps: eps, affine: false, bias: false)
    self._attn = ModuleInfo(key: "attn")
    self._attn.wrappedValue = QwenImage21Attention(dim: dim, heads: heads, headDim: headDim, eps: eps)
    self._imgMLP = ModuleInfo(key: "img_mlp")
    self._imgMLP.wrappedValue = QwenImage21SwiGLU(hiddenSize: dim, mlpHiddenSize: dim * mlpRatio)
    super.init()
  }

  /// - Parameters:
  ///   - hidden: `[1, S, dim]`.
  ///   - modulation: per-token rows `[1, S, 4 * dim]`.
  ///   - attend: closure running attention (rope, masks, KV cache) over a
  ///     modulated `[1, S, dim]` input for this block's layer index.
  func callAsFunction(
    _ hidden: MLXArray,
    modulation: MLXArray,
    attend: (QwenImage21Attention, MLXArray) -> MLXArray
  ) -> MLXArray {
    let pairWidth = 2 * innerDim
    let mod1Scale = modulation[0..., 0..., 0..<innerDim]
    let mod1Gate = modulation[0..., 0..., innerDim..<pairWidth]
    let mod2Scale = modulation[0..., 0..., pairWidth..<(pairWidth + innerDim)]
    let mod2Gate = modulation[0..., 0..., (pairWidth + innerDim)..<(2 * pairWidth)]

    let normed1 = norm1(hidden)
    let modulated1 = normed1 + (normed1 * mod1Scale)
    var x = hidden + MLX.tanh(mod1Gate) * attend(attn, modulated1)

    let normed2 = norm2(x)
    let modulated2 = normed2 + (normed2 * mod2Scale)
    x = x + MLX.tanh(mod2Gate) * imgMLP(modulated2)
    return x
  }
}

/// Final adaptive norm. Scale only: `LN(x) * (1 + linear(silu(temb)))`.
final class QwenImage21AdaLayerNormContinuous: Module {
  @ModuleInfo(key: "linear") var linear: Linear
  let norm: LayerNorm

  init(embeddingDim: Int, conditioningDim: Int, eps: Float) {
    self._linear = ModuleInfo(key: "linear")
    self._linear.wrappedValue = Linear(conditioningDim, embeddingDim, bias: false)
    self.norm = LayerNorm(dimensions: embeddingDim, eps: eps, affine: false, bias: false)
    super.init()
  }

  /// - Parameters:
  ///   - conditioning: per-token scale rows `[1, S, dim]`.
  func callAsFunction(_ x: MLXArray, scale: MLXArray) -> MLXArray {
    let normed = norm(x)
    return normed * (scale + 1.0)
  }
}

// MARK: - KV cache

/// Per-layer cached prefix keys and values (post-RoPE), extracted on the
/// first denoising step and reused on every later one.
public final class QwenImage21LayerKVCache {
  var keys: MLXArray?
  var values: MLXArray?
}

public final class QwenImage21KVCache {
  let layerCaches: [QwenImage21LayerKVCache]

  init(numLayers: Int) {
    self.layerCaches = (0..<numLayers).map { _ in QwenImage21LayerKVCache() }
  }
}

// MARK: - Model

public final class QwenImage21Transformer2DModel: Module {
  public let configuration: QwenImage21TransformerConfiguration

  @ModuleInfo(key: "txt_in") var txtIn: QwenImage21TextProjection
  @ModuleInfo(key: "img_in") var imgIn: Linear
  @ModuleInfo(key: "time_text_embed") var timeTextEmbed: QwenImage21TimestepProjEmbeddings
  @ModuleInfo(key: "modulation") var modulation: Sequential
  @ModuleInfo(key: "transformer_blocks") var blocks: [QwenImage21TransformerBlock]
  @ModuleInfo(key: "norm_out") var normOut: QwenImage21AdaLayerNormContinuous
  @ModuleInfo(key: "proj_out") var projOut: Linear

  private let rope: QwenImage21Rope

  public init(configuration: QwenImage21TransformerConfiguration) {
    self.configuration = configuration
    let inner = configuration.innerDim
    self._txtIn = ModuleInfo(key: "txt_in")
    self._txtIn.wrappedValue = QwenImage21TextProjection(
      contextInDim: configuration.contextInDim, hiddenSize: inner, eps: configuration.eps)
    self._imgIn = ModuleInfo(key: "img_in")
    self._imgIn.wrappedValue = Linear(configuration.inChannels, inner, bias: false)
    self._timeTextEmbed = ModuleInfo(key: "time_text_embed")
    self._timeTextEmbed.wrappedValue = QwenImage21TimestepProjEmbeddings(embeddingDim: inner)
    self._modulation = ModuleInfo(key: "modulation")
    self._modulation.wrappedValue = Sequential(layers: [Linear(inner, 4 * inner, bias: false)])
    self._blocks = ModuleInfo(key: "transformer_blocks")
    self._blocks.wrappedValue = (0..<configuration.numLayers).map { _ in
      QwenImage21TransformerBlock(
        dim: inner,
        heads: configuration.numAttentionHeads,
        headDim: configuration.attentionHeadDim,
        mlpRatio: configuration.mlpRatio,
        eps: configuration.eps
      )
    }
    self._normOut = ModuleInfo(key: "norm_out")
    self._normOut.wrappedValue = QwenImage21AdaLayerNormContinuous(
      embeddingDim: inner, conditioningDim: inner, eps: configuration.eps)
    self._projOut = ModuleInfo(key: "proj_out")
    self._projOut.wrappedValue = Linear(inner, configuration.outChannels, bias: false)
    self.rope = QwenImage21Rope(theta: 10000, axesDim: configuration.axesDimsRope)
  }

  public struct ForwardLayout: Sendable {
    public let textLength: Int
    public let imageHeight: Int
    public let imageWidth: Int

    public var targetTokens: Int { imageHeight * imageWidth }
    public var sequenceLength: Int { textLength + targetTokens }
  }

  /// Prefill forward for the pure T2I layout: `textLength` text tokens
  /// followed by the target image block, `[text | image]` contiguous. With
  /// `cache` the prefix (text) keys and values are stored for reuse on
  /// later steps.
  public func forwardPrefill(
    latents: MLXArray,
    textEmbeddings: MLXArray,
    timestep: MLXArray,
    layout: ForwardLayout,
    cache: QwenImage21KVCache?
  ) -> MLXArray {
    let projectedText = txtIn(textEmbeddings)
    let projectedImage = imgIn(latents)
    var hidden = MLX.concatenated([projectedText, projectedImage], axis: 1)

    let indices = QwenImage21Rope.t2iTokenIndices(
      textLength: layout.textLength,
      height: layout.imageHeight,
      width: layout.imageWidth
    )
    let (cos, sin) = rope.frequencyTables(for: indices)
    let modulation = perTokenModulation(timestep, textLength: layout.textLength, imageTokens: layout.targetTokens)
    let attentionScale = self.attentionScale

    for (index, block) in blocks.enumerated() {
      let layerCache = cache?.layerCaches[index]
      hidden = block(hidden, modulation: modulation) { attn, x in
        let (query, key, value) = attn.project(x)
        let queryRotated = attn.applyRope(query, cos: cos, sin: sin)
        let keyRotated = attn.applyRope(key, cos: cos, sin: sin)

        if let layerCache {
          // Only the text prefix is timestep-independent and cacheable.
          layerCache.keys = keyRotated[0..., 0..., 0..<layout.textLength, 0...]
          layerCache.values = value[0..., 0..., 0..<layout.textLength, 0...]
          MLX.eval(layerCache.keys!, layerCache.values!)
        }

        // Block-causal rule for this layout: the text prefix is causal
        // over its own keys, and the target image attends to everything.
        let textAttention = MLXFast.scaledDotProductAttention(
          queries: queryRotated[0..., 0..., 0..<layout.textLength, 0...],
          keys: keyRotated[0..., 0..., 0..<layout.textLength, 0...],
          values: value[0..., 0..., 0..<layout.textLength, 0...],
          scale: attentionScale,
          mask: .causal
        )
        let imageAttention = MLXFast.scaledDotProductAttention(
          queries: queryRotated[0..., 0..., layout.textLength..., 0...],
          keys: keyRotated,
          values: value,
          scale: attentionScale,
          mask: .none
        )
        let attended = MLX.concatenated([textAttention, imageAttention], axis: 2)
        return attn.output(attended)
      }
    }

    let scale = perTokenFinalScale(timestep, textLength: layout.textLength, imageTokens: layout.targetTokens)
    let normed = normOut(hidden, scale: scale)
    let output = projOut(normed)
    return output[0..., layout.textLength..., 0...]
  }

  /// Cached decode forward: only the target image tokens are recomputed,
  /// attending to the stored prefix keys and values plus their own block.
  public func forwardCached(
    latents: MLXArray,
    timestep: MLXArray,
    layout: ForwardLayout,
    cache: QwenImage21KVCache
  ) -> MLXArray {
    var hidden = imgIn(latents)

    let allIndices = QwenImage21Rope.t2iTokenIndices(
      textLength: layout.textLength,
      height: layout.imageHeight,
      width: layout.imageWidth
    )
    let targetIndices = QwenImage21Rope.TokenIndices(
      frame: Array(allIndices.frame.suffix(layout.targetTokens)),
      height: Array(allIndices.height.suffix(layout.targetTokens)),
      width: Array(allIndices.width.suffix(layout.targetTokens))
    )
    let (cos, sin) = rope.frequencyTables(for: targetIndices)
    let modulation = perTokenModulation(timestep, textLength: 0, imageTokens: layout.targetTokens)
    let attentionScale = self.attentionScale

    for (index, block) in blocks.enumerated() {
      let layerCache = cache.layerCaches[index]
      hidden = block(hidden, modulation: modulation) { attn, x in
        let (query, key, value) = attn.project(x)
        let queryRotated = attn.applyRope(query, cos: cos, sin: sin)
        let keyRotated = attn.applyRope(key, cos: cos, sin: sin)

        guard let cachedKeys = layerCache.keys, let cachedValues = layerCache.values else {
          fatalError("forwardCached called before forwardPrefill populated the KV cache")
        }
        let keys = MLX.concatenated([cachedKeys, keyRotated], axis: 2)
        let values = MLX.concatenated([cachedValues, value], axis: 2)
        let attended = MLXFast.scaledDotProductAttention(
          queries: queryRotated, keys: keys, values: values, scale: attentionScale, mask: .none)
        return attn.output(attended)
      }
    }

    // Every cached token is a target token, so the real timestep row
    // applies to the whole sequence.
    let scale = perTokenFinalScale(timestep, textLength: 0, imageTokens: layout.targetTokens)
    let normed = normOut(hidden, scale: scale)
    return projOut(normed)
  }

  // MARK: - Modulation helpers

  /// The shared modulation projection evaluated on `[t, 0]`: row 0 is the
  /// real timestep, row 1 the `t = 0` row prefix tokens modulate from.
  private func modulationProjection(_ timestep: MLXArray) -> MLXArray {
    let withZero = MLX.concatenated([timestep.reshaped(-1).asType(.float32), MLXArray([Float(0.0)])], axis: 0)
    let temb = timeTextEmbed(withZero)
    guard let linear = modulation.layers.first as? Linear else {
      fatalError("the shared modulation projection is missing its linear layer")
    }
    return linear(silu(temb))
  }

  /// Per-token modulation rows `[1, textLength + imageTokens, 4 * dim]`:
  /// prefix tokens read the `t = 0` row, target-image tokens the real row.
  private func perTokenModulation(_ timestep: MLXArray, textLength: Int, imageTokens: Int) -> MLXArray {
    let rows = modulationProjection(timestep)
    return tileModulationRows(rows, realRow: 0, zeroRow: 1, textLength: textLength, imageTokens: imageTokens)
  }

  /// Per-token final-norm scale rows `[1, S, dim]` with the same row
  /// selection.
  private func perTokenFinalScale(_ timestep: MLXArray, textLength: Int, imageTokens: Int) -> MLXArray {
    let withZero = MLX.concatenated([timestep.reshaped(-1).asType(.float32), MLXArray([Float(0.0)])], axis: 0)
    let temb = timeTextEmbed(withZero)
    let rows = normOut.linear(silu(temb))
    return tileModulationRows(rows, realRow: 0, zeroRow: 1, textLength: textLength, imageTokens: imageTokens)
  }

  private func tileModulationRows(
    _ rows: MLXArray,
    realRow: Int,
    zeroRow: Int,
    textLength: Int,
    imageTokens: Int
  ) -> MLXArray {
    let width = rows.dim(1)
    let sequence = textLength + imageTokens
    if textLength == 0 {
      let real = MLX.repeated(rows[realRow..<realRow + 1], count: imageTokens, axis: 0)
      return real.reshaped(1, sequence, width)
    }
    let zero = MLX.repeated(rows[zeroRow..<zeroRow + 1], count: textLength, axis: 0)
    let real = MLX.repeated(rows[realRow..<realRow + 1], count: imageTokens, axis: 0)
    return MLX.concatenated([zero, real], axis: 0).reshaped(1, sequence, width)
  }

  private var attentionScale: Float {
    pow(Float(configuration.attentionHeadDim), -0.5)
  }

  /// GELU with the tanh approximation, matching `nn.GELU(approximate:
  /// "tanh")`.
  static func tanhGELU(_ x: MLXArray) -> MLXArray {
    let float = x.asType(.float32)
    let cubic = float * float * float * 0.044715 + float
    let tanh = MLX.tanh(MLXArray(Float(0.797_884_56)) * cubic)
    let result = 0.5 * float * (tanh + 1.0)
    return result.asType(x.dtype)
  }
}

/// Sinusoidal timestep projection: `cos` fills the first half of the
/// channels and `sin` the second, then two bias-free linears with SiLU.
final class QwenImage21TimestepProjEmbeddings: Module {
  @ModuleInfo(key: "linear_1") var linear1: Linear
  @ModuleInfo(key: "linear_2") var linear2: Linear

  private let timestepDim = 256
  private let timeFactor: Float = 1000.0

  init(embeddingDim: Int) {
    self._linear1 = ModuleInfo(key: "linear_1")
    self._linear1.wrappedValue = Linear(timestepDim, embeddingDim, bias: false)
    self._linear2 = ModuleInfo(key: "linear_2")
    self._linear2.wrappedValue = Linear(embeddingDim, embeddingDim, bias: false)
    super.init()
  }

  private var freqs: MLXArray {
    let half = timestepDim / 2
    return MLXArray((0..<half).map { exp(-log(10000.0) * Float($0) / Float(half)) })
  }

  func callAsFunction(_ timestep: MLXArray) -> MLXArray {
    let scaled = timestep.asType(.float32) * timeFactor
    let args = scaled.reshaped(-1, 1) * freqs.reshaped(1, -1)
    let embedding = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    var hidden = linear1(embedding)
    hidden = silu(hidden)
    return linear2(hidden)
  }
}
