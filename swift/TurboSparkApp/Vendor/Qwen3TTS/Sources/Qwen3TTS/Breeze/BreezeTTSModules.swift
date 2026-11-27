import Foundation
import MLX
import MLXLMCommon
import MLXNN

// Breeze TTS 2 stacks, ported from mlx-audio 0.5.7 `breeze_tts.py`. Property
// keys follow the checkpoint's module tree (`backbone_model.*`,
// `depth_decoder.*`, `text_encoder.*`) so `update(parameters:verify:.all)`
// catches any missing or extra tensor. Linears and embeddings are
// `@ModuleInfo` so `quantize(model:)` can swap in quantized layers.

// MARK: - Qwen3 / Llama decoder block (backbone and depth decoder)

final class BreezeAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    // Qwen3 normalizes q and k per head; Llama does not.
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm?
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm?
    // Held as a closure so Module reflection does not treat the RoPE as a
    // child: Llama3RoPE caches `freqs`, which would read as a missing weight.
    let rope: (MLXArray, Int) -> MLXArray

    init(hidden: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float, qkNorm: Bool, rope: OffsetLayer) {
        self.heads = heads; self.kvHeads = kvHeads; self.headDim = headDim
        scale = pow(Float(headDim), -0.5)
        _qProj.wrappedValue = Linear(hidden, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(hidden, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, hidden, bias: false)
        _qNorm.wrappedValue = qkNorm ? RMSNorm(dimensions: headDim, eps: eps) : nil
        _kNorm.wrappedValue = qkNorm ? RMSNorm(dimensions: headDim, eps: eps) : nil
        self.rope = { rope($0, offset: $1) }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qProj(x).reshaped(b, l, heads, headDim)
        var k = kProj(x).reshaped(b, l, kvHeads, headDim)
        if let qNorm, let kNorm { q = qNorm(q); k = kNorm(k) }
        q = q.transposed(0, 2, 1, 3)
        k = k.transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let offset = cache?.offset ?? 0
        q = rope(q, offset)
        k = rope(k, offset)
        let out = attentionWithCacheUpdate(queries: q, keys: k, values: v, cache: cache, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class BreezeMLP: Module, UnaryLayer {
    @ModuleInfo(key: "gate_proj") var gateProj: Linear
    @ModuleInfo(key: "up_proj") var upProj: Linear
    @ModuleInfo(key: "down_proj") var downProj: Linear
    let gelu: Bool

    init(hidden: Int, intermediate: Int, gelu: Bool = false) {
        _gateProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _upProj.wrappedValue = Linear(hidden, intermediate, bias: false)
        _downProj.wrappedValue = Linear(intermediate, hidden, bias: false)
        self.gelu = gelu
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let gate = gateProj(x)
        // T5Gemma2 uses the tanh GELU (`gelu_pytorch_tanh`); Qwen3/Llama use SiLU.
        return downProj((gelu ? geluApproximate(gate) : silu(gate)) * upProj(x))
    }
}

final class BreezeDecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: BreezeAttention
    @ModuleInfo var mlp: BreezeMLP
    @ModuleInfo(key: "input_layernorm") var inputLayerNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postAttentionLayerNorm: RMSNorm

    init(hidden: Int, intermediate: Int, heads: Int, kvHeads: Int, headDim: Int, eps: Float, qkNorm: Bool, rope: OffsetLayer) {
        _selfAttn.wrappedValue = BreezeAttention(hidden: hidden, heads: heads, kvHeads: kvHeads, headDim: headDim, eps: eps, qkNorm: qkNorm, rope: rope)
        _mlp.wrappedValue = BreezeMLP(hidden: hidden, intermediate: intermediate)
        _inputLayerNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
        _postAttentionLayerNorm.wrappedValue = RMSNorm(dimensions: hidden, eps: eps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode, cache: KVCache?) -> MLXArray {
        let h = x + selfAttn(inputLayerNorm(x), mask: mask, cache: cache)
        return h + mlp(postAttentionLayerNorm(h))
    }
}

/// A fresh prompt or a cacheless pass is causal over its own length; a
/// single cached step attends to everything before it.
func breezeCausalMask(length: Int) -> MLXFast.ScaledDotProductAttentionMaskMode {
    length > 1 ? .causal : .none
}

// MARK: - Backbone (qwen3 over summed audio codebook embeddings)

final class BreezeAudioEmbedding: Module {
    let numCodebooks: Int, vocabSize: Int
    @ModuleInfo(key: "embed_audio_tokens") var embedAudioTokens: Embedding

    init(numCodebooks: Int, vocabSize: Int, hidden: Int) {
        self.numCodebooks = numCodebooks; self.vocabSize = vocabSize
        _embedAudioTokens.wrappedValue = Embedding(embeddingCount: numCodebooks * vocabSize, dimensions: hidden)
    }

    /// codebooks: [batch, time, numCodebooks] -> [batch, time, hidden]
    func callAsFunction(_ codebooks: MLXArray) -> MLXArray {
        let offsets = (MLXArray(0 ..< numCodebooks).asType(codebooks.dtype) * Int32(vocabSize)).reshaped(1, 1, numCodebooks)
        return embedAudioTokens(codebooks + offsets).sum(axis: -2)
    }
}

final class BreezeBackbone: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: BreezeAudioEmbedding
    @ModuleInfo var layers: [BreezeDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: BreezeTTSConfig) {
        let b = config.backbone
        _embedTokens.wrappedValue = BreezeAudioEmbedding(numCodebooks: config.numCodebooks, vocabSize: config.vocabSize, hidden: b.hiddenSize)
        _layers.wrappedValue = (0 ..< b.layers).map { _ in
            BreezeDecoderLayer(hidden: b.hiddenSize, intermediate: b.intermediateSize, heads: b.heads, kvHeads: b.kvHeads,
                               headDim: b.headDim, eps: b.rmsNormEps, qkNorm: true,
                               rope: initializeRope(dims: b.headDim, base: b.ropeTheta, traditional: false, scalingConfig: nil, maxPositionEmbeddings: nil))
        }
        _norm.wrappedValue = RMSNorm(dimensions: b.hiddenSize, eps: b.rmsNormEps)
    }

    func callAsFunction(embeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        var h = embeddings
        let mask = breezeCausalMask(length: h.dim(1))
        for (layer, c) in zip(layers, cache) { h = layer(h, mask: mask, cache: c) }
        return norm(h)
    }

    func makeCache() -> [KVCache] { layers.map { _ in KVCacheSimple() } }
}

// MARK: - T5Gemma2 bidirectional text encoder

/// T5Gemma2's zero-initialized offset RMSNorm: scale is `1 + weight`.
final class BreezeOffsetRMSNorm: Module, UnaryLayer {
    var weight: MLXArray
    let eps: Float
    init(dimensions: Int, eps: Float) {
        weight = MLXArray.zeros([dimensions]); self.eps = eps
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.rmsNorm(x, weight: 1 + weight, eps: eps)
    }
}

final class BreezeTextEmbedding: Module {
    var weight: MLXArray
    @ParameterInfo(key: "eoi_embedding") var eoiEmbedding: MLXArray
    let eoiTokenIndex: Int

    init(vocabSize: Int, hidden: Int, eoiTokenIndex: Int) {
        weight = MLXArray.zeros([vocabSize, hidden])
        _eoiEmbedding.wrappedValue = MLXArray.zeros([hidden])
        self.eoiTokenIndex = eoiTokenIndex
    }

    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let embeds = weight[ids] * sqrt(Float(weight.dim(-1)))
        return MLX.where(expandedDimensions(ids .== Int32(eoiTokenIndex), axis: -1), eoiEmbedding, embeds)
    }
}

final class BreezeTextAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float
    let layerType: String
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: BreezeOffsetRMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: BreezeOffsetRMSNorm
    // Held as a closure so Module reflection does not treat the RoPE as a
    // child: Llama3RoPE caches `freqs`, which would read as a missing weight.
    let rope: (MLXArray, Int) -> MLXArray

    init(_ t: BreezeTTSConfig.TextEncoder, layerType: String) {
        heads = t.heads; kvHeads = t.kvHeads; headDim = t.headDim
        scale = pow(t.queryPreAttnScalar, -0.5)
        self.layerType = layerType
        _qProj.wrappedValue = Linear(t.hiddenSize, heads * headDim, bias: false)
        _kProj.wrappedValue = Linear(t.hiddenSize, kvHeads * headDim, bias: false)
        _vProj.wrappedValue = Linear(t.hiddenSize, kvHeads * headDim, bias: false)
        _oProj.wrappedValue = Linear(heads * headDim, t.hiddenSize, bias: false)
        _qNorm.wrappedValue = BreezeOffsetRMSNorm(dimensions: headDim, eps: t.rmsNormEps)
        _kNorm.wrappedValue = BreezeOffsetRMSNorm(dimensions: headDim, eps: t.rmsNormEps)
        let spec = t.rope[layerType]!
        let layer = initializeRope(dims: headDim, base: spec.base, traditional: false, scalingConfig: spec.scaling,
                                   maxPositionEmbeddings: t.maxPositionEmbeddings)
        rope = { layer($0, offset: $1) }
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let q = rope(qNorm(qProj(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3), 0)
        let k = rope(kNorm(kProj(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3), 0)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: mask)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class BreezeTextLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: BreezeTextAttention
    @ModuleInfo var mlp: BreezeMLP
    @ModuleInfo(key: "pre_self_attn_layernorm") var preSelfAttn: BreezeOffsetRMSNorm
    @ModuleInfo(key: "post_self_attn_layernorm") var postSelfAttn: BreezeOffsetRMSNorm
    @ModuleInfo(key: "pre_feedforward_layernorm") var preFeedforward: BreezeOffsetRMSNorm
    @ModuleInfo(key: "post_feedforward_layernorm") var postFeedforward: BreezeOffsetRMSNorm

    init(_ t: BreezeTTSConfig.TextEncoder, layerType: String) {
        _selfAttn.wrappedValue = BreezeTextAttention(t, layerType: layerType)
        _mlp.wrappedValue = BreezeMLP(hidden: t.hiddenSize, intermediate: t.intermediateSize, gelu: true)
        _preSelfAttn.wrappedValue = BreezeOffsetRMSNorm(dimensions: t.hiddenSize, eps: t.rmsNormEps)
        _postSelfAttn.wrappedValue = BreezeOffsetRMSNorm(dimensions: t.hiddenSize, eps: t.rmsNormEps)
        _preFeedforward.wrappedValue = BreezeOffsetRMSNorm(dimensions: t.hiddenSize, eps: t.rmsNormEps)
        _postFeedforward.wrappedValue = BreezeOffsetRMSNorm(dimensions: t.hiddenSize, eps: t.rmsNormEps)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXFast.ScaledDotProductAttentionMaskMode) -> MLXArray {
        let h = x + postSelfAttn(selfAttn(preSelfAttn(x), mask: mask))
        return h + postFeedforward(mlp(preFeedforward(h)))
    }
}

/// The T5Gemma2 sliding pattern: `(window + 1) / 2` positions to the left and
/// `window / 2 + 1` to the right, both counting the query itself
/// (`_t5gemma2_attention_mask`). Full-attention layers are unmasked.
func breezeSlidingMask(length: Int, window: Int) -> MLXArray {
    let query = MLXArray(0 ..< length).reshaped(length, 1)
    let key = MLXArray(0 ..< length).reshaped(1, length)
    let distance = query - key
    let left = Int32((window + 1) / 2), right = Int32(window / 2 + 1)
    return ((distance .>= 0) .&& (distance .< left)) .|| ((distance .< 0) .&& ((-distance) .< right))
}

final class BreezeTextEncoder: Module {
    let config: BreezeTTSConfig.TextEncoder
    @ModuleInfo(key: "embed_tokens") var embedTokens: BreezeTextEmbedding
    @ModuleInfo var layers: [BreezeTextLayer]
    @ModuleInfo var norm: BreezeOffsetRMSNorm

    init(_ t: BreezeTTSConfig.TextEncoder) {
        config = t
        _embedTokens.wrappedValue = BreezeTextEmbedding(vocabSize: t.vocabSize, hidden: t.hiddenSize, eoiTokenIndex: t.eoiTokenIndex)
        _layers.wrappedValue = t.layerTypes.map { BreezeTextLayer(t, layerType: $0) }
        _norm.wrappedValue = BreezeOffsetRMSNorm(dimensions: t.hiddenSize, eps: t.rmsNormEps)
    }

    /// ids: [1, length] -> [1, length, hidden]. Single unpadded sequence.
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var h = embedTokens(ids)
        let sliding = MLXFast.ScaledDotProductAttentionMaskMode.array(breezeSlidingMask(length: ids.dim(1), window: config.slidingWindow))
        for layer in layers {
            h = layer(h, mask: layer.selfAttn.layerType == "sliding_attention" ? sliding : .none)
        }
        return norm(h)
    }
}

// MARK: - Depth decoder (llama over the codebook sequence)

final class BreezeDepthModel: Module {
    let vocabSize: Int
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    @ModuleInfo(key: "inputs_embeds_projector") var inputsEmbedsProjector: Linear
    @ModuleInfo var layers: [BreezeDecoderLayer]
    @ModuleInfo var norm: RMSNorm

    init(_ config: BreezeTTSConfig) {
        let d = config.depth
        vocabSize = config.vocabSize
        let embedSize = config.backbone.hiddenSize
        _embedTokens.wrappedValue = Embedding(embeddingCount: config.numCodebooks * config.vocabSize, dimensions: embedSize)
        _inputsEmbedsProjector.wrappedValue = Linear(embedSize, d.hiddenSize, bias: false)
        _layers.wrappedValue = (0 ..< d.layers).map { _ in
            BreezeDecoderLayer(hidden: d.hiddenSize, intermediate: d.intermediateSize, heads: d.heads, kvHeads: d.kvHeads,
                               headDim: d.headDim, eps: d.rmsNormEps, qkNorm: false,
                               rope: initializeRope(dims: d.headDim, base: d.ropeTheta, traditional: false,
                                                    scalingConfig: d.ropeScaling, maxPositionEmbeddings: d.maxPositionEmbeddings))
        }
        _norm.wrappedValue = RMSNorm(dimensions: d.hiddenSize, eps: d.rmsNormEps)
    }

    /// tokens: [1, T] starting with a placeholder that the backbone state
    /// replaces; later positions carry codebook offsets. Cacheless and causal.
    func callAsFunction(_ tokens: MLXArray, backboneHidden: MLXArray) -> MLXArray {
        let t = tokens.dim(1)
        let positions = MLX.maximum(MLXArray(0 ..< t) - 1, 0).asType(.int32)
        let embeds = embedTokens(tokens + (positions * Int32(vocabSize)).reshaped(1, t))
        let joined = concatenated([expandedDimensions(backboneHidden, axis: 1).asType(embeds.dtype), embeds[0..., 1...]], axis: 1)
        var h = inputsEmbedsProjector(joined)
        let mask = breezeCausalMask(length: t)
        for layer in layers { h = layer(h, mask: mask, cache: nil) }
        return norm(h)
    }
}

final class BreezeDepthDecoder: Module {
    @ModuleInfo var model: BreezeDepthModel
    /// `[numCodebooks - 1, hidden, vocab]`: one output projection per depth position.
    @ModuleInfo(key: "codebooks_head") var codebooksHead: BreezeCodebooksHead

    init(_ config: BreezeTTSConfig) {
        _model.wrappedValue = BreezeDepthModel(config)
        _codebooksHead.wrappedValue = BreezeCodebooksHead(heads: config.numCodebooks - 1, hidden: config.depth.hiddenSize, vocab: config.vocabSize)
    }

    /// The last input token is codebook N-1 and predicts codebook N.
    func nextLogits(_ tokens: MLXArray, backboneHidden: MLXArray) -> MLXArray {
        let head = tokens.dim(1) - 2
        precondition(head >= 0 && head < codebooksHead.weight.dim(0), "depth input needs the placeholder and one codebook")
        let hidden = model(tokens, backboneHidden: backboneHidden)[0..., -1, 0...]
        return matmul(hidden, codebooksHead.weight[head])
    }
}

final class BreezeCodebooksHead: Module {
    var weight: MLXArray
    init(heads: Int, hidden: Int, vocab: Int) { weight = MLXArray.zeros([heads, hidden, vocab]) }
}
