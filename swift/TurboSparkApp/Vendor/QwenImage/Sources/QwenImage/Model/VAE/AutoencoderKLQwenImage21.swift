import Foundation
import MLX
import MLXFast
import MLXNN

/// Decoder-only port of diffusers `AutoencoderKLQwenImage21` for
/// single-frame image decoding.
///
/// The reference's `CausalConv3d` extends `Conv2d` and folds the single
/// frame away, so the whole decoder is a 2D network here. The temporal
/// upsamplers and DupUp3D shortcuts keep their shape semantics: the
/// channel-duplicating shortcut doubles the temporal axis and the
/// `first_chunk` trim keeps exactly one frame, which reduces to a spatial
/// 2x channel-interleaved upsample.
public struct QwenImage21VAEConfiguration: Sendable {
  public var zDim: Int
  public var decoderBaseDim: Int
  public var dimMult: [Int]
  public var numResBlocks: Int
  public var outChannels: Int
  public var latentsMean: [Float]
  public var latentsStd: [Float]

  public init(
    zDim: Int = 64,
    decoderBaseDim: Int = 144,
    dimMult: [Int] = [1, 2, 4, 8, 8],
    numResBlocks: Int = 2,
    outChannels: Int = 4,
    latentsMean: [Float] = [],
    latentsStd: [Float] = []
  ) {
    self.zDim = zDim
    self.decoderBaseDim = decoderBaseDim
    self.dimMult = dimMult
    self.numResBlocks = numResBlocks
    self.outChannels = outChannels
    self.latentsMean = latentsMean
    self.latentsStd = latentsStd
  }

  public static func fromConfigJSON(_ config: [String: Any]) -> QwenImage21VAEConfiguration {
    func int(_ key: String, _ fallback: Int) -> Int {
      (config[key] as? Int) ?? fallback
    }
    let mean = (config["latents_mean"] as? [Any])?.compactMap { ($0 as? Double).map(Float.init) } ?? []
    let std = (config["latents_std"] as? [Any])?.compactMap { ($0 as? Double).map(Float.init) } ?? []
    return QwenImage21VAEConfiguration(
      zDim: int("z_dim", 64),
      decoderBaseDim: int("decoder_base_dim", 144),
      dimMult: (config["dim_mult"] as? [Any])?.compactMap { $0 as? Int } ?? [1, 2, 4, 8, 8],
      numResBlocks: int("num_res_blocks", 2),
      outChannels: int("out_channels", 4),
      latentsMean: mean,
      latentsStd: std
    )
  }
}

/// Runs an MLXNN `Conv2d` on channel-first `[B, C, H, W]` input. MLX
/// convolutions are channel-last, so the wrapper transposes on the way in
/// and out; the weights stay in the checkpoint's MLX layout.
enum QwenImage21Conv {
  static func apply(_ x: MLXArray, _ conv: Conv2d) -> MLXArray {
    let channelLast = x.transposed(0, 2, 3, 1)
    var output = MLX.conv2d(
      channelLast,
      conv.weight,
      stride: .init(conv.stride),
      padding: .init(conv.padding)
    )
    if let bias = conv.bias {
      output = output + bias
    }
    return output.transposed(0, 3, 1, 2)
  }
}

/// Channel-first RMS normalization: L2-normalize over the channel axis,
/// scale by `sqrt(dim)` and the learned gamma, computed in float32.
final class QwenImage21RMSNorm: Module {
  @ParameterInfo(key: "gamma") var gamma: MLXArray

  let dim: Int
  let scale: Float

  init(dim: Int) {
    self.dim = dim
    self.scale = pow(Float(dim), 0.5)
    self._gamma = ParameterInfo(key: "gamma")
    self._gamma.wrappedValue = MLX.ones([dim])
    super.init()
  }

  /// `x` is `[B, C, H, W]`.
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let float = x.asType(.float32)
    // F.normalize(x, dim=1): divide by the channel-axis L2 norm.
    let squaredSum = float * float
    let norm = MLX.sqrt(squaredSum.sum(axis: 1, keepDims: true))
    let normalized = float / MLX.maximum(norm, 1e-12)
    let gammaBroadcast = gamma.asType(.float32).reshaped(1, dim, 1, 1)
    return (normalized * scale * gammaBroadcast).asType(x.dtype)
  }
}

final class QwenImage21ResidualBlock: Module {
  @ModuleInfo(key: "norm1") var norm1: QwenImage21RMSNorm
  @ModuleInfo(key: "conv1") var conv1: Conv2d
  @ModuleInfo(key: "norm2") var norm2: QwenImage21RMSNorm
  @ModuleInfo(key: "conv2") var conv2: Conv2d
  @ModuleInfo(key: "conv_shortcut") var convShortcut: Conv2d?

  init(inDim: Int, outDim: Int) {
    self._norm1 = ModuleInfo(key: "norm1")
    self._norm1.wrappedValue = QwenImage21RMSNorm(dim: inDim)
    self._conv1 = ModuleInfo(key: "conv1")
    self._conv1.wrappedValue = Conv2d(inputChannels: inDim, outputChannels: outDim, kernelSize: 3, padding: 1)
    self._norm2 = ModuleInfo(key: "norm2")
    self._norm2.wrappedValue = QwenImage21RMSNorm(dim: outDim)
    self._conv2 = ModuleInfo(key: "conv2")
    self._conv2.wrappedValue = Conv2d(inputChannels: outDim, outputChannels: outDim, kernelSize: 3, padding: 1)
    self._convShortcut = ModuleInfo(key: "conv_shortcut")
    if inDim != outDim {
      self._convShortcut.wrappedValue = Conv2d(inputChannels: inDim, outputChannels: outDim, kernelSize: 1)
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let shortcut = convShortcut.map { QwenImage21Conv.apply(x, $0) } ?? x
    var hidden = silu(norm1(x))
    hidden = QwenImage21Conv.apply(hidden, conv1)
    hidden = silu(norm2(hidden))
    hidden = QwenImage21Conv.apply(hidden, conv2)
    return hidden + shortcut
  }
}

/// Single-head spatial self-attention with 1x1 convolutions.
final class QwenImage21AttentionBlock: Module {
  let dim: Int
  @ModuleInfo(key: "norm") var norm: QwenImage21RMSNorm
  @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
  @ModuleInfo(key: "proj") var proj: Conv2d

  init(dim: Int) {
    self.dim = dim
    self._norm = ModuleInfo(key: "norm")
    self._norm.wrappedValue = QwenImage21RMSNorm(dim: dim)
    self._toQKV = ModuleInfo(key: "to_qkv")
    self._toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim * 3, kernelSize: 1)
    self._proj = ModuleInfo(key: "proj")
    self._proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {

    let batch = x.dim(0)
    let height = x.dim(2)
    let width = x.dim(3)
    let tokens = height * width

    let normed = norm(x)
    let qkv = QwenImage21Conv.apply(normed, toQKV)
    // [B, 3C, H, W] -> three [B, tokens, C] streams.
    let flattened = qkv.transposed(0, 2, 3, 1).reshaped(batch, tokens, 3, dim)
    let query = flattened[0..., 0..., 0, 0...].reshaped(batch, 1, tokens, dim)
    let key = flattened[0..., 0..., 1, 0...].reshaped(batch, 1, tokens, dim)
    let value = flattened[0..., 0..., 2, 0...].reshaped(batch, 1, tokens, dim)

    var attended = MLXFast.scaledDotProductAttention(
      queries: query, keys: key, values: value, scale: pow(Float(dim), -0.5), mask: .none)
    attended = attended.reshaped(batch, height, width, dim).transposed(0, 3, 1, 2)
    let projected = QwenImage21Conv.apply(attended, proj)
    return projected + x
  }
}

final class QwenImage21MidBlock: Module {
  @ModuleInfo(key: "resnets") var resnets: [QwenImage21ResidualBlock]
  @ModuleInfo(key: "attentions") var attentions: [QwenImage21AttentionBlock]

  init(dim: Int, numLayers: Int = 1) {
    var resnets: [QwenImage21ResidualBlock] = [QwenImage21ResidualBlock(inDim: dim, outDim: dim)]
    var attentions: [QwenImage21AttentionBlock] = []
    for _ in 0..<numLayers {
      attentions.append(QwenImage21AttentionBlock(dim: dim))
      resnets.append(QwenImage21ResidualBlock(inDim: dim, outDim: dim))
    }
    self._resnets = ModuleInfo(key: "resnets")
    self._resnets.wrappedValue = resnets
    self._attentions = ModuleInfo(key: "attentions")
    self._attentions.wrappedValue = attentions
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {

    var hidden = x
    hidden = resnets[0](hidden)
    for (attention, resnet) in zip(attentions, resnets.dropFirst()) {
      hidden = attention(hidden)
      hidden = resnet(hidden)
    }
    return hidden
  }
}

/// Nearest 2x spatial upsample followed by a 3x3 convolution. The temporal
/// branch of the reference `upsample3d` mode is inert for single-frame
/// decodes without a feature cache, so both modes share this path here.
final class QwenImage21Resample: Module {
  @ModuleInfo(key: "resample") var resample: Sequential

  init(dim: Int, outDim: Int) {
    let upsample = QwenImage21NearestUpsample()
    let conv = Conv2d(inputChannels: dim, outputChannels: outDim, kernelSize: 3, padding: 1, bias: true)
    self._resample = ModuleInfo(key: "resample")
    self._resample.wrappedValue = Sequential(layers: [upsample, conv])
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    // layers[0] is the parameter-free upsample; layers[1] the convolution.
    let upsample = resample.layers[0]
    let conv = resample.layers[1] as! Conv2d
    return QwenImage21Conv.apply(upsample(x), conv)
  }
}

/// Parameter-free nearest 2x upsample module so the Sequential parameter
/// indices line up with the reference (`resample.1` is the convolution).
final class QwenImage21NearestUpsample: Module, UnaryLayer {
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let heightRepeated = MLX.repeated(x, count: 2, axis: 2)
    return MLX.repeated(heightRepeated, count: 2, axis: 3)
  }
}

/// Parameter-free channel-duplicating shortcut of the residual up blocks.
///
/// Channels are repeated `factor` times, regrouped so every duplicate maps
/// to a `(frame, height, width)` neighborhood slot, and the first
/// `factorT - 1` frames are trimmed, exactly as the reference DupUp3D with
/// `first_chunk = true` does for a single-frame input.
final class QwenImage21DupUp {
  let inChannels: Int
  let outChannels: Int
  let factorT: Int
  let factorS: Int

  init(inChannels: Int, outChannels: Int, factorT: Int, factorS: Int) {
    precondition(outChannels * factorT * factorS * factorS % inChannels == 0)
    self.inChannels = inChannels
    self.outChannels = outChannels
    self.factorT = factorT
    self.factorS = factorS
  }

  var repeats: Int {
    outChannels * factorT * factorS * factorS / inChannels
  }

  /// `x` is channel-first `[B, inChannels, H, W]` with the single frame
  /// folded away; returns `[B, outChannels, H*factorS, W*factorS]`. The
  /// `first_chunk` trim keeps exactly one frame of the doubled temporal
  /// axis, which is what single-frame decoding needs.
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let batch = x.dim(0)
    let height = x.dim(2)
    let width = x.dim(3)

    let repeated = MLX.repeated(x, count: repeats, axis: 1)
    // Channel regrouping without the singleton frame axis: the reference
    // shape is (B, out, f_t, f_s, f_s, T=1, H, W).
    let grouped = repeated.reshaped(batch, outChannels, factorT, factorS, factorS, height, width)
    let permuted = grouped.transposed(0, 1, 2, 5, 3, 6, 4)
    let merged = permuted.reshaped(batch, outChannels, factorT, height * factorS, width * factorS)
    // first_chunk trim: keep temporal slot [factorT - 1], leaving one frame.
    let trimmed = merged[0..., 0..., (factorT - 1)..., 0..., 0...]
    return trimmed.reshaped(batch, outChannels, height * factorS, width * factorS)
  }
}

final class QwenImage21ResidualUpBlock: Module {
  @ModuleInfo(key: "resnets") var resnets: [QwenImage21ResidualBlock]
  @ModuleInfo(key: "upsampler") var upsampler: QwenImage21Resample?
  let duplicateShortcut: QwenImage21DupUp?

  init(inDim: Int, outDim: Int, numResBlocks: Int, temporalUpsample: Bool, upsample: Bool) {
    var resnets: [QwenImage21ResidualBlock] = []
    var currentDim = inDim
    for _ in 0..<(numResBlocks + 1) {
      resnets.append(QwenImage21ResidualBlock(inDim: currentDim, outDim: outDim))
      currentDim = outDim
    }
    self._resnets = ModuleInfo(key: "resnets")
    self._resnets.wrappedValue = resnets
    self._upsampler = ModuleInfo(key: "upsampler")
    if upsample {
      self._upsampler.wrappedValue = QwenImage21Resample(dim: outDim, outDim: outDim)
      self.duplicateShortcut = QwenImage21DupUp(
        inChannels: inDim, outChannels: outDim,
        factorT: temporalUpsample ? 2 : 1, factorS: 2)
    } else {
      self.duplicateShortcut = nil
    }
    super.init()
  }

  func callAsFunction(_ x: MLXArray) -> MLXArray {
    let input = x
    var hidden = x
    for resnet in resnets {
      hidden = resnet(hidden)
    }
    if let upsampler {
      hidden = upsampler(hidden)
    }
    if let shortcut = duplicateShortcut {
      return hidden + shortcut(input)
    }
    return hidden
  }
}

final class QwenImage21Decoder3d: Module {
  @ModuleInfo(key: "conv_in") var convIn: Conv2d
  @ModuleInfo(key: "mid_block") var midBlock: QwenImage21MidBlock
  @ModuleInfo(key: "up_blocks") var upBlocks: [QwenImage21ResidualUpBlock]
  @ModuleInfo(key: "norm_out") var normOut: QwenImage21RMSNorm
  @ModuleInfo(key: "conv_out") var convOut: Conv2d

  init(configuration: QwenImage21VAEConfiguration) {
    let dim = configuration.decoderBaseDim
    let dimMult = configuration.dimMult
    let dims = [dim * dimMult[dimMult.count - 1]] + dimMult.reversed().map { dim * $0 }

    self._convIn = ModuleInfo(key: "conv_in")
    self._convIn.wrappedValue = Conv2d(inputChannels: configuration.zDim, outputChannels: dims[0], kernelSize: 3, padding: 1, bias: true)
    self._midBlock = ModuleInfo(key: "mid_block")
    self._midBlock.wrappedValue = QwenImage21MidBlock(dim: dims[0])

    // The reference reverses the config's `temperal_downsample` flag list
    // for the decoder's up blocks; the trailing block does not upsample.
    let temporalUpsampleFlags: [Bool] = [true, true, true, false]
    var blocks: [QwenImage21ResidualUpBlock] = []
    for (index, (inDim, outDim)) in zip(dims, dims.dropFirst()).enumerated() {
      let upFlag = index != dimMult.count - 1
      let temporal = upFlag ? (temporalUpsampleFlags[index] ?? false) : false
      blocks.append(
        QwenImage21ResidualUpBlock(
          inDim: inDim, outDim: outDim,
          numResBlocks: configuration.numResBlocks,
          temporalUpsample: temporal, upsample: upFlag))
    }
    self._upBlocks = ModuleInfo(key: "up_blocks")
    self._upBlocks.wrappedValue = blocks

    let finalDim = dims[dims.count - 1]
    self._normOut = ModuleInfo(key: "norm_out")
    self._normOut.wrappedValue = QwenImage21RMSNorm(dim: finalDim)
    self._convOut = ModuleInfo(key: "conv_out")
    self._convOut.wrappedValue = Conv2d(inputChannels: finalDim, outputChannels: configuration.outChannels, kernelSize: 3, padding: 1, bias: true)
    super.init()
  }

  /// `z` is `[B, zDim, 1, H, W]`; returns `[B, outChannels, 1, H*16, W*16]`.
  func callAsFunction(_ z: MLXArray) -> MLXArray {
    let frames = z.dim(2)
    precondition(frames == 1, "the single-frame decoder received \(frames) frames")


    var hidden = QwenImage21Conv.apply(z[0..., 0..., 0, 0..., 0...], convIn)
    hidden = midBlock(hidden)
    for block in upBlocks {
      hidden = block(hidden)
    }
    hidden = silu(normOut(hidden))
    let pixels = QwenImage21Conv.apply(hidden, convOut)
    return MLX.expandedDimensions(pixels, axis: 2)
  }
}

public final class AutoencoderKLQwenImage21: Module {
  public let configuration: QwenImage21VAEConfiguration

  @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
  @ModuleInfo(key: "decoder") var decoder: QwenImage21Decoder3d

  public init(configuration: QwenImage21VAEConfiguration) {
    self.configuration = configuration
    self._postQuantConv = ModuleInfo(key: "post_quant_conv")
    self._postQuantConv.wrappedValue = Conv2d(inputChannels: configuration.zDim, outputChannels: configuration.zDim, kernelSize: 1, bias: true)
    self._decoder = ModuleInfo(key: "decoder")
    self._decoder.wrappedValue = QwenImage21Decoder3d(configuration: configuration)
    super.init()
  }

  /// Decodes denormalized-latent-ready input `z` `[B, zDim, 1, H, W]` to
  /// `[B, outChannels, 1, H*16, W*16]` in `[-1, 1]`.
  public func decode(_ z: MLXArray) -> MLXArray {
    let projected = QwenImage21Conv.apply(z[0..., 0..., 0, 0..., 0...], postQuantConv)
    let restored = MLX.expandedDimensions(projected, axis: 2)
    let decoded = decoder(restored)
    return MLX.minimum(MLX.maximum(decoded, -1.0), 1.0)
  }

  /// Per-channel denormalization: `latents * std + mean`.
  public func denormalizeLatents(_ latents: MLXArray) -> MLXArray {
    precondition(!configuration.latentsMean.isEmpty && !configuration.latentsStd.isEmpty)
    let mean = MLXArray(configuration.latentsMean).reshaped(1, configuration.zDim, 1, 1, 1)
    let std = MLXArray(configuration.latentsStd).reshaped(1, configuration.zDim, 1, 1, 1)
    return latents * std.asType(latents.dtype) + mean.asType(latents.dtype)
  }
}
