import Foundation
import MLX
import MLXNN

final class SiLUModule: Module, UnaryLayer {
  func callAsFunction(_ x: MLXArray) -> MLXArray {
    MLXNN.silu(x)
  }
}

final class ZImageTimestepEmbedder: Module {
  let frequencyEmbeddingSize: Int

  @ModuleInfo(key: "mlp") var mlp: (Linear, SiLUModule, Linear)

  init(outSize: Int, midSize: Int? = nil, frequencyEmbeddingSize: Int = 256) {
    self.frequencyEmbeddingSize = frequencyEmbeddingSize
    let hidden = midSize ?? outSize
    self._mlp.wrappedValue = (
      Linear(frequencyEmbeddingSize, hidden, bias: true),
      SiLUModule(),
      Linear(hidden, outSize, bias: true)
    )
    super.init()
  }

  private func timestepEmbedding(_ timesteps: MLXArray, dim: Int, maxPeriod: Float = 10_000.0) -> MLXArray {
    let dtype = timesteps.dtype
    let halfDim = dim / 2
    var exponent = -MLX.log(MLXArray(maxPeriod))
    exponent = exponent * MLXArray(0..<halfDim).asType(dtype)
    exponent = exponent / MLXArray(Float(halfDim))
    let freqs = MLX.exp(exponent)

    var args = timesteps[.ellipsis, .newAxis] * freqs[.newAxis]
    let cosPart = MLX.cos(args)
    let sinPart = MLX.sin(args)
    var emb = MLX.concatenated([cosPart, sinPart], axis: -1)

    if dim % 2 != 0 {
      let pad = MLX.zeros([emb.dim(0), 1], dtype: dtype)
      emb = MLX.concatenated([emb, pad], axis: -1)
    }
    return emb
  }

  func castFrequencyEmbeddingToMLPInputDTypeIfNeeded(_ input: MLXArray) -> MLXArray {
    let weight = mlp.0.weight
    let dtype: DType?
    if weight.dtype == .uint32, let quantized = mlp.0 as? QuantizedLinear {
      // The packed plane is U32; compute in the scales precision. Only affine
      // scales are floats: mxfp4 scales are uint8, and casting the sinusoidal
      // embedding to them makes quantizedMM throw (a fatalError in mlx-swift).
      // Leave the input as is for the other modes.
      dtype = quantized.mode == .affine ? quantized.scales.dtype : nil
    } else {
      switch weight.dtype {
      case .float16, .bfloat16, .float32, .float64:
        dtype = weight.dtype
      default:
        dtype = nil
      }
    }
    guard let dtype, input.dtype != dtype else {
      return input
    }
    return input.asType(dtype)
  }

  func callAsFunction(_ timesteps: MLXArray) -> MLXArray {
    let tFreq = castFrequencyEmbeddingToMLPInputDTypeIfNeeded(
      timestepEmbedding(timesteps, dim: frequencyEmbeddingSize)
    )
    var out = mlp.0(tFreq)
    out = mlp.1(out)
    out = mlp.2(out)
    return out
  }
}
