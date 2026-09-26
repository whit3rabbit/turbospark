import Foundation
import Logging
import MLX
import MLXNN

public enum ZImageWeightsMapping {
  public struct Partition {
    public let transformer: [String: MLXArray]
    public let textEncoder: [String: MLXArray]
    public let vae: [String: MLXArray]
    public let unassigned: [String: MLXArray]
  }

  public static func partition(weights: [String: MLXArray], logger _: Logger? = nil) -> Partition {
    var transformer: [String: MLXArray] = [:]
    var textEncoder: [String: MLXArray] = [:]
    var vae: [String: MLXArray] = [:]
    var unassigned: [String: MLXArray] = [:]

    for (key, tensor) in weights {
      if key.hasPrefix("transformer.") {
        transformer[String(key.dropFirst("transformer.".count))] = tensor
      } else if key.hasPrefix("text_encoder.") {
        textEncoder[String(key.dropFirst("text_encoder.".count))] = tensor
      } else if key.hasPrefix("vae.") {
        vae[String(key.dropFirst("vae.".count))] = tensor
      } else {
        unassigned[key] = tensor
      }
    }

    return Partition(
      transformer: transformer,
      textEncoder: textEncoder,
      vae: vae,
      unassigned: unassigned
    )
  }

  private static func transformerMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      mapped["transformer.\(k)"] = v
    }
    return mapped
  }

  private static func textEncoderMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      if k.hasPrefix("model.") {
        let remainder = String(k.dropFirst("model.".count))
        mapped["text_encoder.encoder.\(remainder)"] = v
      } else {
        mapped["text_encoder.\(k)"] = v
      }
    }
    return mapped
  }

  private static func vaeMapping(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      var tensor = v
      if tensor.ndim == 4 {
        tensor = tensor.transposed(0, 2, 3, 1)
      }
      mapped["vae.\(k)"] = tensor
    }
    return mapped
  }

  public static func applyTransformer(
    weights: [String: MLXArray],
    to model: ZImageTransformer2DModel,
    manifest: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) {
    if weights.isEmpty {
      logger.warning("Transformer weights empty; nothing to apply.")
      return
    }

    if let manifest {
      let availableKeys = Set(weights.keys)
      ZImageQuantizer.applyQuantization(
        to: model,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.transformerTensorName
      )
    }

    let runtimeWeights = castFloat16ToBFloat16(weights)
    let mapped = transformerMapping(runtimeWeights)
    ZImageModuleWeightsApplier.applyToModule(model, weights: mapped, prefix: "transformer", logger: logger)

    let auxiliaryWeights = denseAuxiliaryWeights(weights: runtimeWeights, manifest: manifest)
    let groupSize = manifest?.groupSize ?? 32
    let bits = manifest?.bits ?? 8
    model.loadCapEmbedderWeights(from: auxiliaryWeights)
    model.loadXEmbedderWeights(from: auxiliaryWeights, groupSize: groupSize, bits: bits)
    model.loadFinalLayerWeights(from: auxiliaryWeights, groupSize: groupSize, bits: bits)

    model.setPadTokens(
      xPad: auxiliaryWeights["x_pad_token"],
      capPad: auxiliaryWeights["cap_pad_token"]
    )
  }

  private static func denseAuxiliaryWeights(
    weights: [String: MLXArray],
    manifest: ZImageQuantizationManifest?
  ) -> [String: MLXArray] {
    guard let manifest else { return weights }
    let mode: QuantizationMode = manifest.mode == "mxfp4" ? .mxfp4 : .affine
    var result = weights
    let auxiliaryWeights = [
      (weight: "x_pad_token", scales: "x_pad_token.scales", biases: "x_pad_token.biases"),
      (weight: "cap_pad_token", scales: "cap_pad_token.scales", biases: "cap_pad_token.biases"),
      (weight: "all_x_embedder.2-1.weight", scales: "all_x_embedder.2-1.scales", biases: "all_x_embedder.2-1.biases"),
      (weight: "all_final_layer.2-1.linear.weight", scales: "all_final_layer.2-1.linear.scales", biases: "all_final_layer.2-1.linear.biases"),
      (weight: "all_final_layer.2-1.adaLN_modulation.1.weight", scales: "all_final_layer.2-1.adaLN_modulation.1.scales", biases: "all_final_layer.2-1.adaLN_modulation.1.biases"),
    ]
    for item in auxiliaryWeights {
      guard let packed = weights[item.weight], let scales = weights[item.scales] else { continue }
      result[item.weight] = dequantized(
        packed,
        scales: scales,
        biases: weights[item.biases],
        groupSize: manifest.groupSize,
        bits: manifest.bits,
        mode: mode,
        dtype: .bfloat16
      )
    }
    return result
  }

  private static func castFloat16ToBFloat16(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    weights.mapValues { value in
      value.dtype == .float16 ? value.asType(.bfloat16) : value
    }
  }

  public static func applyTextEncoder(
    weights: [String: MLXArray],
    to model: QwenTextEncoder,
    manifest: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) {
    if weights.isEmpty {
      logger.warning("Text encoder weights empty; nothing to apply.")
      return
    }

    if let manifest {
      let availableKeys = Set(weights.keys)
      ZImageQuantizer.applyQuantization(
        to: model,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.textEncoderTensorName
      )
    }

    let mapped = textEncoderMapping(castFloat16ToBFloat16(weights))
    ZImageModuleWeightsApplier.applyToModule(model, weights: mapped, prefix: "text_encoder", logger: logger)
  }

  public static func applyVAE(
    weights: [String: MLXArray],
    to model: Module,
    manifest _: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) {
    if weights.isEmpty {
      logger.warning("VAE weights empty; nothing to apply.")
      return
    }

    let mapped = vaeMapping(weights)
    ZImageModuleWeightsApplier.applyToModule(model, weights: mapped, prefix: "vae", logger: logger)
  }
}
