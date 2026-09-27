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

  /// Rewrites the mflux v0.17.5 tensor spellings onto the canonical names the
  /// Diffusers conversions use. mflux names the timestep MLP linears
  /// `linear1`/`linear2` and indexes the final-layer adaLN Sequential at 0;
  /// the canonical checkpoint names are `mlp.0`/`mlp.2` and index 1. Per-block
  /// `adaLN_modulation.0` keys are canonical already and are left alone.
  static func canonicalizeTransformerKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var result = weights
    for (key, value) in weights {
      var target: String?
      if key.hasPrefix("t_embedder.linear1.") {
        target = "t_embedder.mlp.0" + key.dropFirst("t_embedder.linear1".count)
      } else if key.hasPrefix("t_embedder.linear2.") {
        target = "t_embedder.mlp.2" + key.dropFirst("t_embedder.linear2".count)
      } else if key.hasPrefix("all_final_layer."),
        key.contains(".adaLN_modulation.0.")
      {
        target = key.replacingOccurrences(of: ".adaLN_modulation.0.", with: ".adaLN_modulation.1.")
      }
      if let target, result[target] == nil {
        result[target] = value
      }
    }
    return result
  }

  /// Adds the `model.` prefix mflux v0.17.5 drops from text-encoder tensor
  /// names (`embed_tokens`, `layers.*`, `norm.weight`).
  static func canonicalizeTextEncoderKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let hasPrefixedKey = weights.keys.contains { $0.hasPrefix("model.") }
    if hasPrefixedKey {
      return weights
    }
    var result: [String: MLXArray] = [:]
    for (key, value) in weights {
      result["model.\(key)"] = value
    }
    return result
  }

  /// Rewrites the mflux VAE boundary spellings onto the canonical parameter
  /// names (`conv_in.conv.weight` -> `conv_in.weight`, `conv_norm_out.norm.*`
  /// -> `conv_norm_out.*`). The boundary modules expose their weight directly,
  /// so the extra inner segment would silently miss every parameter and leave
  /// the decoder boundary randomly initialized.
  static func canonicalizeVAEKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    let heads = [
      "decoder.conv_in.", "encoder.conv_in.",
      "decoder.conv_out.", "encoder.conv_out.",
      "decoder.conv_norm_out.", "encoder.conv_norm_out.",
    ]
    let innerSegments = ["conv.", "conv2d.", "norm."]
    var result: [String: MLXArray] = [:]
    for (key, value) in weights {
      var canonical = key
      for head in heads where canonical.hasPrefix(head) {
        let rest = canonical.dropFirst(head.count)
        for inner in innerSegments where rest.hasPrefix(inner) {
          canonical = head + rest.dropFirst(inner.count)
        }
      }
      if result[canonical] == nil {
        result[canonical] = value
      }
    }
    return result
  }

  /// Dequantizes every packed affine linear in the dictionary back to dense
  /// BF16. Used for components whose modules stay dense, such as the VAE
  /// mid-block attention linears that mflux quantizes.
  static func dequantizeAffineTensors(
    _ weights: [String: MLXArray],
    manifest: ZImageQuantizationManifest
  ) -> [String: MLXArray] {
    let mode: QuantizationMode = manifest.mode == "mxfp4" ? .mxfp4 : .affine
    var result = weights
    for (key, value) in weights {
      guard key.hasSuffix(".weight"), value.dtype == .uint32 else { continue }
      let base = String(key.dropLast(".weight".count))
      guard let scales = weights["\(base).scales"] else { continue }
      result[key] = dequantized(
        value,
        scales: scales,
        biases: weights["\(base).biases"],
        groupSize: manifest.groupSize,
        bits: manifest.bits,
        mode: mode,
        dtype: .bfloat16
      )
    }
    return result
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

  private static func vaeMapping(
    _ weights: [String: MLXArray],
    targetShapes: [String: [Int]]? = nil
  ) -> [String: MLXArray] {
    var mapped: [String: MLXArray] = [:]
    for (k, v) in weights {
      var tensor = v
      // Diffusers conv weights arrive as PyTorch OIHW and are transposed to
      // the MLX NHWC layout. mflux conversions already store NHWC; transposing
      // those again corrupts them, so only transpose when the target module
      // disagrees with the stored shape.
      if tensor.ndim == 4, let targetShapes,
        let target = targetShapes[k], target != tensor.shape,
        tensor.transposed(0, 2, 3, 1).shape == target
      {
        tensor = tensor.transposed(0, 2, 3, 1)
      } else if tensor.ndim == 4, targetShapes == nil {
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

    let canonical = canonicalizeTransformerKeys(weights)

    if let manifest {
      let availableKeys = Set(canonical.keys)
      ZImageQuantizer.applyQuantization(
        to: model,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.transformerTensorName
      )
    }

    let runtimeWeights = castFloat16ToBFloat16(canonical)
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
      (weight: "cap_embedder.1.weight", scales: "cap_embedder.1.scales", biases: "cap_embedder.1.biases"),
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

    let canonical = canonicalizeTextEncoderKeys(weights)

    if let manifest {
      let availableKeys = Set(canonical.keys)
      ZImageQuantizer.applyQuantization(
        to: model,
        manifest: manifest,
        availableKeys: availableKeys,
        tensorNameTransform: ZImageQuantizer.textEncoderTensorName
      )
    }

    let mapped = textEncoderMapping(castFloat16ToBFloat16(canonical))
    ZImageModuleWeightsApplier.applyToModule(model, weights: mapped, prefix: "text_encoder", logger: logger)
  }

  public static func applyVAE(
    weights: [String: MLXArray],
    to model: Module,
    manifest: ZImageQuantizationManifest? = nil,
    logger: Logger
  ) {
    if weights.isEmpty {
      logger.warning("VAE weights empty; nothing to apply.")
      return
    }

    // VAE modules stay dense, but mflux conversions quantize the mid-block
    // attention linears; restore them before the dense apply.
    let canonical = canonicalizeVAEKeys(weights)
    let runtimeWeights: [String: MLXArray]
    if let manifest {
      runtimeWeights = dequantizeAffineTensors(castFloat16ToBFloat16(canonical), manifest: manifest)
    } else {
      runtimeWeights = castFloat16ToBFloat16(canonical)
    }

    let targetShapes =
      Dictionary(
        uniqueKeysWithValues: model.parameters().flattened().lazy.map { ($0.0, $0.1.shape) }
      ) as [String: [Int]]
    let mapped = vaeMapping(runtimeWeights, targetShapes: targetShapes)
    ZImageModuleWeightsApplier.applyToModule(model, weights: mapped, prefix: "vae", logger: logger)
  }
}
