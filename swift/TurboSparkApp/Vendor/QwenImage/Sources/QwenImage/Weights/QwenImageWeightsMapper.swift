import Foundation
import Logging
import MLX
import MLXNN

/// Loads the MLX-format Qwen-Image-2.1 components from a snapshot
/// directory, quantizes the module trees to match the packed affine
/// tensors, and applies the weights.
///
/// The module trees use the checkpoint's own parameter nesting, so tensor
/// names apply without renaming. Only the name identity between the MLX
/// conversion and the module trees is relied on; mismatches fail the
/// shape-verified update instead of passing silently.
enum QwenImageWeightsMapper {
  struct ComponentConfigs {
    let transformer: QwenImage21TransformerConfiguration
    let textEncoder: Qwen3TextEncoderConfiguration
    let vae: QwenImage21VAEConfiguration
    let scheduler: QwenImageSchedulerConfig

    static func load(from snapshot: URL) throws -> ComponentConfigs {
      let transformer = try loadJSON(snapshot.appending(path: "transformer/config.json"))
      let textEncoder = try loadJSON(snapshot.appending(path: "text_encoder/config.json"))
      let vae = try loadJSON(snapshot.appending(path: "vae/config.json"))
      let scheduler = try loadJSON(snapshot.appending(path: "scheduler/scheduler_config.json"))
      return ComponentConfigs(
        transformer: QwenImage21TransformerConfiguration.fromConfigJSON(transformer),
        textEncoder: Qwen3TextEncoderConfiguration.fromQwen3VLConfig(textEncoder),
        vae: QwenImage21VAEConfiguration.fromConfigJSON(vae),
        scheduler: QwenImageSchedulerConfig.fromConfigJSON(scheduler)
      )
    }
  }

  static func loadJSON(_ url: URL) throws -> [String: Any] {
    let data = try Data(contentsOf: url)
    let object = try JSONSerialization.jsonObject(with: data, options: [])
    guard let dictionary = object as? [String: Any] else {
      throw QwenImagePipelineError.weightsMissing("\(url.lastPathComponent) is not a JSON object")
    }
    return dictionary
  }

  // MARK: - Weight loading

  /// Loads every tensor of one component. `skipPrefixes` drops tensor
  /// groups the text-to-image pipeline never uses (the vision tower and
  /// language-model head).
  static func loadComponent(_ directory: URL, skipPrefixes: [String] = []) throws -> [String: MLXArray] {
    let safetensors = try FileManager.default
      .contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
      .filter { $0.pathExtension == "safetensors" }
      .sorted { $0.lastPathComponent < $1.lastPathComponent }
    guard !safetensors.isEmpty else {
      throw QwenImagePipelineError.missingSnapshotComponent(
        "safetensors weights under \(directory.lastPathComponent)/")
    }
    var weights: [String: MLXArray] = [:]
    for file in safetensors {
      let tensors = try MLX.loadArrays(url: file)
      for (name, tensor) in tensors {
        if skipPrefixes.contains(where: { name.hasPrefix($0) }) { continue }
        weights[name] = tensor
      }
    }
    return weights
  }

  // MARK: - Application

  /// Quantizes the module's linears and embeddings wherever the checkpoint
  /// carries a matching `.scales` companion, then applies the weights.
  static func apply(
    _ weights: [String: MLXArray],
    to model: Module,
    logger: Logger
  ) {
    MLXNN.quantize(model: model) { path, _ in
      // Sequential children appear as `x.layers.0` while the checkpoint
      // stores `x.0.scales`.
      let candidates = [
        "\(path).scales",
        path.replacingOccurrences(of: ".layers.", with: ".") + ".scales",
      ]
      guard candidates.contains(where: { weights[$0] != nil }) else { return nil }
      return (64, 4, QuantizationMode.affine)
    }
    applyParameters(weights, to: model, logger: logger)
  }

  /// Applies dense weights (VAE) with a layout fallback for PyTorch-order
  /// conv kernels.
  static func applyDense(
    _ weights: [String: MLXArray],
    to model: Module,
    logger: Logger
  ) {
    applyParameters(weights, to: model, transposeConvIfNeeded: true, logger: logger)
  }

  private static func applyParameters(
    _ weights: [String: MLXArray],
    to model: Module,
    transposeConvIfNeeded: Bool = false,
    logger: Logger
  ) {
    let parameters = model.parameters().flattened()
    var updates: [(String, MLXArray)] = []
    updates.reserveCapacity(parameters.count)
    var missing: [String] = []

    for (key, parameter) in parameters {
      // `Sequential` nests its children under `layers.`, which the
      // checkpoint spellings do not carry (`modulation.0.weight`).
      let sequentialKey = key.replacingOccurrences(of: ".layers.", with: ".")
      let candidates = sequentialKey == key ? [key] : [key, sequentialKey]
      if let tensor = candidates.compactMap({ weights[$0] }).first {
        var value = tensor
        if value.shape != parameter.shape, transposeConvIfNeeded, value.ndim == 4,
          value.transposed(0, 2, 3, 1).shape == parameter.shape
        {
          value = value.transposed(0, 2, 3, 1)
        }
        updates.append((key, value))
      } else {
        missing.append(key)
      }
    }

    if updates.isEmpty {
      logger.error("no checkpoint tensors matched the module tree")
      return
    }
    if !missing.isEmpty {
      logger.warning("module parameters without checkpoint tensors (\(missing.count)): \(missing.prefix(8).joined(separator: ", "))")
    }

    do {
      let nested = ModuleParameters.unflattened(updates)
      try model.update(parameters: nested, verify: [.shapeMismatch])
    } catch {
      logger.error("failed to apply weights: \(error)")
    }
  }
}
