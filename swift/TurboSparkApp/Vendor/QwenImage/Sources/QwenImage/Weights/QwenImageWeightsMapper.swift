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
      var schedulerConfig = QwenImageSchedulerConfig.fromConfigJSON(scheduler)
      // Turbo publishes its fixed sigma ladder in model_index.json. The base
      // checkpoints have no such key, or have a model_index.json we do not
      // need, so a missing index is not an error.
      let indexURL = snapshot.appending(path: "model_index.json")
      if FileManager.default.fileExists(atPath: indexURL.path) {
        schedulerConfig.sampleSigmas = try QwenImageSchedulerConfig.sampleSigmas(
          fromModelIndex: try loadJSON(indexURL))
      }
      return ComponentConfigs(
        transformer: QwenImage21TransformerConfiguration.fromConfigJSON(transformer),
        textEncoder: Qwen3TextEncoderConfiguration.fromQwen3VLConfig(textEncoder),
        vae: QwenImage21VAEConfiguration.fromConfigJSON(vae),
        scheduler: schedulerConfig
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

  /// Maps a transformer tensor name to the module tree's spelling.
  ///
  /// The module tree follows the layout of the earlier MLX conversion: the
  /// shared modulation projection is `modulation.0` (the SiLU in front of it
  /// is applied in code, not stored as a layer) and the timestep MLP sits
  /// directly under `time_text_embed`. The Diffusers-named Turbo checkpoint
  /// spells them `modulation.1` and `time_text_embed.timestep_embedder`.
  /// Names already in the module tree's spelling pass through.
  static func canonicalTransformerName(_ name: String) -> String {
    if name.hasPrefix("modulation.1.") {
      return "modulation.0." + name.dropFirst("modulation.1.".count)
    }
    let embedder = "time_text_embed.timestep_embedder."
    if name.hasPrefix(embedder) {
      return "time_text_embed." + name.dropFirst(embedder.count)
    }
    return name
  }

  /// Maps a text-encoder tensor name to the module tree's spelling, or nil
  /// when the pipeline never uses the tensor.
  ///
  /// The module tree follows the mlx-vlm layout (`language_model.model.*`).
  /// Checkpoints exported from transformers 5 spell the same tensors
  /// `model.language_model.*` and put the vision tower under
  /// `model.visual.*`; the Turbo checkpoint is one of them. Names already in
  /// the mlx-vlm layout pass through unchanged, so the older checkpoint
  /// loads as before. Dropping the vision tower and LM head here keeps them
  /// out of memory (the vision tower is about 1 GiB at Q8).
  static func canonicalTextEncoderName(_ name: String) -> String? {
    if name.hasPrefix("model.visual.") || name.hasPrefix("vision_tower.") { return nil }
    if name.hasPrefix("language_model.lm_head") || name.hasPrefix("lm_head") { return nil }
    let transformersPrefix = "model.language_model."
    if name.hasPrefix(transformersPrefix) {
      return "language_model.model." + name.dropFirst(transformersPrefix.count)
    }
    return name
  }

  /// Loads every tensor of one component. `skipPrefixes` drops tensor
  /// groups the text-to-image pipeline never uses (the vision tower and
  /// language-model head).
  static func loadComponent(
    _ directory: URL,
    skipPrefixes: [String] = [],
    rename: (String) -> String? = { $0 }
  ) throws -> [String: MLXArray] {
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
        guard let canonical = rename(name) else { continue }
        weights[canonical] = tensor
      }
    }
    return weights
  }

  // MARK: - Application

  /// Group size every published Qwen-Image-2.1 MLX checkpoint uses. The
  /// bit width is not fixed: the Turbo checkpoint mixes Q4 transformer
  /// blocks with a Q8 text encoder, and a component config carries only one
  /// `quantization` block.
  static let quantizationGroupSize = 64
  private static let supportedQuantizationBits: Set<Int> = [2, 3, 4, 5, 6, 8]

  /// Bit width of one MLX affine layer, derived from its tensor shapes.
  ///
  /// A packed weight is `[out, in * bits / 32]` (U32) and its scales are
  /// `[out, in / groupSize]`, so `bits = 32 * weightLast / (groupSize *
  /// scalesLast)`. Returns nil when the shapes are not an exact MLX affine
  /// layout, so a corrupt or mismatched pair fails instead of loading as
  /// the wrong width.
  static func inferQuantizationBits(
    packedWeightLastDim: Int,
    scalesLastDim: Int,
    groupSize: Int = quantizationGroupSize
  ) -> Int? {
    guard packedWeightLastDim > 0, scalesLastDim > 0 else { return nil }
    let numerator = 32 * packedWeightLastDim
    let denominator = groupSize * scalesLastDim
    guard numerator % denominator == 0 else { return nil }
    let bits = numerator / denominator
    return supportedQuantizationBits.contains(bits) ? bits : nil
  }

  /// Quantizes the module's linears and embeddings wherever the checkpoint
  /// carries a matching `.scales` companion, at the width its shapes imply,
  /// then applies the weights.
  static func apply(
    _ weights: [String: MLXArray],
    to model: Module,
    logger: Logger
  ) throws {
    var shapeFailures: [String] = []
    MLXNN.quantize(model: model) { path, _ in
      // Sequential children appear as `x.layers.0` while the checkpoint
      // stores `x.0.scales`.
      let bases = [path, path.replacingOccurrences(of: ".layers.", with: ".")]
      guard let base = bases.first(where: { weights["\($0).scales"] != nil }),
        let scales = weights["\(base).scales"]
      else { return nil }
      guard let packed = weights["\(base).weight"], packed.ndim >= 1, scales.ndim >= 1,
        let bits = inferQuantizationBits(
          packedWeightLastDim: packed.dim(-1), scalesLastDim: scales.dim(-1))
      else {
        shapeFailures.append(base)
        return nil
      }
      return (quantizationGroupSize, bits, QuantizationMode.affine)
    }
    if !shapeFailures.isEmpty {
      throw QwenImagePipelineError.weightsMissing(
        "quantized layers with an unreadable affine layout (\(shapeFailures.count)): "
          + shapeFailures.prefix(4).joined(separator: ", "))
    }
    try applyParameters(weights, to: model, logger: logger)
  }

  /// Applies dense weights (VAE) with a layout fallback for PyTorch-order
  /// conv kernels.
  static func applyDense(
    _ weights: [String: MLXArray],
    to model: Module,
    logger: Logger
  ) throws {
    try applyParameters(weights, to: model, transposeConvIfNeeded: true, logger: logger)
  }

  private static func squeezedShape(_ shape: [Int]) -> [Int] {
    shape.filter { $0 != 1 }
  }

  private static func applyParameters(
    _ weights: [String: MLXArray],
    to model: Module,
    transposeConvIfNeeded: Bool = false,
    logger: Logger
  ) throws {
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
        } else if value.shape != parameter.shape, transposeConvIfNeeded,
          squeezedShape(value.shape) == squeezedShape(parameter.shape)
        {
          // Diffusers stores RMS-norm `gamma` as `[C, 1, 1, 1]`; the module
          // holds `[C]`. Only singleton axes may differ, so this cannot
          // reinterpret a real layout difference.
          value = value.reshaped(parameter.shape)
        }
        updates.append((key, value))
      } else {
        missing.append(key)
      }
    }

    if updates.isEmpty {
      logger.error("no checkpoint tensors matched the module tree")
      // Returning here left the layers randomly initialized and the model
      // marked loaded, so every generation produced a noise image.
      throw QwenImagePipelineError.weightsMissing("no checkpoint tensors matched the module tree")
    }
    // A parameter the checkpoint does not supply keeps its random
    // initialization, and the pipeline then runs and returns noise with no
    // error. A faithful port has no parameter without a tensor, so a gap is
    // a naming mismatch, not something to log and continue past.
    if !missing.isEmpty {
      logger.error("module parameters without checkpoint tensors (\(missing.count)): \(missing.prefix(8).joined(separator: ", "))")
      throw QwenImagePipelineError.weightsMissing(
        "\(missing.count) module parameters have no checkpoint tensor: "
          + missing.prefix(6).joined(separator: ", "))
    }

    do {
      let nested = ModuleParameters.unflattened(updates)
      try model.update(parameters: nested, verify: [.shapeMismatch])
    } catch {
      logger.error("failed to apply weights: \(error)")
      throw QwenImagePipelineError.weightsMissing("failed to apply weights: \(error)")
    }
  }
}
