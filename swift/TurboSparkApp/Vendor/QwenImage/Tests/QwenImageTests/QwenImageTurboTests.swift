import Logging
import MLX
import MLXNN
import XCTest

@testable import QwenImage

/// Checks for the mixed-precision Turbo checkpoint: per-layer affine bit
/// inference and the published 8-step sigma ladder. No weights needed.
final class QwenImageTurboTests: XCTestCase {
  // The `sample_sigmas` value published in the Turbo model_index.json.
  private static let publishedLadder: [Float] = [
    1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568,
  ]

  // MARK: - Scheduler

  func testStaticLadderReplacesTheComputedSchedule() {
    let config = QwenImageSchedulerConfig(
      useDynamicShifting: false, shiftTerminal: nil, sampleSigmas: Self.publishedLadder)
    let scheduler = QwenImage21Scheduler(
      numInferenceSteps: 8, imageSequenceLength: 4096, config: config)

    XCTAssertEqual(scheduler.sigmas.count, 9)
    for (index, expected) in Self.publishedLadder.enumerated() {
      XCTAssertEqual(scheduler.sigmas[index], expected, accuracy: 1e-7)
      XCTAssertEqual(scheduler.timesteps[index], expected * 1000, accuracy: 1e-3)
    }
    XCTAssertEqual(scheduler.sigmas[8], 0.0, accuracy: 1e-9)
  }

  /// The ladder is ignored while dynamic shifting is on: the base
  /// checkpoint must keep its computed schedule even if a stray key exists.
  func testLadderIsIgnoredWhenDynamicShiftingIsOn() {
    var config = QwenImageSchedulerConfig()
    config.sampleSigmas = Self.publishedLadder
    XCTAssertNil(config.staticSigmaLadder)
    let scheduler = QwenImage21Scheduler(
      numInferenceSteps: 8, imageSequenceLength: 1024, config: config)
    XCTAssertEqual(scheduler.sigmas[1], 0.913085, accuracy: 1e-4)
  }

  func testModelIndexLadderParsing() throws {
    let parsed = try QwenImageSchedulerConfig.sampleSigmas(
      fromModelIndex: ["sample_sigmas": [1.0, 0.5, 0.25]])
    XCTAssertEqual(parsed, [1.0, 0.5, 0.25])
    XCTAssertNil(try QwenImageSchedulerConfig.sampleSigmas(fromModelIndex: [:]))
  }

  func testModelIndexLadderRejectsMalformedValues() {
    let bad: [[String: Any]] = [
      ["sample_sigmas": "1.0"],
      ["sample_sigmas": []],
      ["sample_sigmas": [0.5, 1.0]],  // increasing
      ["sample_sigmas": [1.0, 1.0]],  // not strict
      ["sample_sigmas": [1.0, 0.0]],  // zero
      ["sample_sigmas": [1.5, 0.5]],  // above 1
      ["sample_sigmas": [1.0, "x"]],
    ]
    for index in bad {
      XCTAssertThrowsError(
        try QwenImageSchedulerConfig.sampleSigmas(fromModelIndex: index), "\(index)")
    }
  }

  // MARK: - Quantization bit inference

  /// Shapes copied from the Turbo safetensors headers.
  func testBitInferenceMatchesTheTurboTensorShapes() {
    func bits(_ weight: Int, _ scales: Int) -> Int? {
      QwenImageWeightsMapper.inferQuantizationBits(
        packedWeightLastDim: weight, scalesLastDim: scales)
    }
    XCTAssertEqual(bits(512, 64), 4)  // transformer to_q [4096,512] / scales [4096,64]
    XCTAssertEqual(bits(1536, 192), 4)  // transformer img_mlp.out
    XCTAssertEqual(bits(1024, 64), 8)  // encoder q_proj [4096,1024]
    XCTAssertEqual(bits(3072, 192), 8)  // encoder down_proj [4096,3072]
    XCTAssertEqual(bits(288, 18), 8)  // vision qkv [3456,288] / scales [3456,18]
    XCTAssertEqual(bits(384, 64), 3)
  }

  func testBitInferenceRejectsInexactLayouts() {
    func bits(_ weight: Int, _ scales: Int) -> Int? {
      QwenImageWeightsMapper.inferQuantizationBits(
        packedWeightLastDim: weight, scalesLastDim: scales)
    }
    XCTAssertNil(bits(500, 64))  // not a whole number of bits
    XCTAssertNil(bits(0, 64))
    XCTAssertNil(bits(512, 0))
    XCTAssertNil(bits(64, 64))  // 1 bit is not an MLX affine width
    XCTAssertNil(bits(2048, 64))  // 16 bits
  }

  // MARK: - Transformer key canonicalization

  /// Names copied from the Turbo transformer safetensors index.
  func testTransformerNamesMapOntoTheModuleTree() {
    let map = QwenImageWeightsMapper.canonicalTransformerName
    XCTAssertEqual(map("modulation.1.weight"), "modulation.0.weight")
    XCTAssertEqual(
      map("time_text_embed.timestep_embedder.linear_1.weight"),
      "time_text_embed.linear_1.weight")
    XCTAssertEqual(
      map("time_text_embed.timestep_embedder.linear_2.weight"),
      "time_text_embed.linear_2.weight")
    // Names in the module tree's own spelling, and untouched tensors, pass
    // through, so the earlier checkpoint loads as before.
    XCTAssertEqual(map("modulation.0.weight"), "modulation.0.weight")
    XCTAssertEqual(map("time_text_embed.linear_1.weight"), "time_text_embed.linear_1.weight")
    XCTAssertEqual(
      map("transformer_blocks.3.attn.to_q.scales"), "transformer_blocks.3.attn.to_q.scales")
    XCTAssertEqual(map("txt_in.in_layer.weight"), "txt_in.in_layer.weight")
  }

  // MARK: - Text encoder key canonicalization

  /// Names copied from the Turbo text_encoder safetensors header.
  func testTextEncoderNamesMapOntoTheModuleTree() {
    let map = QwenImageWeightsMapper.canonicalTextEncoderName
    XCTAssertEqual(
      map("model.language_model.layers.0.self_attn.q_proj.scales"),
      "language_model.model.layers.0.self_attn.q_proj.scales")
    XCTAssertEqual(
      map("model.language_model.embed_tokens.weight"),
      "language_model.model.embed_tokens.weight")
    XCTAssertEqual(
      map("model.language_model.norm.weight"), "language_model.model.norm.weight")
    // The mlx-vlm spelling the older checkpoint uses passes through.
    XCTAssertEqual(
      map("language_model.model.layers.3.mlp.up_proj.weight"),
      "language_model.model.layers.3.mlp.up_proj.weight")
  }

  func testTextEncoderDropsTheVisionTowerAndHead() {
    let map = QwenImageWeightsMapper.canonicalTextEncoderName
    XCTAssertNil(map("model.visual.blocks.0.attn.qkv.weight"))
    XCTAssertNil(map("model.visual.patch_embed.proj.weight"))
    XCTAssertNil(map("vision_tower.blocks.0.attn.qkv.weight"))
    XCTAssertNil(map("language_model.lm_head.weight"))
    XCTAssertNil(map("lm_head.weight"))
  }

  // MARK: - Mixed-width apply

  private final class TinyMixed: Module {
    @ModuleInfo(key: "wide") var wide: Linear
    @ModuleInfo(key: "narrow") var narrow: Linear
    @ModuleInfo(key: "dense") var dense: Linear

    override init() {
      self._wide = ModuleInfo(key: "wide")
      self._wide.wrappedValue = Linear(128, 64, bias: false)
      self._narrow = ModuleInfo(key: "narrow")
      self._narrow.wrappedValue = Linear(128, 64, bias: false)
      self._dense = ModuleInfo(key: "dense")
      self._dense.wrappedValue = Linear(128, 64, bias: false)
      super.init()
    }
  }

  private func packed(_ prefix: String, bits: Int) -> [String: MLXArray] {
    let dense = MLXRandom.normal([64, 128])
    let (wq, scales, biases) = MLX.quantized(dense, groupSize: 64, bits: bits)
    return ["\(prefix).weight": wq, "\(prefix).scales": scales, "\(prefix).biases": biases!]
  }

  /// One module with an 8-bit layer, a 4-bit layer, and a dense layer, the
  /// shape of the Turbo mix. Forcing every layer to 4 bits (the old loader)
  /// makes the 8-bit plane fail the shape-verified update.
  func testApplyQuantizesEachLayerAtItsOwnWidth() throws {
    var weights = packed("wide", bits: 8)
    weights.merge(packed("narrow", bits: 4)) { $1 }
    weights["dense.weight"] = MLXRandom.normal([64, 128])

    let model = TinyMixed()
    try QwenImageWeightsMapper.apply(
      weights, to: model, logger: Logger(label: "qwen-image.test"))

    let wide = try XCTUnwrap(model.wide as? QuantizedLinear)
    let narrow = try XCTUnwrap(model.narrow as? QuantizedLinear)
    XCTAssertEqual(wide.bits, 8)
    XCTAssertEqual(narrow.bits, 4)
    XCTAssertEqual(wide.groupSize, 64)
    XCTAssertFalse(model.dense is QuantizedLinear)

    // The quantized layers must run, not just construct.
    let out = model.wide(MLXArray.ones([1, 128]))
    MLX.eval(out)
    XCTAssertEqual(out.shape, [1, 64])
  }

  /// A parameter with no checkpoint tensor used to log a warning and keep its
  /// random initialization, so generation returned noise with no error.
  func testApplyRejectsAModuleParameterTheCheckpointDoesNotSupply() {
    var weights = packed("wide", bits: 8)
    weights.merge(packed("narrow", bits: 4)) { $1 }
    // `dense.weight` deliberately absent.
    let model = TinyMixed()
    XCTAssertThrowsError(
      try QwenImageWeightsMapper.apply(
        weights, to: model, logger: Logger(label: "qwen-image.test"))
    ) { error in
      XCTAssertTrue("\(error)".contains("dense.weight"), "\(error)")
    }
  }

  func testApplyRejectsAScalesTensorWithoutAReadableWeight() {
    var weights = packed("wide", bits: 8)
    weights["wide.weight"] = nil
    weights["narrow.weight"] = MLXRandom.normal([64, 128])
    let model = TinyMixed()
    XCTAssertThrowsError(
      try QwenImageWeightsMapper.apply(
        weights, to: model, logger: Logger(label: "qwen-image.test")))
  }
}
