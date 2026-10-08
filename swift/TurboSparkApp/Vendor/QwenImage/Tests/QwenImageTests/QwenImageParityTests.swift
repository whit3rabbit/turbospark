import MLX
import XCTest

@testable import QwenImage

/// Component-level parity checks against the diffusers reference
/// (`qwenimage21` pipeline and `FlowMatchEulerDiscreteScheduler`). A
/// PyTorch reference run was infeasible on this checkout, so the frozen
/// values below were computed independently from the reference formulas
/// (float64 NumPy) and recorded here. Full-image parity remains unverified;
/// see UPSTREAM.md.
final class QwenImageParityTests: XCTestCase {
  // MARK: - Scheduler

  /// Reference values for N = 8 at a 1024-token image sequence (512x512),
  /// computed from the diffusers formulas: base ladder
  /// `linspace(1, 1/1000, 8)`, `mu = 0.538710`, exponential time shift,
  /// stretch to terminal 0.02.
  func testSchedulerMatchesTheReferenceScheduleAtEightSteps() {
    let scheduler = QwenImage21Scheduler(
      numInferenceSteps: 8,
      imageSequenceLength: 1024,
      config: QwenImageSchedulerConfig()
    )

    XCTAssertEqual(scheduler.sigmas.count, 9)
    XCTAssertEqual(scheduler.timesteps.count, 8)
    XCTAssertEqual(scheduler.sigmas[0], 1.0, accuracy: 1e-6)
    XCTAssertEqual(scheduler.sigmas[1], 0.913085, accuracy: 1e-4)
    XCTAssertEqual(scheduler.sigmas[4], 0.571009, accuracy: 1e-4)
    XCTAssertEqual(scheduler.sigmas[7], 0.02, accuracy: 1e-4)
    XCTAssertEqual(scheduler.sigmas[8], 0.0, accuracy: 1e-9)
    XCTAssertEqual(scheduler.timesteps[0], 1000.0, accuracy: 1e-3)
    XCTAssertEqual(scheduler.timesteps[7], 20.0, accuracy: 1e-2)
  }

  /// A one-step ladder is [1.0]; the stretch-to-terminal used to divide
  /// 0/0 and produce a NaN timestep.
  func testOneStepScheduleIsFiniteAndTerminatesAtZero() {
    let scheduler = QwenImage21Scheduler(
      numInferenceSteps: 1,
      imageSequenceLength: 4096,
      config: QwenImageSchedulerConfig()
    )
    XCTAssertEqual(scheduler.sigmas.count, 2)
    XCTAssertEqual(scheduler.sigmas[0], 1.0, accuracy: 1e-6)
    XCTAssertEqual(scheduler.sigmas[1], 0.0, accuracy: 1e-9)
    XCTAssertTrue(scheduler.timesteps[0].isFinite)
  }

  /// Reference values for N = 40 at a 4096-token image sequence
  /// (1024x1024), `mu = 0.693548`.
  func testSchedulerMatchesTheReferenceScheduleAtFortySteps() {
    let scheduler = QwenImage21Scheduler(
      numInferenceSteps: 40,
      imageSequenceLength: 4096,
      config: QwenImageSchedulerConfig()
    )

    XCTAssertEqual(scheduler.sigmas.count, 41)
    XCTAssertEqual(scheduler.sigmas[0], 1.0, accuracy: 1e-6)
    XCTAssertEqual(scheduler.sigmas[1], 0.987265, accuracy: 1e-4)
    XCTAssertEqual(scheduler.sigmas[20], 0.661936, accuracy: 1e-4)
    XCTAssertEqual(scheduler.sigmas[39], 0.02, accuracy: 1e-4)
    XCTAssertEqual(scheduler.sigmas[40], 0.0, accuracy: 1e-9)
  }

  /// A longer sequence shifts more (larger mu pushes mid-schedule sigmas
  /// up); the endpoints stay pinned at 1 and the terminal.
  func testSchedulerSequenceLengthRaisesMidScheduleSigmas() {
    let short = QwenImage21Scheduler(
      numInferenceSteps: 8, imageSequenceLength: 256,
      config: QwenImageSchedulerConfig())
    let long = QwenImage21Scheduler(
      numInferenceSteps: 8, imageSequenceLength: 4096,
      config: QwenImageSchedulerConfig())

    XCTAssertGreaterThan(long.sigmas[4], short.sigmas[4])
    XCTAssertEqual(long.sigmas[0], short.sigmas[0], accuracy: 1e-9)
    XCTAssertEqual(long.sigmas[7], short.sigmas[7], accuracy: 1e-6)
  }

  /// The Euler update: `prev = sample + (sigma_next - sigma) * model_output`.
  func testSchedulerStepIsTheFlowMatchEulerUpdate() {
    let scheduler = QwenImage21Scheduler(
      numInferenceSteps: 4, imageSequenceLength: 256,
      config: QwenImageSchedulerConfig())
    let sample = MLXArray([Float](repeating: 0.5, count: 2), [2])
    let modelOutput = MLXArray([Float(0.25), Float(-0.25)])
    let updated = scheduler.step(modelOutput: modelOutput, stepIndex: 0, sample: sample)
    let expectedDelta = scheduler.sigmas[1] - scheduler.sigmas[0]
    let values = updated.asArray(Float.self)
    XCTAssertEqual(Double(values[0]), 0.5 + 0.25 * Double(expectedDelta), accuracy: 1e-5)
    XCTAssertEqual(Double(values[1]), 0.5 - 0.25 * Double(expectedDelta), accuracy: 1e-5)
  }

  // MARK: - Prompt template

  /// The T2I framing must stay byte-identical to the diffusers constant:
  /// the checkpoint was trained on this exact string, not on the chat
  /// template.
  func testPromptTemplateMatchesTheReferenceConstant() {
    let expected = "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n"
      + "<|im_start|>user\na red kite<|im_end|>\n"
      + "<|im_start|>assistant\n"
    XCTAssertEqual(
      String(format: QwenImageModelMetadata.promptTemplateT2I, "a red kite"),
      expected)
    XCTAssertEqual(
      QwenImageModelMetadata.systemSegmentT2I,
      "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n")
  }

  // MARK: - RoPE

  /// The frame axis freezes image tokens at the text length while the h/w
  /// axes use grids centered on zero, matching `get_rope_indices`.
  func testRopeIndicesCenterTheImageGridsAndFreezeTheFrameAxis() {
    let indices = QwenImage21Rope.t2iTokenIndices(textLength: 4, height: 3, width: 2)

    XCTAssertEqual(indices.frame, [0, 1, 2, 3, 4, 4, 4, 4, 4, 4])
    XCTAssertEqual(indices.height, [0, 1, 2, 3, -2, -2, -1, -1, 0, 0])
    XCTAssertEqual(indices.width, [0, 1, 2, 3, -1, 0, -1, 0, -1, 0])
  }

  /// Reference frequencies are `pos * theta^(-2i/dim)`: slot 0 always has
  /// angle `pos`, and the frame axis (dim 16) slot 1 has
  /// `1 / 10000^(1/8)`. Asserted through the cos table.
  func testRopeFrequenciesMatchTheReferenceFormula() {
    let rope = QwenImage21Rope(theta: 10000, axesDim: [16, 56, 56])
    let indices = QwenImage21Rope.TokenIndices(
      frame: [1], height: [0], width: [0])
    let (cosTable, _) = rope.frequencyTables(for: indices)

    XCTAssertEqual(cosTable.dim(0), 1)
    XCTAssertEqual(cosTable.dim(1), 8 + 28 + 28)
    let row = cosTable[0].asArray(Float.self)
    // Frame axis, slot 0: angle = 1 / theta^0 = 1.
    XCTAssertEqual(Double(row[0]), cos(1.0), accuracy: 1e-4)
    // Frame axis, slot 1: angle = 1 / 10000^(2/16) = 1 / 10^(1/2).
    let angle = pow(10.0, -0.5)
    XCTAssertEqual(Double(row[1]), cos(angle), accuracy: 1e-4)
    // The height/width axes sit at position 0, so every angle is 0.
    XCTAssertEqual(Double(row[8]), 1.0, accuracy: 1e-6)
    XCTAssertEqual(Double(row[8 + 28]), 1.0, accuracy: 1e-6)
  }
}
