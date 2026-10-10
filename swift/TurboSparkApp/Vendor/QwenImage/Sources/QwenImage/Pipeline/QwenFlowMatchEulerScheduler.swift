import Foundation
import MLX

/// Scheduler configuration read from the snapshot's
/// `scheduler/scheduler_config.json`.
public struct QwenImageSchedulerConfig: Sendable {
  public var numTrainTimesteps: Int
  public var baseShift: Float
  public var maxShift: Float
  public var baseImageSeqLen: Int
  public var maxImageSeqLen: Int
  public var useDynamicShifting: Bool
  public var timeShiftType: String
  public var shiftTerminal: Float?
  public var shift: Float
  /// Fixed sigma ladder (without the trailing zero) the Turbo checkpoint
  /// publishes as `sample_sigmas` in `model_index.json`. It applies only
  /// when `useDynamicShifting` is false; the ladder then replaces the
  /// computed schedule outright and its length fixes the step count.
  public var sampleSigmas: [Float]?

  public init(
    numTrainTimesteps: Int = 1000,
    baseShift: Float = 0.5,
    maxShift: Float = 0.9,
    baseImageSeqLen: Int = 256,
    maxImageSeqLen: Int = 8192,
    useDynamicShifting: Bool = true,
    timeShiftType: String = "exponential",
    shiftTerminal: Float? = 0.02,
    shift: Float = 1.0,
    sampleSigmas: [Float]? = nil
  ) {
    self.numTrainTimesteps = numTrainTimesteps
    self.baseShift = baseShift
    self.maxShift = maxShift
    self.baseImageSeqLen = baseImageSeqLen
    self.maxImageSeqLen = maxImageSeqLen
    self.useDynamicShifting = useDynamicShifting
    self.timeShiftType = timeShiftType
    self.shiftTerminal = shiftTerminal
    self.shift = shift
    self.sampleSigmas = sampleSigmas
  }

  /// The static ladder in force for this config, or nil when the schedule
  /// is computed (dynamic shifting on, or no published ladder).
  public var staticSigmaLadder: [Float]? {
    guard !useDynamicShifting, let sampleSigmas, !sampleSigmas.isEmpty else { return nil }
    return sampleSigmas
  }

  /// Parses `sample_sigmas` from `model_index.json`. Returns nil when the
  /// key is absent; throws on a malformed ladder so a corrupt index fails
  /// loudly instead of silently falling back to the 40-step schedule.
  public static func sampleSigmas(fromModelIndex index: [String: Any]) throws -> [Float]? {
    guard let raw = index["sample_sigmas"] else { return nil }
    guard let numbers = raw as? [Any] else {
      throw QwenImagePipelineError.weightsMissing("model_index.json sample_sigmas is not an array")
    }
    let values: [Float] = try numbers.map { element in
      if let double = element as? Double { return Float(double) }
      if let int = element as? Int { return Float(int) }
      throw QwenImagePipelineError.weightsMissing("model_index.json sample_sigmas holds a non-number")
    }
    guard !values.isEmpty,
      values.allSatisfy({ $0.isFinite && $0 > 0 && $0 <= 1 }),
      zip(values, values.dropFirst()).allSatisfy({ $0 > $1 })
    else {
      throw QwenImagePipelineError.weightsMissing(
        "model_index.json sample_sigmas must be strictly decreasing values in (0, 1]")
    }
    return values
  }

  public static func fromConfigJSON(_ config: [String: Any]) -> QwenImageSchedulerConfig {
    func int(_ key: String, _ fallback: Int) -> Int {
      if let value = config[key] as? Int { return value }
      if let value = config[key] as? Double { return Int(value) }
      return fallback
    }
    func float(_ key: String, _ fallback: Float) -> Float {
      if let value = config[key] as? Double { return Float(value) }
      if let value = config[key] as? Int { return Float(value) }
      return fallback
    }
    let terminal: Float?
    if config.keys.contains("shift_terminal") {
      terminal = float("shift_terminal", 0.02)
    } else {
      terminal = nil
    }
    return QwenImageSchedulerConfig(
      numTrainTimesteps: int("num_train_timesteps", 1000),
      baseShift: float("base_shift", 0.5),
      maxShift: float("max_shift", 0.9),
      baseImageSeqLen: int("base_image_seq_len", 256),
      maxImageSeqLen: int("max_image_seq_len", 8192),
      useDynamicShifting: (config["use_dynamic_shifting"] as? Bool) ?? true,
      timeShiftType: (config["time_shift_type"] as? String) ?? "exponential",
      shiftTerminal: terminal,
      shift: float("shift", 1.0)
    )
  }
}

/// Flow-match Euler scheduler with dynamic exponential time shifting and
/// the stretch-to-terminal sigma adjustment the Qwen-Image-2.1 config uses.
///
/// The schedule follows the diffusers pipeline exactly: the base ladder
/// runs `linspace(sigma_max, sigma_min, N)` with `sigma_min = 1 /
/// num_train_timesteps` (the reference derives both endpoints from its
/// reversed training schedule), shifted by `exp(mu) / (exp(mu) + 1/t - 1)`,
/// stretched so the final sigma lands on `shift_terminal`, then a trailing
/// zero sigma is appended.
public struct QwenImage21Scheduler: Sendable {
  public let sigmas: [Float]
  public let timesteps: [Float]
  public let numInferenceSteps: Int
  private let numTrainTimesteps: Int

  public init(
    numInferenceSteps: Int,
    imageSequenceLength: Int,
    config: QwenImageSchedulerConfig
  ) {
    precondition(numInferenceSteps > 0)
    let n = numInferenceSteps
    let numTrain = Float(config.numTrainTimesteps)

    // A published ladder is the whole schedule: no linspace, shift, or
    // terminal stretch. Callers validate the step count against it first
    // (`QwenImage21Pipeline` throws), so a mismatch here is a programmer
    // error, not an input to paper over.
    if let ladder = config.staticSigmaLadder {
      precondition(ladder.count == n, "static sigma ladder length must equal the step count")
      self.numTrainTimesteps = config.numTrainTimesteps
      self.timesteps = ladder.map { $0 * numTrain }
      self.sigmas = ladder + [0.0]
      self.numInferenceSteps = n
      return
    }

    // 1. Base sigmas: linspace(sigma_max = 1, sigma_min = 1/num_train, N).
    let sigmaMin = 1.0 / numTrain
    var sigmas = (0..<n).map { index in
      1.0 + (sigmaMin - 1.0) * Float(index) / Float(max(n - 1, 1))
    }

    // 2. Dynamic shift.
    let mu = Self.calculateShift(
      imageSeqLen: imageSequenceLength,
      baseSeqLen: config.baseImageSeqLen,
      maxSeqLen: config.maxImageSeqLen,
      baseShift: config.baseShift,
      maxShift: config.maxShift
    )
    if config.useDynamicShifting {
      sigmas = sigmas.map { sigma in
        Self.timeShift(mu: mu, sigma: 1.0, t: sigma, type: config.timeShiftType)
      }
    } else if abs(config.shift - 1.0) > Float.ulpOfOne {
      sigmas = sigmas.map { sigma in
        let numerator = config.shift * sigma
        let denominator = 1 + (config.shift - 1) * sigma
        return denominator > 0 ? numerator / denominator : sigma
      }
    }

    // 3. Stretch so the final sigma terminates at shift_terminal.
    if let terminal = config.shiftTerminal, terminal > 0, !sigmas.isEmpty {
      let oneMinusZ = sigmas.map { 1 - $0 }
      let scaleFactor = oneMinusZ[oneMinusZ.count - 1] / (1 - terminal)
      // A one-step ladder is [1.0], so 1 - sigma is 0 and the stretch would
      // be 0/0 = NaN. Leave it unstretched ([1.0, 0.0] after the append).
      if scaleFactor != 0 {
        sigmas = oneMinusZ.map { 1 - $0 / scaleFactor }
      }
    }

    self.numTrainTimesteps = config.numTrainTimesteps
    self.timesteps = sigmas.map { $0 * numTrain }
    sigmas.append(0.0)
    self.sigmas = sigmas
    self.numInferenceSteps = n
  }

  /// `mu = image_seq_len * m + b` from the base/max shift interpolation.
  public static func calculateShift(
    imageSeqLen: Int,
    baseSeqLen: Int,
    maxSeqLen: Int,
    baseShift: Float,
    maxShift: Float
  ) -> Float {
    let slope = (maxShift - baseShift) / Float(maxSeqLen - baseSeqLen)
    let intercept = baseShift - slope * Float(baseSeqLen)
    return Float(imageSeqLen) * slope + intercept
  }

  static func timeShift(mu: Float, sigma: Float, t: Float, type: String) -> Float {
    let inverse = 1 / t - 1
    let denominator = pow(inverse, sigma)
    if type == "exponential" {
      return exp(mu) / (exp(mu) + denominator)
    }
    return mu / (mu + denominator)
  }

  /// Euler flow-matching update: `sample + (sigma_next - sigma) * output`.
  public func step(modelOutput: MLXArray, stepIndex: Int, sample: MLXArray) -> MLXArray {
    precondition(stepIndex >= 0 && stepIndex + 1 < sigmas.count, "invalid step index")
    let sigma = sigmas[stepIndex]
    let sigmaNext = sigmas[stepIndex + 1]
    let dt = MLXArray(sigmaNext - sigma, dtype: .float32).asType(sample.dtype)
    return sample + modelOutput * dt
  }

  public var timestepValues: [Float] { timesteps }
}
