import Foundation
import Logging
import MLX
import MLXNN

enum ZImageModuleWeightsApplyError: LocalizedError {
  case noMatchingWeights(String)
  case updateFailed(String, String)

  var errorDescription: String? {
    switch self {
    case .noMatchingWeights(let prefix):
      return "No matching checkpoint weights were found for \(prefix)."
    case .updateFailed(let prefix, let reason):
      return "Failed to apply checkpoint weights to \(prefix): \(reason)"
    }
  }
}

enum ZImageModuleWeightsApplier {
  @discardableResult
  static func applyToModule(
    _ module: Module,
    weights: [String: MLXArray],
    prefix: String,
    logger: Logger,
    tensorNameTransform: ((String) -> String)? = nil,
    parameterKeyTransform: ((String) -> String)? = nil
  ) -> Error? {
    let params = module.parameters().flattened()
    var updates: [(String, MLXArray)] = []
    updates.reserveCapacity(params.count)

    for (key, _) in params {
      let candidate2Base = "\(prefix).\(key)"
      let candidate2 = tensorNameTransform.map { $0(candidate2Base) } ?? candidate2Base
      let candidates = [key, candidate2]
      if let found = candidates.compactMap({ weights[$0] }).first {
        updates.append((key, found))
      }
    }

    let expectedPrefix = tensorNameTransform.map { $0(prefix) } ?? prefix
    let expectedPrefixWithDot = "\(expectedPrefix)."
    for (weightKey, tensor) in weights {
      guard weightKey.hasPrefix(expectedPrefixWithDot) else { continue }
      guard weightKey.hasSuffix(".scales") || weightKey.hasSuffix(".biases") else { continue }

      var paramKey = String(weightKey.dropFirst(expectedPrefixWithDot.count))
      if let parameterKeyTransform {
        paramKey = parameterKeyTransform(paramKey)
      }

      if !updates.contains(where: { $0.0 == paramKey }) {
        updates.append((paramKey, tensor))
      }
    }

    if updates.isEmpty {
      logger.warning("\(prefix) received no matching weights; skipping apply.")
      return ZImageModuleWeightsApplyError.noMatchingWeights(prefix)
    }

    do {
      let nd = ModuleParameters.unflattened(updates)
      try module.update(parameters: nd, verify: [.shapeMismatch])
      return nil
    } catch {
      logger.error("Failed to apply weights to \(prefix): \(error)")
      return ZImageModuleWeightsApplyError.updateFailed(prefix, String(describing: error))
    }
  }
}
