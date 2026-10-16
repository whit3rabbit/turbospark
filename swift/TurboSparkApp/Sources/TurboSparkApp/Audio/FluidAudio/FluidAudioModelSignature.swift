import CoreML
import Foundation

/// One input or output of a compiled model as FluidAudio uses it: the feature
/// name, the array data type, and the shapes it accepts. `shapes` is nil when
/// the model takes a range (the waveform preprocessors), which is not checked.
struct FluidAudioFeature: Sendable, Equatable {
    let name: String
    let dataType: MLMultiArrayDataType
    let shapes: Set<[Int]>?

    init(_ name: String, _ dataType: MLMultiArrayDataType, _ shapes: [[Int]]? = nil) {
        self.name = name
        self.dataType = dataType
        self.shapes = shapes.map(Set.init)
    }
}

/// The inputs and outputs FluidAudio feeds and reads for one compiled model,
/// recorded from the pinned export (`FluidAudioSignatures`). FluidAudio
/// answers a model that differs with `preconditionFailure`, and only on first
/// use, which ends the app. The loader compares first and throws instead.
struct FluidAudioModelSignature: Sendable, Equatable {
    let file: String
    let inputs: [FluidAudioFeature]
    let outputs: [FluidAudioFeature]

    /// Every way `found` falls short of this signature, empty when it
    /// matches. Inputs must match name, type and, when recorded, the exact
    /// set of accepted shapes. Outputs must match name and type. Extra
    /// features are allowed: FluidAudio never feeds or reads them.
    func mismatches(inputs foundInputs: [FluidAudioFeature], outputs foundOutputs: [FluidAudioFeature]) -> [String] {
        func compare(_ expected: [FluidAudioFeature], _ found: [FluidAudioFeature], role: String,
                     checkShapes: Bool) -> [String] {
            let byName = Dictionary(found.map { ($0.name, $0) }, uniquingKeysWith: { first, _ in first })
            return expected.compactMap { want in
                guard let have = byName[want.name] else { return "\(file): no \(role) \(want.name)" }
                if have.dataType != want.dataType {
                    return "\(file): \(role) \(want.name) is \(have.dataType.rawValue), not \(want.dataType.rawValue)"
                }
                if checkShapes, let shapes = want.shapes, have.shapes != shapes {
                    return "\(file): \(role) \(want.name) accepts \(have.shapes.map { Array($0) } ?? []), not \(Array(shapes))"
                }
                return nil
            }
        }
        return compare(inputs, foundInputs, role: "input", checkShapes: true)
            + compare(outputs, foundOutputs, role: "output", checkShapes: false)
    }

    /// The features a loaded model declares, in the same form.
    static func features(_ descriptions: [String: MLFeatureDescription]) -> [FluidAudioFeature] {
        descriptions.values.compactMap { description in
            guard let array = description.multiArrayConstraint else { return nil }
            let constraint = array.shapeConstraint
            let shapes: [[Int]]?
            switch constraint.type {
            case .enumerated: shapes = constraint.enumeratedShapes.map { $0.map(\.intValue) }
            case .unspecified: shapes = array.shape.isEmpty ? nil : [array.shape.map(\.intValue)]
            default: shapes = nil
            }
            return FluidAudioFeature(description.name, array.dataType, shapes)
        }
    }
}

enum FluidAudioModelError: LocalizedError, Equatable {
    /// A model in the install is not the one its family expects.
    case mismatch([String])
    /// The vocabulary file is neither a JSON array nor an id-keyed object.
    case vocabulary(String)
    /// The family has no speech recognizer in this build.
    case notSpeech(FluidAudioFamily)
    /// The window is longer than the recognizer decodes without dropping text.
    case windowTooLong(samples: Int, limit: Int)

    /// What the workspace shows. A model that fails its signature or
    /// vocabulary check is a damaged or replaced install, which a fresh
    /// download fixes. The other two are routing faults, not user errors.
    var errorDescription: String? {
        switch self {
        case .mismatch, .vocabulary:
            return AppStrings.localized("The installed model does not match this app. Delete it and download it again.")
        case .notSpeech:
            return AppStrings.localized("Runtime unavailable")
        case .windowTooLong:
            return nil
        }
    }
}

enum FluidAudioModelLoader {
    /// Loads one compiled model from a verified install and checks it against
    /// its recorded signature before anything hands it to FluidAudio.
    static func load(_ signature: FluidAudioModelSignature, in directory: URL,
                     units: MLComputeUnits) throws -> MLModel {
        let config = MLModelConfiguration()
        config.computeUnits = units
        let model = try MLModel(contentsOf: directory.appendingPathComponent(signature.file), configuration: config)
        let problems = signature.mismatches(
            inputs: FluidAudioModelSignature.features(model.modelDescription.inputDescriptionsByName),
            outputs: FluidAudioModelSignature.features(model.modelDescription.outputDescriptionsByName))
        guard problems.isEmpty else { throw FluidAudioModelError.mismatch(problems) }
        return model
    }

    /// Token id to text, from the `vocab.json` beside the models: a JSON array
    /// (index is the id) or an object keyed by the id, as FluidAudio reads it.
    static func vocabulary(in directory: URL, file: String = "vocab.json") throws -> [Int: String] {
        let data = try Data(contentsOf: directory.appendingPathComponent(file))
        let json = try JSONSerialization.jsonObject(with: data)
        if let tokens = json as? [String] {
            return Dictionary(uniqueKeysWithValues: tokens.enumerated().map { ($0.offset, $0.element) })
        }
        if let pairs = json as? [String: String] {
            var out: [Int: String] = [:]
            for (key, token) in pairs {
                guard let id = Int(key) else { throw FluidAudioModelError.vocabulary(file) }
                out[id] = token
            }
            return out
        }
        throw FluidAudioModelError.vocabulary(file)
    }
}
