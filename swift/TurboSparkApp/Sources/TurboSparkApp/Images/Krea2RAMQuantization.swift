import Foundation
import ZImage

struct Krea2RAMQuantization: Sendable {
    static let supportedWidths: Set<Int> = [4, 5, 6, 8]

    let groupSize: Int
    let bitsByModule: [String: Int]

    static func load(modelRoot: URL) throws -> Krea2RAMQuantization {
        try validateSourceComponents(at: modelRoot)

        let mapURL = modelRoot.appendingPathComponent("ram_bits.json")
        let mapData = try Data(contentsOf: mapURL)
        let manifest = try JSONDecoder().decode(RAMManifest.self, from: mapData)
        guard manifest.groupSize == 64 else {
            throw Krea2RAMQuantizationError.invalidGroupSize(manifest.groupSize)
        }
        guard !manifest.bits.isEmpty else {
            throw Krea2RAMQuantizationError.emptyMap
        }

        let indexURL = modelRoot.appendingPathComponent("model.safetensors.index.json")
        let indexData = try Data(contentsOf: indexURL)
        let index = try JSONDecoder().decode(SafeTensorsIndex.self, from: indexData)
        try validateShardNames(index.weightMap.values, under: modelRoot)

        var readers: [String: SafeTensorsReader] = [:]
        for (module, width) in manifest.bits {
            guard Self.supportedWidths.contains(width) else {
                throw Krea2RAMQuantizationError.unsupportedWidth(module: module, width: width)
            }
            guard Self.isSafeModuleName(module) else {
                throw Krea2RAMQuantizationError.invalidModuleName(module)
            }

            let tensorNames = ["weight", "scales", "biases"].map { "\(module).\($0)" }
            guard tensorNames.allSatisfy({ index.weightMap[$0] != nil }) else {
                throw Krea2RAMQuantizationError.unmatchedAssignment(module)
            }
            let shardNames = Set(tensorNames.compactMap { index.weightMap[$0] })
            guard shardNames.count == 1, let shardName = shardNames.first else {
                throw Krea2RAMQuantizationError.splitAssignment(module)
            }
            let reader: SafeTensorsReader
            if let cached = readers[shardName] {
                reader = cached
            } else {
                reader = try SafeTensorsReader(
                    fileURL: modelRoot.appendingPathComponent(shardName)
                )
                readers[shardName] = reader
            }
            try Self.validateGeometry(
                module: module,
                width: width,
                groupSize: manifest.groupSize,
                reader: reader
            )
        }

        return Krea2RAMQuantization(
            groupSize: manifest.groupSize,
            bitsByModule: manifest.bits
        )
    }

    func width(for module: String) throws -> Int {
        guard let width = bitsByModule[module] else {
            throw Krea2RAMQuantizationError.unassignedModule(module)
        }
        return width
    }

    private static func validateSourceComponents(at root: URL) throws {
        for path in [
            "model.safetensors.index.json",
            "ram_bits.json",
            "LICENSE.pdf",
            "Notice",
            "text_encoder/model.safetensors.index.json",
            "tokenizer/chat_template.jinja",
            "tokenizer/tokenizer.json",
            "tokenizer/tokenizer_config.json",
            "vae/model.safetensors.index.json",
        ] {
            guard FileManager.default.fileExists(atPath: root.appendingPathComponent(path).path) else {
                throw Krea2RAMQuantizationError.missingComponent(path)
            }
        }

        for component in ["text_encoder", "vae"] {
            let indexURL = root.appendingPathComponent("\(component)/model.safetensors.index.json")
            let index = try JSONDecoder().decode(
                SafeTensorsIndex.self,
                from: Data(contentsOf: indexURL)
            )
            guard !index.weightMap.isEmpty else {
                throw Krea2RAMQuantizationError.emptyComponentIndex(component)
            }
            try validateShardNames(index.weightMap.values, under: root.appendingPathComponent(component))
        }
    }

    private static func validateShardNames<S: Sequence>(_ names: S, under root: URL) throws
    where S.Element == String {
        for name in Set(names) {
            guard isSafeRelativePath(name) else {
                throw Krea2RAMQuantizationError.invalidShardName(name)
            }
            guard FileManager.default.fileExists(atPath: root.appendingPathComponent(name).path) else {
                throw Krea2RAMQuantizationError.missingComponent(root.appendingPathComponent(name).path)
            }
        }
    }

    private static func validateGeometry(
        module: String,
        width: Int,
        groupSize: Int,
        reader: SafeTensorsReader
    ) throws {
        let weightName = "\(module).weight"
        let scaleName = "\(module).scales"
        let biasName = "\(module).biases"
        guard let weight = reader.metadata(for: weightName),
              let scales = reader.metadata(for: scaleName),
              let biases = reader.metadata(for: biasName) else {
            throw Krea2RAMQuantizationError.missingQuantizedTensor(module)
        }
        guard weight.shape.count == 2,
              scales.shape.count == 2,
              biases.shape == scales.shape,
              weight.shape[0] == scales.shape[0],
              scales.shape[1] > 0 else {
            throw Krea2RAMQuantizationError.invalidQuantizedGeometry(module)
        }

        let (inputWidth, inputOverflow) = scales.shape[1].multipliedReportingOverflow(by: groupSize)
        let (packedBits, bitsOverflow) = inputWidth.multipliedReportingOverflow(by: width)
        guard !inputOverflow, !bitsOverflow, packedBits % 32 == 0 else {
            throw Krea2RAMQuantizationError.invalidQuantizedGeometry(module)
        }
        guard weight.shape[1] == packedBits / 32 else {
            throw Krea2RAMQuantizationError.invalidQuantizedGeometry(module)
        }
    }

    private static func isSafeModuleName(_ name: String) -> Bool {
        !name.isEmpty
            && !name.contains("..")
            && !name.contains("/")
            && name.unicodeScalars.allSatisfy {
                CharacterSet(charactersIn: "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_.")
                    .contains($0)
            }
    }

    private static func isSafeRelativePath(_ path: String) -> Bool {
        let parts = path.split(separator: "/", omittingEmptySubsequences: false)
        return !path.isEmpty
            && !path.hasPrefix("/")
            && parts.allSatisfy { !$0.isEmpty && $0 != "." && $0 != ".." }
    }
}

private struct RAMManifest: Decodable {
    let groupSize: Int
    let bits: [String: Int]

    enum CodingKeys: String, CodingKey {
        case groupSize = "group_size"
        case bits
    }
}

private struct SafeTensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

enum Krea2RAMQuantizationError: Error, LocalizedError, Equatable {
    case missingComponent(String)
    case emptyComponentIndex(String)
    case invalidGroupSize(Int)
    case emptyMap
    case unsupportedWidth(module: String, width: Int)
    case invalidModuleName(String)
    case unmatchedAssignment(String)
    case splitAssignment(String)
    case invalidShardName(String)
    case unassignedModule(String)
    case missingQuantizedTensor(String)
    case invalidQuantizedGeometry(String)

    var errorDescription: String? {
        switch self {
        case .missingComponent(let path): return "Krea 2 source is missing required file \(path)."
        case .emptyComponentIndex(let component): return "Krea 2 \(component) index contains no tensors."
        case .invalidGroupSize(let size): return "Krea 2 RAM map uses group size \(size); expected 64."
        case .emptyMap: return "Krea 2 RAM map contains no assignments."
        case .unsupportedWidth(let module, let width):
            return "Krea 2 RAM map assigns unsupported \(width)-bit quantization to \(module)."
        case .invalidModuleName(let name): return "Krea 2 RAM map contains an invalid module name: \(name)."
        case .unmatchedAssignment(let module):
            return "Krea 2 RAM assignment \(module) does not match weight, scales, and biases in the transformer index."
        case .splitAssignment(let module):
            return "Krea 2 RAM assignment \(module) has tensors split across safetensors shards."
        case .invalidShardName(let name): return "Krea 2 index contains an unsafe shard path: \(name)."
        case .unassignedModule(let module): return "Krea 2 model requested an unassigned quantized module: \(module)."
        case .missingQuantizedTensor(let module): return "Krea 2 assignment \(module) is missing a quantized tensor."
        case .invalidQuantizedGeometry(let module):
            return "Krea 2 assignment \(module) has a tensor layout that does not match its RAM bit width."
        }
    }
}
