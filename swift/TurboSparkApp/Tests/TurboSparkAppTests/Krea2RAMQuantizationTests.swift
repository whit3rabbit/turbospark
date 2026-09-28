import Foundation
import XCTest
@testable import TurboSparkApp

final class Krea2RAMQuantizationTests: XCTestCase {
    func testPinnedRAMAssignmentsResolveAndGeometryMatchesTheDeclaredWidth() throws {
        let fixture = try Krea2SourceFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        let map = try Krea2RAMQuantization.load(modelRoot: fixture.root)

        XCTAssertEqual(map.groupSize, 64)
        XCTAssertEqual(try map.width(for: "blocks.0.attn.wq"), 4)
        XCTAssertEqual(map.bitsByModule.count, 1)
    }

    func testMissingComponentsAreRejectedBeforeModelConstruction() throws {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("krea2-missing-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }

        XCTAssertThrowsError(try Krea2RAMQuantization.load(modelRoot: root)) { error in
            XCTAssertEqual(
                error as? Krea2RAMQuantizationError,
                .missingComponent("model.safetensors.index.json")
            )
        }
    }

    func testUnsupportedWidthsAndUnmatchedMapEntriesAreRejected() throws {
        let fixture = try Krea2SourceFixture()
        defer { try? FileManager.default.removeItem(at: fixture.root) }

        try fixture.writeRAMMap(module: "blocks.0.attn.wq", width: 3)
        XCTAssertThrowsError(try Krea2RAMQuantization.load(modelRoot: fixture.root)) { error in
            XCTAssertEqual(
                error as? Krea2RAMQuantizationError,
                .unsupportedWidth(module: "blocks.0.attn.wq", width: 3)
            )
        }

        try fixture.writeRAMMap(module: "blocks.1.attn.wq", width: 4)
        XCTAssertThrowsError(try Krea2RAMQuantization.load(modelRoot: fixture.root)) { error in
            XCTAssertEqual(
                error as? Krea2RAMQuantizationError,
                .unmatchedAssignment("blocks.1.attn.wq")
            )
        }
    }

    func testEveryMixedPrecisionWidthInThePublishedFormatIsAccepted() throws {
        for width in Krea2RAMQuantization.supportedWidths {
            let fixture = try Krea2SourceFixture(width: width)
            defer { try? FileManager.default.removeItem(at: fixture.root) }
            let map = try Krea2RAMQuantization.load(modelRoot: fixture.root)
            XCTAssertEqual(try map.width(for: "blocks.0.attn.wq"), width)
        }
    }
}

private struct Krea2SourceFixture {
    let root: URL
    private let module = "blocks.0.attn.wq"

    init(width: Int = 4) throws {
        root = FileManager.default.temporaryDirectory
            .appendingPathComponent("krea2-source-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try writeRAMMap(module: module, width: width)

        for file in ["LICENSE.pdf", "Notice", "tokenizer/chat_template.jinja",
                     "tokenizer/tokenizer.json", "tokenizer/tokenizer_config.json"] {
            try write(Data([0x20]), to: file)
        }

        let transformerTensors = [
            "\(module).weight": TensorDescription(
                dtype: "U32", shape: [2, 2 * width], offsets: [0, 16 * width]
            ),
            "\(module).scales": TensorDescription(
                dtype: "F32", shape: [2, 1], offsets: [16 * width, 16 * width + 8]
            ),
            "\(module).biases": TensorDescription(
                dtype: "F32", shape: [2, 1], offsets: [16 * width + 8, 16 * width + 16]
            )
        ]
        try writeSafetensors(transformerTensors, to: "0.safetensors")
        try writeIndex(
            ["\(module).weight", "\(module).scales", "\(module).biases"],
            shard: "0.safetensors",
            to: "model.safetensors.index.json"
        )
        try writeComponentIndex("text_encoder")
        try writeComponentIndex("vae")
    }

    func writeRAMMap(module: String, width: Int) throws {
        let json: [String: Any] = ["group_size": 64, "bits": [module: width]]
        try writeJSON(json, to: "ram_bits.json")
    }

    private func writeComponentIndex(_ component: String) throws {
        let directory = root.appendingPathComponent(component, isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        try write(Data([0x20]), to: "\(component)/0.safetensors")
        try writeIndex(["fixture.weight"], shard: "0.safetensors", to: "\(component)/model.safetensors.index.json")
    }

    private func writeIndex(_ names: [String], shard: String, to path: String) throws {
        let map = Dictionary(uniqueKeysWithValues: names.map { ($0, shard) })
        try writeJSON(["weight_map": map], to: path)
    }

    private func writeSafetensors(_ tensors: [String: TensorDescription], to path: String) throws {
        let header: [String: Any] = Dictionary(uniqueKeysWithValues: tensors.map { name, tensor in
            (name, ["dtype": tensor.dtype, "shape": tensor.shape, "data_offsets": tensor.offsets])
        })
        let headerData = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var headerLength = UInt64(headerData.count).littleEndian
        var data = Data()
        withUnsafeBytes(of: &headerLength) { data.append(contentsOf: $0) }
        data.append(headerData)
        let payloadLength = tensors.values.map { $0.offsets[1] }.max() ?? 0
        data.append(Data(repeating: 0, count: payloadLength))
        try write(data, to: path)
    }

    private func writeJSON(_ object: [String: Any], to path: String) throws {
        let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
        try write(data, to: path)
    }

    private func write(_ data: Data, to path: String) throws {
        let url = root.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        try data.write(to: url)
    }
}

private struct TensorDescription {
    let dtype: String
    let shape: [Int]
    let offsets: [Int]
}
