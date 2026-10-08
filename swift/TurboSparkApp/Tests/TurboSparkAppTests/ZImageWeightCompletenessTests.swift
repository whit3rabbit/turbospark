import Foundation
import XCTest
import ZImage

/// A snapshot whose index names a shard the disk lacks must resolve to no
/// weights, not to the shards that happen to exist (the missing layers would
/// run with random weights).
final class ZImageWeightCompletenessTests: XCTestCase {
    private func makeSnapshot(shards: [String], present: [String]) throws -> URL {
        let root = FileManager.default.temporaryDirectory
            .appendingPathComponent("zimage-snapshot-\(UUID().uuidString)", isDirectory: true)
        let dir = root.appendingPathComponent("transformer", isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var map: [String: String] = [:]
        for (i, shard) in shards.enumerated() { map["layer\(i).weight"] = shard }
        let index = try JSONSerialization.data(withJSONObject: ["weight_map": map])
        try index.write(to: dir.appendingPathComponent("diffusion_pytorch_model.safetensors.index.json"))
        for name in present { try Data("x".utf8).write(to: dir.appendingPathComponent(name)) }
        return root
    }

    func testIndexNamingAMissingShardResolvesToNothing() throws {
        let shards = [
            "diffusion_pytorch_model-00001-of-00003.safetensors",
            "diffusion_pytorch_model-00002-of-00003.safetensors",
            "diffusion_pytorch_model-00003-of-00003.safetensors",
        ]
        let partial = try makeSnapshot(shards: shards, present: Array(shards.prefix(2)))
        defer { try? FileManager.default.removeItem(at: partial) }
        XCTAssertTrue(ZImageFiles.resolveTransformerWeights(at: partial).isEmpty)

        let complete = try makeSnapshot(shards: shards, present: shards)
        defer { try? FileManager.default.removeItem(at: complete) }
        XCTAssertEqual(ZImageFiles.resolveTransformerWeights(at: complete).count, 3)
    }
}
