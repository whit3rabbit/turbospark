import CoreML
import TurboSpark
import XCTest
@testable import TurboSparkApp

/// The check that stands between a model file and FluidAudio's
/// `preconditionFailure`, and the catalog rows it guards.
final class FluidAudioSignatureTests: XCTestCase {
    private let signature = FluidAudioModelSignature(
        file: "Encoder.mlmodelc",
        inputs: [
            FluidAudioFeature("speech", .float32, [[1, 128, 560], [1, 256, 560]]),
            FluidAudioFeature("speech_lengths", .int32, [[1]]),
            FluidAudioFeature("waveform", .float32),
        ],
        outputs: [FluidAudioFeature("logits", .float16)])

    private var matching: [FluidAudioFeature] {
        [
            FluidAudioFeature("speech", .float32, [[1, 256, 560], [1, 128, 560]]),
            FluidAudioFeature("speech_lengths", .int32, [[1]]),
            FluidAudioFeature("waveform", .float32, [[1, 3200]]),
            FluidAudioFeature("unused_extra", .float32),
        ]
    }

    func testTheRecordedModelPassesAndOrderOrExtrasDoNotMatter() {
        let outputs = [FluidAudioFeature("logits", .float16, [[1, 128, 8404]]), FluidAudioFeature("other", .int32)]
        XCTAssertEqual(signature.mismatches(inputs: matching, outputs: outputs), [])
    }

    func testEachWayAModelCanDifferIsReported() {
        func problems(_ edit: (inout [FluidAudioFeature]) -> Void,
                      outputs: [FluidAudioFeature] = [FluidAudioFeature("logits", .float16)]) -> [String] {
            var inputs = matching
            edit(&inputs)
            return signature.mismatches(inputs: inputs, outputs: outputs)
        }
        // A missing input: what OfflineEmbeddingExtractor trapped on.
        XCTAssertEqual(problems { $0.removeAll { $0.name == "speech_lengths" } },
                       ["Encoder.mlmodelc: no input speech_lengths"])
        XCTAssertEqual(problems { $0[1] = FluidAudioFeature("speech_lengths", .float32, [[1]]) }.count, 1)
        // A different bucket set is a different export.
        XCTAssertEqual(problems { $0[0] = FluidAudioFeature("speech", .float32, [[1, 128, 560]]) }.count, 1)
        // A range-shaped input is not shape-checked, only named and typed.
        XCTAssertEqual(problems { $0[2] = FluidAudioFeature("waveform", .float32, [[1, 9]]) }, [])
        // Outputs: name and type.
        XCTAssertEqual(problems({ _ in }, outputs: []), ["Encoder.mlmodelc: no output logits"])
        XCTAssertEqual(problems({ _ in }, outputs: [FluidAudioFeature("logits", .float32)]).count, 1)
    }

    func testVocabularyReadsArraysAndIdKeyedObjectsAndRefusesTheRest() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("vocab-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        func read(_ json: String) throws -> [Int: String] {
            try Data(json.utf8).write(to: dir.appendingPathComponent("vocab.json"))
            return try FluidAudioModelLoader.vocabulary(in: dir)
        }
        XCTAssertEqual(try read(#"["<unk>","a","b"]"#), [0: "<unk>", 1: "a", 2: "b"])
        XCTAssertEqual(try read(#"{"0":"<blank>","7":"x"}"#), [0: "<blank>", 7: "x"])
        XCTAssertThrowsError(try read(#"{"zero":"x"}"#))
        XCTAssertThrowsError(try read(#"[1,2]"#))
    }

    /// Every FluidAudio row the catalog ships names a family this app knows,
    /// with the layout that family's loader reads.
    func testCatalogRowsMatchTheirFamilies() throws {
        let rows = try AudioCatalog.profiles().filter { $0.capabilities.backend == "swift_coreml" }
        XCTAssertFalse(rows.isEmpty)
        for row in rows {
            let family = try XCTUnwrap(FluidAudioFamily(rawValue: row.family), row.family)
            XCTAssertEqual(family.expectedInstallSubdir, row.installSubdir, row.identity.alias)
            XCTAssertEqual(row.capabilities.httpApi, false, row.identity.alias)
        }
    }
}
