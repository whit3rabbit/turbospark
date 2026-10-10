import Foundation
import MLX
import XCTest
@testable import Qwen3TTS

final class Qwen3TTSTests: XCTestCase {
    struct Golden: Decodable { let samples: [Float]; let shape: [Int]; let mel: [Float] }
    func config(_ changes: (inout [String: Any]) -> Void = { _ in }) throws -> Data {
        let url = Bundle.module.url(forResource: "base-config", withExtension: "json")!
        var value = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as! [String: Any]
        changes(&value)
        return try JSONSerialization.data(withJSONObject: value)
    }
    func testSpeakerFrontendMatchesPinnedPython() throws {
        let fixture = try JSONDecoder().decode(Golden.self, from: Data(contentsOf: Bundle.module.url(forResource: "speaker-mel", withExtension: "json")!))
        let mel = speakerMel(MLXArray(fixture.samples))
        eval(mel)
        XCTAssertEqual(mel.shape, fixture.shape)
        let actual = mel.asArray(Float.self)
        XCTAssertEqual(actual.count, fixture.mel.count)
        XCTAssertLessThan(zip(actual, fixture.mel).map { abs($0 - $1) }.max()!, 0.0001)
    }
    func testVariantsAndLanguagesRejectInvalidRequests() throws {
        let base = try Qwen3TTSCheckpoint(data: config())
        let reference = Array(repeating: Float(0.1), count: 24_000)
        XCTAssertNoThrow(try base.validate(text: "Hello", voice: nil, reference: reference, referenceText: "Reference", language: "en"))
        XCTAssertThrowsError(try base.validate(text: "Hello", voice: nil, reference: nil, referenceText: nil, language: "en"))
        XCTAssertThrowsError(try base.validate(text: "Hello", voice: nil, reference: reference, referenceText: nil, language: "xx"))
        XCTAssertThrowsError(try base.validate(text: "Hello", voice: nil, reference: Array(repeating: 0, count: 24_000), referenceText: nil, language: "auto"))
        XCTAssertThrowsError(try Qwen3TTSCheckpoint(data: config { $0["model_type"] = "qwen3_asr" }))
        XCTAssertThrowsError(try Qwen3TTSCheckpoint(data: config { $0.removeValue(forKey: "model_type") }))
        XCTAssertThrowsError(try Qwen3TTSCheckpoint(data: config { $0["tts_model_type"] = "voice_design" }))
        XCTAssertThrowsError(try Qwen3TTSCheckpoint(data: config { $0["quantization"] = ["bits": 3, "group_size": 64, "mode": "affine"] }))
    }
    func testIncompleteCheckpointPreservesExistingFiles() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try config().write(to: root.appendingPathComponent("config.json"))
        let sentinel = Data("Existing checkpoint data".utf8)
        try sentinel.write(to: root.appendingPathComponent("speech_tokenizer"))
        do { _ = try await Qwen3TTSModel.fromModelDirectory(root); XCTFail("Incomplete checkpoint loaded") }
        catch { XCTAssertEqual(try Data(contentsOf: root.appendingPathComponent("speech_tokenizer")), sentinel) }
        XCTAssertTrue(FileManager.default.fileExists(atPath: root.appendingPathComponent("config.json").path))
    }
    func testCustomVoiceStyleRequires17B() throws {
        let custom = try Qwen3TTSCheckpoint(data: config {
            $0["tts_model_type"] = "custom_voice"
            var talker = $0["talker_config"] as! [String: Any]; talker["spk_id"] = ["ryan": 3061]; $0["talker_config"] = talker
        })
        XCTAssertNoThrow(try custom.validate(text: "Hello", voice: "Ryan", reference: nil, referenceText: nil, language: "english"))
        XCTAssertThrowsError(try custom.validate(text: "Hello", voice: "Ryan,whispering", reference: nil, referenceText: nil, language: "en"))
    }

    func testRealCheckpointRefusesExhaustedTokenBudget() async throws {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_QWEN3_TTS_CUSTOM"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_QWEN3_TTS_CUSTOM for the real token-exhaustion gate")
        }
        let model = try await Qwen3TTSModel.fromModelDirectory(URL(fileURLWithPath: path))
        MLXRandom.seed(42)
        XCTAssertThrowsError(try model.synthesize(
            text: "The sky is clear. We are ready for the next flight.", voice: "Ryan", reference: nil,
            referenceText: nil, language: "en", parameters: GenerateParameters(maxTokens: 1, temperature: 0))) {
                XCTAssertTrue($0.localizedDescription.contains("token limit"), $0.localizedDescription)
            }
    }

    func testRealCodecKeepsZeroValuedFrames() async throws {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_QWEN3_TTS_CUSTOM"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_QWEN3_TTS_CUSTOM for the real zero-codec-token gate")
        }
        let model = try await Qwen3TTSModel.fromModelDirectory(URL(fileURLWithPath: path))
        let codec = try XCTUnwrap(model.speechTokenizer)
        let (audio, lengths) = try codec.decode(MLXArray.zeros([1, 4, 16], dtype: .int32))
        eval(audio, lengths)
        XCTAssertEqual(audio.shape, [1, 4 * 1920])
        XCTAssertEqual(lengths.item(Int32.self), 4 * 1920, "Codec token zero is speech, not padding")
        XCTAssertTrue(audio.asArray(Float.self).allSatisfy(\.isFinite))
    }
}
