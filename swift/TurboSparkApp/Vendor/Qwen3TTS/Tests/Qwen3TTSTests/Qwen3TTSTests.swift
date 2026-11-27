import Foundation
import MLX
import MLXNN
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

    // Streaming must match one call for any chunking. The overlap-add tail
    // once carried the transposed-conv bias into the next chunk, whose head
    // already has it (mlx-audio PR 1003). A zero bias would hide that.
    func testUpsampleStepMatchesFullCallForAnyChunking() throws {
        for rate in [2, 4, 8] {
            MLXRandom.seed(0)
            let block = DecoderBlockUpsample(inDim: 6, outDim: 5, upsampleRate: rate)
            try block.conv.update(parameters: ModuleParameters.unflattened(["bias": MLXRandom.normal([5])]), verify: .none)
            let x = MLXRandom.normal([1, 6, 12])  // NCL
            let full = block(x)
            for chunks in [[Int](repeating: 1, count: 12), [Int](repeating: 2, count: 6), [3, 1, 4, 4], [12]] {
                block.resetState()
                var parts: [MLXArray] = []
                var start = 0
                for size in chunks {
                    parts.append(block.step(x[0..., 0..., start ..< (start + size)]))
                    start += size
                }
                let streamed = concatenated(parts, axis: -1)
                eval(full, streamed)
                XCTAssertEqual(streamed.shape, full.shape)
                let diff = abs(streamed - full).max().item(Float.self)
                XCTAssertLessThan(diff, 1e-5, "rate \(rate), chunks \(chunks)")
            }
        }
    }
}

final class BreezeTTSTests: XCTestCase {
    // mlx-swift 0.30.6 QuantizedLinear drops the mode when it packs, so an
    // mxfp4 linear came out affine with a biases array the checkpoint lacks.
    func testMxfp4LinearHasNoBiasesAndKeepsMode() throws {
        let layer = try XCTUnwrap(BreezeTTSModel.quantizeLayer(Linear(64, 64, bias: false), groupSize: 32, bits: 4, mode: .mxfp4) as? QuantizedLinear)
        XCTAssertEqual(layer.mode, .mxfp4)
        XCTAssertNil(layer.biases)
        XCTAssertEqual(Set(layer.parameters().flattened().map(\.0)), ["scales", "weight"])
        let affine = try XCTUnwrap(BreezeTTSModel.quantizeLayer(Linear(64, 64, bias: false), groupSize: 64, bits: 8, mode: .affine) as? QuantizedLinear)
        XCTAssertNotNil(affine.biases)
        let embedding = try XCTUnwrap(BreezeTTSModel.quantizeLayer(Embedding(embeddingCount: 8, dimensions: 64), groupSize: 32, bits: 4, mode: .mxfp4) as? QuantizedEmbedding)
        XCTAssertNil(embedding.biases)
    }

    // Real-checkpoint gate: every activation stays bf16 like mlx-audio, so
    // greedy near-tie picks follow the same rounding.
    func testRealCheckpointActivationsStayBF16() async throws {
        guard let path = ProcessInfo.processInfo.environment["TURBOSPARK_TEST_BREEZE"] else {
            throw XCTSkip("Set TURBOSPARK_TEST_BREEZE to a Breeze TTS 2 checkpoint")
        }
        let model = try await BreezeTTSModel.fromModelDirectory(URL(fileURLWithPath: path))
        let depthEmbed = model.depthDecoder.model.embedTokens(MLXArray([Int32(1), 2]).reshaped(1, 2))
        let audioEmbed = model.backbone.embedTokens(MLXArray.zeros([1, 1, 16], dtype: .int32))
        let prompt = try model.promptEmbeddings(text: "Hi.", speakerTag: "[S0]", instruction: nil, reference: nil)
        let hidden = model.backbone(embeddings: prompt, cache: model.backbone.makeCache())[0..., -1, 0...]
        let depth = model.depthDecoder.nextLogits(MLXArray([Int32(0), 5]).reshaped(1, 2), backboneHidden: hidden)
        print("DTYPES depthEmbed=\(depthEmbed.dtype) audioEmbed=\(audioEmbed.dtype) prompt=\(prompt.dtype) hidden=\(hidden.dtype) lm=\(model.lmHead(hidden).dtype) depth=\(depth.dtype)")
        for array in [depthEmbed, audioEmbed, prompt, hidden, model.lmHead(hidden), depth] {
            XCTAssertEqual(array.dtype, .bfloat16)
        }
    }

    // A minimal config with the pinned checkpoints' shapes (small layer
    // counts; validation does not depend on depth).
    static func breezeConfig(_ change: (inout [String: Any]) -> Void = { _ in }) -> [String: Any] {
        var c: [String: Any] = [
            "model_type": "breeze_tts", "audio_num_codebooks": 16, "audio_vocab_size": 2051, "codebook_eos_token_id": 0,
            "codec_config": ["codebook_size": 2048, "sampling_rate": 24000],
            "backbone_config": ["model_type": "qwen3", "hidden_size": 64, "num_hidden_layers": 1, "intermediate_size": 96,
                                "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "rms_norm_eps": 1e-6, "rope_theta": 1000000],
            "text_encoder_config": ["model_type": "t5gemma2_text", "vocab_size": 300, "hidden_size": 32, "intermediate_size": 48,
                                    "num_hidden_layers": 2, "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 16,
                                    "sliding_window": 512, "layer_types": ["sliding_attention", "full_attention"],
                                    "rope_parameters": ["full_attention": ["rope_type": "linear", "factor": 8.0, "rope_theta": 1000000],
                                                        "sliding_attention": ["rope_type": "default", "rope_theta": 10000]]],
            "depth_decoder_config": ["hidden_size": 32, "num_hidden_layers": 1, "intermediate_size": 48, "num_attention_heads": 2,
                                     "num_key_value_heads": 1, "head_dim": 16, "max_position_embeddings": 33, "rope_theta": 500000,
                                     "rope_scaling": ["factor": 32.0, "high_freq_factor": 0.0078125, "low_freq_factor": 0.001953125,
                                                      "original_max_position_embeddings": 16, "rope_type": "llama3"]],
            "quantization": ["group_size": 32, "bits": 4, "mode": "mxfp4"],
        ]
        c["audio_embed_size"] = 64
        change(&c)
        return c
    }

    func testConfigAcceptsPinnedFormatsAndRefusesOthers() throws {
        let mx = try BreezeTTSConfig(json: Self.breezeConfig())
        XCTAssertEqual(mx.quantization, .init(groupSize: 32, bits: 4, mode: .mxfp4))
        XCTAssertEqual(mx.codecVocabSize, 2048)
        XCTAssertEqual(mx.text.rope["full_attention"]?.scaling?["factor"], .float(8))
        XCTAssertNil(mx.text.rope["sliding_attention"]?.scaling)
        XCTAssertNoThrow(try BreezeTTSConfig(json: Self.breezeConfig { $0["quantization"] = ["group_size": 64, "bits": 8, "mode": "affine"] }))
        XCTAssertNil(try BreezeTTSConfig(json: Self.breezeConfig { $0.removeValue(forKey: "quantization") }).quantization)
        for bad: [String: Any] in [["group_size": 32, "bits": 4, "mode": "nvfp4"], ["group_size": 64, "bits": 4, "mode": "affine"],
                                   ["group_size": 64, "bits": 4], ["mode": "mxfp4"]] {
            XCTAssertThrowsError(try BreezeTTSConfig(json: Self.breezeConfig { $0["quantization"] = bad })) {
                XCTAssertEqual($0 as? BreezeTTSError, .invalidQuantization, "\(bad)")
            }
        }
        // Two disagreeing blocks are refused rather than picking one.
        XCTAssertThrowsError(try BreezeTTSConfig(json: Self.breezeConfig { $0["quantization_config"] = ["group_size": 64, "bits": 8, "mode": "affine"] }))
        XCTAssertThrowsError(try BreezeTTSConfig(json: Self.breezeConfig { $0["model_type"] = "qwen3_tts" }))
        XCTAssertThrowsError(try BreezeTTSConfig(json: Self.breezeConfig { $0["audio_num_codebooks"] = 8 }))
        XCTAssertThrowsError(try BreezeTTSConfig(json: Self.breezeConfig { $0["audio_embed_size"] = 32 }))
        XCTAssertThrowsError(try BreezeTTSConfig(json: Self.breezeConfig { $0["codec_config"] = ["codebook_size": 4096] }))
        XCTAssertThrowsError(try BreezeTTSConfig(json: Self.breezeConfig {
            var t = $0["text_encoder_config"] as! [String: Any]; t["layer_types"] = ["sliding_attention"]; $0["text_encoder_config"] = t
        }))
    }

    func testSanitizeDropsUnusedTablesAndFillsTiedBackboneEmbedding() {
        let w = MLXArray([UInt32(1)]), s = MLXArray([UInt8(2)])
        let out = BreezeTTSModel.sanitize([
            "depth_decoder.model.embed_tokens.weight": w, "depth_decoder.model.embed_tokens.scales": s,
            "embed_text_tokens.weight": w, "embed_text_tokens.scales": s, "codec_model.x": w,
            "a.rotary_emb.inv_freq": w, "q.codebook.initialized": w,
        ])
        XCTAssertEqual(Set(out.keys), ["depth_decoder.model.embed_tokens.weight", "depth_decoder.model.embed_tokens.scales",
                                       "backbone_model.embed_tokens.embed_audio_tokens.weight", "backbone_model.embed_tokens.embed_audio_tokens.scales"])
        // A stored backbone copy is kept as is.
        let kept = BreezeTTSModel.sanitize(["backbone_model.embed_tokens.embed_audio_tokens.weight": s, "depth_decoder.model.embed_tokens.weight": w])
        XCTAssertEqual(kept["backbone_model.embed_tokens.embed_audio_tokens.weight"]?.dtype, .uint8)
    }

    func testSegmentsSplitOnNewlinesAndAt600Characters() {
        XCTAssertEqual(BreezeTTSModel.segments("One.\n\n  Two. "), ["One.", "Two."])
        let sentence = String(repeating: "word ", count: 30) + "end. "
        let long = String(repeating: sentence, count: 6)
        let chunks = BreezeTTSModel.splitLongSegment(long)
        XCTAssertGreaterThan(chunks.count, 1)
        XCTAssertTrue(chunks.allSatisfy { $0.count <= 600 && !$0.isEmpty })
        XCTAssertTrue(chunks.dropLast().allSatisfy { $0.hasSuffix("end.") }, "sentence ends stay attached")
        let cjk = String(repeating: "\u{5929}", count: 1500)
        XCTAssertEqual(BreezeTTSModel.splitLongSegment(cjk).map(\.count), [600, 600, 300])
        XCTAssertEqual(BreezeTTSModel.splitLongSegment("short"), ["short"])
    }

    func testSlidingMaskMatchesT5Gemma2Window() {
        // window 4: 2 left (incl. self), 3 right (incl. self)
        let mask = breezeSlidingMask(length: 6, window: 4).asArray(Bool.self)
        let expected: [[Int]] = (0 ..< 6).map { q in (0 ..< 6).map { k in (q - k >= 0 && q - k < 2) || (q - k < 0 && k - q < 3) ? 1 : 0 } }
        XCTAssertEqual(mask.map { $0 ? 1 : 0 }, expected.flatMap { $0 })
        XCTAssertEqual(expected[2], [0, 1, 1, 1, 1, 0])
    }

    func testRepetitionPenaltyIsSignAwareAndDeduplicated() {
        let logits = MLXArray([Float(2), -2, 4, 1]).reshaped(1, 4)
        let out = BreezeTTSModel.repetitionPenalty(logits, history: [0, 1, 1, 9, -1], penalty: 2).asArray(Float.self)
        XCTAssertEqual(out, [1, -4, 4, 1])
        XCTAssertEqual(BreezeTTSModel.repetitionPenalty(logits, history: [0], penalty: 1).asArray(Float.self), [2, -2, 4, 1])
    }

    func testRequestValidationRequiresExactTranscriptForCloning() {
        let reference = (0 ..< 24_000).map { Float(sin(Double($0) / 10)) * 0.1 }
        XCTAssertNoThrow(try BreezeTTSRequest(text: "Hi", referenceAudio: reference, referenceText: "Words").validate())
        XCTAssertThrowsError(try BreezeTTSRequest(text: "Hi", referenceAudio: reference, referenceText: "  ").validate())
        XCTAssertThrowsError(try BreezeTTSRequest(text: "Hi", referenceAudio: reference).validate())
        XCTAssertThrowsError(try BreezeTTSRequest(text: "Hi", referenceAudio: Array(reference.prefix(1000)), referenceText: "W").validate())
        XCTAssertThrowsError(try BreezeTTSRequest(text: "Hi", referenceText: "orphan").validate())
        XCTAssertThrowsError(try BreezeTTSRequest(text: "Hi", speaker: 10).validate())
        XCTAssertThrowsError(try BreezeTTSRequest(text: "   ").validate())
        XCTAssertThrowsError(try BreezeTTSRequest(text: "Hi", cfgScale: .infinity).validate())
        XCTAssertEqual(BreezeTTSRequest(text: "Hi", instruction: "  ").trimmedInstruction, nil)
    }
}
