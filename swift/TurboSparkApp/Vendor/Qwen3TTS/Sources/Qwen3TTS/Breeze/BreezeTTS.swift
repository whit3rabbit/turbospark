import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

/// One Breeze TTS 2 request. The mode follows the fields: a reference clones
/// (with an instruction it directs), an instruction alone designs a voice,
/// and neither speaks with the preset speaker tag.
public struct BreezeTTSRequest: Sendable {
    public var text: String
    /// Preset speaker tag `[S0]`..`[S9]`.
    public var speaker: Int
    public var instruction: String?
    /// Mono 24 kHz reference, 1 to 30 s, with its exact transcript.
    public var referenceAudio: [Float]?
    public var referenceText: String?
    /// Classifier-free guidance; it runs only with an instruction and a
    /// scale other than 1 (`breeze_tts.py:1068`). Upstream recommends 4.
    public var cfgScale: Float?
    public var maxFramesPerSegment: Int
    public var temperature: Float
    public var topP: Float
    public var topK: Int
    public var repetitionPenalty: Float
    public var seed: UInt64?

    public init(text: String, speaker: Int = 0, instruction: String? = nil, referenceAudio: [Float]? = nil,
                referenceText: String? = nil, cfgScale: Float? = nil, maxFramesPerSegment: Int = 750,
                temperature: Float = 0.9, topP: Float = 1.0, topK: Int = 50, repetitionPenalty: Float = 1.0,
                seed: UInt64? = nil) {
        self.text = text; self.speaker = speaker; self.instruction = instruction
        self.referenceAudio = referenceAudio; self.referenceText = referenceText; self.cfgScale = cfgScale
        self.maxFramesPerSegment = maxFramesPerSegment; self.temperature = temperature; self.topP = topP
        self.topK = topK; self.repetitionPenalty = repetitionPenalty; self.seed = seed
    }

    public var trimmedInstruction: String? {
        let value = instruction?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return value.isEmpty ? nil : value
    }

    public func validate() throws {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, text.count <= 6000 else { throw BreezeTTSError.invalidRequest("text must be 1 to 6000 characters") }
        guard (0 ... 9).contains(speaker) else { throw BreezeTTSError.invalidRequest("speaker must be S0 to S9") }
        guard maxFramesPerSegment > 0, topK >= 0, temperature >= 0, temperature.isFinite,
              (0 ... 1).contains(topP), repetitionPenalty > 0, repetitionPenalty.isFinite,
              cfgScale.map(\.isFinite) ?? true else {
            throw BreezeTTSError.invalidRequest("sampling settings are out of range")
        }
        if let referenceAudio {
            guard (24_000 ... 720_000).contains(referenceAudio.count), referenceAudio.allSatisfy(\.isFinite),
                  referenceAudio.contains(where: { abs($0) > 1e-5 }) else { throw BreezeTTSError.invalidReference }
            // Breeze speaks a mismatched transcript; it needs the exact words.
            guard !(referenceText ?? "").trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw BreezeTTSError.invalidReference
            }
        } else if referenceText != nil {
            throw BreezeTTSError.invalidRequest("a reference transcript needs reference audio")
        }
    }
}

public struct BreezeTTSResult: Sendable {
    public let samples: [Float]
    public let sampleRate: Int
    public let frames: Int
    /// A segment reached `maxFramesPerSegment` before the model's end class.
    public let hitFrameLimit: Bool
    /// Every generated frame (`numCodebooks` codes), for parity checks.
    public let codes: [[Int]]
}

// Unchecked like Qwen3TTSModel: the app hands one resident model to one
// producer at a time (Qwen3TTSSpeechJob), never to two at once.
public final class BreezeTTSModel: Module, @unchecked Sendable {
    public let config: BreezeTTSConfig
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    @ModuleInfo(key: "backbone_model") var backbone: BreezeBackbone
    @ModuleInfo(key: "depth_decoder") var depthDecoder: BreezeDepthDecoder
    @ModuleInfo(key: "text_encoder") var textEncoder: BreezeTextEncoder
    @ModuleInfo(key: "text_encoder_proj") var textEncoderProj: Linear
    var tokenizer: Tokenizer?
    var codec: Qwen3TTSSpeechTokenizer?

    init(config: BreezeTTSConfig) {
        self.config = config
        // One extra backbone class is the end-of-speech id `vocabSize`.
        _lmHead.wrappedValue = Linear(config.backbone.hiddenSize, config.vocabSize + 1, bias: false)
        _backbone.wrappedValue = BreezeBackbone(config)
        _depthDecoder.wrappedValue = BreezeDepthDecoder(config)
        _textEncoder.wrappedValue = BreezeTextEncoder(config.text)
        _textEncoderProj.wrappedValue = Linear(config.text.hiddenSize, config.backbone.hiddenSize, bias: false)
    }

    public var sampleRate: Int { config.sampleRate }
    var numCodebooks: Int { config.numCodebooks }
    var vocabSize: Int { config.vocabSize }

    // MARK: Loading

    /// Checks the folder layout and configs without reading weights.
    @discardableResult
    public static func validateDirectory(_ dir: URL) throws -> BreezeTTSConfig {
        let fm = FileManager.default
        let config = try BreezeTTSConfig(data: Data(contentsOf: dir.appendingPathComponent("config.json")))
        let codec = dir.appendingPathComponent("audio_tokenizer")
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: codec.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw BreezeTTSError.invalidCheckpoint("missing audio_tokenizer folder")
        }
        let codecConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: codec.appendingPathComponent("config.json"))) as? [String: Any]
        // Cloning needs the encoder, so it is required here, unlike Qwen3-TTS
        // CustomVoice checkpoints.
        guard codecConfig?["model_type"] as? String == "qwen3_tts_tokenizer_12hz",
              codecConfig?["input_sample_rate"] as? Int == 24_000,
              codecConfig?["output_sample_rate"] as? Int == 24_000,
              codecConfig?["decode_upsample_rate"] as? Int == 1920,
              codecConfig?["encoder_valid_num_quantizers"] as? Int == config.numCodebooks,
              codecConfig?["decoder_config"] is [String: Any],
              codecConfig?["encoder_config"] is [String: Any] else {
            throw BreezeTTSError.invalidCheckpoint("audio_tokenizer is not the Qwen3-TTS 12 Hz codec")
        }
        for folder in [dir, codec] {
            guard try fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil).contains(where: { $0.pathExtension == "safetensors" }) else {
                throw BreezeTTSError.invalidCheckpoint("missing safetensors weights")
            }
        }
        for name in ["tokenizer.json", "tokenizer_config.json"] {
            guard fm.fileExists(atPath: dir.appendingPathComponent(name).path) else {
                throw BreezeTTSError.invalidCheckpoint("missing \(name)")
            }
        }
        return config
    }

    /// Keeps main-model tensors (`Model.sanitize`): drops codec, runtime flag,
    /// and rotary tensors, and the unused `embed_text_tokens` table, which
    /// generation never reads. The backbone audio table is tied to the depth
    /// table; a checkpoint that stores only the depth copy gets the backbone
    /// entry (weight, scales, biases) from it.
    static func sanitize(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        var out: [String: MLXArray] = [:]
        for (key, value) in weights {
            if key.hasPrefix("codec_model.") || key.hasSuffix(".initialized") || key.contains("rotary_emb.inv_freq")
                || key.hasPrefix("embed_text_tokens.") {
                continue
            }
            out[key] = value
        }
        let depth = "depth_decoder.model.embed_tokens"
        let backbone = "backbone_model.embed_tokens.embed_audio_tokens"
        if out["\(backbone).weight"] == nil {
            for suffix in ["weight", "scales", "biases"] {
                if let value = out["\(depth).\(suffix)"] { out["\(backbone).\(suffix)"] = value }
            }
        }
        return out
    }

    /// `quantizeSingle` with one fix: mlx-swift 0.30.6
    /// `QuantizedLinear.init(weight:bias:groupSize:bits:mode:)` packs with the
    /// default affine mode whatever `mode` says, which gives an mxfp4 layer an
    /// affine `biases` array. Packing here with the requested mode and using
    /// the array initializer keeps mode and parameters consistent.
    static func quantizeLayer(_ layer: Module, groupSize: Int, bits: Int, mode: QuantizationMode) -> Module? {
        if let linear = layer as? Linear, !(layer is QuantizedLinear) {
            let (wq, scales, biases) = MLX.quantized(linear.weight, groupSize: groupSize, bits: bits, mode: mode)
            let quantized = QuantizedLinear(weight: wq, bias: linear.bias, scales: scales, biases: biases,
                                            groupSize: groupSize, bits: bits, mode: mode)
            quantized.freeze()
            return quantized
        }
        return quantizeSingle(layer: layer, groupSize: groupSize, bits: bits, mode: mode)
    }

    public static func fromModelDirectory(_ dir: URL) async throws -> BreezeTTSModel {
        try Task.checkCancellation()
        let config = try validateDirectory(dir)
        let model = BreezeTTSModel(config: config)

        var all = [String: MLXArray]()
        let fm = FileManager.default
        for file in try fm.contentsOfDirectory(at: dir, includingPropertiesForKeys: nil) where file.pathExtension == "safetensors" {
            try Task.checkCancellation()
            all.merge(try MLX.loadArrays(url: file)) { _, new in new }
        }
        let weights = sanitize(all)
        all.removeAll()
        // Quantize exactly the layers the checkpoint stores packed. Gust keeps
        // everything but the depth decoder in BF16, so this is per tensor.
        if let quantization = config.quantization {
            quantize(model: model, filter: { path, _ in
                weights["\(path).scales"] != nil ? quantization.asTuple : nil
            }, apply: quantizeLayer)
        }
        try model.update(parameters: ModuleParameters.unflattened(weights.map { ($0.key, $0.value) }), verify: .all)
        eval(model.parameters())

        model.tokenizer = try await loadTokenizer(dir)
        try Task.checkCancellation()
        model.codec = try Qwen3TTSModel.loadSpeechTokenizer(path: dir.appendingPathComponent("audio_tokenizer"))
        guard model.codec?.hasEncoder == true else {
            throw BreezeTTSError.invalidCheckpoint("audio_tokenizer has no encoder")
        }
        return model
    }

    /// Builds the tokenizer in scratch storage so the model folder stays
    /// untouched. `model_max_length` is dropped as the reference does
    /// (`post_load_hook`); its 1e30 value is not a usable length.
    static func loadTokenizer(_ dir: URL) async throws -> Tokenizer {
        let fm = FileManager.default
        let scratch = fm.temporaryDirectory.appendingPathComponent("turbospark-breeze-tokenizer-" + UUID().uuidString)
        try fm.createDirectory(at: scratch, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        defer { try? fm.removeItem(at: scratch) }
        try fm.copyItem(at: dir.appendingPathComponent("tokenizer.json"), to: scratch.appendingPathComponent("tokenizer.json"))
        var tokenizerConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: dir.appendingPathComponent("tokenizer_config.json"))) as? [String: Any] ?? [:]
        tokenizerConfig.removeValue(forKey: "model_max_length")
        try JSONSerialization.data(withJSONObject: tokenizerConfig).write(to: scratch.appendingPathComponent("tokenizer_config.json"))
        let special = dir.appendingPathComponent("special_tokens_map.json")
        if fm.fileExists(atPath: special.path) {
            try fm.copyItem(at: special, to: scratch.appendingPathComponent("special_tokens_map.json"))
        }
        return try await AutoTokenizer.from(modelFolder: scratch)
    }

    // MARK: Prompt

    func textIds(_ text: String) throws -> MLXArray {
        guard let tokenizer else { throw BreezeTTSError.invalidCheckpoint("tokenizer not loaded") }
        let ids = tokenizer.encode(text: text, addSpecialTokens: true)
        return MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
    }

    func embedText(_ text: String) throws -> MLXArray {
        textEncoderProj(textEncoder(try textIds(text)))
    }

    /// Reference codes `[1, T, K]` from mono 24 kHz audio.
    func encodeReference(_ audio: [Float]) throws -> MLXArray {
        guard let codec else { throw BreezeTTSError.invalidCheckpoint("audio tokenizer not loaded") }
        let codes = codec.encode(MLXArray(audio).reshaped(1, 1, audio.count))
        return codes.transposed(0, 2, 1).asType(.int32)
    }

    /// `_prompt_embeddings`: an optional `{speaker}{ref_text}` turn with its
    /// codes and a codebook-EOS frame, then the target turn.
    func promptEmbeddings(text: String, speakerTag: String, instruction: String?, reference: (codes: MLXArray, text: String)?) throws -> MLXArray {
        var parts: [MLXArray] = []
        if let reference {
            parts.append(try embedText("\(speakerTag)\(reference.text)"))
            parts.append(backbone.embedTokens(reference.codes))
            let eos = MLXArray.full([1, 1, numCodebooks], values: MLXArray(Int32(config.codebookEosTokenId)))
            parts.append(backbone.embedTokens(eos))
        }
        let target = instruction.map { "\(speakerTag)<ins_bos>\($0)<ins_eos>\(text)" } ?? "\(speakerTag)\(text)"
        parts.append(try embedText(target))
        let dtype = parts[0].dtype
        return concatenated(parts.map { $0.asType(dtype) }, axis: 1)
    }

    // MARK: Sampling

    /// Codec ids `[codecVocabSize, vocabSize)` are reserved (padding and
    /// control) and masked in every distribution; the backbone EOS column at
    /// `vocabSize` survives this mask.
    func maskReserved(_ logits: MLXArray) -> MLXArray {
        let width = logits.dim(-1)
        let start = config.codecVocabSize, end = min(vocabSize, width)
        guard start < end else { return logits }
        var mask = [Float](repeating: 0, count: width)
        for i in start ..< end { mask[i] = -Float.infinity }
        // In the logits dtype: a float32 mask would promote bf16 logits and
        // change near-tie argmax picks against the reference.
        return logits + MLXArray(mask).asType(logits.dtype)
    }

    static func repetitionPenalty(_ logits: MLXArray, history: [Int], penalty: Float) -> MLXArray {
        guard penalty != 1, !history.isEmpty else { return logits }
        let ids = Array(Set(history.filter { $0 >= 0 && $0 < logits.dim(-1) })).sorted()
        guard !ids.isEmpty else { return logits }
        let index = MLXArray(ids.map(Int32.init)).reshaped(1, ids.count)
        let selected = takeAlong(logits, index, axis: -1)
        let penalized = MLX.where(selected .< 0, selected * penalty, selected / penalty)
        return putAlong(logits, index, values: penalized, axis: -1)
    }

    /// `_sample` over `make_sampler`: reserved ids masked, slice to the
    /// valid width, log-softmax, then top-p, top-k, and a temperature draw
    /// (argmax at temperature 0). Stays in the logits dtype (bf16 for the
    /// pinned checkpoints) like the reference: greedy picks on near ties
    /// follow bf16 rounding, and float32 diverged within a few frames.
    func sample(_ logits: MLXArray, temperature: Float, topP: Float, topK: Int, allowEOS: Bool) -> Int {
        let valid = vocabSize + (allowEOS ? 1 : 0)
        let sliced = maskReserved(logits)[0..., ..<valid]
        var logprobs = sliced - logSumExp(sliced, axis: -1, keepDims: true)
        if temperature == 0 { return argMax(logprobs, axis: -1).item(Int.self) }
        if topP > 0 && topP < 1 {
            // HF nucleus: keep the smallest high-probability set whose mass
            // exceeds top_p (ascending cumulative mass above 1 - top_p).
            let order = argSort(logprobs, axis: -1)
            let sortedProbs = takeAlong(exp(logprobs), order, axis: -1)
            let keep = cumsum(sortedProbs, axis: -1) .> (1 - topP)
            let sortedLog = MLX.where(keep, takeAlong(logprobs, order, axis: -1), MLXArray(-Float.infinity).asType(logprobs.dtype))
            logprobs = putAlong(logprobs, order, values: sortedLog, axis: -1)
        }
        var k = topK > 0 ? min(topK, valid) : 0
        if k == valid { k = 0 }
        if k > 0 {
            let drop = argPartition(-logprobs, kth: k - 1, axis: -1)[0..., k...]
            logprobs = putAlong(logprobs, drop, values: MLXArray(-Float.infinity).asType(logprobs.dtype), axis: -1)
        }
        return categorical(logprobs * (1 / temperature)).item(Int.self)
    }

    static func combine(cond: MLXArray, uncond: MLXArray?, scale: Float) -> MLXArray {
        guard let uncond else { return cond }
        return uncond + scale * (cond - uncond)
    }

    func depthTokens(first: Int, cond: MLXArray, uncond: MLXArray?, scale: Float, request: BreezeTTSRequest) -> [Int] {
        var tokens = [0, first]
        for _ in 0 ..< (numCodebooks - 1) {
            let ids = MLXArray(tokens.map(Int32.init)).reshaped(1, tokens.count)
            let condLogits = depthDecoder.nextLogits(ids, backboneHidden: cond)
            let uncondLogits = uncond.map { depthDecoder.nextLogits(ids, backboneHidden: $0) }
            let logits = maskReserved(Self.combine(cond: condLogits, uncond: uncondLogits, scale: scale))
            tokens.append(sample(logits, temperature: request.temperature, topP: request.topP, topK: request.topK, allowEOS: false))
        }
        return Array(tokens.dropFirst())
    }

    // MARK: Generation

    /// Generates one request, segment by segment. `onAudio` receives decoded
    /// chunks as they stream (about 2 s each); `onFrame` reports frames per
    /// segment. Cancellation is checked every frame.
    public func generate(_ request: BreezeTTSRequest,
                         onFrame: ((_ frames: Int, _ segment: Int, _ segments: Int) -> Void)? = nil,
                         onAudio: (([Float]) -> Void)? = nil) throws -> BreezeTTSResult {
        try request.validate()
        guard let codec else { throw BreezeTTSError.invalidCheckpoint("audio tokenizer not loaded") }
        if let seed = request.seed { MLXRandom.seed(seed) }
        let speakerTag = "[S\(request.speaker)]"
        let instruction = request.trimmedInstruction
        let reference = try request.referenceAudio.map { audio in
            (codes: try encodeReference(audio), text: request.referenceText!.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        let useCFG = instruction != nil && request.cfgScale != nil && request.cfgScale != 1
        let scale = request.cfgScale ?? 1
        let segments = Self.segments(request.text)
        let chunkFrames = max(1, Int(2.0 * Double(sampleRate) / Double(codec.decodeUpsampleRate)))

        var samples: [Float] = []
        var allCodes: [[Int]] = []
        var hitLimit = false
        for (index, segmentText) in segments.enumerated() {
            try Task.checkCancellation()
            let condPrompt = try promptEmbeddings(text: segmentText, speakerTag: speakerTag, instruction: instruction, reference: reference)
            let uncondPrompt = useCFG ? try promptEmbeddings(text: segmentText, speakerTag: speakerTag, instruction: nil, reference: reference) : nil
            let condCache = backbone.makeCache()
            let uncondCache = useCFG ? backbone.makeCache() : []
            var cond = backbone(embeddings: condPrompt, cache: condCache)[0..., -1, 0...]
            var uncond = uncondPrompt.map { backbone(embeddings: $0, cache: uncondCache)[0..., -1, 0...] }

            var frames: [[Int]] = []
            var pending: [[Int]] = []
            codec.decoder.resetStreamingState()
            var segmentAudio: [Float] = []
            var reachedEOS = false
            for _ in 0 ..< request.maxFramesPerSegment {
                try Task.checkCancellation()
                var logits = Self.combine(cond: lmHead(cond), uncond: uncond.map { lmHead($0) }, scale: scale)
                logits = Self.repetitionPenalty(logits, history: frames.map { $0[0] }, penalty: request.repetitionPenalty)
                let first = sample(maskReserved(logits), temperature: request.temperature, topP: request.topP, topK: request.topK, allowEOS: true)
                if first == vocabSize { reachedEOS = true; break }
                let frame = depthTokens(first: first, cond: cond, uncond: uncond, scale: scale, request: request)
                frames.append(frame)
                pending.append(frame)
                onFrame?(frames.count, index, segments.count)
                if onAudio != nil, pending.count >= chunkFrames {
                    let chunk = streamDecode(pending, codec: codec)
                    pending.removeAll()
                    segmentAudio += chunk
                    onAudio?(chunk)
                }
                let codes = MLXArray(frame.map(Int32.init)).reshaped(1, 1, numCodebooks)
                cond = backbone(embeddings: backbone.embedTokens(codes), cache: condCache)[0..., -1, 0...]
                uncond = useCFG ? backbone(embeddings: backbone.embedTokens(codes), cache: uncondCache)[0..., -1, 0...] : nil
            }
            hitLimit = hitLimit || !reachedEOS
            allCodes += frames
            guard !frames.isEmpty else { continue }
            if onAudio != nil {
                if !pending.isEmpty {
                    let chunk = streamDecode(pending, codec: codec)
                    segmentAudio += chunk
                    onAudio?(chunk)
                }
                codec.decoder.resetStreamingState()
            } else {
                let codes = MLXArray(frames.flatMap { $0 }.map(Int32.init)).reshaped(1, frames.count, numCodebooks)
                let (audio, lengths) = try codec.decode(codes)
                eval(audio, lengths)
                let valid = max(0, min(audio.dim(-1), lengths.reshaped(-1)[0].item(Int.self)))
                segmentAudio = audio.reshaped(-1)[..<valid].asArray(Float.self)
            }
            samples += segmentAudio
        }
        return BreezeTTSResult(samples: samples, sampleRate: sampleRate, frames: allCodes.count, hitFrameLimit: hitLimit, codes: allCodes)
    }

    func streamDecode(_ frames: [[Int]], codec: Qwen3TTSSpeechTokenizer) -> [Float] {
        let codes = MLXArray(frames.flatMap { $0 }.map(Int32.init)).reshaped(1, frames.count, numCodebooks)
        let audio = codec.decoder.streamingStep(codes.transposed(0, 2, 1))
        eval(audio)
        return audio.reshaped(-1).asArray(Float.self)
    }

    // MARK: Text segmentation

    /// Newline segments, each split to at most 600 characters
    /// (`generate` with `split_pattern="\n"` and `_split_long_segment`).
    static func segments(_ text: String) -> [String] {
        let raw = text.components(separatedBy: "\n").map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }.filter { !$0.isEmpty }
        let segments = raw.flatMap { splitLongSegment($0) }
        return segments.isEmpty ? [text] : segments
    }

    static func splitLongSegment(_ text: String, maxChars: Int = 600) -> [String] {
        guard text.count > maxChars else { return [text] }
        // Sentence ends stay attached to the preceding sentence; full-width
        // terminators cover Chinese text without spaces.
        var sentences: [String] = []
        var current = ""
        for ch in text {
            current.append(ch)
            if ".!?\u{3002}\u{FF01}\u{FF1F}".contains(ch) { sentences.append(current); current = "" }
        }
        if !current.isEmpty { sentences.append(current) }

        var chunks: [String] = []
        var buffer = ""
        for var sentence in sentences {
            while !sentence.isEmpty {
                if !buffer.isEmpty && buffer.count + sentence.count > maxChars {
                    chunks.append(buffer.trimmingCharacters(in: .whitespacesAndNewlines))
                    buffer = ""
                    sentence = String(sentence.drop(while: \.isWhitespace))
                    continue
                }
                let available = maxChars - buffer.count
                if sentence.count <= available { buffer += sentence; break }
                // Prefer the last whitespace in the second half of the window,
                // else hard split (unspaced CJK text).
                let window = Array(sentence.prefix(available + 1))
                var splitAt = available
                if let ws = window.lastIndex(where: \.isWhitespace) {
                    var start = ws
                    while start > 0 && window[start - 1].isWhitespace { start -= 1 }
                    if start >= available / 2 { splitAt = start }
                }
                let chunk = (buffer + String(sentence.prefix(splitAt))).trimmingCharacters(in: .whitespacesAndNewlines)
                if !chunk.isEmpty { chunks.append(chunk) }
                buffer = ""
                sentence = String(sentence.dropFirst(splitAt).drop(while: \.isWhitespace))
            }
        }
        let tail = buffer.trimmingCharacters(in: .whitespacesAndNewlines)
        if !tail.isEmpty { chunks.append(tail) }
        return chunks
    }
}
