import Foundation
import MLX
import MLXLMCommon

/// The Breeze TTS 2 wrapper config (`model_type` `breeze_tts`), resolved to
/// the values the MLX port reads. Ported from mlx-audio 0.5.7
/// `tts/models/breeze_tts/config.py` and the module constructors in
/// `breeze_tts.py`. Shapes the two pinned checkpoints do not use (audio
/// embedding projectors, a backbone/audio width mismatch) are refused rather
/// than ported, so an unexpected export fails at load instead of decoding noise.
public struct BreezeTTSConfig: Sendable {
    public struct Quantization: Sendable, Equatable {
        public let groupSize: Int
        public let bits: Int
        public let mode: QuantizationMode
        var asTuple: (groupSize: Int, bits: Int, mode: QuantizationMode) { (groupSize, bits, mode) }
    }

    struct Backbone: Sendable {
        let hiddenSize, layers, intermediateSize, heads, kvHeads, headDim: Int
        let rmsNormEps, ropeTheta: Float
    }

    struct TextEncoder: Sendable {
        let vocabSize, hiddenSize, intermediateSize, layers, heads, kvHeads, headDim: Int
        let rmsNormEps, queryPreAttnScalar: Float
        let slidingWindow, eoiTokenIndex, maxPositionEmbeddings: Int
        let layerTypes: [String]
        /// Per layer type: (base, rope scaling config for `initializeRope`).
        let rope: [String: RopeSpec]
    }

    struct RopeSpec: Sendable {
        let base: Float
        let scaling: [String: StringOrNumber]?
    }

    struct DepthDecoder: Sendable {
        let hiddenSize, layers, intermediateSize, heads, kvHeads, headDim: Int
        let rmsNormEps, ropeTheta: Float
        let ropeScaling: [String: StringOrNumber]?
        let maxPositionEmbeddings: Int
    }

    public let numCodebooks: Int
    /// Wrapper vocabulary per codebook (codec entries plus reserved ids).
    public let vocabSize: Int
    /// Codec entries before Breeze's reserved ids; ids in
    /// `[codecVocabSize, vocabSize)` are masked at every sampling step.
    public let codecVocabSize: Int
    public let codebookEosTokenId: Int
    public let sampleRate: Int
    public let quantization: Quantization?
    let backbone: Backbone
    let text: TextEncoder
    let depth: DepthDecoder

    /// Accepted weight formats: BF16, the mlx-community mxfp4 export (4-bit,
    /// group 32), and the Gust int8 affine depth decoder (8-bit, group 64).
    static let acceptedQuantizations: [Quantization] = [
        Quantization(groupSize: 32, bits: 4, mode: .mxfp4),
        Quantization(groupSize: 64, bits: 8, mode: .affine),
    ]

    public init(data: Data) throws {
        guard let json = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw BreezeTTSError.invalidCheckpoint("config.json is not an object")
        }
        try self.init(json: json)
    }

    init(json: [String: Any]) throws {
        guard json["model_type"] as? String == "breeze_tts" else {
            throw BreezeTTSError.invalidCheckpoint("model_type is not breeze_tts")
        }
        // The audio_* aliases win when present (config.py from_dict).
        numCodebooks = try Self.int(json["audio_num_codebooks"] ?? json["num_codebooks"], "num_codebooks")
        vocabSize = try Self.int(json["audio_vocab_size"] ?? json["vocab_size"], "vocab_size")
        codebookEosTokenId = (json["codebook_eos_token_id"] as? Int) ?? 0
        let codec = json["codec_config"] as? [String: Any] ?? [:]
        codecVocabSize = (codec["codebook_size"] as? Int) ?? (vocabSize - 3)
        sampleRate = (codec["sampling_rate"] as? Int) ?? (codec["output_sample_rate"] as? Int) ?? (json["sample_rate"] as? Int) ?? 24_000
        guard numCodebooks == 16, vocabSize > 0, 0 < codecVocabSize, codecVocabSize <= vocabSize,
              (0 ..< codecVocabSize).contains(codebookEosTokenId), sampleRate == 24_000 else {
            throw BreezeTTSError.invalidCheckpoint("unexpected codebook layout or sample rate")
        }

        quantization = try Self.quantization(json)

        guard let b = json["backbone_config"] as? [String: Any], b["model_type"] as? String == "qwen3" else {
            throw BreezeTTSError.invalidCheckpoint("backbone_config must be qwen3")
        }
        backbone = Backbone(
            hiddenSize: try Self.int(b["hidden_size"], "backbone hidden_size"),
            layers: try Self.int(b["num_hidden_layers"], "backbone num_hidden_layers"),
            intermediateSize: try Self.int(b["intermediate_size"], "backbone intermediate_size"),
            heads: try Self.int(b["num_attention_heads"], "backbone num_attention_heads"),
            kvHeads: try Self.int(b["num_key_value_heads"], "backbone num_key_value_heads"),
            headDim: (b["head_dim"] as? Int) ?? 128,
            rmsNormEps: Self.float(b["rms_norm_eps"]) ?? 1e-6,
            ropeTheta: Self.float(b["rope_theta"]) ?? 1_000_000)
        if let scaling = b["rope_scaling"], !(scaling is NSNull) {
            throw BreezeTTSError.invalidCheckpoint("backbone rope_scaling is not supported")
        }
        let audioEmbed = (json["audio_embed_size"] as? Int) ?? backbone.hiddenSize
        guard audioEmbed == backbone.hiddenSize, backbone.heads % backbone.kvHeads == 0 else {
            throw BreezeTTSError.invalidCheckpoint("audio embedding projector is not supported")
        }

        guard let t = json["text_encoder_config"] as? [String: Any], t["model_type"] as? String == "t5gemma2_text" else {
            throw BreezeTTSError.invalidCheckpoint("text_encoder_config must be t5gemma2_text")
        }
        let textLayers = (t["num_hidden_layers"] as? Int) ?? 26
        let layerTypes: [String]
        if let types = t["layer_types"] as? [String], !types.isEmpty {
            layerTypes = types
        } else {
            let pattern = (t["_sliding_window_pattern"] as? Int) ?? (t["sliding_window_pattern"] as? Int) ?? 6
            guard pattern > 0 else { throw BreezeTTSError.invalidCheckpoint("sliding_window_pattern must be positive") }
            layerTypes = (0 ..< textLayers).map { ($0 + 1) % pattern == 0 ? "full_attention" : "sliding_attention" }
        }
        guard layerTypes.count >= textLayers,
              layerTypes.allSatisfy({ $0 == "full_attention" || $0 == "sliding_attention" }) else {
            throw BreezeTTSError.invalidCheckpoint("text encoder layer_types do not cover every layer")
        }
        // Upstream defaults, then the checkpoint's per-type mapping, then the
        // legacy rope_scaling for full attention (`_normalize_config`).
        var ropeRaw: [String: [String: Any]] = [
            "sliding_attention": ["rope_type": "default", "rope_theta": 10_000.0],
            "full_attention": ["rope_type": "default", "rope_theta": 1_000_000.0],
        ]
        let provided = t["rope_parameters"] as? [String: Any] ?? [:]
        for (key, value) in provided { ropeRaw[key] = value as? [String: Any] ?? [:] }
        if let legacy = t["rope_scaling"] as? [String: Any], provided["full_attention"] == nil {
            ropeRaw["full_attention"]?.merge(legacy) { _, new in new }
        }
        var rope: [String: RopeSpec] = [:]
        for kind in ["sliding_attention", "full_attention"] {
            let raw = ropeRaw[kind] ?? [:]
            let base = Self.float(raw["rope_theta"]) ?? (kind == "sliding_attention" ? 10_000 : 1_000_000)
            let type = raw["rope_type"] as? String ?? "default"
            guard type == "default" || type == "linear" else {
                throw BreezeTTSError.invalidCheckpoint("text encoder rope_type \(type) is not supported")
            }
            // Only linear scaling reaches initialize_rope (breeze_tts.py:330).
            rope[kind] = RopeSpec(base: base, scaling: type == "linear" ? Self.ropeScaling(raw) : nil)
        }
        text = TextEncoder(
            vocabSize: (t["vocab_size"] as? Int) ?? 262_208,
            hiddenSize: (t["hidden_size"] as? Int) ?? 2304,
            intermediateSize: (t["intermediate_size"] as? Int) ?? 9216,
            layers: textLayers,
            heads: (t["num_attention_heads"] as? Int) ?? 8,
            kvHeads: (t["num_key_value_heads"] as? Int) ?? 4,
            headDim: (t["head_dim"] as? Int) ?? 256,
            rmsNormEps: Self.float(t["rms_norm_eps"]) ?? 1e-6,
            queryPreAttnScalar: Self.float(t["query_pre_attn_scalar"]) ?? 256,
            slidingWindow: (t["sliding_window"] as? Int) ?? 4096,
            eoiTokenIndex: (t["eoi_token_index"] as? Int) ?? 256_000,
            maxPositionEmbeddings: (t["max_position_embeddings"] as? Int) ?? 131_072,
            layerTypes: Array(layerTypes.prefix(textLayers)),
            rope: rope)
        guard text.slidingWindow > 0, text.heads % text.kvHeads == 0 else {
            throw BreezeTTSError.invalidCheckpoint("invalid text encoder attention shape")
        }

        let d = json["depth_decoder_config"] as? [String: Any] ?? [:]
        guard (d["vocab_size"] as? Int ?? vocabSize) == vocabSize,
              (d["num_codebooks"] as? Int ?? numCodebooks) == numCodebooks,
              (d["audio_embed_size"] as? Int ?? audioEmbed) == audioEmbed,
              (d["backbone_hidden_size"] as? Int ?? backbone.hiddenSize) == audioEmbed else {
            throw BreezeTTSError.invalidCheckpoint("depth decoder shape does not match the wrapper")
        }
        depth = DepthDecoder(
            hiddenSize: (d["hidden_size"] as? Int) ?? 1024,
            layers: (d["num_hidden_layers"] as? Int) ?? 12,
            intermediateSize: (d["intermediate_size"] as? Int) ?? 8192,
            heads: (d["num_attention_heads"] as? Int) ?? 8,
            kvHeads: (d["num_key_value_heads"] as? Int) ?? 2,
            headDim: (d["head_dim"] as? Int) ?? 128,
            rmsNormEps: Self.float(d["rms_norm_eps"]) ?? 1e-5,
            ropeTheta: Self.float(d["rope_theta"]) ?? 500_000,
            ropeScaling: (d["rope_scaling"] as? [String: Any]).map(Self.ropeScaling),
            maxPositionEmbeddings: (d["max_position_embeddings"] as? Int) ?? 33)
        guard depth.heads % depth.kvHeads == 0, numCodebooks + 1 <= depth.maxPositionEmbeddings else {
            throw BreezeTTSError.invalidCheckpoint("invalid depth decoder attention shape")
        }
    }

    static func quantization(_ json: [String: Any]) throws -> Quantization? {
        var found: Quantization?
        for key in ["quantization", "quantization_config"] {
            guard let block = json[key] as? [String: Any] else { continue }
            guard let bits = block["bits"] as? Int, let group = block["group_size"] as? Int else {
                // Per-layer overrides are not part of either pinned export.
                throw BreezeTTSError.invalidQuantization
            }
            let modeName = block["mode"] as? String ?? "affine"
            let mode: QuantizationMode
            switch modeName {
            case "affine": mode = .affine
            case "mxfp4": mode = .mxfp4
            default: throw BreezeTTSError.invalidQuantization
            }
            let q = Quantization(groupSize: group, bits: bits, mode: mode)
            guard acceptedQuantizations.contains(q), found == nil || found == q else {
                throw BreezeTTSError.invalidQuantization
            }
            found = q
        }
        return found
    }

    static func ropeScaling(_ raw: [String: Any]) -> [String: StringOrNumber] {
        var out: [String: StringOrNumber] = [:]
        for (key, value) in raw {
            if let s = value as? String { out[key] = .string(s) }
            else if let n = value as? NSNumber {
                // JSONSerialization gives NSNumber; keep integers as ints so
                // `asInt()` lookups (original_max_position_embeddings) work.
                if CFNumberIsFloatType(n) { out[key] = .float(n.floatValue) } else { out[key] = .int(n.intValue) }
            }
        }
        return out
    }

    static func int(_ value: Any?, _ name: String) throws -> Int {
        guard let v = value as? Int else { throw BreezeTTSError.invalidCheckpoint("missing \(name)") }
        return v
    }

    static func float(_ value: Any?) -> Float? {
        (value as? NSNumber)?.floatValue
    }
}

public enum BreezeTTSError: LocalizedError, Equatable {
    case invalidCheckpoint(String)
    case invalidQuantization
    case invalidRequest(String)
    case invalidReference

    public var errorDescription: String? {
        switch self {
        case .invalidCheckpoint(let why): return "Choose a complete Breeze TTS 2 checkpoint (\(why))."
        case .invalidQuantization: return "This loader supports Breeze TTS 2 in BF16, MXFP4 4-bit (group 32), or affine 8-bit (group 64)."
        case .invalidRequest(let why): return "Check the Breeze TTS 2 request: \(why)."
        case .invalidReference: return "Cloning requires 1 to 30 seconds of non-silent mono reference audio at 24 kHz and its exact transcript."
        }
    }
}
