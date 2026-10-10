import Foundation

public struct Qwen3TTSCheckpoint: Sendable {
    public enum Variant: String, Sendable { case base, customVoice = "custom_voice", voiceDesign = "voice_design" }
    public let variant: Variant
    public let size: String
    public let speakers: [String]
    public let languages: [String]
    public var supportsStyle: Bool { variant == .voiceDesign || (variant == .customVoice && size == "1b7") }
    public static func languageName(_ language: String) -> String {
        let codes = ["en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean", "de": "german",
                     "fr": "french", "ru": "russian", "pt": "portuguese", "es": "spanish", "it": "italian"]
        return codes[language.lowercased()] ?? language.lowercased()
    }

    public init(data: Data) throws {
        let json = try JSONSerialization.jsonObject(with: data) as? [String: Any]
        for key in ["quantization", "quantization_config"] {
            if let settings = json?[key] as? [String: Any] {
                let blocks = settings["bits"] == nil ? settings.values.compactMap { $0 as? [String: Any] } : [settings]
                for block in blocks {
                    guard let bits = block["bits"] as? Int, [4, 8].contains(bits),
                          block["group_size"] as? Int == 64, (block["mode"] as? String ?? "affine") == "affine" else {
                        throw Qwen3TTSError.invalidQuantization
                    }
                }
            }
        }
        try self.init(config: JSONDecoder().decode(Qwen3TTSModelConfig.self, from: data))
    }

    public static func validateDirectory(_ directory: URL) throws -> Self {
        let fm = FileManager.default
        let descriptor = try Self(data: Data(contentsOf: directory.appendingPathComponent("config.json")))
        let codec = directory.appendingPathComponent("speech_tokenizer")
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: codec.path, isDirectory: &isDirectory), isDirectory.boolValue else {
            throw Qwen3TTSError.invalidCheckpoint
        }
        let codecConfig = try JSONSerialization.jsonObject(with: Data(contentsOf: codec.appendingPathComponent("config.json"))) as? [String: Any]
        guard codecConfig?["model_type"] as? String == "qwen3_tts_tokenizer_12hz",
              codecConfig?["input_sample_rate"] as? Int == 24_000,
              codecConfig?["output_sample_rate"] as? Int == 24_000,
              codecConfig?["decode_upsample_rate"] as? Int == 1920,
              codecConfig?["decoder_config"] is [String: Any],
              descriptor.variant != .base || codecConfig?["encoder_config"] is [String: Any] else {
            throw Qwen3TTSError.invalidCheckpoint
        }
        for folder in [directory, codec] {
            guard try fm.contentsOfDirectory(at: folder, includingPropertiesForKeys: nil).contains(where: { $0.pathExtension == "safetensors" }) else {
                throw Qwen3TTSError.invalidCheckpoint
            }
        }
        guard fm.fileExists(atPath: directory.appendingPathComponent("tokenizer_config.json").path),
              fm.fileExists(atPath: directory.appendingPathComponent("tokenizer.json").path)
                || (fm.fileExists(atPath: directory.appendingPathComponent("vocab.json").path)
                    && fm.fileExists(atPath: directory.appendingPathComponent("merges.txt").path)) else {
            throw Qwen3TTSError.invalidCheckpoint
        }
        return descriptor
    }

    init(config: Qwen3TTSModelConfig) throws {
        guard config.modelType == "qwen3_tts", let variant = Variant(rawValue: config.ttsModelType),
              ["0b6", "1b7"].contains(config.ttsModelSize),
              config.tokenizerType == "qwen3_tts_tokenizer_12hz", config.sampleRate == 24_000,
              let talker = config.talkerConfig, talker.numCodeGroups == 16,
              talker.numHiddenLayers == 28, talker.hiddenSize == (config.ttsModelSize == "0b6" ? 1024 : 2048),
              talker.headDim == 128, talker.numKeyValueHeads > 0,
              talker.numAttentionHeads % talker.numKeyValueHeads == 0,
              variant != .voiceDesign || config.ttsModelSize == "1b7" else {
            throw Qwen3TTSError.invalidCheckpoint
        }
        if let quantization = config.quantization {
            guard [4, 8].contains(quantization.bits), quantization.groupSize == 64,
                  quantization.mode == .affine else { throw Qwen3TTSError.invalidQuantization }
        }
        self.variant = variant; size = config.ttsModelSize
        speakers = (talker.spkId ?? [:]).keys.sorted()
        languages = ["auto"] + (talker.codecLanguageId ?? [:]).keys.sorted()
    }

    public func validate(text: String, voice: String?, reference: [Float]?, referenceText: String?, language: String) throws {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, text.count <= 6000,
              languages.contains(Self.languageName(language)) else { throw Qwen3TTSError.invalidRequest }
        switch variant {
        case .base:
            guard let reference, (24_000...720_000).contains(reference.count),
                  reference.allSatisfy(\.isFinite), reference.contains(where: { abs($0) > 1e-5 }), voice == nil else {
                throw Qwen3TTSError.invalidReference
            }
        case .voiceDesign:
            guard reference == nil, let voice, !voice.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
                throw Qwen3TTSError.invalidRequest
            }
        case .customVoice:
            let parts = (voice ?? "").split(separator: ",", maxSplits: 1, omittingEmptySubsequences: false)
            let speaker = parts.first.map { $0.trimmingCharacters(in: .whitespacesAndNewlines).lowercased() } ?? ""
            guard reference == nil, speakers.contains(speaker), supportsStyle || parts.count == 1 else {
                throw Qwen3TTSError.invalidRequest
            }
        }
    }
}

public enum Qwen3TTSError: LocalizedError, Equatable {
    case invalidCheckpoint, invalidQuantization, invalidRequest, invalidReference, incompatibleProfile
    public var errorDescription: String? {
        switch self {
        case .invalidCheckpoint: return "Choose a Qwen3-TTS 12Hz Base, CustomVoice, or 1.7B VoiceDesign checkpoint with its complete talker configuration."
        case .invalidQuantization: return "This loader supports BF16 and MLX affine 4-bit or 8-bit weights with group size 64."
        case .invalidRequest: return "Check the text, language, speaker, and voice description against the selected Qwen3-TTS checkpoint. Style instructions require CustomVoice 1.7B or VoiceDesign."
        case .invalidReference: return "Cloning requires 1 to 30 seconds of non-silent mono reference audio at 24 kHz."
        case .incompatibleProfile: return "Choose a local checkpoint with the same Qwen3-TTS role and model size as the selected model. Its precision may differ."
        }
    }
}
