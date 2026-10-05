import CTurboSpark
import Foundation

/// Whether one engine audio task can run, and if not, why.
public struct AudioTaskStatus: Codable, Sendable, Equatable {
    public let active: Bool
    public let reason: String?

    public init(active: Bool, reason: String?) {
        self.active = active
        self.reason = reason
    }
}

/// The engine's audio capability table (`ts_audio_capabilities_json`).
public struct AudioEngineCapabilities: Codable, Sendable, Equatable {
    /// File extensions decode, peaks and convert accept.
    public let decodeExtensions: [String]
    /// Audio extensions refused with a "convert it" sentence.
    public let refusedExtensions: [String]
    public let speechToText: AudioTaskStatus
    public let textToSpeech: AudioTaskStatus
    public let music: AudioTaskStatus
}

/// What `ts_audio_probe_json` read from a file.
public struct AudioProbe: Codable, Sendable, Equatable {
    public let sampleRate: Int
    public let channels: Int
    public let frames: Int
    public let durationSeconds: Double
    public let codec: String
}

/// Waveform peaks for a whole file.
public struct AudioPeaks: Codable, Sendable, Equatable {
    /// One value per bucket in 0...1, loudest bucket = 1.
    public let peaks: [Float]
    public let durationSeconds: Double
    public let sampleRate: Int
    public let channels: Int
}

/// What a conversion wrote.
public struct AudioConversionReport: Codable, Sendable, Equatable {
    public let sampleRate: Int
    public let channels: Int
    public let frames: Int
    public let durationSeconds: Double
}

/// WAV sample encoding for `TurboSparkAudio.convert`.
public enum AudioSampleFormat: String, Codable, Sendable {
    case int16
    case float32
}

/// Options for `TurboSparkAudio.convert`. Nil keeps the source's value.
public struct AudioConvertOptions: Codable, Sendable, Equatable {
    public var sampleRate: Int?
    /// 1 downmixes; otherwise must equal the source's count.
    public var channels: Int?
    public var sampleFormat: AudioSampleFormat
    public var startSeconds: Double?
    public var endSeconds: Double?

    public init(
        sampleRate: Int? = nil, channels: Int? = nil,
        sampleFormat: AudioSampleFormat = .int16,
        startSeconds: Double? = nil, endSeconds: Double? = nil
    ) {
        self.sampleRate = sampleRate
        self.channels = channels
        self.sampleFormat = sampleFormat
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
    }

    /// 16 kHz mono 16-bit: the input shape speech models read.
    public static let speech = AudioConvertOptions(sampleRate: 16_000, channels: 1, sampleFormat: .int16)
}

/// One timed span of a transcript.
public struct AudioTranscriptSegment: Codable, Sendable, Equatable {
    public let startSeconds: Double
    public let endSeconds: Double
    public let text: String
}

/// A finished engine transcription.
public struct AudioTranscript: Codable, Sendable, Equatable {
    public let text: String
    public let language: String?
    public let segments: [AudioTranscriptSegment]
}

/// What an engine synthesis wrote.
public struct AudioSynthesisReport: Codable, Sendable, Equatable {
    public let sampleRate: Int
    public let frames: Int
    public let durationSeconds: Double
}

/// The engine's stateless audio calls. Every one blocks for its whole run
/// (a peaks pass streams the full file), so call them off the main actor.
/// Synchronous on purpose: no async overload with the same signature, which
/// would make an `async` caller resolve back into itself.
public enum TurboSparkAudio {
    public static func capabilities() throws -> AudioEngineCapabilities {
        let json = try takeString { out in ts_audio_capabilities_json(out) }
        return try decode(AudioEngineCapabilities.self, from: json)
    }

    public static func probe(_ url: URL) throws -> AudioProbe {
        let json = try takeString { out in
            url.path.withCString { ts_audio_probe_json($0, out) }
        }
        return try decode(AudioProbe.self, from: json)
    }

    public static func peaks(_ url: URL, buckets: Int) throws -> AudioPeaks {
        let json = try takeString { out in
            url.path.withCString { ts_audio_peaks_json($0, max(0, buckets), out) }
        }
        return try decode(AudioPeaks.self, from: json)
    }

    /// Converts any decodable file to WAV at `destination`, overwriting it.
    @discardableResult
    public static func convert(
        _ source: URL, to destination: URL, options: AudioConvertOptions = AudioConvertOptions()
    ) throws -> AudioConversionReport {
        let optionsJSON = try encodeJSON(options)
        let json = try takeString { out in
            source.path.withCString { src in
                destination.path.withCString { dst in
                    optionsJSON.withCString { ts_audio_convert_json(src, dst, $0, out) }
                }
            }
        }
        return try decode(AudioConversionReport.self, from: json)
    }

    /// Meter level in 0...1 over -50...0 dBFS. Real-time safe: a capture
    /// tap may call this per buffer.
    public static func displayLevel(_ samples: UnsafePointer<Float>, count: Int) -> Float {
        ts_audio_display_level(samples, max(0, count))
    }

    static func encodeJSON<T: Encodable>(_ value: T) throws -> String {
        let data = try JSONEncoder().encode(value)
        guard let json = String(data: data, encoding: .utf8) else {
            throw TurboSparkError(code: .json, message: "failed to serialize options to JSON")
        }
        return json
    }
}

/// An engine audio model (speech-to-text or text-to-speech).
///
/// Every open is refused today (`AudioEngineCapabilities.speechToText.reason`
/// says why); the type exists so the app's engine-first routing is already
/// written against the real ABI. Calls are serialized on a private queue,
/// mirroring `TurboSparkImageSession`.
public final class TurboSparkAudioSession: @unchecked Sendable {
    private struct Handle: @unchecked Sendable {
        let raw: OpaquePointer
    }

    private let handle: Handle
    private let queue: DispatchQueue

    public init(modelPath: String) async throws {
        let modelPath = NSString(string: modelPath).expandingTildeInPath
        let queue = DispatchQueue(label: "com.turbospark.audio-session", qos: .userInitiated)
        let raw: OpaquePointer = try await withCheckedThrowingContinuation { continuation in
            queue.async {
                var out: OpaquePointer?
                let status = modelPath.withCString { ts_audio_session_open($0, &out) }
                guard status == 0, let out else {
                    continuation.resume(throwing: TurboSparkError.fromLastError(status))
                    return
                }
                continuation.resume(returning: out)
            }
        }
        handle = Handle(raw: raw)
        self.queue = queue
    }

    deinit {
        let raw = handle.raw
        queue.sync { ts_audio_session_close(raw) }
    }

    public func transcribe(_ audio: URL, language: String? = nil) async throws -> AudioTranscript {
        struct Options: Encodable { let language: String?; let timestamps: Bool }
        let optionsJSON = try TurboSparkAudio.encodeJSON(Options(language: language, timestamps: true))
        let handle = self.handle
        return try await run { () throws -> AudioTranscript in
            let json = try takeString { out in
                audio.path.withCString { path in
                    optionsJSON.withCString { ts_audio_transcribe_json(handle.raw, path, $0, out) }
                }
            }
            return try decode(AudioTranscript.self, from: json)
        }
    }

    public func synthesize(
        _ text: String, to destination: URL, voice: String? = nil, rate: Float? = nil
    ) async throws -> AudioSynthesisReport {
        struct Options: Encodable { let voice: String?; let rate: Float? }
        let optionsJSON = try TurboSparkAudio.encodeJSON(Options(voice: voice, rate: rate))
        let handle = self.handle
        return try await run { () throws -> AudioSynthesisReport in
            let json = try takeString { out in
                text.withCString { textPtr in
                    optionsJSON.withCString { opts in
                        destination.path.withCString { dst in
                            ts_audio_synthesize_json(handle.raw, textPtr, opts, dst, out)
                        }
                    }
                }
            }
            return try decode(AudioSynthesisReport.self, from: json)
        }
    }

    private func run<T: Sendable>(_ body: @escaping @Sendable () throws -> T) async throws -> T {
        try await withCheckedThrowingContinuation { continuation in
            queue.async {
                continuation.resume(with: Result { try body() })
            }
        }
    }
}
