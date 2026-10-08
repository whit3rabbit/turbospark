import CTurboSpark
import Foundation

/// Options for opening a streaming STT session.
public struct STTOptions: Encodable, Sendable {
    /// Language code (e.g. "en", "de"), or nil/"auto" for auto-detection.
    public var language: String?

    public init(language: String? = nil) {
        self.language = language
    }
}

/// Progress returned after appending a PCM chunk to an STT stream.
public struct STTAppendProgress: Decodable, Sendable, Equatable {
    public let bufferedSamples: Int
    public let bufferedSeconds: Double

    public init(bufferedSamples: Int, bufferedSeconds: Double) {
        self.bufferedSamples = bufferedSamples
        self.bufferedSeconds = bufferedSeconds
    }
}

/// One transcribed segment with timing and text.
public struct STTSegment: Decodable, Sendable, Equatable {
    public let index: Int
    public let startSeconds: Double
    public let endSeconds: Double
    public let text: String

    public init(index: Int, startSeconds: Double, endSeconds: Double, text: String) {
        self.index = index
        self.startSeconds = startSeconds
        self.endSeconds = endSeconds
        self.text = text
    }
}

/// Full transcription result emitted on stream finish.
public struct STTTranscription: Decodable, Sendable, Equatable {
    public let segments: [STTSegment]
    public let languageDetected: String

    public var text: String {
        segments.map(\.text).joined()
    }

    public init(segments: [STTSegment], languageDetected: String) {
        self.segments = segments
        self.languageDetected = languageDetected
    }
}

/// A resident speech-to-text model loaded into memory.
///
/// Dispatches across Whisper and Qwen3-ASR families based on model configuration.
/// Model closing is gated while any child streams remain open.
public final class TurboSparkSTTModel: @unchecked Sendable {
    private struct Handle: @unchecked Sendable {
        let raw: OpaquePointer
    }

    private let handle: Handle
    private let lock = NSLock()
    private var closed = false

    public init(modelPath: String) throws {
        let path = NSString(string: modelPath).expandingTildeInPath
        var out: OpaquePointer?
        let status = path.withCString { cPath in
            ts_stt_open(cPath, &out)
        }
        guard status == 0, let out else {
            throw TurboSparkError.fromLastError(status)
        }
        self.handle = Handle(raw: out)
    }

    deinit {
        close()
    }

    /// Releases the resident model weights and resources.
    public func close() {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        // Rust refuses while a stream is still open and keeps the model. Latch
        // closed only on success, or deinit would be a no-op and the resident
        // weights would leak for the process lifetime.
        if ts_stt_close(handle.raw) == 0 {
            closed = true
        }
    }

    /// Opens a new streaming audio transcription session.
    public func createStream(options: STTOptions = STTOptions()) throws -> TurboSparkSTTStream {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else {
            throw TurboSparkError(code: .invalidArgument, message: "model is closed")
        }
        let data = try JSONEncoder().encode(options)
        let json = String(decoding: data, as: UTF8.self)
        var out: OpaquePointer?
        let status = json.withCString { opt in
            ts_stt_stream_open(handle.raw, opt, &out)
        }
        guard status == 0, let out else {
            throw TurboSparkError.fromLastError(status)
        }
        return TurboSparkSTTStream(model: self, raw: out)
    }
}

/// One streaming speech transcription session.
///
/// Audio is appended in bounded PCM chunks (16 kHz mono f32 little-endian, up to 2 seconds).
/// On finish, synchronous decoding produces timestamped transcript segments.
public final class TurboSparkSTTStream: @unchecked Sendable {
    private struct Handle: @unchecked Sendable {
        let raw: OpaquePointer
    }

    private let handle: Handle
    private let model: TurboSparkSTTModel
    private let lock = NSLock()
    private var closed = false

    init(model: TurboSparkSTTModel, raw: OpaquePointer) {
        self.model = model
        self.handle = Handle(raw: raw)
    }

    deinit {
        close()
    }

    /// Closes the stream handle and releases its buffered audio.
    public func close() {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        closed = true
        ts_stt_stream_close(handle.raw)
    }

    /// Cancels transcription and discards buffered PCM. Idempotent.
    public func cancel() {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else { return }
        ts_stt_stream_cancel(handle.raw)
    }

    /// Appends a base64-encoded chunk of 16 kHz f32 mono PCM audio.
    public func append(pcmBase64: String) throws -> STTAppendProgress {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else {
            throw TurboSparkError(code: .invalidArgument, message: "stream is closed")
        }
        let json = try takeString { out in
            pcmBase64.withCString { pcm in
                ts_stt_stream_append(handle.raw, pcm, out)
            }
        }
        return try JSONDecoder().decode(STTAppendProgress.self, from: Data(json.utf8))
    }

    /// Appends an array of 16 kHz f32 mono PCM samples.
    public func append(samples: [Float]) throws -> STTAppendProgress {
        let data = samples.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
        let base64 = data.base64EncodedString()
        return try append(pcmBase64: base64)
    }

    /// Transcribes all buffered audio and marks the stream finished.
    public func finish() throws -> STTTranscription {
        lock.lock()
        defer { lock.unlock() }
        guard !closed else {
            throw TurboSparkError(code: .invalidArgument, message: "stream is closed")
        }
        let json = try takeString { out in
            ts_stt_stream_finish(handle.raw, out)
        }
        return try JSONDecoder().decode(STTTranscription.self, from: Data(json.utf8))
    }
}
