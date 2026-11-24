import CTurboSpark
import Foundation

/// What one push released: the isolated voice that became final, and timing.
public struct AudioEnhancePush: Sendable, Equatable {
    /// 48 kHz mono voice. Sample `i` of everything a stream releases is the
    /// cleaned sample `i` of everything pushed: output is time-aligned to the
    /// input and only leaves later (about 63 ms plus compute).
    public let voice: [Float]
    /// The push took longer than the audio it carried. A host feeding the
    /// voice onward should use the raw input while pushes run late.
    public let late: Bool
    public let processingMs: Double
    public init(voice: [Float], late: Bool, processingMs: Double) {
        self.voice = voice; self.late = late; self.processingMs = processingMs
    }
}

/// Live voice isolation (background-noise reduction) over the realtime
/// voice-isolator install (`ts_audio_enhance_stream_*`). Calls are
/// synchronous and run on the calling thread: keep them off the main actor.
/// Like `AudioStreamSession`, a stream takes no native heavy-work permit.
/// One call runs at a time; a second waits for the first.
public final class AudioEnhanceStream: @unchecked Sendable {
    public enum Backend: String, Sendable, CaseIterable {
        /// Core ML when the step program is installed on macOS 15+, else Metal.
        case auto, coreml, metal, cpu
    }

    /// The catalog alias of the realtime install this stream opens.
    public static let realtimeAlias = "voice-isolator-v1-realtime"
    public static let family = "voice_isolator"
    public static let sampleRate = 48_000
    /// Most samples one native push takes; longer arrays are split.
    public static let maxPushSamples = 32_000

    /// Held for push, flush and close, so close waits for a call to end.
    private let operation = NSLock()
    private var handle: OpaquePointer?
    /// The network the stream runs: `coreml`, `metal` or `cpu`.
    public let backend: String

    /// Opens and, for a managed install, verifies the model first.
    /// `onProgress` receives byte progress while files are hashed;
    /// `openToken` stops only the verify.
    public init(path: String, backend: Backend = .auto, openToken: AudioOpenToken? = nil,
                onProgress: (@Sendable (AudioProgress) -> Void)? = nil) throws {
        struct Options: Encodable { let backend: String }
        let json = try audioEncode(Options(backend: backend.rawValue))
        let path = NSString(string: path).expandingTildeInPath
        var native: OpaquePointer?
        let box = AudioCallbackBox(progress: onProgress ?? { _ in })
        let context = Unmanaged.passUnretained(box).toOpaque()
        let call: (OpaquePointer?) -> Int32 = { token in
            path.withCString { path in
                json.withCString { options in
                    ts_audio_enhance_stream_open(path, options, token, onProgress == nil ? nil : audioProgressTrampoline,
                                                 context, &native)
                }
            }
        }
        let status = openToken.map { $0.withRaw(call) } ?? call(nil)
        withExtendedLifetime(box) {}
        guard status == 0, let native else { throw TurboSparkError.fromLastError(status) }
        struct Info: Decodable { let backend: String }
        do {
            let info = try takeString { ts_audio_enhance_stream_info(native, $0) }
            self.backend = try audioDecode(Info.self, info).backend
        } catch {
            _ = ts_audio_enhance_stream_close(native)
            throw error
        }
        handle = native
    }

    /// Opens on a detached task: a first Core ML load can take a while.
    public static func open(path: String, backend: Backend = .auto,
                            openToken: AudioOpenToken? = nil) async throws -> AudioEnhanceStream {
        try await Task.detached { try AudioEnhanceStream(path: path, backend: backend, openToken: openToken) }.value
    }

    deinit { close() }

    /// Adds 48 kHz mono samples and returns the voice they released. An
    /// empty array releases nothing and never reaches the library.
    public func push(_ pcm: [Float]) throws -> AudioEnhancePush {
        operation.lock(); defer { operation.unlock() }
        guard let handle else { throw Self.closedError }
        var voice: [Float] = []
        var late = false
        var processingMs = 0.0
        struct Wire: Decodable { let released: Int; let processingMs: Double; let late: Bool }
        for start in stride(from: 0, to: pcm.count, by: Self.maxPushSamples) {
            let count = min(Self.maxPushSamples, pcm.count - start)
            let box = AudioCallbackBox(progress: { _ in })
            let context = Unmanaged.passUnretained(box).toOpaque()
            let json = try takeString { out in
                pcm.withUnsafeBufferPointer {
                    ts_audio_enhance_stream_push(handle, $0.baseAddress!.advanced(by: start), count,
                                                 audioProgressTrampoline, context, out)
                }
            }
            withExtendedLifetime(box) {}
            let wire = try audioDecode(Wire.self, json)
            guard wire.released == box.samples.count else {
                throw TurboSparkError(code: .invalidArgument, message: "Native voice isolation output count mismatch")
            }
            voice += box.samples
            late = late || wire.late
            processingMs += wire.processingMs
        }
        return AudioEnhancePush(voice: voice, late: late, processingMs: processingMs)
    }

    /// Ends the stream and returns the remaining voice. Later pushes fail;
    /// close and open a new stream for the next take.
    public func flush() throws -> [Float] {
        operation.lock(); defer { operation.unlock() }
        guard let handle else { throw Self.closedError }
        let box = AudioCallbackBox(progress: { _ in })
        let context = Unmanaged.passUnretained(box).toOpaque()
        try check(ts_audio_enhance_stream_flush(handle, audioProgressTrampoline, context))
        withExtendedLifetime(box) {}
        return box.samples
    }

    /// Waits for any running call to return, then frees the stream.
    public func close() {
        operation.lock(); defer { operation.unlock() }
        let native = handle; handle = nil
        if let native { _ = ts_audio_enhance_stream_close(native) }
    }

    private static let closedError = TurboSparkError(code: .invalidArgument, message: "Voice isolation stream is closed")
}
