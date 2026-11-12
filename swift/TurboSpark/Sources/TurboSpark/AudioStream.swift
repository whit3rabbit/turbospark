import CTurboSpark
import Foundation

/// A word a live stream has finished, timed from the stream's start.
public struct AudioStreamWord: Codable, Sendable, Equatable {
    public let text: String
    public let startSeconds: Double
    public let endSeconds: Double
    public init(text: String, startSeconds: Double, endSeconds: Double) {
        self.text = text; self.startSeconds = startSeconds; self.endSeconds = endSeconds
    }
}

/// What one append or finish committed. Joining `appendedText` across a
/// stream's updates gives its transcript; committed text is never revised.
public struct AudioStreamUpdate: Codable, Sendable, Equatable {
    public let appendedText: String
    /// Words completed by this update.
    public let closedWords: [AudioStreamWord]
    /// Committed pieces of a word later audio may extend.
    public let openWord: String?
    public let receivedSeconds: Double
    public let committedThroughSeconds: Double
    public let isFinal: Bool
    public init(appendedText: String = "", closedWords: [AudioStreamWord] = [], openWord: String? = nil,
                receivedSeconds: Double = 0, committedThroughSeconds: Double = 0, isFinal: Bool = false) {
        self.appendedText = appendedText; self.closedWords = closedWords; self.openWord = openWord
        self.receivedSeconds = receivedSeconds; self.committedThroughSeconds = committedThroughSeconds
        self.isFinal = isFinal
    }
    /// `self` followed by `later`, as if both arrived in one call.
    public func merged(with later: AudioStreamUpdate) -> AudioStreamUpdate {
        AudioStreamUpdate(appendedText: appendedText + later.appendedText,
                          closedWords: closedWords + later.closedWords, openWord: later.openWord,
                          receivedSeconds: later.receivedSeconds,
                          committedThroughSeconds: later.committedThroughSeconds,
                          isFinal: isFinal || later.isFinal)
    }
}

/// Live transcription over a streaming speech install (Parakeet Unified EN
/// on Core ML). Calls are synchronous and run on the calling thread: keep
/// them off the main actor. Unlike `AudioSession`, a stream takes no native
/// heavy-work permit and never uses Metal, so chat and image work keep
/// running beside it.
public final class AudioStreamSession: @unchecked Sendable {
    /// Held for append, finish and close, so close waits for a call to end.
    private let operation = NSLock()
    /// Held only to read or clear the handle, so cancel never waits.
    private let control = NSLock()
    private var handle: OpaquePointer?
    public static let family = "parakeet_unified"
    /// Most samples one native append takes; longer arrays are split.
    public static let maxAppendSamples = 32_000

    /// Opens and, for a managed install, verifies the model first.
    /// `onProgress` receives byte progress while files are hashed. The first
    /// load compiles the encoder for the Neural Engine, which can take
    /// minutes and cannot be stopped; `openToken` stops only the verify.
    public init(modelPath: String, expectedFamily: String? = AudioStreamSession.family,
                openToken: AudioOpenToken? = nil, onProgress: (@Sendable (AudioProgress) -> Void)? = nil) throws {
        struct Options: Encodable { let expectedFamily: String? }
        let json = try audioEncode(Options(expectedFamily: expectedFamily))
        let path = NSString(string: modelPath).expandingTildeInPath
        var native: OpaquePointer?
        let box = AudioCallbackBox(progress: onProgress ?? { _ in })
        let context = Unmanaged.passUnretained(box).toOpaque()
        let call: (OpaquePointer?) -> Int32 = { token in
            path.withCString { path in
                json.withCString { options in
                    ts_audio_stream_open(path, options, token, onProgress == nil ? nil : audioProgressTrampoline,
                                         context, &native)
                }
            }
        }
        let status = openToken.map { $0.withRaw(call) } ?? call(nil)
        withExtendedLifetime(box) {}
        guard status == 0, let native else { throw TurboSparkError.fromLastError(status) }
        handle = native
    }
    deinit { close() }

    /// Adds 16 kHz mono samples and returns what they committed.
    public func append(_ pcm: [Float]) throws -> AudioStreamUpdate {
        operation.lock(); defer { operation.unlock() }
        guard let handle = currentHandle() else { throw Self.closedError }
        var update = AudioStreamUpdate()
        for start in stride(from: 0, to: pcm.count, by: Self.maxAppendSamples) {
            let count = min(Self.maxAppendSamples, pcm.count - start)
            let json = try takeString { out in
                pcm.withUnsafeBufferPointer { ts_audio_stream_append(handle, $0.baseAddress!.advanced(by: start), count, out) }
            }
            update = update.merged(with: try audioDecode(AudioStreamUpdate.self, json))
        }
        return update
    }

    /// Commits the held-back tail and closes the last word. The stream then
    /// restarts at time zero for a new segment.
    public func finish() throws -> AudioStreamUpdate {
        operation.lock(); defer { operation.unlock() }
        guard let handle = currentHandle() else { throw Self.closedError }
        let json = try takeString { ts_audio_stream_finish(handle, $0) }
        return try audioDecode(AudioStreamUpdate.self, json)
    }

    /// Stops a running append or finish before its next encoder step. Never
    /// waits. Terminal: later calls fail; close and open a new stream.
    public func cancel() {
        control.lock(); defer { control.unlock() }
        if let handle { _ = ts_audio_stream_cancel(handle) }
    }

    /// Cancels, waits for any running call to return, then frees the stream.
    public func close() {
        cancel()
        operation.lock(); defer { operation.unlock() }
        control.lock(); let native = handle; handle = nil; control.unlock()
        if let native { _ = ts_audio_stream_close(native) }
    }

    private func currentHandle() -> OpaquePointer? {
        control.lock(); defer { control.unlock() }
        return handle
    }
    private static let closedError = TurboSparkError(code: .invalidArgument, message: "Audio stream is closed")
}
