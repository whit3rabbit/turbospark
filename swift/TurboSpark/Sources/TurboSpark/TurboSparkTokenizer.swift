import CTurboSpark
import Foundation

/// A tokenizer opened without the engine behind it.
///
/// `TurboSparkSession` maps gigabytes and compiles pipelines at open; counting
/// tokens for a context meter or a prompt-budget preview before any model is
/// loaded should not cost that. This reads only the tokenizer files.
///
/// Text-level calls only. Rendering a conversation and fitting a window need
/// the family and dialect the engine resolves at open, so `renderPrompt` and
/// `fitWindow` stay on `TurboSparkSession`.
///
/// Calls are synchronous and cheap (microseconds to milliseconds), so unlike
/// the session they do not hop to a queue. They are safe from any thread: the
/// native tokenizer is read-only after open and `close()` waits for calls in
/// flight.
public final class TurboSparkTokenizer: @unchecked Sendable {
    private var raw: OpaquePointer?
    private let lock = NSLock()

    /// Opens the tokenizer of a `.gturbo` directory or an installed alias. An
    /// existing directory wins over an alias, as in `TurboSparkSession`.
    public init(modelPath: String) throws {
        try TurboSparkRuntime.verifyABIOnce()
        let path = NSString(string: modelPath).expandingTildeInPath
        var out: OpaquePointer?
        let status = path.withCString { ts_tokenizer_open($0, &out) }
        guard status == 0, let out else { throw TurboSparkError.fromLastError(status) }
        raw = out
    }

    deinit { close() }

    /// Releases the native tokenizer. Safe to call more than once.
    public func close() {
        lock.lock(); defer { lock.unlock() }
        if let raw { ts_tokenizer_close(raw); self.raw = nil }
    }

    private func withHandle<T>(_ body: (OpaquePointer) throws -> T) throws -> T {
        lock.lock(); defer { lock.unlock() }
        guard let raw else {
            throw TurboSparkError(code: .invalidArgument, message: "tokenizer is closed")
        }
        return try body(raw)
    }

    /// How many tokens `text` is.
    public func count(_ text: String, addSpecialTokens: Bool = false) throws -> Int {
        try withHandle { handle in
            var count: UInt32 = 0
            let status = text.withCString {
                ts_tokenizer_count_text_tokens(handle, $0, addSpecialTokens, &count)
            }
            try check(status)
            return Int(count)
        }
    }

    /// `text` as token ids.
    public func tokenize(_ text: String, addSpecialTokens: Bool = false) throws -> [Int32] {
        try withHandle { handle in
            let json = try takeString { out in
                text.withCString { ts_tokenizer_tokenize_json(handle, $0, addSpecialTokens, out) }
            }
            return try decode([Int32].self, from: json)
        }
    }

    /// Token ids back to text.
    public func detokenize(_ tokens: [Int32], skipSpecialTokens: Bool = false) throws -> String {
        let json = String(decoding: try JSONEncoder().encode(tokens), as: UTF8.self)
        return try withHandle { handle in
            try takeString { out in
                json.withCString {
                    ts_tokenizer_detokenize_json(handle, $0, skipSpecialTokens, out)
                }
            }
        }
    }
}
