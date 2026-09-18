import CTurboSpark
import Foundation

/// Anything the C layer refused to do.
///
/// The message comes from the library's per-thread error slot, read
/// immediately after the failing call and on the same thread, which is what
/// its contract requires.
public struct TurboSparkError: Error, CustomStringConvertible {
    public enum Code: Int32, Sendable {
        case invalidArgument = 1
        case open = 2
        case generate = 3
        case json = 4
        case unsupportedPlatform = 5
        /// A panic was caught inside the library. The process is intact and
        /// the operation did not happen; this is a library bug rather than
        /// anything the caller did.
        case panic = 6
        case unknown = -1
    }

    public let code: Code
    public let message: String

    public init(code: Code, message: String) {
        self.code = code
        self.message = message
    }

    public var description: String { "\(code): \(message)" }

    /// Reads the library's error slot. Call ONLY after a non-zero return,
    /// on the thread that made the call.
    static func fromLastError(_ status: Int32) -> TurboSparkError {
        // Ask for the length, then read: the return value is the message's
        // own length rather than the bytes written, so this cannot truncate.
        let needed = ts_last_error(nil, 0)
        var buffer = [CChar](repeating: 0, count: needed + 1)
        _ = ts_last_error(&buffer, buffer.count)
        return TurboSparkError(
            code: Code(rawValue: status) ?? .unknown,
            message: String(cString: buffer)
        )
    }
}

/// Runs a C call that returns a status code, throwing on failure.
@inline(__always)
func check(_ status: Int32) throws {
    guard status == 0 else { throw TurboSparkError.fromLastError(status) }
}

/// Runs a C call that fills a `char **`, decoding and freeing the result.
///
/// The free is in a `defer`, so a decoding failure cannot leak the string.
func takeString(_ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32) throws
    -> String
{
    var out: UnsafeMutablePointer<CChar>?
    try check(body(&out))
    guard let out else {
        throw TurboSparkError(code: .unknown, message: "the library reported success but wrote no result")
    }
    defer { ts_string_free(out) }
    return String(cString: out)
}

/// Runs a C call that fills a `char **`, returning nil if `out` is null.
func takeOptionalString(_ body: (UnsafeMutablePointer<UnsafeMutablePointer<CChar>?>) -> Int32)
    throws -> String?
{
    var out: UnsafeMutablePointer<CChar>?
    try check(body(&out))
    guard let out else { return nil }
    defer { ts_string_free(out) }
    return String(cString: out)
}

/// Decodes JSON the library produced.
///
/// `keyDecodingStrategy` is deliberately NOT set: the Rust side already
/// emits camelCase for exactly this reason, so the two spellings match with
/// no transformation and nothing can drift between them.
func decode<T: Decodable>(_ type: T.Type, from json: String) throws -> T {
    guard let data = json.data(using: .utf8) else {
        throw TurboSparkError(code: .json, message: "result was not valid UTF-8")
    }
    do {
        return try JSONDecoder().decode(type, from: data)
    } catch {
        throw TurboSparkError(code: .json, message: "\(error)")
    }
}
