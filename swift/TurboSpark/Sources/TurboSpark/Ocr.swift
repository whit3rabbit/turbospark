import CTurboSpark
import Foundation

/// Result of one `TurboSparkOcr.read` call.
public struct OcrReadResult: Decodable, Sendable, Equatable {
    /// Plain text; pages are separated by a blank line.
    public let text: String
    /// Model output with its `<PAGE>` and `<|det|>` layout markup.
    public let rawText: String
    public let pages: Int
    public let promptTokens: Int
    public let outputTokens: Int
    /// False when the read hit the token limit and the text may be cut off.
    public let stoppedAtEos: Bool

    enum CodingKeys: String, CodingKey {
        case text, pages
        case rawText = "raw_text"
        case promptTokens = "prompt_tokens"
        case outputTokens = "output_tokens"
        case stoppedAtEos = "stopped_at_eos"
    }
}

/// Document OCR on Core ML (Unlimited OCR, macOS 15+).
///
/// Add one or more pages, then `read` them as one document; every read
/// clears the pending pages. Opening compiles the decoder for this machine
/// the first time, and a loaded model uses several gigabytes.
public final class TurboSparkOcr: @unchecked Sendable {
    private struct Handle: @unchecked Sendable {
        let raw: OpaquePointer
    }

    private let handle: Handle

    /// Opens a directory holding the Core ML parts of
    /// `FluidInference/unlimited-ocr-mlx-coreml`.
    public init(modelPath: String) throws {
        let path = NSString(string: modelPath).expandingTildeInPath
        var out: OpaquePointer?
        let status = path.withCString { ts_ocr_open($0, &out) }
        guard status == 0, let out else {
            throw TurboSparkError.fromLastError(status)
        }
        handle = Handle(raw: out)
    }

    deinit {
        ts_ocr_close(handle.raw)
    }

    /// Adds a page from a PNG or JPEG file.
    public func addPage(fileAt path: String) throws {
        try path.withCString { try check(ts_ocr_add_page_file(handle.raw, $0)) }
    }

    /// Adds a page of interleaved 8-bit RGB pixels. `rowBytes` is the
    /// stride of one row, at least `width * 3`.
    public func addPage(rgb: Data, width: Int, height: Int, rowBytes: Int) throws {
        guard width > 0, height > 0, rowBytes >= width * 3, rgb.count >= rowBytes * height else {
            throw TurboSparkError(code: .invalidArgument, message: "pixel buffer does not match its size")
        }
        try rgb.withUnsafeBytes { raw in
            guard let base = raw.bindMemory(to: UInt8.self).baseAddress else {
                throw TurboSparkError(code: .invalidArgument, message: "empty pixel buffer")
            }
            try check(ts_ocr_add_page_rgb(handle.raw, base, width, height, rowBytes))
        }
    }

    /// Reads the pending pages. Runs off the calling thread; `cancel()`
    /// from another thread stops it with a `.cancelled` error.
    public func read(maxNewTokens: Int? = nil) async throws -> OcrReadResult {
        let options = maxNewTokens.map { "{\"max_new_tokens\": \($0)}" }
        // The task holds `self`, so the session cannot be closed by deinit
        // while the read is still running on it.
        return try await Task.detached { [self] in
            let json = try takeString { out in
                if let options {
                    return options.withCString { ts_ocr_read_json(self.handle.raw, $0, out) }
                }
                return ts_ocr_read_json(self.handle.raw, nil, out)
            }
            return try JSONDecoder().decode(OcrReadResult.self, from: Data(json.utf8))
        }.value
    }

    /// Asks an in-flight read to stop. Safe from any thread.
    public func cancel() {
        ts_ocr_cancel(handle.raw)
    }
}
