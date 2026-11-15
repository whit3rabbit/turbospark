import Foundation

/// OC-10: Content-Length framing for language server stdio. LSP is not
/// newline-delimited like the MCP stdio sessions (`LineBuffer`), so it has its
/// own codec. A server is third-party code writing to a pipe: every limit here
/// is enforced while reading, before a message is allocated.
enum LSPFraming {
    /// One message above this is a protocol violation, not a large reply. The
    /// largest legitimate payloads are a workspace symbol page and a
    /// documentSymbol tree, both far smaller.
    static let maximumFrameBytes = 8 * 1_024 * 1_024
    /// The header block (`Content-Length` and `Content-Type`) is tiny. A
    /// server streaming bytes with no blank line is cut off at this size.
    static let maximumHeaderBytes = 4 * 1_024

    enum Failure: Error, Equatable {
        case oversizeFrame(Int)
        case malformedHeader
    }

    static func encode(_ json: Data) -> Data {
        Data("Content-Length: \(json.count)\r\n\r\n".utf8) + json
    }

    static func encode(object: [String: Any]) throws -> Data {
        // Slashes stay unescaped: URIs read as written, and a server that
        // matches method names textually (a shell fixture) is not surprised.
        encode(try JSONSerialization.data(withJSONObject: object, options: [.withoutEscapingSlashes]))
    }
}

/// Incremental decoder: feed it whatever chunk the pipe delivered and it
/// returns every complete message body, keeping a partial tail for the next
/// chunk. A throw is terminal for the stream; the caller kills the server.
struct LSPFrameDecoder {
    private var buffer = Data()
    private let terminator = Data("\r\n\r\n".utf8)

    mutating func append(_ chunk: Data) throws -> [Data] {
        buffer.append(chunk)
        var messages: [Data] = []
        while true {
            guard let headerEnd = buffer.range(of: terminator) else {
                if buffer.count > LSPFraming.maximumHeaderBytes { throw LSPFraming.Failure.malformedHeader }
                break
            }
            guard headerEnd.lowerBound <= LSPFraming.maximumHeaderBytes else {
                throw LSPFraming.Failure.malformedHeader
            }
            let header = String(decoding: buffer[buffer.startIndex..<headerEnd.lowerBound], as: UTF8.self)
            guard let length = Self.contentLength(in: header) else { throw LSPFraming.Failure.malformedHeader }
            guard length <= LSPFraming.maximumFrameBytes else { throw LSPFraming.Failure.oversizeFrame(length) }
            let bodyStart = headerEnd.upperBound
            guard buffer.endIndex - bodyStart >= length else { break }
            messages.append(Data(buffer[bodyStart..<(bodyStart + length)]))
            buffer.removeSubrange(buffer.startIndex..<(bodyStart + length))
        }
        return messages
    }

    private static func contentLength(in header: String) -> Int? {
        for line in header.components(separatedBy: "\r\n") {
            guard let colon = line.firstIndex(of: ":") else { continue }
            let name = line[line.startIndex..<colon].trimmingCharacters(in: .whitespaces).lowercased()
            guard name == "content-length" else { continue }
            let value = line[line.index(after: colon)...].trimmingCharacters(in: .whitespaces)
            // Digits only: "+5", "-1" and "0x10" are not lengths.
            guard !value.isEmpty, value.allSatisfy(\.isASCII), value.allSatisfy(\.isNumber),
                  let number = Int(value) else { return nil }
            return number
        }
        return nil
    }
}
