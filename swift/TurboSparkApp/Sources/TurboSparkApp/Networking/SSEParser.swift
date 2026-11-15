import Foundation

/// One dispatched Server-Sent Event.
struct SSEEvent: Equatable, Sendable {
    /// The `event:` field, nil when the stream did not name one (the
    /// OpenAI-compatible shape never does).
    var event: String?
    /// The `data:` lines joined with a newline.
    var data: String
}

/// Why `SSEParser` stopped.
enum SSEParseError: Error, Equatable {
    /// One event grew past `maxEventBytes` without a blank line. A server
    /// that never ends an event would otherwise grow this buffer without
    /// bound.
    case eventTooLarge
}

/// Incremental parser for `text/event-stream` bytes (the WHATWG rules the two
/// family clients need): lines end in LF, CR or CRLF, a blank line dispatches
/// the event, a leading colon is a comment, one leading space after the colon
/// is dropped, and `id` / `retry` are ignored because the clients never
/// reconnect on their own.
///
/// **BYTES, NOT `AsyncBytes.lines`.** `URLSession.AsyncBytes.lines` drops empty
/// lines, and the empty line is the dispatch signal. The parser takes raw
/// chunks and splits on the newline bytes itself, which also keeps a UTF-8
/// scalar that straddles two chunks whole.
struct SSEParser {
    static let defaultMaxEventBytes = 1 << 20

    let maxEventBytes: Int
    private var lineBuffer = Data()
    private var pendingCR = false
    private var eventName: String?
    private var dataLines: [String] = []
    private var eventBytes = 0

    init(maxEventBytes: Int = SSEParser.defaultMaxEventBytes) {
        self.maxEventBytes = maxEventBytes
    }

    /// Feeds one chunk and returns every event it completed.
    mutating func feed(_ chunk: Data) throws -> [SSEEvent] {
        var events: [SSEEvent] = []
        for byte in chunk {
            if pendingCR {
                pendingCR = false
                // The LF of a CRLF pair was already handled at the CR.
                if byte == 0x0A { continue }
            }
            switch byte {
            case 0x0A:
                try endLine(into: &events)
            case 0x0D:
                pendingCR = true
                try endLine(into: &events)
            default:
                lineBuffer.append(byte)
                eventBytes += 1
                if eventBytes > maxEventBytes { throw SSEParseError.eventTooLarge }
            }
        }
        return events
    }

    /// Whether a partial event is buffered. A stream that ends with this set
    /// ended mid-event; the clients treat that as a bad stream.
    var hasPartialEvent: Bool {
        !lineBuffer.isEmpty || !dataLines.isEmpty
    }

    private mutating func endLine(into events: inout [SSEEvent]) throws {
        let line = String(decoding: lineBuffer, as: UTF8.self)
        lineBuffer.removeAll(keepingCapacity: true)
        if line.isEmpty {
            // Dispatch. An event with no data lines is dropped, as the spec
            // says.
            if !dataLines.isEmpty {
                events.append(SSEEvent(event: eventName, data: dataLines.joined(separator: "\n")))
            }
            eventName = nil
            dataLines.removeAll(keepingCapacity: true)
            eventBytes = 0
            return
        }
        if line.hasPrefix(":") { return }
        let field: Substring
        var value: Substring
        if let colon = line.firstIndex(of: ":") {
            field = line[line.startIndex..<colon]
            value = line[line.index(after: colon)...]
            if value.hasPrefix(" ") { value = value.dropFirst() }
        } else {
            field = Substring(line)
            value = ""
        }
        switch field {
        case "event": eventName = String(value)
        case "data": dataLines.append(String(value))
        default: break
        }
    }
}
