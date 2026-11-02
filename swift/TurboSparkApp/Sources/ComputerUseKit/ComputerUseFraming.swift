import Foundation

/// Length-prefixed JSON frames over a blocking socket descriptor: a 4-byte
/// big-endian length, then that many bytes of JSON.
public enum ComputerUseFraming {
    /// A screenshot of a 5K window as PNG in base64 stays well under this.
    public static let maximumFrameBytes = 48 * 1_024 * 1_024

    public enum FramingError: Error, Equatable {
        case closed
        case tooLarge(Int)
        case io(Int32)
    }

    public static func encode<T: Encodable>(_ value: T) throws -> Data {
        let body = try JSONEncoder().encode(value)
        guard body.count <= maximumFrameBytes else { throw FramingError.tooLarge(body.count) }
        var length = UInt32(body.count).bigEndian
        var frame = Data(bytes: &length, count: 4)
        frame.append(body)
        return frame
    }

    /// Splits complete frames off the front of `buffer`.
    public static func decodeFrames(_ buffer: inout Data) throws -> [Data] {
        var frames: [Data] = []
        while buffer.count >= 4 {
            let header = buffer.prefix(4)
            let length = header.reduce(0) { ($0 << 8) | Int($1) }
            guard length <= maximumFrameBytes else { throw FramingError.tooLarge(length) }
            guard buffer.count >= 4 + length else { break }
            frames.append(Data(buffer.dropFirst(4).prefix(length)))
            buffer = Data(buffer.dropFirst(4 + length))
        }
        return frames
    }

    /// A write to a socket whose peer died raises SIGPIPE, which kills the
    /// process. Both ends call this on every socket they own.
    public static func disableSigpipe(_ fd: Int32) {
        var on: Int32 = 1
        _ = setsockopt(fd, SOL_SOCKET, SO_NOSIGPIPE, &on, socklen_t(MemoryLayout<Int32>.size))
    }

    public static func write<T: Encodable>(_ value: T, to fd: Int32) throws {
        let frame = try encode(value)
        try frame.withUnsafeBytes { raw in
            var offset = 0
            while offset < raw.count {
                let n = Darwin.write(fd, raw.baseAddress!.advanced(by: offset), raw.count - offset)
                if n < 0 {
                    if errno == EINTR { continue }
                    throw FramingError.io(errno)
                }
                offset += n
            }
        }
    }

    /// Blocks until one whole frame arrives.
    public static func read<T: Decodable>(_ type: T.Type, from fd: Int32) throws -> T {
        let header = try readExactly(4, from: fd)
        let length = header.reduce(0) { ($0 << 8) | Int($1) }
        guard length <= maximumFrameBytes else { throw FramingError.tooLarge(length) }
        let body = try readExactly(length, from: fd)
        return try JSONDecoder().decode(T.self, from: body)
    }

    private static func readExactly(_ count: Int, from fd: Int32) throws -> Data {
        var data = Data(count: count)
        var offset = 0
        try data.withUnsafeMutableBytes { raw in
            while offset < count {
                let n = Darwin.read(fd, raw.baseAddress!.advanced(by: offset), count - offset)
                if n == 0 { throw FramingError.closed }
                if n < 0 {
                    if errno == EINTR { continue }
                    throw FramingError.io(errno)
                }
                offset += n
            }
        }
        return data
    }
}
