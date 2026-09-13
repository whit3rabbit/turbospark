import Foundation
#if canImport(Darwin)
import Darwin
#elseif canImport(Glibc)
import Glibc
#endif

enum HttpRequestDestinationValidator {
    static func validate(_ url: URL) throws {
        guard let host = url.host, !host.isEmpty else {
            throw error(2, "Malformed URL with missing host.")
        }
        if AppToolSandbox.isPrivateOrMetadataHost(host) {
            throw error(3, "Access to private network or metadata host '\(host)' is denied.")
        }
        try AppToolSandbox.validateDomain(host)

        var hints = addrinfo()
        hints.ai_family = AF_UNSPEC
#if os(Linux)
        hints.ai_socktype = Int32(SOCK_STREAM.rawValue)
#else
        hints.ai_socktype = SOCK_STREAM
#endif
        var result: UnsafeMutablePointer<addrinfo>?
        let status = getaddrinfo(host, nil, &hints, &result)
        guard status == 0, let first = result else {
            throw error(8, "Could not resolve HTTP host '\(host)'.")
        }
        defer { freeaddrinfo(first) }

        var cursor: UnsafeMutablePointer<addrinfo>? = first
        var foundAddress = false
        while let info = cursor?.pointee {
            defer { cursor = info.ai_next }
            guard let address = info.ai_addr else { continue }
            if isPrivate(address) {
                throw error(3, "HTTP host '\(host)' resolves to a private or metadata address.")
            }
            var buffer = [CChar](repeating: 0, count: Int(NI_MAXHOST))
            guard getnameinfo(
                address,
                info.ai_addrlen,
                &buffer,
                socklen_t(buffer.count),
                nil,
                0,
                NI_NUMERICHOST
            ) == 0 else { continue }
            foundAddress = true
            let numericHost = String(cString: buffer)
            if AppToolSandbox.isPrivateOrMetadataHost(numericHost) {
                throw error(3, "HTTP host '\(host)' resolves to a private or metadata address.")
            }
        }
        guard foundAddress else {
            throw error(8, "HTTP host '\(host)' resolved without a usable address.")
        }
    }

    private static func error(_ code: Int, _ message: String) -> NSError {
        NSError(
            domain: "TurboSparkHttpRequest",
            code: code,
            userInfo: [NSLocalizedDescriptionKey: message]
        )
    }

    private static func isPrivate(_ address: UnsafePointer<sockaddr>) -> Bool {
        if Int32(address.pointee.sa_family) == AF_INET {
            let ipv4 = address.withMemoryRebound(to: sockaddr_in.self, capacity: 1) {
                UInt32(bigEndian: $0.pointee.sin_addr.s_addr)
            }
            return isPrivateIPv4(ipv4)
        }
        guard Int32(address.pointee.sa_family) == AF_INET6 else { return true }
        let bytes = address.withMemoryRebound(to: sockaddr_in6.self, capacity: 1) {
            withUnsafeBytes(of: $0.pointee.sin6_addr) { Array($0) }
        }
        if bytes == Array(repeating: 0, count: 16) { return true }
        if bytes.dropLast().allSatisfy({ $0 == 0 }) && bytes.last == 1 { return true }
        if bytes[0] & 0xFE == 0xFC || (bytes[0] == 0xFE && bytes[1] & 0xC0 == 0x80) {
            return true
        }
        if bytes[0..<10].allSatisfy({ $0 == 0 }) && bytes[10] == 0xFF && bytes[11] == 0xFF {
            let ipv4 = bytes[12...15].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
            return isPrivateIPv4(ipv4)
        }
        return false
    }

    private static func isPrivateIPv4(_ address: UInt32) -> Bool {
        let first = address >> 24
        let second = (address >> 16) & 0xFF
        return first == 0 || first == 10 || first == 127
            || (first == 100 && (64...127).contains(second))
            || (first == 169 && second == 254)
            || (first == 172 && (16...31).contains(second))
            || (first == 192 && second == 168)
    }
}
