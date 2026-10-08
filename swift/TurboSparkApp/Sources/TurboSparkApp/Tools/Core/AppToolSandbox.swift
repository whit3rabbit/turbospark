import Foundation

/// Sandbox configuration and validation rules for safe workspace execution.
public struct SandboxConfig: Sendable, Equatable {
    public var allowedWritePaths: [URL]
    public var deniedWritePaths: [String]
    public var allowedDomains: [String]
    public var deniedDomains: [String]
    public var networkAllowed: Bool

    public static let systemDeniedPaths: [String] = [
        "/etc", "/usr", "/sys", "/proc", "/System", "/Library", "/Applications", "/bin", "/sbin", "/private/etc"
    ]

    public init(
        allowedWritePaths: [URL] = [],
        deniedWritePaths: [String] = SandboxConfig.systemDeniedPaths,
        allowedDomains: [String] = [],
        deniedDomains: [String] = [],
        networkAllowed: Bool = true
    ) {
        self.allowedWritePaths = allowedWritePaths
        self.deniedWritePaths = deniedWritePaths
        self.allowedDomains = allowedDomains
        self.deniedDomains = deniedDomains
        self.networkAllowed = networkAllowed
    }
}

/// Sandbox validator enforcing filesystem boundaries and network access rules.
public enum AppToolSandbox {
    /// Canonical form of `url` for containment checks, valid even when the
    /// leaf (or several trailing components) does not exist yet.
    ///
    /// **`resolvingSymlinksInPath` DOES NOTHING FOR A PATH THAT DOES NOT
    /// EXIST.** A new file under a symlinked directory (`link/new.txt`
    /// with `link -> /outside`) therefore came back unresolved, passed the
    /// lexical prefix check, and the write then followed the symlink out of
    /// the project. This resolves the deepest ancestor that exists (using
    /// `lstat`, so a dangling symlink counts as existing and is followed by
    /// hand) and re-appends the not-yet-created components, so the result
    /// contains no symlink a later `createDirectory`/write could traverse.
    public static func resolvedForContainment(_ url: URL) -> URL {
        resolvedForContainment(url, depth: 0)
    }

    private static func resolvedForContainment(_ url: URL, depth: Int) -> URL {
        let std = url.standardizedFileURL
        let path = std.path
        var info = stat()
        if lstat(path, &info) == 0 {
            let isLink = (info.st_mode & S_IFMT) == S_IFLNK
            var target = stat()
            if isLink, stat(path, &target) != 0, depth < 16,
                let dest = try? FileManager.default.destinationOfSymbolicLink(atPath: path)
            {
                // Dangling symlink: a write through it would create the
                // destination, so judge the destination instead.
                let parent = resolvedForContainment(std.deletingLastPathComponent(), depth: depth + 1)
                let next = dest.hasPrefix("/")
                    ? URL(fileURLWithPath: dest)
                    : parent.appendingPathComponent(dest)
                return resolvedForContainment(next, depth: depth + 1)
            }
            return std.resolvingSymlinksInPath()
        }
        // `/` always exists, so the walk terminates; the guard is belt and braces.
        guard path != "/", !path.isEmpty else { return std }
        let parent = resolvedForContainment(std.deletingLastPathComponent(), depth: depth)
        return parent.appendingPathComponent(std.lastPathComponent)
    }

    /// Validates whether a file path is permitted for write operations.
    public static func validateWritePath(_ url: URL, rootURL: URL, config: SandboxConfig = SandboxConfig()) throws {
        let path = resolvedForContainment(url).path
        let rootPath = resolvedForContainment(rootURL).path
        let rootPrefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"

        // Must stay inside workspace root unless explicitly allowed
        let isInsideRoot = path == rootPath || path.hasPrefix(rootPrefix)
        let isExplicitlyAllowed = config.allowedWritePaths.contains {
            let allowed = resolvedForContainment($0).path
            return path == allowed || path.hasPrefix(allowed.hasSuffix("/") ? allowed : allowed + "/")
        }

        guard isInsideRoot || isExplicitlyAllowed else {
            throw NSError(domain: "TurboSparkSandbox", code: 1, userInfo: [
                NSLocalizedDescriptionKey: "Sandbox write violation: Path '\(path)' is outside the workspace root."
            ])
        }

        // Check denied system paths
        for denied in config.deniedWritePaths {
            if path == denied || path.hasPrefix(denied.hasSuffix("/") ? denied : denied + "/") {
                throw NSError(domain: "TurboSparkSandbox", code: 2, userInfo: [
                    NSLocalizedDescriptionKey: "Sandbox write violation: Write to system directory '\(denied)' is denied."
                ])
            }
        }
    }

    /// Whether `host` names this machine, a private network, or a cloud
    /// metadata service.
    ///
    /// Called from `ToolRiskClassifier`'s web arm. The check it replaced knew
    /// six literals (`localhost`, `127.0.0.1`, `169.254.169.254`,
    /// `metadata.google.internal`, and the `192.168.`/`10.` prefixes), which
    /// left every other spelling of the same address reading as an ordinary
    /// outbound fetch: IPv6 loopback, the whole `172.16/12` block,
    /// link-local, `.local` mDNS names, `0.0.0.0`, and the decimal and octal
    /// forms of an IPv4 address that most HTTP clients resolve happily
    /// (`http://2130706433/` is `127.0.0.1`).
    public static func isPrivateOrMetadataHost(_ host: String) -> Bool {
        var clean = host.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        // `localhost.` and `127.0.0.1.` are the same hosts as the unqualified
        // spellings (a rooted DNS name), and URLSession resolves them.
        while clean.hasSuffix(".") { clean.removeLast() }
        guard !clean.isEmpty else { return false }

        let namedHosts: Set<String> = [
            "localhost", "metadata.google.internal", "metadata", "instance-data",
            "::1", "::", "0:0:0:0:0:0:0:1",
        ]
        if namedHosts.contains(clean) { return true }
        if clean.hasSuffix(".localhost") || clean.hasSuffix(".local")
            || clean.hasSuffix(".internal")
        {
            return true
        }

        // Any IPv6 literal is parsed to bytes rather than prefix-matched on
        // the string: `::ffff:127.0.0.1`, `::ffff:7f00:1` and `0::1` are all
        // loopback and no textual prefix test sees them all.
        if clean.contains(":") {
            return isPrivateIPv6Literal(clean)
        }

        if let packed = packedIPv4(clean) {
            return isPrivateIPv4(packed)
        }

        return false
    }

    private static func isPrivateIPv4(_ packed: UInt32) -> Bool {
        let a = (packed >> 24) & 0xFF
        let b = (packed >> 16) & 0xFF
        if a == 0 || a == 10 || a == 127 { return true }
        if a == 169 && b == 254 { return true }  // link-local, incl. 169.254.169.254
        if a == 172 && (16...31).contains(b) { return true }
        if a == 192 && b == 168 { return true }
        if a == 100 && (64...127).contains(b) { return true }  // CGNAT
        return false
    }

    /// Fails closed: a string with a colon that is not a valid IPv6 literal
    /// is not a host this app should be connecting to anyway.
    private static func isPrivateIPv6Literal(_ literal: String) -> Bool {
        // Drop a zone id (`fe80::1%en0`); inet_pton rejects it.
        let address = literal.split(separator: "%", maxSplits: 1).first.map(String.init) ?? literal
        var storage = in6_addr()
        guard inet_pton(AF_INET6, address, &storage) == 1 else { return true }
        let bytes = withUnsafeBytes(of: &storage) { Array($0) }

        func embedded(_ offset: Int) -> UInt32 {
            bytes[offset..<(offset + 4)].reduce(UInt32(0)) { ($0 << 8) | UInt32($1) }
        }

        if bytes.dropLast().allSatisfy({ $0 == 0 }) && bytes.last! <= 1 { return true }  // :: and ::1
        if bytes[0] & 0xFE == 0xFC { return true }  // fc00::/7 unique-local
        if bytes[0] == 0xFE && bytes[1] & 0xC0 == 0x80 { return true }  // fe80::/10 link-local
        // ::ffff:a.b.c.d (mapped) and ::a.b.c.d (deprecated compatible).
        if bytes[0..<10].allSatisfy({ $0 == 0 })
            && ((bytes[10] == 0xFF && bytes[11] == 0xFF) || (bytes[10] == 0 && bytes[11] == 0))
        {
            return isPrivateIPv4(embedded(12))
        }
        // 64:ff9b::/96 NAT64 carries an IPv4 address in the low 32 bits.
        if bytes[0] == 0x00, bytes[1] == 0x64, bytes[2] == 0xFF, bytes[3] == 0x9B,
            bytes[4..<12].allSatisfy({ $0 == 0 })
        {
            return isPrivateIPv4(embedded(12))
        }
        // 2002::/16 6to4 carries one in bytes 2-5.
        if bytes[0] == 0x20 && bytes[1] == 0x02 { return isPrivateIPv4(embedded(2)) }
        return false
    }

    /// Parses dotted-quad, bare-decimal and octal/hex IPv4 spellings into a
    /// packed address. Returns nil for anything that is not an IPv4 literal.
    ///
    /// The non-dotted forms are the point: `http://2130706433/` and
    /// `http://0177.0.0.1/` both reach 127.0.0.1 through URLSession, and a
    /// prefix test on the string "127." sees neither.
    private static func packedIPv4(_ host: String) -> UInt32? {
        let parts = host.components(separatedBy: ".")
        guard (1...4).contains(parts.count) else { return nil }

        func value(_ s: String) -> UInt64? {
            guard !s.isEmpty else { return nil }
            if s.hasPrefix("0x") || s.hasPrefix("0X") {
                return UInt64(s.dropFirst(2), radix: 16)
            }
            if s.hasPrefix("0") && s.count > 1 {
                return UInt64(s.dropFirst(), radix: 8)
            }
            return UInt64(s, radix: 10)
        }

        let values = parts.compactMap(value)
        guard values.count == parts.count else { return nil }

        // A single number is the whole 32-bit address; the dotted forms fill
        // from the left with the last part covering the remainder.
        if values.count == 1 {
            guard values[0] <= 0xFFFF_FFFF else { return nil }
            return UInt32(values[0])
        }
        guard values.dropLast().allSatisfy({ $0 <= 0xFF }) else { return nil }
        let remainingBytes = 4 - values.count
        guard let last = values.last, last < (UInt64(1) << UInt64(8 * (remainingBytes + 1)))
        else { return nil }

        var packed: UInt64 = 0
        for v in values.dropLast() { packed = (packed << 8) | v }
        packed = (packed << UInt64(8 * (remainingBytes + 1))) | last
        guard packed <= 0xFFFF_FFFF else { return nil }
        return UInt32(packed)
    }

    /// Validates whether a domain is permitted for network requests.
    public static func validateDomain(_ domain: String, config: SandboxConfig = SandboxConfig()) throws {
        guard config.networkAllowed else {
            throw NSError(domain: "TurboSparkSandbox", code: 3, userInfo: [
                NSLocalizedDescriptionKey: "Sandbox network violation: Outbound network access is disabled."
            ])
        }

        let cleanDomain = domain.lowercased().trimmingCharacters(in: .whitespacesAndNewlines)

        for denied in config.deniedDomains {
            let cleanDenied = denied.lowercased()
            if cleanDomain == cleanDenied || cleanDomain.hasSuffix("." + cleanDenied) {
                throw NSError(domain: "TurboSparkSandbox", code: 4, userInfo: [
                    NSLocalizedDescriptionKey: "Sandbox network violation: Access to domain '\(cleanDomain)' is blocked."
                ])
            }
        }

        if !config.allowedDomains.isEmpty {
            let isAllowed = config.allowedDomains.contains { allowed in
                let cleanAllowed = allowed.lowercased()
                return cleanDomain == cleanAllowed || cleanDomain.hasSuffix("." + cleanAllowed)
            }
            if !isAllowed {
                throw NSError(domain: "TurboSparkSandbox", code: 5, userInfo: [
                    NSLocalizedDescriptionKey: "Sandbox network violation: Domain '\(cleanDomain)' is not in allowed list."
                ])
            }
        }
    }
}
