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
    /// Validates whether a file path is permitted for write operations.
    public static func validateWritePath(_ url: URL, rootURL: URL, config: SandboxConfig = SandboxConfig()) throws {
        let path = url.standardizedFileURL.resolvingSymlinksInPath().path
        let rootPath = rootURL.standardizedFileURL.resolvingSymlinksInPath().path
        let rootPrefix = rootPath.hasSuffix("/") ? rootPath : rootPath + "/"

        // Must stay inside workspace root unless explicitly allowed
        let isInsideRoot = path == rootPath || path.hasPrefix(rootPrefix)
        let isExplicitlyAllowed = config.allowedWritePaths.contains {
            let allowed = $0.standardizedFileURL.resolvingSymlinksInPath().path
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
        let clean = host.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
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

        // IPv6 unique-local (fc00::/7) and link-local (fe80::/10).
        if clean.hasPrefix("fc") || clean.hasPrefix("fd") || clean.hasPrefix("fe8")
            || clean.hasPrefix("fe9") || clean.hasPrefix("fea") || clean.hasPrefix("feb")
        {
            if clean.contains(":") { return true }
        }

        if let packed = packedIPv4(clean) {
            let a = (packed >> 24) & 0xFF
            let b = (packed >> 16) & 0xFF
            if a == 0 || a == 10 || a == 127 { return true }
            if a == 169 && b == 254 { return true }  // link-local, incl. 169.254.169.254
            if a == 172 && (16...31).contains(b) { return true }
            if a == 192 && b == 168 { return true }
            if a == 100 && (64...127).contains(b) { return true }  // CGNAT
        }

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
