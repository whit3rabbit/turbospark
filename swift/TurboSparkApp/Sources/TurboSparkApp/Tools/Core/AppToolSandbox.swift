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
