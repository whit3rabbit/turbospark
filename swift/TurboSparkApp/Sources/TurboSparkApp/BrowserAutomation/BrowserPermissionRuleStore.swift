import Foundation

/// A canonical HTTP(S) origin, including the scheme's effective port.
public struct BrowserOrigin: Hashable, Sendable {
    public let canonicalString: String
    public let scheme: String
    public let host: String
    public let effectivePort: Int

    /// Parses an origin grant, rejecting paths, query strings, fragments, and userinfo.
    public init?(origin: String) {
        guard !origin.isEmpty,
              origin.rangeOfCharacter(from: .whitespacesAndNewlines) == nil,
              let components = URLComponents(string: origin),
              components.percentEncodedPath.isEmpty || components.percentEncodedPath == "/",
              components.percentEncodedQuery == nil,
              components.fragment == nil,
              !Self.hasEmptyPortSuffix(origin),
              let parsed = Self.parse(components)
        else {
            return nil
        }
        self = parsed
    }

    /// Extracts the origin from a navigation URL without carrying its path or query.
    public init?(url: URL) {
        guard let components = URLComponents(url: url, resolvingAgainstBaseURL: false),
              let parsed = Self.parse(components)
        else {
            return nil
        }
        self = parsed
    }

    private init(scheme: String, host: String, effectivePort: Int) {
        self.scheme = scheme
        self.host = host
        self.effectivePort = effectivePort
        let defaultPort = scheme == "https" ? 443 : 80
        let portSuffix = effectivePort == defaultPort ? "" : ":\(effectivePort)"
        let serializedHost = host.contains(":") ? "[\(host)]" : host
        self.canonicalString = "\(scheme)://\(serializedHost)\(portSuffix)"
    }

    private static func parse(_ components: URLComponents) -> BrowserOrigin? {
        guard let rawScheme = components.scheme?.lowercased(),
              rawScheme == "http" || rawScheme == "https",
              components.user == nil,
              components.password == nil,
              let url = components.url,
              let host = url.host?.lowercased(),
              !host.isEmpty,
              !host.contains("*"),
              host.rangeOfCharacter(
                  from: .whitespacesAndNewlines.union(CharacterSet(charactersIn: "/?#@"))) == nil
        else {
            return nil
        }
        let effectivePort = components.port ?? (rawScheme == "https" ? 443 : 80)
        guard (0...65_535).contains(effectivePort) else { return nil }
        return BrowserOrigin(scheme: rawScheme, host: host, effectivePort: effectivePort)
    }

    private static func hasEmptyPortSuffix(_ origin: String) -> Bool {
        guard let schemeSeparator = origin.range(of: "://") else { return true }
        let afterScheme = origin[schemeSeparator.upperBound...]
        let authority = afterScheme.prefix { $0 != "/" && $0 != "?" && $0 != "#" }
        return authority.hasSuffix(":")
    }
}

public enum BrowserPermissionOwner: Sendable {
    case agent
    case user
}

/// Typed origin and ownership context supplied by the browser authorization boundary.
public struct BrowserPermissionContext: Sendable {
    public let origin: BrowserOrigin?
    public let owner: BrowserPermissionOwner
    public let currentActionApproved: Bool

    public init(
        origin: BrowserOrigin?,
        owner: BrowserPermissionOwner = .agent,
        currentActionApproved: Bool = false
    ) {
        self.origin = origin
        self.owner = owner
        self.currentActionApproved = currentActionApproved
    }
}

public enum BrowserPermissionRuleDecision: Sendable {
    case allow
    case ask
    case deny
    case userOwnedNavigation
}

public enum BrowserPermissionGrantResult: Equatable, Sendable {
    case added
    case alreadyGranted
    case invalidOrigin
    case limitReached
}

/// Validates and queries project origin grants stored in AppProjectPermissions.
public enum BrowserPermissionRuleStore {
    public static let maximumGrantCount = 256

    public static func decision(
        for origin: BrowserOrigin?,
        in permissions: AppProjectPermissions,
        owner: BrowserPermissionOwner = .agent,
        currentActionApproved: Bool = false
    ) -> BrowserPermissionRuleDecision {
        // User-owned navigation is outside the agent permission policy. The
        // engine handles this before applying agent risk and permission gates.
        if case .user = owner { return .userOwnedNavigation }
        guard permissions.mode != .readOnly, permissions.browser != .deny else { return .deny }
        guard let origin else { return .ask }
        if currentActionApproved { return .allow }
        if permissions.browserOriginAllowlist.contains(origin.canonicalString) { return .allow }
        return .ask
    }

    @discardableResult
    public static func grant(
        origin rawOrigin: String,
        to permissions: inout AppProjectPermissions
    ) -> BrowserPermissionGrantResult {
        guard let origin = BrowserOrigin(origin: rawOrigin) else { return .invalidOrigin }

        permissions.browserOriginAllowlist = normalizedAllowlist(permissions.browserOriginAllowlist)
        if permissions.browserOriginAllowlist.contains(origin.canonicalString) { return .alreadyGranted }
        guard permissions.browserOriginAllowlist.count < maximumGrantCount else { return .limitReached }
        permissions.browserOriginAllowlist.append(origin.canonicalString)
        return .added
    }

    @discardableResult
    public static func revoke(
        origin: BrowserOrigin,
        from permissions: inout AppProjectPermissions
    ) -> Bool {
        permissions.browserOriginAllowlist = normalizedAllowlist(permissions.browserOriginAllowlist)
        let existingCount = permissions.browserOriginAllowlist.count
        permissions.browserOriginAllowlist.removeAll { $0 == origin.canonicalString }
        return permissions.browserOriginAllowlist.count != existingCount
    }

    static func normalizedAllowlist(_ origins: [String]) -> [String] {
        var seen = Set<String>()
        var normalized: [String] = []
        normalized.reserveCapacity(min(origins.count, maximumGrantCount))
        for rawOrigin in origins {
            guard let origin = BrowserOrigin(origin: rawOrigin),
                  seen.insert(origin.canonicalString).inserted
            else {
                continue
            }
            normalized.append(origin.canonicalString)
            if normalized.count == maximumGrantCount { break }
        }
        return normalized
    }
}
