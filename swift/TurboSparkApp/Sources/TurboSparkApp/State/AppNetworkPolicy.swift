import Foundation

/// How tool calls reach the network (Settings > Network).
public enum AppNetworkMode: String, Codable, CaseIterable, Identifiable, Sendable {
    /// Allowlisted domains run, every other domain asks.
    case ask
    /// Every public domain runs except the blocklist.
    case allowAll
    /// No tool network access. Shell commands run under a Seatbelt profile.
    case blockAll
    /// Allowlist runs, blocklist is refused, unlisted domains follow `customUnlisted`.
    case custom

    public var id: String { rawValue }
}

/// Outcome of the network policy for one destination.
public enum AppNetworkDecision: Equatable, Sendable {
    case allow
    case ask(reason: String)
    case deny(reason: String)
}

/// The app-wide network policy for tool calls.
///
/// **A CEILING OVER PROJECT PERMISSIONS, NOT A DEFAULT UNDER THEM.**
/// `AppToolPermissionEngine` applies a deny here in every project mode,
/// Full access included, and an ask here turns any project-level allow back
/// into an ask. A project can only be stricter (`web = .deny`).
///
/// Coverage is narrower than the name suggests and the Settings pane says
/// so: URL tools, remote MCP servers and agent browser navigation follow the
/// whole policy; shell commands are confined only under Block all; stdio MCP
/// servers, plugins and hooks are not covered.
public struct AppNetworkPolicy: Codable, Equatable, Sendable {
    public var mode: AppNetworkMode
    /// Loopback (`localhost`, 127/8, `::1`) is reachable in every mode,
    /// Block all included. On by default: local dev servers are the common
    /// case and loopback cannot exfiltrate on its own.
    public var allowLocalhost: Bool
    /// RFC 1918, CGNAT, `.local` and other non-public addresses. Off by
    /// default: a LAN holds routers and NAS admin pages.
    public var allowLocalNetwork: Bool
    /// `example.com` also matches its subdomains, `*.example.com` only them.
    public var allowedDomains: [String]
    /// Wins over `allowedDomains`.
    public var deniedDomains: [String]
    /// Custom mode only: what an unlisted domain does (`.allow` is not offered).
    public var customUnlisted: AppToolPermission
    /// Custom mode only: refuse any HTTP method other than GET, HEAD, OPTIONS.
    public var readOnlyMethods: Bool

    public init(
        mode: AppNetworkMode = .ask,
        allowLocalhost: Bool = true,
        allowLocalNetwork: Bool = false,
        allowedDomains: [String] = [],
        deniedDomains: [String] = [],
        customUnlisted: AppToolPermission = .ask,
        readOnlyMethods: Bool = false
    ) {
        self.mode = mode
        self.allowLocalhost = allowLocalhost
        self.allowLocalNetwork = allowLocalNetwork
        self.allowedDomains = allowedDomains
        self.deniedDomains = deniedDomains
        self.customUnlisted = customUnlisted
        self.readOnlyMethods = readOnlyMethods
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        let defaults = AppNetworkPolicy()
        mode = c.decodeLenient(AppNetworkMode.self, forKey: .mode, fallback: defaults.mode)
        allowLocalhost = c.decodeLenient(Bool.self, forKey: .allowLocalhost, fallback: defaults.allowLocalhost)
        allowLocalNetwork = c.decodeLenient(
            Bool.self, forKey: .allowLocalNetwork, fallback: defaults.allowLocalNetwork)
        allowedDomains = c.decodeLenient([String].self, forKey: .allowedDomains, fallback: [])
            .compactMap(Self.normalizedDomainEntry)
        deniedDomains = c.decodeLenient([String].self, forKey: .deniedDomains, fallback: [])
            .compactMap(Self.normalizedDomainEntry)
        let unlisted = c.decodeLenient(
            AppToolPermission.self, forKey: .customUnlisted, fallback: defaults.customUnlisted)
        // An edited file must not turn Custom into Allow all by the back door.
        customUnlisted = unlisted == .allow ? .ask : unlisted
        readOnlyMethods = c.decodeLenient(Bool.self, forKey: .readOnlyMethods, fallback: false)
    }

    private enum CodingKeys: String, CodingKey {
        case mode, allowLocalhost, allowLocalNetwork, allowedDomains, deniedDomains
        case customUnlisted, readOnlyMethods
    }

    /// Domains the Custom pane offers in one click: source hosts and package
    /// registries a coding agent routinely needs.
    public static let commonDeveloperDomains = [
        "github.com", "raw.githubusercontent.com", "pypi.org", "files.pythonhosted.org",
        "registry.npmjs.org", "crates.io", "static.crates.io", "huggingface.co",
        "developer.apple.com", "docs.rs",
    ]

    static let readOnlyHTTPMethods: Set<String> = ["GET", "HEAD", "OPTIONS"]

    /// Whether any tool network traffic is permitted at all. Used for calls
    /// with no model-chosen host, such as a web search provider endpoint.
    public var permitsOutboundTraffic: Bool { mode != .blockAll }

    /// The decision for a request to `host`. `method` is the HTTP method when
    /// the tool sends one, nil for tools that only read.
    public func decision(host rawHost: String, method: String? = nil) -> AppNetworkDecision {
        let host = Self.normalizedHost(rawHost)
        guard !host.isEmpty else { return .deny(reason: "Network request has no host.") }

        switch AppToolSandbox.hostScope(host) {
        case .metadata:
            return .deny(reason: "Access to cloud metadata or link-local host '\(host)' is always blocked.")
        case .loopback:
            // Loopback skips the domain lists and the mode: "Allow localhost"
            // is its whole policy, so Block all still reaches a dev server.
            return allowLocalhost
                ? .allow
                : .deny(reason: "Localhost access is turned off in Network settings ('\(host)').")
        case .privateNetwork:
            guard allowLocalNetwork else {
                return .deny(reason: "Local network access is turned off in Network settings ('\(host)').")
            }
        case .publicHost:
            break
        }

        if Self.matches(host, deniedDomains) {
            return .deny(reason: "Domain '\(host)' is blocked in Network settings.")
        }
        if mode == .custom, readOnlyMethods,
           let method = method?.uppercased(), !Self.readOnlyHTTPMethods.contains(method)
        {
            return .deny(reason: "Network settings allow only read-only requests (GET, HEAD, OPTIONS), not \(method).")
        }

        switch mode {
        case .blockAll:
            return .deny(reason: "Network access for tools is blocked in Network settings ('\(host)').")
        case .allowAll:
            return .allow
        case .ask:
            return Self.matches(host, allowedDomains)
                ? .allow : .ask(reason: "Network request to '\(host)', which is not on the allowed list.")
        case .custom:
            if Self.matches(host, allowedDomains) { return .allow }
            return customUnlisted == .deny
                ? .deny(reason: "Domain '\(host)' is not on the allowed list in Network settings.")
                : .ask(reason: "Network request to '\(host)', which is not on the allowed list.")
        }
    }

    /// Lowercased, bracket- and trailing-dot-free host for matching and display.
    public static func normalizedHost(_ host: String) -> String {
        var clean = host.lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .trimmingCharacters(in: CharacterSet(charactersIn: "[]"))
        while clean.hasSuffix(".") { clean.removeLast() }
        return clean
    }

    /// Validates and normalizes a list entry typed by the user. Returns nil
    /// for anything that is not a bare host, optionally with a leading `*.`:
    /// a scheme, path, port, credentials, whitespace or an inner wildcard.
    public static func normalizedDomainEntry(_ entry: String) -> String? {
        let clean = normalizedHost(entry)
        let body = clean.hasPrefix("*.") ? String(clean.dropFirst(2)) : clean
        guard !body.isEmpty, body.count <= 253,
              !body.hasPrefix("."), !body.contains(".."),
              body.allSatisfy({ $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "." || $0 == "-") })
        else { return nil }
        return clean
    }

    /// `example.com` matches itself and every subdomain; `*.example.com`
    /// matches only subdomains.
    static func matches(_ host: String, _ entries: [String]) -> Bool {
        entries.contains { entry in
            if entry.hasPrefix("*.") {
                return host.hasSuffix(String(entry.dropFirst(1)))
            }
            return host == entry || host.hasSuffix("." + entry)
        }
    }

    /// Whether a connection the socket really made is acceptable for a URL
    /// whose own host is local (`urlHost` loopback or LAN).
    ///
    /// **A PUBLIC NAME NEVER EARNS THE LOCAL EXCEPTION.** If `urlHost` is a
    /// public name, this returns false and the caller's private-address
    /// refusal stands, so DNS that answers 127.0.0.1 for `evil.example`
    /// (rebinding) is still refused with localhost allowed. Same stance as
    /// Chrome's Private Network Access.
    public func permitsLocalConnection(urlHost: String, address: String) -> Bool {
        let urlScope = AppToolSandbox.hostScope(urlHost)
        let addressScope = AppToolSandbox.hostScope(address)
        guard !address.isEmpty else { return false }
        switch urlScope {
        case .loopback:
            return allowLocalhost && addressScope == .loopback
        case .privateNetwork:
            guard allowLocalNetwork else { return false }
            return addressScope == .privateNetwork || (addressScope == .loopback && allowLocalhost)
        case .metadata, .publicHost:
            return false
        }
    }

    /// The host a web tool call addresses, or nil for a call with no URL
    /// (a search query) or an unparseable one.
    public static func host(forWebCall call: AppToolCall) -> String? {
        guard call.category == .web,
              let raw = ToolApprovalPreviewSummary.httpURL(call.arguments),
              let host = URL(string: raw.trimmingCharacters(in: .whitespacesAndNewlines))?.host,
              !host.isEmpty
        else { return nil }
        return normalizedHost(host)
    }
}

/// Process-wide copy of the active policy for code that holds no `AppModel`
/// (executors, subagents, the permission engine's default argument).
/// `AppModel.networkPolicy` writes it on every change.
public enum AppNetworkPolicyStore {
    private static let lock = NSLock()
    nonisolated(unsafe) private static var value = AppNetworkPolicy()

    public static var current: AppNetworkPolicy {
        get { lock.lock(); defer { lock.unlock() }; return value }
        set { lock.lock(); value = newValue; lock.unlock() }
    }
}
