import Foundation

/// Where a provider request may go.
///
/// The rules mirror the WebFetch validator's posture (TLS, no credentials in
/// the URL, a re-check on every redirect hop) with one deliberate difference:
/// the private-address refusal is NOT applied, because the user configured
/// this host on purpose (a LAN gateway, a local OpenAI-compatible server).
/// What replaces it is explicitness: a provider is per-chat opt-in, and a
/// plain-http URL is accepted only for loopback.
enum RemoteEndpointPolicy {
    static let loopbackHosts: Set<String> = ["localhost", "127.0.0.1", "::1"]

    static func isLoopback(_ host: String) -> Bool {
        var h = host.lowercased()
        if h.hasPrefix("["), h.hasSuffix("]") { h = String(h.dropFirst().dropLast()) }
        return loopbackHosts.contains(h) || h.hasSuffix(".localhost")
    }

    /// Validates a base URL at save time and again before every send.
    /// A query string is refused because it is where a key ends up when a
    /// user pastes a "key=..." URL; a fragment and userinfo likewise.
    static func validate(baseURL raw: String) throws -> URL {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { throw RemoteProviderError.invalidBaseURL("empty") }
        guard let url = URL(string: trimmed), let scheme = url.scheme?.lowercased(),
            let host = url.host, !host.isEmpty
        else { throw RemoteProviderError.invalidBaseURL("not a URL") }
        guard url.user == nil, url.password == nil else {
            throw RemoteProviderError.invalidBaseURL("credentials in the URL")
        }
        guard url.query == nil, url.fragment == nil else {
            throw RemoteProviderError.invalidBaseURL("query or fragment")
        }
        switch scheme {
        case "https":
            break
        case "http":
            guard isLoopback(host) else {
                throw RemoteProviderError.invalidBaseURL("http is only allowed for localhost")
            }
        default:
            throw RemoteProviderError.invalidBaseURL("scheme must be https")
        }
        return url
    }

    /// The request URL for a family under a validated base. A base that
    /// already ends in `/v1` is not doubled.
    static func endpoint(base: URL, family: RemoteAPIFamily, probe: Bool = false) -> URL {
        var path = base.path
        while path.hasSuffix("/") { path.removeLast() }
        let v1 = path.hasSuffix("/v1")
        let tail: String
        switch (family, probe) {
        case (.anthropic, false): tail = v1 ? "/messages" : "/v1/messages"
        case (.anthropic, true): tail = v1 ? "/models" : "/v1/models"
        case (.openAICompatible, false): tail = "/chat/completions"
        case (.openAICompatible, true): tail = "/models"
        }
        var comps = URLComponents(url: base, resolvingAgainstBaseURL: false) ?? URLComponents()
        comps.path = path + tail
        return comps.url ?? base
    }

    /// Whether a redirect hop may be followed. The credentials are attached
    /// per hop by the caller, and only when this says yes: same host, same
    /// port, and never https to http. Anything else is refused outright
    /// (dropping the header and following would still replay the body).
    static func mayFollow(from: URL, to: URL) -> Bool {
        guard let fh = from.host?.lowercased(), let th = to.host?.lowercased(), fh == th else {
            return false
        }
        let fromScheme = from.scheme?.lowercased()
        let toScheme = to.scheme?.lowercased()
        guard toScheme == "https" || (toScheme == "http" && isLoopback(th)) else { return false }
        if fromScheme == "https", toScheme != "https" { return false }
        return effectivePort(from) == effectivePort(to)
    }

    private static func effectivePort(_ url: URL) -> Int {
        if let p = url.port { return p }
        return url.scheme?.lowercased() == "http" ? 80 : 443
    }
}
