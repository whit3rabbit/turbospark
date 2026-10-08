import Foundation

/// What counts as "the default" for the Hugging Face mirror endpoint
/// ($HF_ENDPOINT), shared so the Server Advanced field and
/// `HfAuthTokenCardView`'s own editor cannot answer that question
/// differently.
///
/// Before this existed each view inlined its own trim-and-compare, and the
/// Server Advanced one skipped the live half entirely: it persisted
/// `hfEndpointInput` but never called `TurboSparkCatalog.setHfEndpoint`,
/// which is what every model install, probe and browse call reads outside
/// server context. Editing the SAME setting from that field left the
/// catalog on the stale endpoint until the next app launch, while
/// `HfAuthTokenCardView`'s editor applied it immediately
/// (`swift/docs/SWIFT_SETTINGS_AUDIT.md`).
enum HfEndpointResolution {
    static let defaultEndpoint = "https://huggingface.co"

    /// `nil` for the default endpoint (or an empty/whitespace-only input),
    /// the trimmed mirror URL otherwise -- the exact value
    /// `TurboSparkCatalog.setHfEndpoint` expects.
    ///
    /// An unusable mirror (not a URL, cleartext http to a non-loopback host)
    /// also yields `nil`: the HF bearer token is sent to this host, so an
    /// invalid value fails closed to the default instead of being used.
    static func effectiveEndpoint(from rawInput: String) -> String? {
        let trimmed = rawInput.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, trimmed != defaultEndpoint, isAcceptableMirror(trimmed) else { return nil }
        return trimmed
    }

    /// https with a host, or http only for loopback (a local mirror/proxy).
    static func isAcceptableMirror(_ endpoint: String) -> Bool {
        guard let url = URL(string: endpoint), let host = url.host, !host.isEmpty else { return false }
        switch url.scheme?.lowercased() {
        case "https": return true
        case "http": return ["localhost", "127.0.0.1", "::1"].contains(host.lowercased())
        default: return false
        }
    }

    /// Whether a non-empty input would be rejected by `isAcceptableMirror`.
    static func isRejectedInput(_ rawInput: String) -> Bool {
        let trimmed = rawInput.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.isEmpty && trimmed != defaultEndpoint && !isAcceptableMirror(trimmed)
    }
}
