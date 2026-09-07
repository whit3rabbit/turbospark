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
    static func effectiveEndpoint(from rawInput: String) -> String? {
        let trimmed = rawInput.trimmingCharacters(in: .whitespacesAndNewlines)
        return trimmed.isEmpty || trimmed == defaultEndpoint ? nil : trimmed
    }
}
