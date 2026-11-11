import Foundation

/// How a Hugging Face search ended.
///
/// **A failed request is a status, not a thrown error**, so a UI can say
/// "you are offline" and "slow down" differently from "something broke". A
/// thrown `TurboSparkError` means the CALL was malformed (a one-letter query,
/// an unknown option), which is the caller's bug rather than the Hub's answer.
public enum HubSearchStatus: String, Decodable, Sendable {
    case ok
    case offline
    case rateLimited
    case needsAuth
    case error

    /// An unknown spelling is `error` rather than a decode failure, so a
    /// library newer than this binding degrades to a generic message instead
    /// of an empty pane.
    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: raw) ?? .error
    }
}

/// What this engine's architecture registry says about a search row's
/// architecture string, WITHOUT reading the checkpoint.
///
/// **`recognized` IS NOT "compatible".** It means this port knows the name.
/// Quantization, a missing tokenizer and the dense-versus-MoE split of one
/// GGUF architecture are decided only by `TurboSparkCatalog.hubCheck`. Render
/// it as "architecture recognized". And `unknown` is not a negative: a
/// safetensors row only carries a `model_type`, and the registry's table for
/// those holds supported rows only.
public enum HubArchHint: String, Decodable, Sendable {
    case recognized
    /// Known and not yet ported. `HubSearchResult.archNeeds` is the next step.
    case planned
    /// A GGUF architecture string this port has never seen.
    case unrecognized
    case unknown

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: raw) ?? .unknown
    }
}

/// One live Hugging Face row. **Advisory only**: nothing here says the model
/// runs. Call `TurboSparkCatalog.hubCheck` for the row the user selects.
public struct HubSearchResult: Decodable, Sendable, Equatable, Identifiable {
    public var id: String { repoId }
    public let repoId: String
    /// The immutable commit the Hub reported for this listing.
    public let revision: String
    public let downloads: UInt64?
    public let likes: UInt64?
    public let hasGguf: Bool
    public let hasMlx: Bool
    /// Gated or licence-walled: an unauthenticated check will say `needsAuth`.
    public let gated: Bool
    /// The repository a conversion was made from. Pass it to `hubCheck` as
    /// `sidecarHint`: GGUF repositories usually lack `tokenizer.json`.
    public let baseModel: String?
    /// `general.architecture` for a GGUF row, `model_type` for safetensors.
    public let architecture: String?
    public let archHint: HubArchHint
    /// For a `planned` row, the registry's own missing-work clause.
    public let archNeeds: String?
    /// Curated catalog aliases for this same repository, so a UI can offer the
    /// verified install instead of a probed one.
    public let bundledAliases: [String]
}

/// One page of Hugging Face search results.
public struct HubSearchPage: Decodable, Sendable, Equatable {
    public let status: HubSearchStatus
    /// The failure in words, when `status` is not `ok`.
    public let message: String?
    /// Seconds to wait before retrying, when `status` is `rateLimited` and
    /// the Hub said.
    public let retryAfterSecs: UInt64?
    /// Unix seconds the list was fetched, which for a cached page is earlier
    /// than now.
    public let fetchedAt: UInt64?
    public let fromCache: Bool?
    /// Rows the response validator refused, never shown. Nonzero is worth a
    /// log line, not a UI.
    public let rejected: Int?
    public let results: [HubSearchResult]
}

/// The verdict of `TurboSparkCatalog.hubCheck`.
///
/// **BRANCH ON THIS.** `needsAuth` and `unavailable` are NOT verdicts on the
/// checkpoint: a gated repository, or no answer. Never render them as
/// "incompatible".
public enum HubCheckState: String, Decodable, Sendable {
    case runnable
    /// A probe gate failed. `HubCheck.reason` is the probe's own words.
    case refused
    case needsAuth
    case unavailable

    public init(from decoder: Decoder) throws {
        let raw = try decoder.singleValueContainer().decode(String.self)
        self = Self(rawValue: raw) ?? .unavailable
    }
}

/// What this engine makes of ONE Hugging Face repository.
public struct HubCheck: Decodable, Sendable, Equatable {
    public let repo: String
    public let revision: String
    public let state: HubCheckState
    /// The refusal or failure in words. `nil` when runnable.
    public let reason: String?
    /// `"gguf"` or `"mlx"`, or `nil` when no report was produced.
    public let kind: String?
    /// The representative GGUF the header was read from, which is the best
    /// quality file with kernels. The user's pick may differ; see `variants`.
    public let file: String?
    public let architecture: String?
    public let family: String?
    public let downloadBytes: UInt64?
    public let trainedContext: UInt32?
    public let chatTemplate: String?
    public let warnings: [String]
    public let sidecarsMissing: [String]
    /// Set only when the weights repository had no `tokenizer.json` and the
    /// hint supplied one that does. **An install of this result must pass the
    /// same repository as `sidecarRepo`.**
    public let sidecarRepo: String?
    /// Every pickable `.gguf`, best quality first. Empty for safetensors.
    public let variants: [RepoVariant]
    public let shardedSkipped: Int
    /// This machine's answer, or `nil` when no shape could be read.
    public let fit: ProbeFit?
}

/// Result ordering for `TurboSparkCatalog.hubSearch`.
public enum HubSearchSort: String, Encodable, Sendable, CaseIterable {
    /// The order that surfaces installable repositories rather than new ones.
    case downloads
    case trending
    case likes
}

/// Which published weight format a search is restricted to.
public enum HubSearchFormat: String, Encodable, Sendable, CaseIterable {
    case any
    case gguf
    case mlx
}
