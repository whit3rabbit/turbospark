import Foundation
import TurboSpark

/// Saved history retains the original source so Retry cannot change models.
public struct ModelDownload: Identifiable, Codable, Equatable {
    public enum Request: Codable, Equatable {
        case catalog(alias: String)
        case repository(repo: String, alias: String, file: String?, sidecarRepo: String?)
        case image(alias: String)
        case audio(identity: AudioProfileIdentity)
        /// A pinned embedding profile (`EmbeddingCatalog`), by alias.
        case embedding(alias: String)

        var alias: String {
            switch self {
            case .catalog(let alias), .repository(_, let alias, _, _), .image(let alias),
                 .embedding(let alias):
                return alias
            case .audio(let identity): return identity.alias
            }
        }

        /// Embedding models install only into This Mac and offer no destination
        /// (requirement 2.5). The engine refuses a library for them too.
        var installsOnlyOnThisMac: Bool {
            if case .embedding = self { return true }
            return false
        }

        /// Catalog and custom text installs share an output namespace. Image
        /// installs do not, so the same alias may exist once in each store.
        var queueKey: String {
            let normalized = alias.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            switch self {
            case .catalog, .repository: return "text:\(normalized)"
            case .image: return "image:\(normalized)"
            case .embedding: return "embedding:\(normalized)"
            case .audio(let identity):
                // Pin every identity field so a retry cannot silently adopt a new catalog revision.
                let fields = [identity.task.rawValue, normalized, identity.repository,
                              identity.revision, identity.assetFingerprint]
                return "audio:" + fields.map { "\($0.utf8.count):\($0)" }.joined()
            }
        }
    }

    public enum Status: String, Codable {
        case queued, running, paused, packing, verifying, loading, cancelling
        case completed, cancelled, failed, interrupted

        var isTerminal: Bool {
            self == .completed || self == .cancelled || self == .failed || self == .interrupted
        }

        var canRetry: Bool { self == .failed || self == .cancelled || self == .interrupted }

        var afterRelaunch: Status {
            switch self {
            case .queued, .running, .paused, .packing, .verifying: return .interrupted
            case .cancelling: return .cancelled
            // Loading starts only after the engine has committed the install.
            case .loading: return .completed
            default: return self
            }
        }
    }

    public let id: UUID
    public let request: Request
    /// The `ModelLibrary.id` the model is installed into. Nil is This Mac, and
    /// the key is then absent from the saved history, so a row written before
    /// libraries and a This Mac row are the same bytes.
    ///
    /// **A RETRY GOES WHERE THE FIRST TRY WENT.** The destination lives on the
    /// row with the request, not on the screen that started it, so a queued
    /// install, a failed one and a restart all keep it. It is not part of
    /// `Request.queueKey`: alias plus modality is unique across every library.
    /// Embedding and OCR rows never carry one (requirement 2.5).
    public var library: String?
    public var status: Status
    public var failure: String?
    public var downloadedBytes: UInt64 = 0
    public var totalBytes: UInt64?
    public var startedAt: Date
    public var updatedAt: Date

    public init(
        request: Request, status: Status = .running, library: String? = nil, now: Date = Date()
    ) {
        self.id = UUID()
        self.request = request
        self.library = library
        self.status = status
        self.startedAt = now
        self.updatedAt = now
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        request = try container.decode(Request.self, forKey: .request)
        library = try container.decodeIfPresent(String.self, forKey: .library)
        let savedStatus = try container.decodeIfPresent(String.self, forKey: .status)
        status = savedStatus.flatMap(Status.init(rawValue:)) ?? .interrupted
        failure = try container.decodeIfPresent(String.self, forKey: .failure)
        downloadedBytes = try container.decodeIfPresent(UInt64.self, forKey: .downloadedBytes) ?? 0
        totalBytes = try container.decodeIfPresent(UInt64.self, forKey: .totalBytes)
        startedAt = try container.decodeIfPresent(Date.self, forKey: .startedAt) ?? .distantPast
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? startedAt
    }
}
