import Foundation

/// Saved history retains the original source so Retry cannot change models.
public struct ModelDownload: Identifiable, Codable, Equatable {
    public enum Request: Codable, Equatable {
        case catalog(alias: String)
        case repository(repo: String, alias: String, file: String?, sidecarRepo: String?)
        case image(alias: String)

        var alias: String {
            switch self {
            case .catalog(let alias), .repository(_, let alias, _, _), .image(let alias):
                return alias
            }
        }

        /// Catalog and custom text installs share an output namespace. Image
        /// installs do not, so the same alias may exist once in each store.
        var queueKey: String {
            let normalized = alias.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            switch self {
            case .catalog, .repository: return "text:\(normalized)"
            case .image: return "image:\(normalized)"
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
    public var status: Status
    public var failure: String?
    public var downloadedBytes: UInt64 = 0
    public var totalBytes: UInt64?
    public var startedAt: Date
    public var updatedAt: Date

    public init(request: Request, status: Status = .running, now: Date = Date()) {
        self.id = UUID()
        self.request = request
        self.status = status
        self.startedAt = now
        self.updatedAt = now
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        id = try container.decode(UUID.self, forKey: .id)
        request = try container.decode(Request.self, forKey: .request)
        let savedStatus = try container.decodeIfPresent(String.self, forKey: .status)
        status = savedStatus.flatMap(Status.init(rawValue:)) ?? .interrupted
        failure = try container.decodeIfPresent(String.self, forKey: .failure)
        downloadedBytes = try container.decodeIfPresent(UInt64.self, forKey: .downloadedBytes) ?? 0
        totalBytes = try container.decodeIfPresent(UInt64.self, forKey: .totalBytes)
        startedAt = try container.decodeIfPresent(Date.self, forKey: .startedAt) ?? .distantPast
        updatedAt = try container.decodeIfPresent(Date.self, forKey: .updatedAt) ?? startedAt
    }
}
