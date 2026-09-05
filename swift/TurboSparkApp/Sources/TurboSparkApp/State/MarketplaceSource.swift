import Foundation

/// Where a marketplace catalog, or one entry in it, is acquired from.
///
/// Shared by the skills marketplace and the MCP one. It was
/// `SkillMarketplaceSource` and lived in `SkillMarketplaceManager.swift`
/// until the MCP marketplace needed the identical four cases. One type
/// rather than two, for the reason the single `ToolCallParser` exists:
/// a fix to how a git ref or a sparse path is read must reach both.
public enum MarketplaceSource: Codable, Equatable, Sendable {
    case url(url: String, headers: [String: String]?)
    case github(repo: String, ref: String?, path: String?, sparsePaths: [String]?)
    case git(url: String, ref: String?, path: String?, sparsePaths: [String]?)
    case directory(path: String)

    enum CodingKeys: String, CodingKey {
        case type, url, headers, repo, ref, path, sparsePaths
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type.lowercased() {
        case "url", "https", "http":
            let url = try container.decode(String.self, forKey: .url)
            let headers = try container.decodeIfPresent([String: String].self, forKey: .headers)
            self = .url(url: url, headers: headers)
        case "github":
            let repo = try container.decode(String.self, forKey: .repo)
            let ref = try container.decodeIfPresent(String.self, forKey: .ref)
            let path = try container.decodeIfPresent(String.self, forKey: .path)
            let sparse = try container.decodeIfPresent([String].self, forKey: .sparsePaths)
            self = .github(repo: repo, ref: ref, path: path, sparsePaths: sparse)
        case "git":
            let url = try container.decode(String.self, forKey: .url)
            let ref = try container.decodeIfPresent(String.self, forKey: .ref)
            let path = try container.decodeIfPresent(String.self, forKey: .path)
            let sparse = try container.decodeIfPresent([String].self, forKey: .sparsePaths)
            self = .git(url: url, ref: ref, path: path, sparsePaths: sparse)
        case "directory", "file", "local":
            let path = try container.decode(String.self, forKey: .path)
            self = .directory(path: path)
        default:
            let url = try container.decodeIfPresent(String.self, forKey: .url) ?? ""
            self = .url(url: url, headers: nil)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .url(let url, let headers):
            try container.encode("url", forKey: .type)
            try container.encode(url, forKey: .url)
            try container.encodeIfPresent(headers, forKey: .headers)
        case .github(let repo, let ref, let path, let sparse):
            try container.encode("github", forKey: .type)
            try container.encode(repo, forKey: .repo)
            try container.encodeIfPresent(ref, forKey: .ref)
            try container.encodeIfPresent(path, forKey: .path)
            try container.encodeIfPresent(sparse, forKey: .sparsePaths)
        case .git(let url, let ref, let path, let sparse):
            try container.encode("git", forKey: .type)
            try container.encode(url, forKey: .url)
            try container.encodeIfPresent(ref, forKey: .ref)
            try container.encodeIfPresent(path, forKey: .path)
            try container.encodeIfPresent(sparse, forKey: .sparsePaths)
        case .directory(let path):
            try container.encode("directory", forKey: .type)
            try container.encode(path, forKey: .path)
        }
    }
}
