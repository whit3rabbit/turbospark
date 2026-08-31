import Foundation
import SwiftUI

/// User-defined organization metadata for an installed model.
public struct ModelCustomMetadata: Codable, Equatable, Sendable {
    public var nickname: String?
    public var notes: String
    public var tags: [String]
    public var isFavorite: Bool

    public init(
        nickname: String? = nil,
        notes: String = "",
        tags: [String] = [],
        isFavorite: Bool = false
    ) {
        self.nickname = nickname
        self.notes = notes
        self.tags = tags
        self.isFavorite = isFavorite
    }

    /// Tolerant decode: every field added after the first release is read
    /// with `decodeIfPresent` and a default, for the same reason
    /// `AppChatMessage.init(from:)` is (`swift/CLAUDE.md` Gotcha 13). This
    /// struct decodes as the VALUE of a `[String: ModelCustomMetadata]`
    /// dictionary, so with a synthesized decoder, one entry missing a field
    /// added after that entry was written fails the WHOLE dictionary decode
    /// and silently discards every model's notes, tags, and favorites at once.
    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        nickname = try container.decodeIfPresent(String.self, forKey: .nickname)
        notes = try container.decodeIfPresent(String.self, forKey: .notes) ?? ""
        tags = try container.decodeIfPresent([String].self, forKey: .tags) ?? []
        isFavorite = try container.decodeIfPresent(Bool.self, forKey: .isFavorite) ?? false
    }
}

/// Observable store for managing persistent user notes, tags, and favorites for models.
@MainActor
public final class ModelOrganizationStore: ObservableObject {
    public static let shared = ModelOrganizationStore()
    private static let storageKey = "TurboSpark.modelOrganizationMetadata"

    @Published public private(set) var metadataByModelKey: [String: ModelCustomMetadata] = [:]

    public init() {
        load()
    }

    // MARK: - Key Generation

    /// Stable key identifying a model either by alias or standardized path.
    public func key(for alias: String, path: String? = nil) -> String {
        if let path, !path.isEmpty {
            let std = URL(fileURLWithPath: path).standardizedFileURL.path
            return "\(alias)::\(std)"
        }
        return alias
    }

    // MARK: - Querying

    public func metadata(for alias: String, path: String? = nil) -> ModelCustomMetadata {
        let fullKey = key(for: alias, path: path)
        if let exact = metadataByModelKey[fullKey] {
            return exact
        }
        // Fallback to alias alone
        if let fallback = metadataByModelKey[alias] {
            return fallback
        }
        return ModelCustomMetadata()
    }

    public func isFavorite(for alias: String, path: String? = nil) -> Bool {
        metadata(for: alias, path: path).isFavorite
    }

    public func isFavorite(alias: String, path: String? = nil) -> Bool {
        isFavorite(for: alias, path: path)
    }

    public func notes(for alias: String, path: String? = nil) -> String {
        metadata(for: alias, path: path).notes
    }

    public func notes(alias: String, path: String? = nil) -> String {
        notes(for: alias, path: path)
    }

    public func tags(for alias: String, path: String? = nil) -> [String] {
        metadata(for: alias, path: path).tags
    }

    public func tags(alias: String, path: String? = nil) -> [String] {
        tags(for: alias, path: path)
    }

    public func nickname(for alias: String, path: String? = nil) -> String? {
        metadata(for: alias, path: path).nickname
    }

    public func nickname(alias: String, path: String? = nil) -> String? {
        nickname(for: alias, path: path)
    }

    /// Returns all distinct tags across all organized models.
    public var allKnownTags: [String] {
        var tagsSet = Set<String>()
        for meta in metadataByModelKey.values {
            for tag in meta.tags {
                tagsSet.insert(tag)
            }
        }
        return Array(tagsSet).sorted()
    }

    // MARK: - Mutations

    public func toggleFavorite(alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.isFavorite.toggle()
        metadataByModelKey[k] = meta
        save()
    }

    public func setFavorite(_ favorite: Bool, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.isFavorite = favorite
        metadataByModelKey[k] = meta
        save()
    }

    public func setNotes(_ notes: String, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.notes = notes
        metadataByModelKey[k] = meta
        save()
    }

    public func setNickname(_ nickname: String?, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        let trimmed = nickname?.trimmingCharacters(in: .whitespacesAndNewlines)
        meta.nickname = (trimmed?.isEmpty == false) ? trimmed : nil
        metadataByModelKey[k] = meta
        save()
    }

    public func addTag(_ tag: String, for alias: String, path: String? = nil) {
        let trimmed = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        if !meta.tags.contains(trimmed) {
            meta.tags.append(trimmed)
            metadataByModelKey[k] = meta
            save()
        }
    }

    public func removeTag(_ tag: String, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.tags.removeAll { $0 == tag }
        metadataByModelKey[k] = meta
        save()
    }

    public func removeMetadata(for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        metadataByModelKey.removeValue(forKey: k)
        metadataByModelKey.removeValue(forKey: alias)
        save()
    }

    // MARK: - Persistence

    private func save() {
        if let data = try? JSONEncoder().encode(metadataByModelKey) {
            UserDefaults.standard.set(data, forKey: Self.storageKey)
        }
    }

    private func load() {
        guard let data = UserDefaults.standard.data(forKey: Self.storageKey) else { return }
        do {
            self.metadataByModelKey = try JSONDecoder().decode([String: ModelCustomMetadata].self, from: data)
        } catch {
            // Reported rather than swallowed (`swift/CLAUDE.md` Gotcha 13):
            // a silent `try?` here reads as "my notes and favorites vanished"
            // with nothing to say why, and this store's own `save()` would
            // happily overwrite the UserDefaults entry with that emptiness
            // on the next mutation.
            FileHandle.standardError.write(
                "TurboSpark: model organization metadata failed to decode, starting empty: \(error)\n"
                    .data(using: .utf8)!)
        }
    }
}
