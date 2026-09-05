import Foundation
import SwiftUI

/// User-defined organization metadata for an installed model.
public struct ModelCustomMetadata: Codable, Equatable, Sendable {
    /// Custom display nickname for the model.
    public var nickname: String?
    /// User notes and comments regarding the model.
    public var notes: String
    /// Tags assigned to the model for filtering and categorization.
    public var tags: [String]
    /// Whether the model is pinned or marked as a favorite.
    public var isFavorite: Bool

    /// Initializes custom metadata with optional nickname, notes, tags, and favorite status.
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
    /// Shared singleton instance of the store.
    public static let shared = ModelOrganizationStore()
    private static let storageKey = "TurboSpark.modelOrganizationMetadata"

    /// User metadata indexed by composite model key or alias.
    @Published public private(set) var metadataByModelKey: [String: ModelCustomMetadata] = [:]

    /// Scanned paths the user has removed from TurboSpark (state#88).
    ///
    /// **REMOVING A SCANNED ROW IS NOT DELETING A MODEL.** A row that only
    /// ever came from walking LM Studio's library or a custom folder has no
    /// bytes this app may remove, so "Remove from TurboSpark" dropped its
    /// notes, tags and favorite and left the file alone -- and the very next
    /// scan walked the same directory and put the row back, now stripped.
    /// The exclusion is what the button actually means, it is reversible from
    /// the Storage pane, and the metadata stays where it was.
    ///
    /// Standardized paths, so the same directory reached two ways is one
    /// entry.
    @Published public private(set) var excludedScanPaths: Set<String> = []

    /// Initializes the store and loads stored metadata and exclusions.
    public init() {
        load()
        loadExclusions()
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

    /// Returns custom metadata for a model, falling back to alias alone or a default instance.
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

    /// Returns whether the specified model is marked as a favorite.
    public func isFavorite(for alias: String, path: String? = nil) -> Bool {
        metadata(for: alias, path: path).isFavorite
    }

    /// Convenience overload returning whether the specified model is marked as a favorite.
    public func isFavorite(alias: String, path: String? = nil) -> Bool {
        isFavorite(for: alias, path: path)
    }

    /// Returns user notes recorded for the specified model.
    public func notes(for alias: String, path: String? = nil) -> String {
        metadata(for: alias, path: path).notes
    }

    /// Convenience overload returning user notes recorded for the specified model.
    public func notes(alias: String, path: String? = nil) -> String {
        notes(for: alias, path: path)
    }

    /// Returns user-defined tags assigned to the specified model.
    public func tags(for alias: String, path: String? = nil) -> [String] {
        metadata(for: alias, path: path).tags
    }

    /// Convenience overload returning user-defined tags assigned to the specified model.
    public func tags(alias: String, path: String? = nil) -> [String] {
        tags(for: alias, path: path)
    }

    /// Returns the custom nickname assigned to the specified model, if any.
    public func nickname(for alias: String, path: String? = nil) -> String? {
        metadata(for: alias, path: path).nickname
    }

    /// Convenience overload returning the custom nickname for the specified model.
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

    /// Toggles the favorite status for the specified model and persists changes.
    public func toggleFavorite(alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.isFavorite.toggle()
        metadataByModelKey[k] = meta
        save()
    }

    /// Sets the favorite status for the specified model and persists changes.
    public func setFavorite(_ favorite: Bool, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.isFavorite = favorite
        metadataByModelKey[k] = meta
        save()
    }

    /// Sets user notes for the specified model and persists changes.
    public func setNotes(_ notes: String, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.notes = notes
        metadataByModelKey[k] = meta
        save()
    }

    /// Updates or clears the nickname for the specified model and persists changes.
    public func setNickname(_ nickname: String?, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        let trimmed = nickname?.trimmingCharacters(in: .whitespacesAndNewlines)
        meta.nickname = (trimmed?.isEmpty == false) ? trimmed : nil
        metadataByModelKey[k] = meta
        save()
    }

    /// Adds a unique trimmed tag to the specified model and persists changes.
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

    /// Removes a tag from the specified model and persists changes.
    public func removeTag(_ tag: String, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.tags.removeAll { $0 == tag }
        metadataByModelKey[k] = meta
        save()
    }

    /// Drops one model's metadata.
    ///
    /// **THE BARE-ALIAS ROW IS SHARED AND IS NOT THIS MODEL'S TO DELETE**
    /// (state#64). `key(for:path:)` prefers the path, and `metadata(for:)`
    /// falls back to the bare alias -- which is how a row written before
    /// paths were keys still resolves. Removing it here deleted the metadata
    /// of every OTHER model that resolves through the same fallback: two LM
    /// Studio installs sharing an alias, or the legacy row for a model that
    /// was never re-tagged. It is dropped only when this call is itself the
    /// bare-alias one, i.e. when no path was given.
    public func removeMetadata(for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        metadataByModelKey.removeValue(forKey: k)
        if path == nil || path?.isEmpty == true {
            metadataByModelKey.removeValue(forKey: alias)
        }
        save()
    }

    // MARK: - Scan exclusions

    /// Returns whether a standardized filesystem path is excluded from model scanning.
    public func isExcludedFromScan(path: String) -> Bool {
        excludedScanPaths.contains(Self.standardized(path))
    }

    /// Excludes a filesystem path from subsequent model scans.
    public func excludeFromScan(path: String) {
        excludedScanPaths.insert(Self.standardized(path))
        saveExclusions()
    }

    /// Restores a previously excluded filesystem path to model scans.
    public func restoreToScan(path: String) {
        excludedScanPaths.remove(Self.standardized(path))
        saveExclusions()
    }

    private static func standardized(_ path: String) -> String {
        URL(fileURLWithPath: ModelStorageManager.expandPath(path)).standardizedFileURL.path
    }

    // MARK: - Persistence

    /// The one store that `AppStorageRoot` did not cover.
    ///
    /// **IT WROTE `UserDefaults`, WHICH THE TEST REDIRECT CANNOT REACH.**
    /// `swift/CLAUDE.md` Gotcha 37 is about seven stores that each spelled
    /// their own Application Support path and so wrote real user data from
    /// the test suite; the fix routed all seven through `AppStorageRoot`, and
    /// this one was missed because it is not a path at all. A test that
    /// deletes a model calls `removeMetadata`, which mutated the developer's
    /// real nicknames and favorites. It also moves domains between `swift
    /// run` and the shipped bundle (Gotcha 12), so the two builds disagreed
    /// about the same user's metadata.
    private static var fileURL: URL { AppStorageRoot.file("model_organization.json") }

    /// **A SECOND FILE RATHER THAN A NEW ROOT SHAPE.** This archive's root IS
    /// the `[String: ModelCustomMetadata]` dictionary, so wrapping it to add a
    /// field would make every existing file fail to decode -- and
    /// `AppJSONStore.load` returns the empty default on a decode failure,
    /// which is `swift/CLAUDE.md` Gotcha 13's data loss exactly. Two files,
    /// no migration, and each one independently tolerant.
    private static var exclusionsFileURL: URL { AppStorageRoot.file("excluded_scan_paths.json") }

    private func save() {
        AppJSONStore.save(
            metadataByModelKey, to: Self.fileURL, label: "Model organization metadata")
    }

    private func saveExclusions() {
        AppJSONStore.save(
            excludedScanPaths.sorted(), to: Self.exclusionsFileURL, label: "Scan exclusions")
    }

    private func loadExclusions() {
        let decoded = AppJSONStore.load(
            [String].self, from: Self.exclusionsFileURL, label: "scan exclusions")
        excludedScanPaths = Set(decoded ?? [])
    }

    private func load() {
        if let decoded = AppJSONStore.load(
            [String: ModelCustomMetadata].self, from: Self.fileURL,
            label: "model organization metadata")
        {
            self.metadataByModelKey = decoded
            return
        }
        // One-time migration off `UserDefaults`, so an existing user's
        // nicknames and favorites survive the move. Left in place rather than
        // deleted: reading it costs nothing and removing it would silently
        // discard whatever had not been migrated yet.
        guard let legacy = UserDefaults.standard.data(forKey: Self.storageKey),
            let decoded = try? JSONDecoder().decode(
                [String: ModelCustomMetadata].self, from: legacy)
        else { return }
        self.metadataByModelKey = decoded
        save()
    }
}
