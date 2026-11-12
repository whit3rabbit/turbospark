import Foundation
import SwiftUI
import TurboSpark

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
    /// The reasoning level (`GenerateOptions.Reasoning.rawValue`) the user last
    /// chose for this model. It lived in `settings.json`, keyed by path, until
    /// task 5.5; keeping it in the same record means a move carries it with
    /// the rest of the model's metadata.
    public var reasoningDefault: String?

    /// Initializes custom metadata with optional nickname, notes, tags, favorite status and
    /// reasoning default.
    public init(
        nickname: String? = nil,
        notes: String = "",
        tags: [String] = [],
        isFavorite: Bool = false,
        reasoningDefault: String? = nil
    ) {
        self.nickname = nickname
        self.notes = notes
        self.tags = tags
        self.isFavorite = isFavorite
        self.reasoningDefault = reasoningDefault
    }

    /// **THE MERGE RULE** for two records of one model (both keys present on a
    /// read, or a `rekey` onto a destination that already has an entry). Self
    /// is the preferred record, normally the one under the `install_id`; each
    /// field keeps its value where it has one and takes `other`'s otherwise:
    /// a nickname, notes, tags or reasoning default is a value when non-empty,
    /// and a favorite is a value only when true. Nothing is concatenated or
    /// unioned, so two conflicting non-empty values resolve to the preferred
    /// one and the other is dropped when its key is retired.
    public func filling(gapsFrom other: ModelCustomMetadata) -> ModelCustomMetadata {
        var merged = self
        if (merged.nickname ?? "").isEmpty { merged.nickname = other.nickname }
        if merged.notes.isEmpty { merged.notes = other.notes }
        if merged.tags.isEmpty { merged.tags = other.tags }
        if !merged.isFavorite { merged.isFavorite = other.isFavorite }
        if (merged.reasoningDefault ?? "").isEmpty { merged.reasoningDefault = other.reasoningDefault }
        return merged
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
        reasoningDefault = try container.decodeIfPresent(String.self, forKey: .reasoningDefault)
    }
}

/// Observable store for managing persistent user notes, tags, favorites,
/// nicknames and reasoning defaults for models.
///
/// **KEYED BY `install_id`** (task 5.5, requirement 8.1): the file is still the
/// flat `[String: ModelCustomMetadata]` dictionary it always was, but a model's
/// entry now lives under `<library-id>:<modality>/<alias>`, which is the same
/// before and after a move or a drive reconnecting under another name. Keys
/// written by earlier builds (`alias::path`, the bare alias) still load and
/// are found through `ModelIdentityKey`'s lookup order; the first read copies
/// the entry to the id key and the next save removes the old key, so the file
/// converges without a migration step. An older build reading a newer file
/// sees the id keys as unknown entries and writes them back unchanged.
///
/// The `(alias:path:)` methods below are the pre-5.5 entry point for a caller
/// that holds only an alias or a path. They read and write the legacy keys
/// exactly as before and do not re-key. Callers that hold an `InstalledModel`
/// (or an install id) use the model overloads.
@MainActor
public final class ModelOrganizationStore: ObservableObject {
    /// Shared singleton instance of the store.
    public static let shared = ModelOrganizationStore()
    private static let storageKey = "TurboSpark.modelOrganizationMetadata"

    /// **NOT `@Published`, AND READS MUTATE IT.** A SwiftUI body reads this
    /// store, and the copy-on-first-read below writes it; publishing from
    /// inside a body is a runtime warning. The copy leaves every reader's
    /// answer unchanged, so it has nothing to announce. Every mutation that
    /// can change what a reader sees calls `objectWillChange.send()` first,
    /// which is what `@Published` did for the old property.
    private var table: [String: ModelCustomMetadata] = [:] {
        didSet { crossLibraryIndex = nil }
    }

    /// User metadata indexed by install id, and by the legacy composite key or
    /// alias until the next save retires it.
    public var metadataByModelKey: [String: ModelCustomMetadata] { table }

    /// Legacy keys whose content was copied to an install id key, with the
    /// content as it was at the copy. The next save removes a key only while it
    /// still holds exactly that content, so a legacy-API write that landed in
    /// between is never discarded.
    private var pendingRetirement: [String: ModelCustomMetadata] = [:]

    /// `<modality>/<alias>` to every key carrying it under some library. Built
    /// on the first miss, dropped when `table` changes, so a row with no
    /// metadata (most of them) costs a dictionary lookup rather than a scan.
    private var crossLibraryIndex: [String: [String]]?

    private let metadataFile: URL
    private let exclusionsFile: URL
    private let importsUserDefaults: Bool

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
        self.metadataFile = Self.fileURL
        self.exclusionsFile = Self.exclusionsFileURL
        self.importsUserDefaults = true
        load()
        loadExclusions()
    }

    /// A store over explicit files, for tests. It never reads the legacy
    /// `UserDefaults` copy, which belongs to the Default profile's real store.
    init(metadataFile: URL, exclusionsFile: URL) {
        self.metadataFile = metadataFile
        self.exclusionsFile = exclusionsFile
        self.importsUserDefaults = false
        load()
        loadExclusions()
    }

    // MARK: - Key Generation

    /// The LEGACY key for a caller that holds only an alias and a path:
    /// the standardized path joined to the alias, or the bare alias. Models
    /// are keyed by `ModelIdentityKey.installId` instead.
    public func key(for alias: String, path: String? = nil) -> String {
        if let path, !path.isEmpty {
            return ModelIdentityKey.metadataPathKey(alias: alias, path: path)
        }
        return alias
    }

    // MARK: - Querying by install identity

    /// Returns the model's metadata, looked up by install id, then the legacy
    /// path key, then the bare alias, then the same model under another
    /// library's id (`ModelIdentityKey`). A hit through a legacy key is copied
    /// to the install id key right here and the legacy key is retired on the
    /// next save. An empty record when nothing is found; nothing is created.
    public func metadata(for key: ModelIdentityKey) -> ModelCustomMetadata {
        resolveAndAdopt(key) ?? ModelCustomMetadata()
    }

    /// Returns the metadata of an installed model by its install identity.
    public func metadata(for model: InstalledModel) -> ModelCustomMetadata {
        metadata(for: ModelIdentityKey(model))
    }

    /// Returns whether the model is marked as a favorite.
    public func isFavorite(_ model: InstalledModel) -> Bool {
        metadata(for: model).isFavorite
    }

    /// Returns the user notes recorded for the model.
    public func notes(_ model: InstalledModel) -> String {
        metadata(for: model).notes
    }

    /// Returns the user-defined tags assigned to the model.
    public func tags(_ model: InstalledModel) -> [String] {
        metadata(for: model).tags
    }

    /// Returns the custom nickname assigned to the model, if any.
    public func nickname(_ model: InstalledModel) -> String? {
        metadata(for: model).nickname
    }

    /// Returns the reasoning level the user last chose for the model, if any.
    public func reasoningDefault(for model: InstalledModel) -> String? {
        let value = metadata(for: model).reasoningDefault
        return (value?.isEmpty == false) ? value : nil
    }

    /// Returns the model's reasoning default and retires the legacy
    /// `settings.json` entry it may still have.
    ///
    /// Those entries (`AppModel.modelReasoningDefaults`) were keyed by the raw
    /// path, or the alias before that. The first legacy key that exists is
    /// removed from `legacy` (the caller persists its settings); its value
    /// becomes the model's default in this store only when the store has none,
    /// so an entry already under the install id wins, as everywhere else.
    public func reasoningDefault(
        for model: InstalledModel, migratingLegacyFrom legacy: inout [String: String]
    ) -> String? {
        let key = ModelIdentityKey(model)
        if let legacyKey = key.legacyReasoningKeys.first(where: { legacy[$0] != nil }),
            let legacyValue = legacy.removeValue(forKey: legacyKey)
        {
            let current = resolveAndAdopt(key)?.reasoningDefault
            if (current ?? "").isEmpty, !legacyValue.isEmpty {
                update(key) { $0.reasoningDefault = legacyValue }
            }
        }
        return reasoningDefault(for: model)
    }

    // MARK: - Mutations by install identity

    /// Toggles the favorite status for the model and persists changes.
    public func toggleFavorite(_ model: InstalledModel) {
        update(ModelIdentityKey(model)) { $0.isFavorite.toggle() }
    }

    /// Sets the favorite status for the model and persists changes.
    public func setFavorite(_ favorite: Bool, for model: InstalledModel) {
        update(ModelIdentityKey(model)) { $0.isFavorite = favorite }
    }

    /// Sets user notes for the model and persists changes.
    public func setNotes(_ notes: String, for model: InstalledModel) {
        update(ModelIdentityKey(model)) { $0.notes = notes }
    }

    /// Updates or clears the nickname for the model and persists changes.
    public func setNickname(_ nickname: String?, for model: InstalledModel) {
        let trimmed = nickname?.trimmingCharacters(in: .whitespacesAndNewlines)
        update(ModelIdentityKey(model)) {
            $0.nickname = (trimmed?.isEmpty == false) ? trimmed : nil
        }
    }

    /// Adds a unique trimmed tag to the model and persists changes.
    public func addTag(_ tag: String, for model: InstalledModel) {
        let trimmed = tag.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty, !metadata(for: model).tags.contains(trimmed) else { return }
        update(ModelIdentityKey(model)) { $0.tags.append(trimmed) }
    }

    /// Removes a tag from the model and persists changes.
    public func removeTag(_ tag: String, for model: InstalledModel) {
        update(ModelIdentityKey(model)) { $0.tags.removeAll { $0 == tag } }
    }

    /// Remembers (or with nil, forgets) the reasoning level chosen for the model.
    public func setReasoningDefault(_ level: String?, for model: InstalledModel) {
        update(ModelIdentityKey(model)) { $0.reasoningDefault = level }
    }

    /// Drops one model's metadata: its install id entry, its `alias::path`
    /// entry, and the single entry another library holds for it. The BARE-ALIAS
    /// row is left alone (state#64): it is the shared fallback of every model
    /// that was never re-tagged, and `removeMetadata(for:path:)` spares it too.
    public func removeMetadata(for model: InstalledModel) {
        let key = ModelIdentityKey(model)
        var doomed: Set<String> = [key.installId]
        if let path = key.path {
            doomed.insert(ModelIdentityKey.metadataPathKey(alias: key.alias, path: path))
        }
        doomed.formUnion(resolution(for: key).folded.filter { $0 != key.alias })
        objectWillChange.send()
        for k in doomed {
            table.removeValue(forKey: k)
            pendingRetirement.removeValue(forKey: k)
        }
        save()
    }

    // MARK: - Re-keying

    /// Moves everything stored under `oldInstallId` to `newInstallId`, for the
    /// move flow: a model that moves to another library keeps its name and
    /// changes only the library part of its id (`T7:text/qwen` to
    /// `internal:text/qwen`). Task 6.11 calls this with `MoveOutcome.installId`
    /// after a successful move.
    ///
    /// Every field travels (the record is moved whole). A destination that
    /// already has an entry is not overwritten: the two merge by
    /// `ModelCustomMetadata.filling(gapsFrom:)` with the destination preferred.
    /// Idempotent: with nothing under the old id a second call changes
    /// nothing. A model's entry may still sit under its legacy
    /// `alias::path` key if nothing has read it since an older build wrote it;
    /// pass the path it had BEFORE the move as `legacyPath` to carry that too.
    ///
    /// Never pass a placeholder row's id (`ModelIdentityKey.isPlaceholder`): a
    /// row built from another app's folder has no registry install to move, and
    /// the move flow already refuses it.
    public func rekey(from oldInstallId: String, to newInstallId: String, legacyPath: String? = nil) {
        guard !oldInstallId.isEmpty, !newInstallId.isEmpty, oldInstallId != newInstallId else {
            return
        }
        var sources: [String] = []
        var moved: ModelCustomMetadata?
        if let entry = table[oldInstallId] {
            moved = entry
            sources.append(oldInstallId)
        }
        if let legacyPath, let alias = ModelIdentityKey.alias(inInstallId: oldInstallId) {
            let legacyKey = ModelIdentityKey.metadataPathKey(alias: alias, path: legacyPath)
            if legacyKey != oldInstallId, let entry = table[legacyKey] {
                moved = moved.map { $0.filling(gapsFrom: entry) } ?? entry
                sources.append(legacyKey)
            }
        }
        guard let moved else { return }
        objectWillChange.send()
        table[newInstallId] = table[newInstallId].map { $0.filling(gapsFrom: moved) } ?? moved
        for source in sources {
            table.removeValue(forKey: source)
            pendingRetirement.removeValue(forKey: source)
        }
        save()
    }

    // MARK: - Resolution

    private func resolution(for key: ModelIdentityKey) -> ModelIdentityKey.Resolution<ModelCustomMetadata> {
        key.resolve(
            in: table, legacyKeys: key.legacyMetadataKeys,
            crossLibraryCandidates: key.isPlaceholder ? [] : crossLibraryKeys(for: key)
        ) { $0.filling(gapsFrom: $1) }
    }

    /// Resolves `key` and, when the answer came through another key, copies it
    /// to the install id key without announcing anything and queues the old key
    /// for removal on the next save.
    private func resolveAndAdopt(_ key: ModelIdentityKey) -> ModelCustomMetadata? {
        let found = resolution(for: key)
        guard let value = found.value else { return nil }
        guard !found.folded.isEmpty else { return value }
        for folded in found.folded {
            if let original = table[folded] { pendingRetirement[folded] = original }
        }
        table[key.installId] = value
        return value
    }

    private func crossLibraryKeys(for key: ModelIdentityKey) -> [String] {
        guard let part = key.modalityAndAlias else { return [] }
        if crossLibraryIndex == nil {
            var index: [String: [String]] = [:]
            for existing in table.keys {
                if let shared = ModelIdentityKey.libraryIndependentPart(of: existing) {
                    index[shared, default: []].append(existing)
                }
            }
            crossLibraryIndex = index
        }
        return crossLibraryIndex?[part] ?? []
    }

    /// Reads the model's record (folding any legacy entry into it), applies
    /// `body`, writes it under the install id and saves.
    private func update(_ key: ModelIdentityKey, _ body: (inout ModelCustomMetadata) -> Void) {
        var meta = resolveAndAdopt(key) ?? ModelCustomMetadata()
        body(&meta)
        objectWillChange.send()
        table[key.installId] = meta
        save()
    }

    // MARK: - Querying by alias or path (legacy entry point)

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
        storeLegacy(meta, at: k)
        save()
    }

    /// Sets the favorite status for the specified model and persists changes.
    public func setFavorite(_ favorite: Bool, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.isFavorite = favorite
        storeLegacy(meta, at: k)
        save()
    }

    /// Sets user notes for the specified model and persists changes.
    public func setNotes(_ notes: String, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.notes = notes
        storeLegacy(meta, at: k)
        save()
    }

    /// Updates or clears the nickname for the specified model and persists changes.
    public func setNickname(_ nickname: String?, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        let trimmed = nickname?.trimmingCharacters(in: .whitespacesAndNewlines)
        meta.nickname = (trimmed?.isEmpty == false) ? trimmed : nil
        storeLegacy(meta, at: k)
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
            storeLegacy(meta, at: k)
            save()
        }
    }

    /// Removes a tag from the specified model and persists changes.
    public func removeTag(_ tag: String, for alias: String, path: String? = nil) {
        let k = key(for: alias, path: path)
        var meta = metadata(for: alias, path: path)
        meta.tags.removeAll { $0 == tag }
        storeLegacy(meta, at: k)
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
        objectWillChange.send()
        table.removeValue(forKey: k)
        pendingRetirement.removeValue(forKey: k)
        if path == nil || path?.isEmpty == true {
            table.removeValue(forKey: alias)
        }
        save()
    }

    /// The legacy writers' one store point. A key written here is the user's
    /// current word, so it is no longer waiting to be retired.
    private func storeLegacy(_ meta: ModelCustomMetadata, at key: String) {
        objectWillChange.send()
        table[key] = meta
        pendingRetirement.removeValue(forKey: key)
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
        retirePendingLegacyKeys()
        AppJSONStore.save(table, to: metadataFile, label: "Model organization metadata")
    }

    /// Removes the legacy keys whose content was copied to an install id key
    /// (re-key on first read, legacy key gone on the next save). A key that no
    /// longer holds what was copied was written to since, and stays.
    private func retirePendingLegacyKeys() {
        for (key, copied) in pendingRetirement where table[key] == copied {
            table.removeValue(forKey: key)
        }
        pendingRetirement.removeAll()
    }

    private func saveExclusions() {
        AppJSONStore.save(
            excludedScanPaths.sorted(), to: exclusionsFile, label: "Scan exclusions")
    }

    private func loadExclusions() {
        let decoded = AppJSONStore.load(
            [String].self, from: exclusionsFile, label: "Scan exclusions")
        excludedScanPaths = Set(decoded ?? [])
    }

    private func load() {
        let failuresBefore = AppJSONStore.readFailureCount
        if let decoded = AppJSONStore.load(
            [String: ModelCustomMetadata].self, from: metadataFile,
            label: "Model organization metadata")
        {
            self.table = decoded
            return
        }
        // A failed read is not an absent record: migrating then would write
        // the stale legacy dictionary over the vault's real data. The legacy
        // key also belongs to the Default profile only; importing it into a
        // later profile leaks the Default user's notes and tags.
        guard importsUserDefaults,
            AppJSONStore.readFailureCount == failuresBefore,
            UserProfileStore.isDefault
        else { return }
        // One-time migration off `UserDefaults`, so an existing user's
        // nicknames and favorites survive the move. Left in place rather than
        // deleted: reading it costs nothing and removing it would silently
        // discard whatever had not been migrated yet.
        guard let legacy = UserDefaults.standard.data(forKey: Self.storageKey),
            let decoded = try? JSONDecoder().decode(
                [String: ModelCustomMetadata].self, from: legacy)
        else { return }
        self.table = decoded
        save()
    }
}
