import Foundation
import TurboSpark

/// How per-model app state finds ONE model, whichever spelling of its key a
/// file was written with (requirement 8.1).
///
/// **THE KEY IS THE `install_id`, NOT THE PATH.** Favorites, tags, notes,
/// nicknames and reasoning defaults were keyed by `alias::path` (or the bare
/// alias). A path changes when the model is moved to another library and when
/// its drive reconnects under another name, and the metadata was orphaned each
/// time. The id (`<library-id>:<modality>/<alias>`) is stable across both.
///
/// Lookup order, shared by every store that follows a model:
///
/// 1. the `install_id`;
/// 2. the first legacy key that exists (`alias::path` before the bare alias
///    for metadata, the raw path before the alias for reasoning defaults). The
///    specific key shadows the shared bare-alias row, as it always did
///    (state#64), so the two are never merged with each other;
/// 3. when nothing answered and the row is not a placeholder, the ONE entry
///    under another library's id for the same `<modality>/<alias>`. The
///    registry keeps that part unique across libraries (tasks.md 1.2), so a
///    row that gained a library later, or was moved by the CLI, finds the
///    entry written under its old id. More than one candidate is ambiguous and
///    is never guessed.
///
/// A hit through 2 or 3 is copied to the `install_id` key by the caller and
/// the old key is retired on the next save. Task 5.9 resolves server
/// favorites through the same type.
public struct ModelIdentityKey: Equatable, Sendable {
    /// `<library-id>:<modality>/<alias>`, for example `T7:text/qwen` or
    /// `internal:text/qwen`.
    public let installId: String
    public let alias: String
    /// The path the row is at now. Only used to spell the legacy keys.
    public let path: String?
    /// True for a row the app built itself for another app's folder (LM
    /// Studio, a custom scan folder, a manually opened file). Its id is
    /// `internal:<modality>/<alias>` by default, which names no registry
    /// install. It keys like any internal row, but it never searches other
    /// libraries for an entry and never produces a library id: it must not be
    /// re-keyed onto one.
    public let isPlaceholder: Bool

    public init(installId: String, alias: String, path: String?, isPlaceholder: Bool = false) {
        self.installId = installId
        self.alias = alias
        self.path = (path?.isEmpty == false) ? path : nil
        self.isPlaceholder = isPlaceholder
    }

    public init(_ model: InstalledModel) {
        self.init(
            installId: model.installId, alias: model.alias, path: model.path,
            isPlaceholder: Self.isPlaceholder(model))
    }

    /// Whether the app built `model` itself rather than receiving it from the
    /// registry. The engine always sends `library` for a This Mac row
    /// (`"internal"`), a library row and an unknown-library row, and sends a
    /// null library only for a custom location, whose `locationKind` says so.
    /// So a nil library with the This Mac kind is the memberwise default the
    /// app's own scanners produce.
    public static func isPlaceholder(_ model: InstalledModel) -> Bool {
        model.library == nil && model.locationKind == .thisMac
    }

    // MARK: Legacy spellings

    /// The metadata key before install ids: `alias::` plus the STANDARDIZED
    /// path (`ModelOrganizationStore.key(for:path:)` has always spelled it so).
    public static func metadataPathKey(alias: String, path: String) -> String {
        "\(alias)::\(URL(fileURLWithPath: path).standardizedFileURL.path)"
    }

    /// Legacy metadata keys in lookup order.
    public var legacyMetadataKeys: [String] {
        var keys: [String] = []
        if let path { keys.append(Self.metadataPathKey(alias: alias, path: path)) }
        keys.append(alias)
        return keys
    }

    /// Legacy reasoning-default keys in lookup order. `setReasoning` wrote the
    /// RAW path (state#96), and the alias before that.
    public var legacyReasoningKeys: [String] {
        var keys: [String] = []
        if let path { keys.append(path) }
        keys.append(alias)
        return keys
    }

    // MARK: Id parts

    /// `text/qwen`: the part of the id that is the same in every library.
    public var modalityAndAlias: String? { Self.libraryIndependentPart(of: installId) }

    /// The `<modality>/<alias>` part of an id, or nil when `id` is not shaped
    /// like one. A legacy key (`qwen::/path`, a bare alias) never is: its text
    /// after the first colon does not start with a modality and a slash.
    public static func libraryIndependentPart(of id: String) -> String? {
        guard let colon = id.firstIndex(of: ":") else { return nil }
        let library = id[..<colon]
        let rest = id[id.index(after: colon)...]
        guard !library.isEmpty, !library.contains("/") else { return nil }
        let modality = rest.prefix { $0.isASCII && $0.isLowercase }
        guard !modality.isEmpty else { return nil }
        let afterModality = rest.dropFirst(modality.count)
        guard afterModality.first == "/", afterModality.count > 1 else { return nil }
        return String(rest)
    }

    /// The alias inside an id, or nil when `id` is not shaped like one.
    public static func alias(inInstallId id: String) -> String? {
        guard let part = libraryIndependentPart(of: id),
            let slash = part.firstIndex(of: "/")
        else { return nil }
        return String(part[part.index(after: slash)...])
    }

    // MARK: Resolution

    /// What a lookup found: the effective value and the keys other than
    /// `installId` whose content it folded in. The caller copies the value to
    /// `installId` and retires `folded` on the next save.
    public struct Resolution<Value> {
        public var value: Value?
        public var folded: [String]
    }

    /// Resolves this model in `table` by the lookup order above.
    ///
    /// - Parameters:
    ///   - legacyKeys: the legacy spellings for this table, in order.
    ///   - crossLibraryCandidates: every key of `table` that carries this
    ///     model's `modalityAndAlias` under some library (the caller indexes
    ///     them once). Ignored for a placeholder.
    ///   - merge: `merge(installIdValue, legacyValue)` for the case where both
    ///     keys exist. The id value is the first argument and wins where it has
    ///     a value (see `ModelCustomMetadata.filling(gapsFrom:)`).
    public func resolve<Value>(
        in table: [String: Value],
        legacyKeys: [String],
        crossLibraryCandidates: [String] = [],
        merge: (Value, Value) -> Value
    ) -> Resolution<Value> {
        var value = table[installId]
        var folded: [String] = []
        if let key = legacyKeys.first(where: { $0 != installId && table[$0] != nil }),
            let legacy = table[key]
        {
            value = value.map { merge($0, legacy) } ?? legacy
            folded.append(key)
        }
        if value == nil, !isPlaceholder {
            let candidates = crossLibraryCandidates.filter { $0 != installId && table[$0] != nil }
            if candidates.count == 1, let other = table[candidates[0]] {
                value = other
                folded.append(candidates[0])
            }
        }
        return Resolution(value: value, folded: folded)
    }
}
