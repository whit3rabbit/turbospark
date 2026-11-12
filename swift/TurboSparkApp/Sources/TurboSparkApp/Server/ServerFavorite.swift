import Foundation
import TurboSpark

/// Server recipes belong to the existing per-user settings store. No keys
/// or captured traffic are serialized with a favorite.
///
/// **A FAVORITE FOLLOWS ITS MODELS BY INSTALL IDENTITY, NOT BY PATH**
/// (requirement 8.1). A path changes when a model is moved to another library
/// and when its drive reconnects under another name, and a favorite keyed by
/// path then reported the model as gone. `modelInstallIds` holds the identity
/// (`<library-id>:<modality>/<alias>`); `modelPaths` stays as the legacy key,
/// still written so an older app can read the favorite, and as the fallback for
/// a favorite saved before identities existed.
///
/// **THE TWO LISTS ARE ALIGNED BY INDEX.** Entry `i` is `modelPaths[i]` and
/// `modelInstallIds[i]`. A model the app built a placeholder row for (another
/// app's folder: LM Studio, a scan folder) has no registry identity, so its
/// slot in `modelInstallIds` is the empty string, never the placeholder's
/// default id; trailing empty slots are dropped, so a favorite of placeholders
/// alone has no identities at all. Index alignment is what lets a moved model
/// (new path, new id) be told apart from a path-only entry, instead of its stale
/// saved path being looked up as if it named a different model.
public struct ServerFavorite: Codable, Identifiable, Equatable, Sendable {
    public var id = UUID()
    public var name: String
    public var host: String
    public var port: UInt16
    public var modelPaths: [String]
    /// Install identities, aligned with `modelPaths`; see the type's note.
    public var modelInstallIds: [String]
    public var contextTokens: Int

    public init(
        id: UUID = UUID(), name: String, host: String, port: UInt16, modelPaths: [String],
        modelInstallIds: [String] = [], contextTokens: Int
    ) {
        self.id = id
        self.name = name
        self.host = host
        self.port = port
        self.modelPaths = modelPaths
        self.modelInstallIds = modelInstallIds
        self.contextTokens = contextTokens
    }

    private enum CodingKeys: String, CodingKey {
        case id, name, host, port, modelPaths, modelInstallIds, contextTokens
    }

    /// `modelInstallIds` is absent from a favorite saved by an older app, and
    /// `MacAppSettings` decodes the whole favorites array or none of it, so an
    /// absent or malformed key must never fail the favorite.
    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        id = try c.decode(UUID.self, forKey: .id)
        name = try c.decode(String.self, forKey: .name)
        host = try c.decode(String.self, forKey: .host)
        port = try c.decode(UInt16.self, forKey: .port)
        modelPaths = try c.decode([String].self, forKey: .modelPaths)
        modelInstallIds = c.decodeLenient([String].self, forKey: .modelInstallIds, fallback: [])
        contextTokens = try c.decode(Int.self, forKey: .contextTokens)
    }
}

/// What one model of a favorite came to when it was looked up.
enum ServerFavoriteModel: Equatable, Sendable {
    /// The model can be opened now.
    case available(InstalledModel)
    /// The model is listed, but its drive is not connected (requirement 8.2).
    /// Both names are raw and untrusted: show them through
    /// `UntrustedText.display`.
    case notConnected(model: String, libraryName: String)
    /// The model is not installed any more, its folder is gone from a connected
    /// drive, or nothing listed matches it. The text names it for the message.
    case missing(String)

    /// "<Model> is on <Drive>, which is not connected." The catalog key, and so
    /// every translation, keeps the model before the drive: the parity gate
    /// compares printf specifiers literally, so positional ones are not used.
    static func notConnectedMessage(
        model: String, library: String, language: AppLanguage? = nil
    ) -> String {
        let model = UntrustedText.display(model)
        let library = UntrustedText.display(library)
        return AppStrings.localized(
            "\(model) is on \(library), which is not connected. Connect the drive and try again.",
            language: language ?? AppStrings.currentLanguage)
    }
}

extension ServerFavorite {
    /// How many models the favorite holds, whichever list says so: a favorite
    /// that carries only identities counts as well as one that carries only
    /// paths.
    var modelCount: Int { max(modelPaths.count, modelInstallIds.count) }

    /// The identities to store for `rows`, aligned with their paths. A
    /// placeholder row (`library == nil` on This Mac, built by the app for
    /// another app's folder) gets the empty slot: its default id names no
    /// registry install and must never be stored as one.
    static func installIds(for rows: [InstalledModel]) -> [String] {
        trimmed(rows.map { ModelIdentityKey.isPlaceholder($0) ? "" : $0.installId })
    }

    private static func trimmed(_ ids: [String]) -> [String] {
        var ids = ids
        while ids.last == "" { ids.removeLast() }
        return ids
    }

    // MARK: Resolution

    /// One model of the favorite as stored.
    struct Entry: Equatable {
        let installId: String?
        let path: String?
    }

    var entries: [Entry] {
        (0..<modelCount).map { index in
            let id = index < modelInstallIds.count ? modelInstallIds[index] : ""
            let path = index < modelPaths.count ? modelPaths[index] : ""
            return Entry(installId: id.isEmpty ? nil : id, path: path.isEmpty ? nil : path)
        }
    }

    /// The row an entry stands for.
    ///
    /// 1. The install id against `rows`, so a model that moved, or whose drive
    ///    came back under another name, is found where it is now. A placeholder
    ///    row never answers an id: its default id is shared with real This Mac
    ///    rows.
    /// 2. The same `<modality>/<alias>` under another library, when exactly one
    ///    row has it (`ModelIdentityKey`'s rule: the registry keeps that part
    ///    unique, and more than one candidate is never guessed).
    /// 3. The legacy path, compared exactly as it always was.
    private static func row(for entry: Entry, in rows: [InstalledModel]) -> InstalledModel? {
        if let id = entry.installId {
            if let exact = rows.first(where: { $0.installId == id && !ModelIdentityKey.isPlaceholder($0) }) {
                return exact
            }
            if let part = ModelIdentityKey.libraryIndependentPart(of: id) {
                let candidates = rows.filter {
                    !ModelIdentityKey.isPlaceholder($0)
                        && ModelIdentityKey.libraryIndependentPart(of: $0.installId) == part
                }
                if Set(candidates.map(\.installId)).count == 1 { return candidates[0] }
            }
        }
        if let path = entry.path { return rows.first { $0.path == path } }
        return nil
    }

    private static func outcome(for entry: Entry, row: InstalledModel?) -> ServerFavoriteModel {
        guard let row else {
            let alias = entry.installId.flatMap { ModelIdentityKey.alias(inInstallId: $0) }
            return .missing(entry.path ?? alias ?? entry.installId ?? "")
        }
        switch row.availability {
        case .available:
            return .available(row)
        case .notConnected:
            // `connectName` is the drive to connect: the library, or for a
            // custom folder the volume.
            return .notConnected(
                model: row.alias, libraryName: row.connectName ?? row.libraryName ?? row.library ?? "")
        case .missing:
            return .missing(entry.path ?? row.path)
        }
    }

    private func lookups(in rows: [InstalledModel]) -> [(entry: Entry, row: InstalledModel?)] {
        entries.map { ($0, Self.row(for: $0, in: rows)) }
    }

    /// Where each model of the favorite stands, in order. `rows` holds every
    /// row the app knows, offline ones included.
    func resolve(in rows: [InstalledModel]) -> [ServerFavoriteModel] {
        lookups(in: rows).map { Self.outcome(for: $0.entry, row: $0.row) }
    }

    /// This favorite with its identities brought up to date from `rows`, or nil
    /// when they already are. The paths are kept as they are.
    ///
    /// A model found in the registry gets that row's id (so a path-only
    /// favorite gains its identities, and a stale id follows the model to its
    /// new library). A model found nowhere keeps the id it had, and a
    /// placeholder never gets one.
    func rekeyed(in rows: [InstalledModel]) -> ServerFavorite? {
        let ids = Self.trimmed(
            lookups(in: rows).map { entry, row in
                if let row, !ModelIdentityKey.isPlaceholder(row) { return row.installId }
                return entry.installId ?? ""
            })
        guard ids != modelInstallIds else { return nil }
        var copy = self
        copy.modelInstallIds = ids
        return copy
    }

    /// This favorite after the model `oldInstallId` (listed at `oldPath`) was moved
    /// to another library and is `newInstallId` now (task 6.11, requirement 8.1), or
    /// nil when it does not hold that model.
    ///
    /// An entry is the model's when it stores the old id, or when it stores no id
    /// (a favorite saved before identities existed) and its path is the old path.
    /// Only the identities are rewritten, aligned by index and without trailing
    /// blanks: the paths stay as they are (the key an older app reads), which is
    /// what `rekeyed(in:)` does too. An entry that already has an id of its own
    /// is never claimed by a path.
    ///
    /// The favorite resolved through the library-independent part of the id
    /// without this (`row(for:in:)`, step 2), so this is the explicit record of
    /// what that fallback finds, not the only thing keeping the favorite alive.
    func followingMove(from oldInstallId: String, oldPath: String, to newInstallId: String) -> ServerFavorite? {
        var ids = (0..<modelCount).map { index in index < modelInstallIds.count ? modelInstallIds[index] : "" }
        var changed = false
        for index in ids.indices {
            let path = index < modelPaths.count ? modelPaths[index] : ""
            let claimed = ids[index] == oldInstallId || (ids[index].isEmpty && !path.isEmpty && path == oldPath)
            if claimed, ids[index] != newInstallId {
                ids[index] = newInstallId
                changed = true
            }
        }
        guard changed else { return nil }
        var copy = self
        copy.modelInstallIds = Self.trimmed(ids)
        return copy
    }
}

extension AppModel {
    func saveServerFavorite(name: String) {
        let name = name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { return }
        let ids = Set(serverInfo?.models ?? [])
        let rows = installed.filter { ids.contains(Self.servedModelID(for: $0)) }
        serverFavorites.append(ServerFavorite(name: name, host: serverHost,
            port: serverPinnedPort, modelPaths: rows.map(\.path),
            modelInstallIds: ServerFavorite.installIds(for: rows), contextTokens: maxContextTokens))
        persistSettingsDebounced()
    }

    /// Every row a favorite can name: the library rows (offline and missing
    /// ones too, which `installed` does not hold) and then the available rows
    /// the app lists that the engine did not (other apps' folders). A model in
    /// both lists is the library row, matched by path.
    func serverFavoriteRows() -> [InstalledModel] {
        var rows = libraryState.allInstalled
        var seen = Set(rows.map(\.path))
        for row in installed where seen.insert(row.path).inserted { rows.append(row) }
        return rows
    }

    /// Looks each model of `favorite` up, and writes the identities of a
    /// favorite that lacked them (or held stale ones) back to the stored
    /// favorite and the settings file. A favorite the user deleted meanwhile is
    /// not brought back.
    func resolveServerFavoriteModels(_ favorite: ServerFavorite) -> [ServerFavoriteModel] {
        let rows = serverFavoriteRows()
        if let updated = favorite.rekeyed(in: rows),
            let index = serverFavorites.firstIndex(where: { $0.id == favorite.id }),
            serverFavorites[index].modelInstallIds != updated.modelInstallIds
        {
            // Only the identities: a rename or a changed path made meanwhile
            // stays as it is.
            serverFavorites[index].modelInstallIds = updated.modelInstallIds
            persistSettingsDebounced()
        }
        return favorite.resolve(in: rows)
    }

    func loadServerFavorite(_ favorite: ServerFavorite) {
        guard server == nil, !serverBusy else { return }
        serverHost = favorite.host
        serverPinnedPort = favorite.port
        serverPortIsValid = true
        maxContextTokens = min(max(0, favorite.contextTokens), Int(UInt32.max))
        persistSettingsDebounced()
        serverBusy = true
        Task {
            await performServerStart(attachChat: false)
            guard let started = server else { return }
            for outcome in resolveServerFavoriteModels(favorite) {
                guard server === started else { return }
                switch outcome {
                case .available(let candidate):
                    if serverInfo?.models.contains(Self.servedModelID(for: candidate)) == true { continue }
                    await attachServerModelAndWait(candidate)
                case .notConnected(let model, let libraryName):
                    // A drive that is away is not a missing model (requirement
                    // 8.2): the server still starts and nothing is attached.
                    showToast(
                        ServerFavoriteModel.notConnectedMessage(model: model, library: libraryName),
                        style: .warning)
                case .missing(let name):
                    // Not localized, as before this task; the sentence is not
                    // part of requirement 8.2.
                    showToast("Model is no longer installed: \(name)", style: .warning)
                }
            }
        }
    }
}
