import Foundation

/// One scope-installation record for a plugin: Claude Code's
/// `installed_plugins.json` v2 shape. Several scopes of one plugin coexist
/// (user plus a project), hence the array per plugin id.
public struct InstalledPluginRecord: Codable, Sendable, Equatable {
    /// "user" or "project". Claude Code also has local/managed/flag; this
    /// app's local settings layer is the per-project archive, so two scopes
    /// cover what exists here.
    public var scope: String
    /// Required when scope is "project".
    public var projectPath: String?
    public var installPath: String
    public var version: String
    public var installedAt: Date
    public var lastUpdated: Date
    public var gitCommitSha: String?

    public init(
        scope: String,
        projectPath: String? = nil,
        installPath: String,
        version: String,
        installedAt: Date = Date(),
        lastUpdated: Date = Date(),
        gitCommitSha: String? = nil
    ) {
        self.scope = scope
        self.projectPath = projectPath
        self.installPath = installPath
        self.version = version
        self.installedAt = installedAt
        self.lastUpdated = lastUpdated
        self.gitCommitSha = gitCommitSha
    }
}

/// The on-disk ledger: `{"version": 2, "plugins": {"<id>": [records]}}`.
public struct InstalledPluginLedger: Codable, Sendable, Equatable {
    public var version: Int
    public var plugins: [String: [InstalledPluginRecord]]

    public init(version: Int = 2, plugins: [String: [InstalledPluginRecord]] = [:]) {
        self.version = version
        self.plugins = plugins
    }
}

/// Reads and writes the plugin install ledger and the local-plugin registry
/// under one root (default `~/.turbospark/plugins`). Injectable root because
/// the managers that hardcode their home path are untestable without
/// writing the developer's real home (`swift/CLAUDE.md` Gotcha 43).
public final class PluginLedgerStore: @unchecked Sendable {
    public let root: URL
    private let lock = NSLock()

    public init(root: URL?) {
        if let root {
            self.root = root
        } else {
            // The Default profile keeps the shared cross-harness root; any
            // other profile holds its plugins inside its own folder.
            self.root = UserProfileStore.isDefault
                ? FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(".turbospark/plugins", isDirectory: true)
                : UserProfileStore.userScopeSubdirectory("plugins")
        }
    }

    public var ledgerURL: URL {
        root.appendingPathComponent("installed_plugins.json")
    }

    public var localRegistryURL: URL {
        root.appendingPathComponent("local_plugins.json")
    }

    // MARK: - Ledger

    public func load() -> InstalledPluginLedger {
        lock.lock()
        defer { lock.unlock() }
        // Dates decode with the SAME strategy save writes (iso8601); a
        // mismatch here made every read of a written ledger decode empty.
        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .iso8601
        guard let data = try? Data(contentsOf: ledgerURL),
            let ledger = try? decoder.decode(InstalledPluginLedger.self, from: data)
        else { return InstalledPluginLedger() }
        return ledger
    }

    public func save(_ ledger: InstalledPluginLedger) {
        try? saveChecked(ledger)
    }

    public func saveChecked(_ ledger: InstalledPluginLedger) throws {
        lock.lock()
        defer { lock.unlock() }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        encoder.dateEncodingStrategy = .iso8601
        try encoder.encode(ledger).write(to: ledgerURL, options: .atomic)
    }

    /// Upserts one record: same plugin, scope and project path replaces.
    public func upsertRecord(_ record: InstalledPluginRecord, for pluginID: String) {
        try? upsertRecordChecked(record, for: pluginID)
    }

    public func upsertRecordChecked(_ record: InstalledPluginRecord, for pluginID: String) throws {
        var ledger = load()
        var list = ledger.plugins[pluginID] ?? []
        list.removeAll {
            $0.scope == record.scope && $0.projectPath == record.projectPath
        }
        list.append(record)
        ledger.plugins[pluginID] = list
        try saveChecked(ledger)
    }

    /// Removes the records matching a scope (and project, for project
    /// scope). Returns whether anything was removed.
    @discardableResult
    public func removeRecords(
        pluginID: String, scope: String, projectPath: String? = nil
    ) -> Bool {
        var ledger = load()
        guard var list = ledger.plugins[pluginID] else { return false }
        let before = list.count
        list.removeAll { record in
            guard record.scope == scope else { return false }
            if scope == "project" {
                return record.projectPath == projectPath
            }
            return true
        }
        guard list.count != before else { return false }
        if list.isEmpty {
            ledger.plugins.removeValue(forKey: pluginID)
        } else {
            ledger.plugins[pluginID] = list
        }
        save(ledger)
        return true
    }

    /// Whether any scope still references a plugin's install path. The cache
    /// directory is deleted only when this answers no, so a user-scope
    /// install survives a project-scope uninstall.
    public func hasRemainingRecords(pluginID: String) -> Bool {
        !(load().plugins[pluginID] ?? []).isEmpty
    }

    // MARK: - Locally-registered plugin folders

    public func loadLocalPluginPaths() -> [String] {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: localRegistryURL),
            let list = try? JSONDecoder().decode([String].self, from: data)
        else { return [] }
        return list
    }

    public func addLocalPluginPath(_ path: String) {
        lock.lock()
        defer { lock.unlock() }
        var list = (try? JSONDecoder().decode([String].self, from: Data(contentsOf: localRegistryURL) )) ?? []
        guard !list.contains(path) else { return }
        list.append(path)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(list) {
            try? data.write(to: localRegistryURL, options: .atomic)
        }
    }

    public func removeLocalPluginPath(_ path: String) {
        lock.lock()
        defer { lock.unlock() }
        guard var list = try? JSONDecoder().decode([String].self, from: Data(contentsOf: localRegistryURL)) else { return }
        list.removeAll { $0 == path }
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(list) {
            try? data.write(to: localRegistryURL, options: .atomic)
        }
    }
}
