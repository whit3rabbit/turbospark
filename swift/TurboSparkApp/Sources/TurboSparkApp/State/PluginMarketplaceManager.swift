import Foundation

/// Installs, updates and removes plugins from marketplaces.
///
/// A marketplace is a git repo (or local directory) containing
/// `.claude-plugin/marketplace.json`: a name, an owner, and a list of plugin
/// entries whose `source` is either a `./relative/path` inside the
/// marketplace checkout or a remote (`github` / `git`). Installs land in the
/// versioned cache `<root>/cache/<marketplace>/<plugin>/<version>/` and are
/// recorded in the v2 ledger; the enable state lives in settings, not here.
///
/// Sources and git are the SHARED `MarketplaceSource` / `MarketplaceGit`
/// pair, per swift/CLAUDE.md: a third copy of clone-and-pull would be a
/// third copy of its defects. The `url` source lists fine but cannot
/// install `./relative` entries (it never materializes a checkout), and it
/// says so rather than failing obscurely.
public final class PluginMarketplaceManager: @unchecked Sendable {
    public static let shared = PluginMarketplaceManager()

    /// The plugin root: marketplaces clone cache, install cache and ledger
    /// all live under here.
    public let root: URL
    let ledger: PluginLedgerStore
    private let lock = NSLock()

    public init(root: URL? = nil) {
        // Profile-aware like `PluginLedgerStore`: the Default profile keeps
        // the shared root, other profiles keep everything under their own.
        let resolved = root
            ?? (UserProfileStore.isDefault
                ? FileManager.default.homeDirectoryForCurrentUser
                    .appendingPathComponent(".turbospark/plugins", isDirectory: true)
                : UserProfileStore.userScopeSubdirectory("plugins"))
        self.root = resolved
        self.ledger = PluginLedgerStore(root: resolved)
    }

    public var marketplacesCacheDirectory: URL {
        root.appendingPathComponent("marketplaces", isDirectory: true)
    }

    public var installCacheDirectory: URL {
        root.appendingPathComponent("cache", isDirectory: true)
    }

    var knownMarketplacesURL: URL {
        marketplacesCacheDirectory.appendingPathComponent("known_marketplaces.json")
    }

    // MARK: - Known marketplaces

    public func loadKnownMarketplaces() -> [String: MarketplaceSource] {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: knownMarketplacesURL),
            let dict = try? JSONDecoder().decode([String: MarketplaceSource].self, from: data)
        else { return [:] }
        return dict
    }

    public func saveKnownMarketplace(name: String, source: MarketplaceSource) throws {
        if let reason = PluginManifestParser.validateMarketplaceName(name) {
            throw PluginLoadError(pluginName: nil, reason: reason)
        }
        lock.lock()
        defer { lock.unlock() }
        var current = (try? JSONDecoder().decode(
            [String: MarketplaceSource].self, from: Data(contentsOf: knownMarketplacesURL))) ?? [:]
        current[name] = source
        try? FileManager.default.createDirectory(
            at: marketplacesCacheDirectory, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(current)
        try data.write(to: knownMarketplacesURL, options: .atomic)
    }

    public func removeKnownMarketplace(name: String) {
        lock.lock()
        defer { lock.unlock() }
        guard var current = try? JSONDecoder().decode(
            [String: MarketplaceSource].self, from: Data(contentsOf: knownMarketplacesURL))
        else { return }
        current.removeValue(forKey: name)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(current) {
            try? data.write(to: knownMarketplacesURL, options: .atomic)
        }
    }

    // MARK: - Fetching a marketplace checkout

    /// A materialized marketplace: a local directory holding the checkout
    /// and the parsed manifest. `directory` is nil for a bare-URL source,
    /// which never touches disk.
    public struct MarketplaceCheckout {
        public var manifest: PluginManifestParser.MarketplaceManifest
        public var directory: URL?
    }

    /// Fetches (and for git/github sources, clones or updates) a
    /// marketplace, then parses its manifest.
    public func fetchMarketplace(
        name: String?, source: MarketplaceSource
    ) async throws -> MarketplaceCheckout {
        switch source {
        case .url(let url, let headers):
            guard let requestURL = URL(string: url) else {
                throw PluginLoadError(pluginName: nil, reason: "Invalid URL: \(url)")
            }
            var request = URLRequest(url: requestURL)
            for (key, value) in headers ?? [:] {
                request.setValue(value, forHTTPHeaderField: key)
            }
            let (data, _) = try await URLSession.shared.data(for: request)
            return MarketplaceCheckout(
                manifest: try PluginManifestParser.parseMarketplace(
                    data: data, sourceDescription: url),
                directory: nil)

        case .github(let repo, let ref, let path, _):
            let dir = marketplacesCacheDirectory.appendingPathComponent(
                MarketplaceGit.cacheDirectoryName(for: "github:\(repo)"), isDirectory: true)
            try await MarketplaceGit.cloneOrPull(
                url: "https://github.com/\(repo).git", targetDir: dir, ref: ref,
                sparsePaths: nil)
            return MarketplaceCheckout(
                manifest: try readManifest(from: dir, declaredPath: path),
                directory: dir)

        case .git(let url, let ref, let path, let sparse):
            let dir = marketplacesCacheDirectory.appendingPathComponent(
                MarketplaceGit.cacheDirectoryName(for: url), isDirectory: true)
            try await MarketplaceGit.cloneOrPull(
                url: url, targetDir: dir, ref: ref, sparsePaths: sparse)
            return MarketplaceCheckout(
                manifest: try readManifest(from: dir, declaredPath: path),
                directory: dir)

        case .directory(let path):
            let dir = URL(fileURLWithPath: path, isDirectory: true)
            guard FileManager.default.fileExists(atPath: dir.path) else {
                throw PluginLoadError(
                    pluginName: nil, reason: "Directory not found: \(path)")
            }
            if path.hasSuffix(".json") {
                let data = try Data(contentsOf: dir)
                return MarketplaceCheckout(
                    manifest: try PluginManifestParser.parseMarketplace(
                        data: data, sourceDescription: path),
                    directory: dir.deletingLastPathComponent())
            }
            return MarketplaceCheckout(
                manifest: try readManifest(from: dir, declaredPath: nil),
                directory: dir)
        }
    }

    /// Reads the manifest from a checkout: the declared path when given,
    /// else `.claude-plugin/marketplace.json`, else the plain
    /// `marketplace.json` some repositories use.
    private func readManifest(from dir: URL, declaredPath: String?) throws -> PluginManifestParser.MarketplaceManifest {
        let candidates: [URL]
        if let declaredPath {
            candidates = [dir.appendingPathComponent(declaredPath)]
        } else {
            candidates = [
                dir.appendingPathComponent(PluginManifestParser.marketplaceRelativePath),
                dir.appendingPathComponent("marketplace.json")
            ]
        }
        for candidate in candidates
        where FileManager.default.fileExists(atPath: candidate.path) {
            let data = try Data(contentsOf: candidate)
            return try PluginManifestParser.parseMarketplace(
                data: data, sourceDescription: candidate.path)
        }
        throw PluginLoadError(
            pluginName: nil,
            reason: "No \(PluginManifestParser.marketplaceRelativePath) found in \(dir.path)")
    }

    // MARK: - Install / uninstall / update

    public struct InstallOutcome: Sendable, Equatable {
        public var pluginID: String
        public var pluginName: String
        public var installPath: String
        public var version: String
    }

    /// Installs one marketplace entry. `scope` is "user" or "project"; a
    /// project install records the path so a second project does not inherit
    /// it.
    @discardableResult
    public func install(
        entry: PluginManifestParser.MarketplaceEntry,
        marketplaceName: String,
        checkoutDirectory: URL?,
        scope: String,
        projectRootURL: URL? = nil
    ) async throws -> InstallOutcome {
        guard let sourceValue = entry.sourceValue else {
            throw PluginLoadError(
                pluginName: entry.name,
                reason: "Marketplace entry '\(entry.name)' has no source")
        }

        // Stage the plugin contents into a temp dir first: the version may
        // only be knowable from the plugin's own manifest, and a half-copied
        // directory must never become the install.
        let staging = FileManager.default.temporaryDirectory
            .appendingPathComponent("plugin_install_\(UUID().uuidString)", isDirectory: true)
        let stagedRoot = staging.appendingPathComponent("plugin")
        try FileManager.default.createDirectory(at: stagedRoot, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: staging) }

        var gitDirForSha: URL?

        if let relative = sourceValue as? String {
            guard relative.hasPrefix("./") else {
                throw PluginLoadError(
                    pluginName: entry.name,
                    reason: "Source '\(relative)' is neither a relative path nor a source object")
            }
            guard let checkoutDirectory else {
                throw PluginLoadError(
                    pluginName: entry.name,
                    reason: "Entry '\(entry.name)' uses a relative source, which needs a materialized marketplace checkout (git, github or directory source). A bare URL marketplace can list it but not install it.")
            }
            // `metadata.pluginRoot` is the base for relative sources.
            let manifest = readManifestIfPossible(from: checkoutDirectory)
            var base = checkoutDirectory
            if let pluginRoot = manifest?.pluginRoot {
                base = checkoutDirectory.appendingPathComponent(pluginRoot)
            }
            let sourceDir = base.appendingPathComponent(relative)
            guard FileManager.default.fileExists(atPath: sourceDir.path) else {
                throw PluginLoadError(
                    pluginName: entry.name,
                    reason: "Source directory \(sourceDir.path) does not exist in the marketplace checkout")
            }
            try Self.copyContents(of: sourceDir, to: stagedRoot)
            gitDirForSha = checkoutDirectory
        } else if let decoded = try? JSONDecoder().decode(
            MarketplaceSource.self, from: JSONSerialization.data(withJSONObject: sourceValue))
        {
            switch decoded {
            case .url:
                throw PluginLoadError(
                    pluginName: entry.name,
                    reason: "The url source is not supported for plugin install; use github or git")
            case .github(let repo, let ref, _, _):
                let temp = staging.appendingPathComponent("clone")
                try await MarketplaceGit.cloneOrPull(
                    url: "https://github.com/\(repo).git", targetDir: temp, ref: ref,
                    sparsePaths: nil)
                try Self.copyContents(of: temp, to: stagedRoot)
                gitDirForSha = temp
            case .git(let url, let ref, let path, let sparse):
                let temp = staging.appendingPathComponent("clone")
                try await MarketplaceGit.cloneOrPull(
                    url: url, targetDir: temp, ref: ref, sparsePaths: sparse)
                let sourceDir = path.map { temp.appendingPathComponent($0) } ?? temp
                try Self.copyContents(of: sourceDir, to: stagedRoot)
                gitDirForSha = temp
            case .directory(let path):
                try Self.copyContents(
                    of: URL(fileURLWithPath: path, isDirectory: true), to: stagedRoot)
                gitDirForSha = nil
            }
        } else {
            throw PluginLoadError(
                pluginName: entry.name,
                reason: "Source of entry '\(entry.name)' has an unusable shape")
        }

        // Version: manifest > entry > git sha12 > unknown, Claude Code's
        // order. The manifest is read from the STAGED copy, so it is what
        // was actually installed.
        let manifestVersion = Self.stagedPluginVersion(directory: stagedRoot)
        let sha: String?
        if let gitDirForSha {
            sha = await Self.currentGitSha(directory: gitDirForSha)
        } else {
            sha = nil
        }
        let version = manifestVersion ?? entry.version
            ?? sha.map { String($0.prefix(12)) } ?? "unknown"

        let cachePath = installCacheDirectory
            .appendingPathComponent(Self.sanitizedComponent(marketplaceName), isDirectory: true)
            .appendingPathComponent(Self.sanitizedComponent(entry.name), isDirectory: true)
            .appendingPathComponent(Self.sanitizedVersion(version), isDirectory: true)
        if FileManager.default.fileExists(atPath: cachePath.path) {
            try FileManager.default.removeItem(at: cachePath)
        }
        try FileManager.default.createDirectory(
            at: cachePath.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.moveItem(at: stagedRoot, to: cachePath)

        let pluginID = "\(entry.name)@\(marketplaceName)"
        ledger.upsertRecord(
            InstalledPluginRecord(
                scope: scope,
                projectPath: projectRootURL?.standardizedFileURL.path,
                installPath: cachePath.standardizedFileURL.path,
                version: version,
                gitCommitSha: sha),
            for: pluginID)

        return InstallOutcome(
            pluginID: pluginID,
            pluginName: entry.name,
            installPath: cachePath.standardizedFileURL.path,
            version: version)
    }

    /// Removes one scope of one plugin and deletes the version cache
    /// directory when nothing references it anymore. Enable-state cleanup is
    /// the CALLER's (settings live with `AppModel`, not here).
    public func uninstall(
        pluginID: String, scope: String, projectRootURL: URL? = nil
    ) throws {
        let records = ledger.load().plugins[pluginID] ?? []
        let matching = records.filter { record in
            guard record.scope == scope else { return false }
            if scope == "project" {
                return record.projectPath == projectRootURL?.standardizedFileURL.path
            }
            return true
        }
        guard !matching.isEmpty else { return }

        ledger.removeRecords(pluginID: pluginID, scope: scope, projectPath: projectRootURL?.standardizedFileURL.path)
        if !ledger.hasRemainingRecords(pluginID: pluginID) {
            for record in matching {
                let dir = URL(fileURLWithPath: record.installPath)
                // Only ever delete INSIDE our own cache root: a ledger row
                // pointing elsewhere (hand-edited, or a seed dir) is left
                // alone rather than removed.
                let standardized = dir.standardizedFileURL.path
                let cachePrefix = installCacheDirectory.standardizedFileURL.path
                if standardized.hasPrefix(cachePrefix + "/") {
                    try? FileManager.default.removeItem(at: dir)
                }
            }
        }
    }

    /// Removes a locally-registered folder plugin (no cache to delete).
    public func uninstallLocal(path: String) {
        ledger.removeLocalPluginPath(path)
    }

    /// MARK: - Helpers

    /// The version declared by the plugin's own manifest inside a staged
    /// copy. Distinct from `readManifestIfPossible`, which looks for
    /// MARKETPLACE metadata.
    static func stagedPluginVersion(directory: URL) -> String? {
        let manifestURL = directory.appendingPathComponent(
            PluginManifestParser.manifestRelativePath)
        guard let data = try? Data(contentsOf: manifestURL),
            let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
        else { return nil }
        return json["version"] as? String
    }

    private func readManifestIfPossible(
        from dir: URL
    ) -> PluginManifestParser.MarketplaceManifest? {
        for candidate in [
            dir.appendingPathComponent(PluginManifestParser.marketplaceRelativePath),
            dir.appendingPathComponent("marketplace.json")
        ] where FileManager.default.fileExists(atPath: candidate.path) {
            if let data = try? Data(contentsOf: candidate),
                let manifest = try? PluginManifestParser.parseMarketplace(
                    data: data, sourceDescription: candidate.path) {
                return manifest
            }
        }
        // A plugin checkout's own manifest is read elsewhere; this only
        // looks for marketplace metadata (pluginRoot).
        return nil
    }

    static func sanitizedComponent(_ value: String) -> String {
        String(value.map { character in
            character.isLetter || character.isNumber || character == "-" || character == "_"
                ? character : "-"
        })
    }

    static func sanitizedVersion(_ value: String) -> String {
        String(value.map { character in
            character.isLetter || character.isNumber || character == "-"
                || character == "_" || character == "."
                ? character : "-"
        })
    }

    static func copyContents(of source: URL, to destination: URL) throws {
        let fm = FileManager.default
        let entries = try fm.contentsOfDirectory(atPath: source.path)
        for entry in entries {
            if entry == ".git" { continue }
            let sourceItem = source.appendingPathComponent(entry)
            let destinationItem = destination.appendingPathComponent(entry)
            if fm.fileExists(atPath: destinationItem.path) {
                try fm.removeItem(at: destinationItem)
            }
            try fm.copyItem(at: sourceItem, to: destinationItem)
        }
    }

    /// The checked-out commit, for the sha12 version fallback. Nil when the
    /// directory has no git history (a `directory` source, for instance).
    static func currentGitSha(directory: URL) async -> String? {
        guard
            let output = try? await ProcessExecutor.run(
                executableURL: MarketplaceGit.executableURL,
                arguments: ["rev-parse", "HEAD"],
                currentDirectoryURL: directory,
                environment: ["GIT_TERMINAL_PROMPT": "0", "PATH": "/usr/bin:/bin:/usr/local/bin"],
                timeoutSeconds: 10),
            output.exitCode == 0
        else { return nil }
        let sha = output.stdout.trimmingCharacters(in: .whitespacesAndNewlines)
        return sha.count == 40 ? sha : nil
    }
}
