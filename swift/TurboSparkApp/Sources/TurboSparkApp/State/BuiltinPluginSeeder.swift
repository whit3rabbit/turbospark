import Foundation

/// Seeds plugins that ship inside the app bundle (ZP-05).
///
/// The bundle holds a read-only directory marketplace
/// (`BuiltinPlugins/.claude-plugin/marketplace.json` plus one folder per
/// plugin, each entry a `./relative` source). Seeding is NOT a second
/// installer: every plugin goes through `PluginMarketplaceManager.install`
/// as a `directory` source, so staging, the versioned cache, the v2 ledger
/// and the hook trust gate are the ones every other plugin uses. Seeding
/// copies files and records state; it never runs code, and a seeded plugin's
/// hooks stay untrusted until the user approves them (`AppHookStore`).
///
/// The bundle's code signature is the integrity anchor (a tampered bundle
/// fails Gatekeeper), so there is no second digest. On top of that the
/// seeder refuses what the generic installer would copy blindly: symlinks,
/// absolute or escaping sources, and trees above the size caps.
///
/// State is ONE file, `builtin_seed.json` beside the ledger, one record per
/// builtin: `seeded` (with the version) or `suppressed`. The ledger stays the
/// truth about what is installed; this file only remembers the user's
/// intent, and `seed()` reconciles the two on every launch:
///
/// - suppressed: never reinstalled (an uninstall must stick across restarts);
/// - seeded but gone from the ledger: the user removed it some other way, so
///   it becomes suppressed rather than coming back;
/// - installed but unrecorded (a crash between the two writes): adopted;
/// - installed at an older version than the bundle: reinstalled (a bundle
///   update reaches existing users).
///
/// A corrupt state file fails closed: nothing is seeded until it is fixed,
/// because treating it as empty would resurrect everything the user removed.
public final class BuiltinPluginSeeder: @unchecked Sendable {
    public static let marketplaceName = "turbospark-plugins-builtin"
    public static let shared = BuiltinPluginSeeder(
        bundleDirectory: Bundle.module.resourceURL?
            .appendingPathComponent("BuiltinPlugins", isDirectory: true))

    /// Caps on one bundled plugin. A bundled plugin is first-party content,
    /// so these are generous; they exist so a packaging mistake cannot turn
    /// into a multi-gigabyte copy at launch.
    static let maxFilesPerPlugin = 2_000
    static let maxBytesPerPlugin: Int64 = 64 * 1024 * 1024

    public enum SeedError: Error, Equatable, LocalizedError {
        case lockTimeout
        case corruptState
        case refused(plugin: String, reason: String)

        public var errorDescription: String? {
            switch self {
            case .lockTimeout:
                return "Another TurboSpark process is seeding builtin plugins."
            case .corruptState:
                return "builtin_seed.json is unreadable; builtin plugins are not seeded until it is repaired or removed."
            case .refused(let plugin, let reason):
                return "Builtin plugin '\(plugin)' refused: \(reason)"
            }
        }
    }

    public struct Record: Codable, Equatable, Sendable {
        public enum State: String, Codable, Sendable { case seeded, suppressed }
        public var state: State
        public var version: String?
    }

    struct StateFile: Codable, Equatable {
        var version = 1
        var plugins: [String: Record] = [:]
    }

    public struct Report: Equatable, Sendable {
        public var seeded: [String] = []
        public var suppressed: [String] = []
        public var refused: [String: String] = [:]
    }

    public let bundleDirectory: URL?
    let manager: PluginMarketplaceManager
    private let lockTimeout: TimeInterval

    public init(
        bundleDirectory: URL?,
        manager: PluginMarketplaceManager = .shared,
        lockTimeout: TimeInterval = 10
    ) {
        self.bundleDirectory = bundleDirectory
        self.manager = manager
        self.lockTimeout = lockTimeout
    }

    var stateURL: URL { manager.root.appendingPathComponent("builtin_seed.json") }
    var lockURL: URL { manager.root.appendingPathComponent(".builtin-seed.lock") }

    // MARK: - The bundled catalog

    /// The bundled marketplace entries; empty (never throwing) when the
    /// app ships no builtins, so the row stays hidden.
    public func entries() -> [PluginManifestParser.MarketplaceEntry] {
        guard let manifest = try? loadManifest() else { return [] }
        return manifest.entries
    }

    /// Whether the app ships any builtin. The marketplace row is shown only
    /// then: an always-present empty "official" row has no user value.
    public var hasBuiltins: Bool { !entries().isEmpty }

    private func loadManifest() throws -> PluginManifestParser.MarketplaceManifest {
        guard let dir = bundleDirectory else {
            throw PluginLoadError(pluginName: nil, reason: "No builtin plugin directory")
        }
        let url = dir.appendingPathComponent(PluginManifestParser.marketplaceRelativePath)
        let data = try Data(contentsOf: url)
        return try PluginManifestParser.parseMarketplace(
            data: data, sourceDescription: url.path)
    }

    // MARK: - State

    private func loadState() throws -> StateFile {
        guard FileManager.default.fileExists(atPath: stateURL.path) else { return StateFile() }
        guard let data = try? Data(contentsOf: stateURL),
            let file = try? JSONDecoder().decode(StateFile.self, from: data),
            file.version == 1
        else { throw SeedError.corruptState }
        return file
    }

    private func saveState(_ file: StateFile) throws {
        try FileManager.default.createDirectory(
            at: manager.root, withIntermediateDirectories: true)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(file).write(to: stateURL, options: .atomic)
    }

    public func record(for pluginName: String) -> Record? {
        try? loadState().plugins[pluginName]
    }

    public func isSuppressed(_ pluginName: String) -> Bool {
        record(for: pluginName)?.state == .suppressed
    }

    /// The user uninstalled a builtin: remember it so a restart does not
    /// seed it again. Only the user scope is the seeded one, so callers do
    /// not call this for a project-scope removal.
    public func recordUninstalled(pluginName: String) throws {
        let lock = try acquireLock()
        defer { lock.release() }
        var file = try loadState()
        file.plugins[pluginName] = Record(state: .suppressed, version: nil)
        try saveState(file)
    }

    /// The user installed a builtin from its row: clears any suppression.
    public func recordInstalled(pluginName: String, version: String) throws {
        let lock = try acquireLock()
        defer { lock.release() }
        var file = try loadState()
        file.plugins[pluginName] = Record(state: .seeded, version: version)
        try saveState(file)
    }

    // MARK: - Seed lock

    /// A cross-process advisory lock on a file in the plugin root. `flock`
    /// is released by the kernel when the holder dies, so a crash never
    /// leaves a stale lock to clean up.
    final class SeedLock {
        private var descriptor: Int32
        init(descriptor: Int32) { self.descriptor = descriptor }
        func release() {
            guard descriptor >= 0 else { return }
            flock(descriptor, LOCK_UN)
            close(descriptor)
            descriptor = -1
        }
        deinit { release() }
    }

    func acquireLock() throws -> SeedLock {
        try FileManager.default.createDirectory(
            at: manager.root, withIntermediateDirectories: true)
        let fd = open(lockURL.path, O_CREAT | O_RDWR | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw SeedError.lockTimeout }
        let deadline = Date().addingTimeInterval(lockTimeout)
        while flock(fd, LOCK_EX | LOCK_NB) != 0 {
            if Date() >= deadline {
                close(fd)
                throw SeedError.lockTimeout
            }
            usleep(20_000)
        }
        return SeedLock(descriptor: fd)
    }

    // MARK: - Source safety

    /// Resolves an entry's source to a directory strictly inside the bundle
    /// and checks it: relative `./` string only, no symlinks anywhere in the
    /// tree, file count and byte caps.
    func validatedSource(of entry: PluginManifestParser.MarketplaceEntry) throws -> URL {
        func refuse(_ reason: String) -> SeedError { .refused(plugin: entry.name, reason: reason) }
        guard let bundle = bundleDirectory else { throw refuse("no bundle directory") }
        guard let relative = entry.sourceValue as? String, relative.hasPrefix("./") else {
            throw refuse("only a ./relative source is allowed")
        }
        let source: URL
        do {
            source = try PluginMarketplaceManager.confinedSource(
                bundle.appendingPathComponent(relative), within: bundle, pluginName: entry.name)
        } catch { throw refuse("source escapes the bundle") }
        let root = bundle.standardizedFileURL.resolvingSymlinksInPath()
        guard source.path != root.path else { throw refuse("source is the bundle root") }

        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: source.path, isDirectory: &isDirectory),
            isDirectory.boolValue
        else { throw refuse("source directory is missing") }

        let keys: [URLResourceKey] = [.isSymbolicLinkKey, .isRegularFileKey, .fileSizeKey]
        guard let walker = FileManager.default.enumerator(
            at: source, includingPropertiesForKeys: keys, options: [])
        else { throw refuse("source cannot be read") }
        var files = 0
        var bytes: Int64 = 0
        for case let item as URL in walker {
            let values = try? item.resourceValues(forKeys: Set(keys))
            if values?.isSymbolicLink != false { throw refuse("contains a symlink or unreadable entry") }
            if values?.isRegularFile == true {
                files += 1
                bytes += Int64(values?.fileSize ?? 0)
            }
            if files > Self.maxFilesPerPlugin || bytes > Self.maxBytesPerPlugin {
                throw refuse("exceeds the size cap")
            }
        }
        return source
    }

    // MARK: - Seeding

    /// Reconciles the bundle against the ledger and state file under the
    /// seed lock. Local only: no network, no code execution. One bad entry
    /// is reported in `refused` and does not stop the rest.
    @discardableResult
    public func seed() async throws -> Report {
        var report = Report()
        let catalog = entries()
        guard !catalog.isEmpty, let bundle = bundleDirectory else { return report }
        let lock = try acquireLock()
        defer { lock.release() }
        var file = try loadState()

        for entry in catalog {
            let name = entry.name
            let pluginID = "\(name)@\(Self.marketplaceName)"
            if file.plugins[name]?.state == .suppressed {
                report.suppressed.append(name)
                continue
            }
            let userRecord = manager.ledger.load().plugins[pluginID]?
                .first { $0.scope == "user" }
            if file.plugins[name]?.state == .seeded, userRecord == nil {
                // Removed outside the app's uninstall path: respect it.
                file.plugins[name] = Record(state: .suppressed, version: nil)
                try saveState(file)
                report.suppressed.append(name)
                continue
            }
            do {
                let source = try validatedSource(of: entry)
                let bundled = PluginMarketplaceManager.stagedPluginVersion(directory: source)
                    ?? entry.version ?? "unknown"
                let current = userRecord.flatMap {
                    FileManager.default.fileExists(atPath: $0.installPath) ? $0.version : nil
                }
                if let current, current == PluginMarketplaceManager.sanitizedVersion(bundled) {
                    if file.plugins[name]?.state != .seeded {
                        // Installed but unrecorded (crash between the writes).
                        file.plugins[name] = Record(state: .seeded, version: current)
                        try saveState(file)
                    }
                    continue
                }
                let outcome = try await manager.install(
                    entry: entry,
                    marketplaceName: Self.marketplaceName,
                    checkoutDirectory: bundle,
                    marketplaceSource: .directory(path: bundle.path),
                    scope: "user")
                file.plugins[name] = Record(state: .seeded, version: outcome.version)
                try saveState(file)
                report.seeded.append(name)
            } catch let error as SeedError {
                if case .refused(_, let reason) = error { report.refused[name] = reason }
                else { throw error }
            } catch {
                report.refused[name] = error.localizedDescription
            }
        }
        return report
    }
}
