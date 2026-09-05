import Foundation

// MARK: - Manifest

/// One MCP server offered by a marketplace catalog.
public struct McpMarketplaceEntry: Identifiable, Codable, Equatable, Sendable {
    public var id: String { "\(name)@\(version ?? "latest")" }
    public var name: String
    public var entryDescription: String
    /// How the server is launched. This is the load-bearing field: it names a
    /// command this app will spawn, which is why `McpMarketplaceManager.install`
    /// lands it DISABLED and why `McpImportSheet` shows the whole command line
    /// before the Install button. It is not tolerant-decoded, for the reason
    /// `McpServerConfig.transport` is not: an entry with no readable transport
    /// cannot be launched, so the honest outcome is that this ROW drops.
    public var transport: McpTransportSpec
    public var category: String?
    public var tags: [String]?
    public var version: String?
    public var author: String?
    public var homepage: String?

    public init(
        name: String,
        entryDescription: String,
        transport: McpTransportSpec,
        category: String? = nil,
        tags: [String]? = nil,
        version: String? = nil,
        author: String? = nil,
        homepage: String? = nil
    ) {
        self.name = name
        self.entryDescription = entryDescription
        self.transport = transport
        self.category = category
        self.tags = tags
        self.version = version
        self.author = author
        self.homepage = homepage
    }

    enum CodingKeys: String, CodingKey {
        case name, description, transport, category, tags, version, author, homepage
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decodeIfPresent(String.self, forKey: .name) ?? ""
        entryDescription = try container.decodeIfPresent(String.self, forKey: .description) ?? ""
        transport = try container.decode(McpTransportSpec.self, forKey: .transport)
        category = try container.decodeIfPresent(String.self, forKey: .category)
        tags = try container.decodeIfPresent([String].self, forKey: .tags)
        version = try container.decodeIfPresent(String.self, forKey: .version)
        author = try container.decodeIfPresent(String.self, forKey: .author)
        homepage = try container.decodeIfPresent(String.self, forKey: .homepage)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encode(entryDescription, forKey: .description)
        try container.encode(transport, forKey: .transport)
        try container.encodeIfPresent(category, forKey: .category)
        try container.encodeIfPresent(tags, forKey: .tags)
        try container.encodeIfPresent(version, forKey: .version)
        try container.encodeIfPresent(author, forKey: .author)
        try container.encodeIfPresent(homepage, forKey: .homepage)
    }

    /// The command line this entry will run, for display before install.
    public var commandSummary: String {
        McpServerConfig(name: name, transport: transport).commandSummary
    }
}

/// A catalog of MCP servers, published as `mcp-marketplace.json` in a repo.
public struct McpMarketplaceManifest: Identifiable, Codable, Equatable, Sendable {
    public var id: String { name }
    public var name: String
    public var manifestDescription: String?
    public var owner: String?
    public var servers: [McpMarketplaceEntry]

    public init(
        name: String,
        manifestDescription: String? = nil,
        owner: String? = nil,
        servers: [McpMarketplaceEntry] = []
    ) {
        self.name = name
        self.manifestDescription = manifestDescription
        self.owner = owner
        self.servers = servers
    }

    enum CodingKeys: String, CodingKey {
        case name, description, owner, servers
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        name = try container.decodeIfPresent(String.self, forKey: .name) ?? "Untitled catalog"
        manifestDescription = try container.decodeIfPresent(String.self, forKey: .description)
        owner = try container.decodeIfPresent(String.self, forKey: .owner)
        // Element-level tolerance: one unreadable entry is one dropped row
        // rather than a rejected catalog. Container-level tolerance would be
        // the bug -- see `decodeLossyArray`'s own note.
        servers = try container.decodeLossyArray(McpMarketplaceEntry.self, forKey: .servers)
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(name, forKey: .name)
        try container.encodeIfPresent(manifestDescription, forKey: .description)
        try container.encodeIfPresent(owner, forKey: .owner)
        try container.encode(servers, forKey: .servers)
    }
}

/// The marketplace sources this app knows about, persisted.
public struct McpMarketplaceArchive: Codable, Sendable {
    public var sources: [String: MarketplaceSource]

    public init(sources: [String: MarketplaceSource] = [:]) {
        self.sources = sources
    }

    public static func empty() -> McpMarketplaceArchive { McpMarketplaceArchive() }
}

/// Filesystem storage for the registered MCP marketplace sources.
///
/// Goes through `AppStorageRoot` and `AppJSONStore` like every other store in
/// this app, which is the deliberate departure from `SkillMarketplaceManager`:
/// that one hand-rolls `Data(contentsOf:)` under a literal `~/.turbospark`, so
/// it is neither quarantine-protected nor test-redirected (Gotcha 43).
public enum McpMarketplaceFileStore {
    private static var archiveFileURL: URL {
        AppStorageRoot.file("mcp_marketplaces.json")
    }

    public static func load() -> McpMarketplaceArchive {
        AppJSONStore.load(
            McpMarketplaceArchive.self, from: archiveFileURL, label: "MCP marketplace archive")
            ?? McpMarketplaceArchive.empty()
    }

    @discardableResult
    public static func save(_ archive: McpMarketplaceArchive) -> Bool {
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        return AppJSONStore.save(
            archive, to: archiveFileURL, label: "MCP marketplaces", encoder: encoder)
    }
}

// MARK: - Manager

/// Fetches MCP server catalogs and turns an entry into a configured server.
public final class McpMarketplaceManager: @unchecked Sendable {
    public static let shared = McpMarketplaceManager()

    /// Where clones are cached. A parameter rather than a constant so tests get
    /// a temp directory: `AppStorageRoot` covers this app's own stores and the
    /// skills marketplace escaped it by hardcoding `~/.turbospark` (Gotcha 43).
    private let cacheRoot: URL
    private let fileManager = FileManager.default

    public init(cacheRoot: URL? = nil) {
        self.cacheRoot = cacheRoot ?? AppStorageRoot.subdirectory("mcp-marketplaces")
    }

    /// The default manifest filename, when a source names none.
    public static let defaultManifestPath = "mcp-marketplace.json"

    public enum MarketplaceError: LocalizedError, Equatable {
        case invalidURL(String)
        case manifestNotFound(String)
        case emptyName
        case commandNotFound(String)
        case nameCollision(String)

        public var errorDescription: String? {
            switch self {
            case .invalidURL(let raw):
                return "Not a valid URL: \(raw)"
            case .manifestNotFound(let path):
                return "No catalog found at \(path)."
            case .emptyName:
                return "This catalog entry has no server name."
            case .commandNotFound(let command):
                return "The command '\(command)' was not found on PATH or in any "
                    + "standard tool directory. Install it first, then add this server."
            case .nameCollision(let name):
                return "A server named '\(name)' already exists. Rename or remove it first."
            }
        }
    }

    // MARK: Registered sources

    public func registeredSources() -> [String: MarketplaceSource] {
        McpMarketplaceFileStore.load().sources
    }

    @discardableResult
    public func register(name: String, source: MarketplaceSource) -> Bool {
        var archive = McpMarketplaceFileStore.load()
        archive.sources[name] = source
        return McpMarketplaceFileStore.save(archive)
    }

    @discardableResult
    public func unregister(name: String) -> Bool {
        var archive = McpMarketplaceFileStore.load()
        archive.sources.removeValue(forKey: name)
        return McpMarketplaceFileStore.save(archive)
    }

    // MARK: Fetching

    public func fetchMarketplace(source: MarketplaceSource) async throws -> McpMarketplaceManifest {
        switch source {
        case .url(let raw, _):
            guard let url = URL(string: raw), url.scheme != nil else {
                throw MarketplaceError.invalidURL(raw)
            }
            let (data, _) = try await URLSession.shared.data(from: url)
            return try JSONDecoder().decode(McpMarketplaceManifest.self, from: data)

        case .github(let repo, let ref, let path, _):
            let branch = ref?.isEmpty == false ? ref! : "main"
            let filePath = path?.isEmpty == false ? path! : Self.defaultManifestPath
            let raw = "https://raw.githubusercontent.com/\(repo)/\(branch)/\(filePath)"
            guard let url = URL(string: raw) else { throw MarketplaceError.invalidURL(raw) }
            let (data, _) = try await URLSession.shared.data(from: url)
            return try JSONDecoder().decode(McpMarketplaceManifest.self, from: data)

        case .git(let raw, let ref, let path, let sparsePaths):
            let cacheDir = cacheRoot.appendingPathComponent(
                MarketplaceGit.cacheDirectoryName(for: raw), isDirectory: true)
            try fileManager.createDirectory(at: cacheRoot, withIntermediateDirectories: true)
            try await MarketplaceGit.cloneOrPull(
                url: raw, targetDir: cacheDir, ref: ref, sparsePaths: sparsePaths)
            return try readManifest(in: cacheDir, path: path)

        case .directory(let path):
            return try readManifest(in: URL(fileURLWithPath: path), path: nil)
        }
    }

    private func readManifest(in directory: URL, path: String?) throws -> McpMarketplaceManifest {
        let candidates = [path, Self.defaultManifestPath, "marketplace.json"]
            .compactMap { $0 }
            .filter { !$0.isEmpty }
        for candidate in candidates {
            let file = directory.appendingPathComponent(candidate)
            if fileManager.fileExists(atPath: file.path) {
                let data = try Data(contentsOf: file)
                return try JSONDecoder().decode(McpMarketplaceManifest.self, from: data)
            }
        }
        throw MarketplaceError.manifestNotFound(
            directory.appendingPathComponent(candidates.first ?? Self.defaultManifestPath).path)
    }

    // MARK: Installing

    /// Turns a catalog entry into a server configuration, or throws.
    ///
    /// **VALIDATE BEFORE THE ROW EXISTS, NOT AFTER.** The skills equivalent
    /// copies files into place and then parses them, which is the defect
    /// state#107 fixed for `importSkill`: a bad entry is already installed by
    /// the time the throw happens. Nothing here reaches the store until the
    /// name, the command and the collision check have all passed.
    ///
    /// **THE RESULT IS DISABLED AND NOT AUTO-APPROVED.** A catalog is a file in
    /// somebody else's repository naming a binary this app will spawn. The
    /// hand-add path in `McpServerEditorSheet` defaults to enabled only because
    /// the user typed that command themselves.
    public func makeServerConfig(
        from entry: McpMarketplaceEntry,
        marketplaceName: String,
        existingNames: [String]
    ) throws -> McpServerConfig {
        let name = entry.name.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !name.isEmpty else { throw MarketplaceError.emptyName }

        guard !McpServerConfig.nameIsTaken(name, among: existingNames) else {
            throw MarketplaceError.nameCollision(name)
        }

        if case .stdio(let command, _, _, _, _) = entry.transport {
            let trimmed = command.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { throw MarketplaceError.emptyName }
            guard McpClientEngine.resolveExecutablePath(trimmed) != nil else {
                throw MarketplaceError.commandNotFound(trimmed)
            }
        }

        return McpServerConfig(
            name: name,
            transport: entry.transport,
            isEnabled: false,
            autoApprove: false,
            sourcePath: "marketplace:\(marketplaceName)",
            serverDescription: entry.entryDescription.isEmpty ? nil : entry.entryDescription)
    }
}
