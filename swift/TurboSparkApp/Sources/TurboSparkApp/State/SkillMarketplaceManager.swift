import Foundation

/// Entry for an individual skill listed within a marketplace manifest.
public struct MarketplaceSkillEntry: Identifiable, Codable, Equatable, Sendable {
    public var id: String { "\(name)@\(version ?? "latest")" }
    public var name: String
    public var description: String
    public var source: MarketplaceSource
    public var category: String?
    public var tags: [String]?
    public var version: String?
    public var author: String?

    public init(
        name: String,
        description: String,
        source: MarketplaceSource,
        category: String? = nil,
        tags: [String]? = nil,
        version: String? = nil,
        author: String? = nil
    ) {
        self.name = name
        self.description = description
        self.source = source
        self.category = category
        self.tags = tags
        self.version = version
        self.author = author
    }
}

/// Top-level manifest for a skill marketplace repository.
public struct MarketplaceManifest: Identifiable, Codable, Equatable, Sendable {
    public var id: String { name }
    public var name: String
    public var description: String?
    public var owner: String?
    public var skills: [MarketplaceSkillEntry]

    public init(
        name: String,
        description: String? = nil,
        owner: String? = nil,
        skills: [MarketplaceSkillEntry] = []
    ) {
        self.name = name
        self.description = description
        self.owner = owner
        self.skills = skills
    }
}

/// Installation ledger entry tracking multi-scope skill provenance.
public struct InstalledSkillRecord: Identifiable, Codable, Equatable, Sendable {
    public var id: String { "\(skillName):\(scope)" }
    public var skillName: String
    public var scope: String // "user" or "project"
    public var projectPath: String?
    public var installPath: String
    public var version: String?
    public var gitCommitSha: String?
    public var installedAt: Date

    public init(
        skillName: String,
        scope: String,
        projectPath: String? = nil,
        installPath: String,
        version: String? = nil,
        gitCommitSha: String? = nil,
        installedAt: Date = Date()
    ) {
        self.skillName = skillName
        self.scope = scope
        self.projectPath = projectPath
        self.installPath = installPath
        self.version = version
        self.gitCommitSha = gitCommitSha
        self.installedAt = installedAt
    }
}

/// Manager handling marketplace discovery, Git clone / sparse-checkout,
/// HTTPS downloading, and multi-scope installation.
public final class SkillMarketplaceManager: @unchecked Sendable {
    public static let shared = SkillMarketplaceManager()

    private let fileManager = FileManager.default
    private let lock = NSLock()

    public init() {}

    // MARK: - Standard Directories

    /// Marketplace clone cache: `~/.turbospark/marketplaces` for the Default
    /// profile, inside that profile's own folder for anyone else.
    public var marketplacesDirectory: URL {
        UserProfileStore.userScopeSubdirectory("marketplaces")
    }

    /// The install ledger for skills added from a marketplace. Per profile,
    /// so two users can hold different versions of the same skill.
    public var installedLedgerURL: URL {
        UserProfileStore.userScopeSubdirectory("plugins")
            .appendingPathComponent("installed_skills.json")
    }

    public var knownMarketplacesURL: URL {
        marketplacesDirectory.appendingPathComponent("known_marketplaces.json")
    }

    // MARK: - Known Marketplaces Configuration

    public func loadKnownMarketplaces() -> [String: MarketplaceSource] {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: knownMarketplacesURL),
              let dict = try? JSONDecoder().decode([String: MarketplaceSource].self, from: data) else {
            return defaultMarketplaces
        }
        return dict
    }

    public func saveKnownMarketplace(name: String, source: MarketplaceSource) throws {
        lock.lock()
        defer { lock.unlock() }
        var current = (try? JSONDecoder().decode([String: MarketplaceSource].self, from: Data(contentsOf: knownMarketplacesURL))) ?? defaultMarketplaces
        current[name] = source
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(current)
        try fileManager.createDirectory(at: marketplacesDirectory, withIntermediateDirectories: true)
        try data.write(to: knownMarketplacesURL, options: .atomic)
    }

    public func removeKnownMarketplace(name: String) throws {
        var sources = loadKnownMarketplaces()
        sources.removeValue(forKey: name)
        try fileManager.createDirectory(at: marketplacesDirectory, withIntermediateDirectories: true)
        try JSONEncoder().encode(sources).write(to: knownMarketplacesURL, options: .atomic)
    }

    private var defaultMarketplaces: [String: MarketplaceSource] {
        [
            "turbospark-official": .github(
                repo: "whit3rabbit/agent-skills",
                ref: "main",
                path: "marketplace.json",
                sparsePaths: nil
            )
        ]
    }

    // MARK: - Marketplace Manifest Fetching

    public func fetchMarketplace(source: MarketplaceSource) async throws -> MarketplaceManifest {
        switch source {
        case .url(let urlStr, _):
            guard let url = URL(string: urlStr) else {
                throw NSError(domain: "SkillMarketplace", code: 1, userInfo: [NSLocalizedDescriptionKey: "Invalid URL: \(urlStr)"])
            }
            let (data, _) = try await URLSession.shared.data(from: url)
            return try JSONDecoder().decode(MarketplaceManifest.self, from: data)

        case .github(let repo, let ref, let path, _):
            let branch = ref ?? "main"
            let filePath = path ?? "marketplace.json"
            let rawURLString = "https://raw.githubusercontent.com/\(repo)/\(branch)/\(filePath)"
            guard let url = URL(string: rawURLString) else {
                throw NSError(domain: "SkillMarketplace", code: 2, userInfo: [NSLocalizedDescriptionKey: "Invalid GitHub raw URL: \(rawURLString)"])
            }
            let (data, _) = try await URLSession.shared.data(from: url)
            return try JSONDecoder().decode(MarketplaceManifest.self, from: data)

        case .git(let urlStr, _, let path, _):
            let cacheDir = marketplacesDirectory.appendingPathComponent(
                MarketplaceGit.cacheDirectoryName(for: urlStr), isDirectory: true)
            try await cloneOrPullGit(
                url: urlStr, targetDir: cacheDir, ref: nil,
                sparsePaths: path != nil ? [path!] : nil)
            let manifestFile = cacheDir.appendingPathComponent(path ?? "marketplace.json")
            let data = try Data(contentsOf: manifestFile)
            return try JSONDecoder().decode(MarketplaceManifest.self, from: data)

        case .directory(let dirPath):
            let dirURL = URL(fileURLWithPath: dirPath)
            let manifestFile = dirURL.appendingPathComponent("marketplace.json")
            let data = try Data(contentsOf: manifestFile)
            return try JSONDecoder().decode(MarketplaceManifest.self, from: data)
        }
    }

    // MARK: - Git Operations

    /// Delegates to the shared `MarketplaceGit`, which bounds the spawn with a
    /// timeout and an output cap and checks EVERY step's exit status. This used
    /// to be four raw `Process()` calls here that checked only the clone, so a
    /// failed `pull` was silent and the caller read a stale cache as fresh.
    private func cloneOrPullGit(
        url: String, targetDir: URL, ref: String?, sparsePaths: [String]?
    ) async throws {
        try await MarketplaceGit.cloneOrPull(
            url: url, targetDir: targetDir, ref: ref, sparsePaths: sparsePaths)
    }

    // MARK: - Skill Installation

    @discardableResult
    public func installSkill(
        entry: MarketplaceSkillEntry,
        targetScope: SkillScope,
        projectRootURL: URL? = nil
    ) async throws -> AppSkill {
        let destBaseDir: URL
        let scopeKey: String
        switch targetScope {
        case .userGlobal, .bundled:
            destBaseDir = SkillManager.shared.defaultUserSkillsDirectory
            scopeKey = "user"
        case .projectLocal(let projectPath):
            destBaseDir = URL(fileURLWithPath: projectPath).appendingPathComponent(".turbospark/skills", isDirectory: true)
            scopeKey = "project"
        case .plugin:
            throw NSError(domain: "SkillMarketplace", code: 6, userInfo: [NSLocalizedDescriptionKey: "A marketplace skill cannot be installed into plugin scope."])
        }

        let targetSkillDir = destBaseDir.appendingPathComponent(entry.name, isDirectory: true)
        try fileManager.createDirectory(at: targetSkillDir, withIntermediateDirectories: true)

        var installedMdURL: URL?

        switch entry.source {
        case .url(let urlStr, _):
            guard let url = URL(string: urlStr) else {
                throw NSError(domain: "SkillMarketplace", code: 3, userInfo: [NSLocalizedDescriptionKey: "Invalid download URL: \(urlStr)"])
            }
            let (data, _) = try await URLSession.shared.data(from: url)
            let destFile = targetSkillDir.appendingPathComponent("SKILL.md")
            try data.write(to: destFile, options: .atomic)
            installedMdURL = destFile

        case .github(let repo, let ref, let path, _):
            let branch = ref ?? "main"
            let skillSubpath = path ?? "skills/\(entry.name)"
            let skillMdPath = skillSubpath.hasSuffix(".md") ? skillSubpath : "\(skillSubpath)/SKILL.md"
            let rawURLString = "https://raw.githubusercontent.com/\(repo)/\(branch)/\(skillMdPath)"
            guard let url = URL(string: rawURLString) else {
                throw NSError(domain: "SkillMarketplace", code: 4, userInfo: [NSLocalizedDescriptionKey: "Invalid GitHub URL"])
            }
            let (data, _) = try await URLSession.shared.data(from: url)
            let destFile = targetSkillDir.appendingPathComponent("SKILL.md")
            try data.write(to: destFile, options: .atomic)
            installedMdURL = destFile

        case .git(let urlStr, let ref, let path, let sparse):
            let tempGitDir = fileManager.temporaryDirectory.appendingPathComponent("git_skill_\(UUID().uuidString)", isDirectory: true)
            defer { try? fileManager.removeItem(at: tempGitDir) }
            let sparseP = sparse ?? (path != nil ? [path!] : nil)
            try await cloneOrPullGit(
                url: urlStr, targetDir: tempGitDir, ref: ref, sparsePaths: sparseP)
            let sourceFolder = path != nil ? tempGitDir.appendingPathComponent(path!) : tempGitDir
            try copyDirectoryContents(from: sourceFolder, to: targetSkillDir)
            installedMdURL = targetSkillDir.appendingPathComponent("SKILL.md")

        case .directory(let dirPath):
            let sourceURL = URL(fileURLWithPath: dirPath)
            try copyDirectoryContents(from: sourceURL, to: targetSkillDir)
            installedMdURL = targetSkillDir.appendingPathComponent("SKILL.md")
        }

        guard let skillFile = installedMdURL, fileManager.fileExists(atPath: skillFile.path) else {
            throw NSError(domain: "SkillMarketplace", code: 5, userInfo: [NSLocalizedDescriptionKey: "SKILL.md was not found after install."])
        }

        let parsedSkill = try SkillParser.parseFile(
            at: skillFile,
            scope: targetScope,
            agentOrigin: .custom,
            containedIn: projectRootURL
        )

        // Record installation in ledger
        recordInstallation(InstalledSkillRecord(
            skillName: entry.name,
            scope: scopeKey,
            projectPath: projectRootURL?.path,
            installPath: targetSkillDir.path,
            version: entry.version,
            gitCommitSha: nil
        ))

        SkillManager.shared.invalidateResolutionCache()
        return parsedSkill
    }

    private func copyDirectoryContents(from source: URL, to destination: URL) throws {
        let entries = try fileManager.contentsOfDirectory(atPath: source.path)
        for entry in entries {
            if entry == ".git" { continue }
            let srcItem = source.appendingPathComponent(entry)
            let dstItem = destination.appendingPathComponent(entry)
            if fileManager.fileExists(atPath: dstItem.path) {
                try? fileManager.removeItem(at: dstItem)
            }
            try fileManager.copyItem(at: srcItem, to: dstItem)
        }
    }

    private func recordInstallation(_ record: InstalledSkillRecord) {
        lock.lock()
        defer { lock.unlock() }
        var ledger: [String: [InstalledSkillRecord]] = [:]
        if let data = try? Data(contentsOf: installedLedgerURL),
           let decoded = try? JSONDecoder().decode([String: [InstalledSkillRecord]].self, from: data) {
            ledger = decoded
        }
        var list = ledger[record.skillName] ?? []
        list.removeAll { $0.scope == record.scope && $0.projectPath == record.projectPath }
        list.append(record)
        ledger[record.skillName] = list

        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        if let data = try? encoder.encode(ledger) {
            try? data.write(to: installedLedgerURL, options: .atomic)
        }
    }
}
