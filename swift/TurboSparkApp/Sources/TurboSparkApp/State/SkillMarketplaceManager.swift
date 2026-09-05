import Foundation

/// Sources from which a skill marketplace or remote skill can be acquired.
public enum SkillMarketplaceSource: Codable, Equatable, Sendable {
    case url(url: String, headers: [String: String]?)
    case github(repo: String, ref: String?, path: String?, sparsePaths: [String]?)
    case git(url: String, ref: String?, path: String?, sparsePaths: [String]?)
    case directory(path: String)

    enum CodingKeys: String, CodingKey {
        case type, url, headers, repo, ref, path, sparsePaths
    }

    public init(from decoder: any Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let type = try container.decode(String.self, forKey: .type)
        switch type.lowercased() {
        case "url", "https", "http":
            let url = try container.decode(String.self, forKey: .url)
            let headers = try container.decodeIfPresent([String: String].self, forKey: .headers)
            self = .url(url: url, headers: headers)
        case "github":
            let repo = try container.decode(String.self, forKey: .repo)
            let ref = try container.decodeIfPresent(String.self, forKey: .ref)
            let path = try container.decodeIfPresent(String.self, forKey: .path)
            let sparse = try container.decodeIfPresent([String].self, forKey: .sparsePaths)
            self = .github(repo: repo, ref: ref, path: path, sparsePaths: sparse)
        case "git":
            let url = try container.decode(String.self, forKey: .url)
            let ref = try container.decodeIfPresent(String.self, forKey: .ref)
            let path = try container.decodeIfPresent(String.self, forKey: .path)
            let sparse = try container.decodeIfPresent([String].self, forKey: .sparsePaths)
            self = .git(url: url, ref: ref, path: path, sparsePaths: sparse)
        case "directory", "file", "local":
            let path = try container.decode(String.self, forKey: .path)
            self = .directory(path: path)
        default:
            let url = try container.decodeIfPresent(String.self, forKey: .url) ?? ""
            self = .url(url: url, headers: nil)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        switch self {
        case .url(let url, let headers):
            try container.encode("url", forKey: .type)
            try container.encode(url, forKey: .url)
            try container.encodeIfPresent(headers, forKey: .headers)
        case .github(let repo, let ref, let path, let sparse):
            try container.encode("github", forKey: .type)
            try container.encode(repo, forKey: .repo)
            try container.encodeIfPresent(ref, forKey: .ref)
            try container.encodeIfPresent(path, forKey: .path)
            try container.encodeIfPresent(sparse, forKey: .sparsePaths)
        case .git(let url, let ref, let path, let sparse):
            try container.encode("git", forKey: .type)
            try container.encode(url, forKey: .url)
            try container.encodeIfPresent(ref, forKey: .ref)
            try container.encodeIfPresent(path, forKey: .path)
            try container.encodeIfPresent(sparse, forKey: .sparsePaths)
        case .directory(let path):
            try container.encode("directory", forKey: .type)
            try container.encode(path, forKey: .path)
        }
    }
}

/// Entry for an individual skill listed within a marketplace manifest.
public struct MarketplaceSkillEntry: Identifiable, Codable, Equatable, Sendable {
    public var id: String { "\(name)@\(version ?? "latest")" }
    public var name: String
    public var description: String
    public var source: SkillMarketplaceSource
    public var category: String?
    public var tags: [String]?
    public var version: String?
    public var author: String?

    public init(
        name: String,
        description: String,
        source: SkillMarketplaceSource,
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

    public var marketplacesDirectory: URL {
        let home = fileManager.homeDirectoryForCurrentUser
        let dir = home.appendingPathComponent(".turbospark/marketplaces", isDirectory: true)
        try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    public var installedLedgerURL: URL {
        let home = fileManager.homeDirectoryForCurrentUser
        let dir = home.appendingPathComponent(".turbospark/plugins", isDirectory: true)
        try? fileManager.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir.appendingPathComponent("installed_skills.json")
    }

    public var knownMarketplacesURL: URL {
        marketplacesDirectory.appendingPathComponent("known_marketplaces.json")
    }

    // MARK: - Known Marketplaces Configuration

    public func loadKnownMarketplaces() -> [String: SkillMarketplaceSource] {
        lock.lock()
        defer { lock.unlock() }
        guard let data = try? Data(contentsOf: knownMarketplacesURL),
              let dict = try? JSONDecoder().decode([String: SkillMarketplaceSource].self, from: data) else {
            return defaultMarketplaces
        }
        return dict
    }

    public func saveKnownMarketplace(name: String, source: SkillMarketplaceSource) throws {
        lock.lock()
        defer { lock.unlock() }
        var current = (try? JSONDecoder().decode([String: SkillMarketplaceSource].self, from: Data(contentsOf: knownMarketplacesURL))) ?? defaultMarketplaces
        current[name] = source
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(current)
        try data.write(to: knownMarketplacesURL, options: .atomic)
    }

    private var defaultMarketplaces: [String: SkillMarketplaceSource] {
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

    public func fetchMarketplace(source: SkillMarketplaceSource) async throws -> MarketplaceManifest {
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
                urlStr.replacingOccurrences(of: "/", with: "-").replacingOccurrences(of: ":", with: "-"),
                isDirectory: true
            )
            try cloneOrPullGit(url: urlStr, targetDir: cacheDir, ref: nil, sparsePaths: path != nil ? [path!] : nil)
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

    private func cloneOrPullGit(url: String, targetDir: URL, ref: String?, sparsePaths: [String]?) throws {
        if fileManager.fileExists(atPath: targetDir.appendingPathComponent(".git").path) {
            // Already cloned, run git pull
            let pull = Process()
            pull.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            pull.currentDirectoryURL = targetDir
            pull.arguments = ["pull", "--quiet"]
            try pull.run()
            pull.waitUntilExit()
            return
        }

        try fileManager.createDirectory(at: targetDir, withIntermediateDirectories: true)
        let isSparse = (sparsePaths != nil && !(sparsePaths?.isEmpty ?? true))
        var args = ["clone", "--depth", "1"]
        if isSparse {
            args.append(contentsOf: ["--filter=blob:none", "--no-checkout"])
        }
        if let ref, !ref.isEmpty {
            args.append(contentsOf: ["--branch", ref])
        }
        args.append(contentsOf: [url, targetDir.path])

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git")
        process.arguments = args
        try process.run()
        process.waitUntilExit()

        guard process.terminationStatus == 0 else {
            throw NSError(domain: "SkillMarketplace", code: 10, userInfo: [NSLocalizedDescriptionKey: "git clone failed with exit code \(process.terminationStatus)"])
        }

        if isSparse, let sparsePaths {
            let setCone = Process()
            setCone.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            setCone.currentDirectoryURL = targetDir
            setCone.arguments = ["sparse-checkout", "set", "--cone", "--"] + sparsePaths
            try setCone.run()
            setCone.waitUntilExit()

            let checkout = Process()
            checkout.executableURL = URL(fileURLWithPath: "/usr/bin/git")
            checkout.currentDirectoryURL = targetDir
            checkout.arguments = ["checkout", "HEAD"]
            try checkout.run()
            checkout.waitUntilExit()
        }
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
            try cloneOrPullGit(url: urlStr, targetDir: tempGitDir, ref: ref, sparsePaths: sparseP)
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
