import Foundation

/// Backup export for user profiles: a profile folder (or the Default user's
/// two roots) zipped with a self-describing manifest, restorable by
/// `ProfileBackupImport` or by hand with any unzip tool.
///
/// The archive is a plain `.zip` on purpose: a backup whose contents need
/// this app to read them back is a weaker artifact than one Finder can open.
/// The manifest at the archive root is what makes it self-describing, and its
/// name is chosen so no profile store file can ever collide with it.
///
/// Everything here takes its inputs as URLs and strings: the AppModel wrapper
/// (`AppModel+ProfileBackup`) resolves this run's profile roots and shows the
/// panels, and the tests drive real temp directories through the same
/// functions.
enum ProfileBackup {
    /// Bump when the archive layout or manifest shape changes. An import that
    /// meets a higher version refuses with a message rather than guessing.
    static let formatVersion = 1
    static let manifestFileName = "turbospark-backup-manifest.json"

    /// `ditto` lives at a fixed path on every macOS this app supports, like
    /// MarketplaceGit's `/usr/bin/git`.
    static let dittoURL = URL(fileURLWithPath: "/usr/bin/ditto")

    // MARK: - Manifest

    /// How the payload is laid out inside the archive.
    enum Layout: String, Codable, CaseIterable {
        /// One self-contained profile folder; its contents sit at the archive
        /// root beside the manifest.
        case profileFolder = "profile-folder"
        /// The Default user: `app-support/` holds the machine-root stores and
        /// `dot-turbospark/` holds the shared-home content, each minus
        /// `defaultExcludedTopLevelNames`.
        case defaultTwoRoot = "default-two-root"
    }

    struct Manifest: Codable, Equatable {
        var formatVersion: Int
        var kind: String
        var profileID: String
        var profileName: String
        var profileCreatedAt: Date
        var isDefault: Bool
        var layout: Layout
        var exportedAt: Date
        var appVersion: String
        /// Every payload path inside the archive, relative to the archive
        /// root and sorted, so a reader can diff what arrived against what
        /// was written.
        var contents: [String]
        /// Which export categories were selected, sorted, or nil for "the
        /// whole profile". Import does not read this -- the contents list
        /// above already says what arrived -- but it makes a partial backup
        /// honest about being partial. Tolerantly decoded: v1.0 archives
        /// have no such key and read as everything.
        var includedCategories: [String]?
    }

    // MARK: - Export categories

    /// One user-facing group of things a profile owns, expressed at
    /// TOP-LEVEL-ENTRY granularity: the sheet toggles a category, and a
    /// category owns whole files and directories inside either source root.
    /// SOUL and personality are settings keys, so they travel under
    /// `settings`; memory spans the `memory` and `projects` directories plus
    /// the Default user's machine-root `profile-memory`.
    struct Category: Identifiable, Equatable {
        let id: String
        let ownedTopLevelNames: Set<String>
    }

    static let categories: [Category] = [
        Category(id: "settings", ownedTopLevelNames: [
            "settings.json", "appearance.json", "disabled_items.json",
            "granted_folders.json", "input_history.json",
        ]),
        Category(id: "chats", ownedTopLevelNames: ["chats_archive.json"]),
        Category(id: "projects", ownedTopLevelNames: ["projects_archive.json"]),
        Category(id: "models", ownedTopLevelNames: [
            "model_organization.json", "excluded_scan_paths.json",
        ]),
        Category(id: "mcp", ownedTopLevelNames: [
            "global_mcp_servers.json", "mcp_marketplaces.json", "mcp-marketplaces",
        ]),
        Category(id: "skills", ownedTopLevelNames: ["skills"]),
        Category(id: "agents", ownedTopLevelNames: ["agents"]),
        Category(id: "tools", ownedTopLevelNames: ["tools"]),
        Category(id: "plugins", ownedTopLevelNames: ["plugins", "marketplaces"]),
        Category(id: "hooks", ownedTopLevelNames: ["hooks.json", "Hooks"]),
        Category(id: "memory", ownedTopLevelNames: ["memory", "projects", "profile-memory"]),
        Category(
            id: "activity",
            ownedTopLevelNames: ["cron_jobs.json", "steering-vectors", "tool-observations"]),
    ]

    static let allCategoryIDs: Set<String> = Set(categories.map(\.id))

    private static let categoryOwning: [String: String] = {
        var map: [String: String] = [:]
        for category in categories {
            for name in category.ownedTopLevelNames {
                map[name] = category.id
            }
        }
        return map
    }()

    /// Whether a top-level entry belongs in an export with `included`
    /// selected. An entry NO category owns always travels: the table above
    /// is a description of the known stores, not an allowlist, so an
    /// unknown file is backed up rather than silently dropped.
    static func shouldInclude(topLevelName: String, included: Set<String>?) -> Bool {
        guard let included else { return true }
        guard let owner = categoryOwning[topLevelName] else { return true }
        return included.contains(owner)
    }

    static let manifestKind = "turbospark-profile-backup"

    /// Decodes and validates manifest data: right kind, exactly this build's
    /// format version. An unknown layout or a malformed shape fails decoding
    /// and reads as corrupt; that is the right verdict for a future archive
    /// too, since a layout change bumps the version it cannot decode under.
    static func validatedManifest(from data: Data) throws -> Manifest {
        guard let manifest = try? JSONDecoder().decode(Manifest.self, from: data),
            manifest.kind == manifestKind
        else { throw ProfileBackupImport.ImportError.manifestCorrupt }
        guard manifest.formatVersion == formatVersion else {
            throw ProfileBackupImport.ImportError.unsupportedVersion(manifest.formatVersion)
        }
        return manifest
    }

    // MARK: - Name sanitization

    /// Makes a profile (or chat) name safe to use as a file name without
    /// touching what the registry stores. Path-hostile punctuation becomes a
    /// dash, control characters (which display as nothing or worse) are
    /// dropped, edge dots and whitespace go away, and the result is capped at
    /// 60 characters. `String.prefix` cuts on whole Characters, so an emoji
    /// at the cap is removed whole rather than torn in half. Emoji otherwise
    /// survive: they are legal in file names and are part of the name.
    static func sanitizedFileName(_ name: String, fallback: String) -> String {
        let invalid = CharacterSet(charactersIn: "/:\\?%*|\"<>")
            .union(.controlCharacters)
        let cleaned = name.components(separatedBy: invalid).joined(separator: "-")
        var trimmed = cleaned.trimmingCharacters(in: .whitespacesAndNewlines)
        while trimmed.hasPrefix(".") || trimmed.hasSuffix(".") {
            trimmed = trimmed.trimmingCharacters(in: CharacterSet(charactersIn: "."))
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let capped = String(trimmed.prefix(60))
        return capped.isEmpty ? fallback : capped
    }

    // MARK: - Default-user source enumeration

    /// Top-level entries of the Default user's two roots that a backup must
    /// NOT carry: the registry and the other profiles' folders at the machine
    /// root, and the shared model downloads and install registry in
    /// `~/.turbospark`. Everything else in either root belongs to the Default
    /// user and travels.
    static let defaultExcludedTopLevelNames: Set<String> = [
        "profiles.json", "profiles", "models", "installed.json",
    ]

    static let appSupportDirectoryName = "app-support"
    static let dotTurbosparkDirectoryName = "dot-turbospark"

    // MARK: - Export

    enum ProcessError: Error, Equatable {
        case sourceMissing
        case stagingFailed
        case processFailed(step: String, exitCode: Int32, stderr: String)
    }

    /// Writes a backup of `profile` to `destination` and returns the manifest
    /// that went inside. Runs synchronously; the AppModel wrapper decides
    /// whether to flush stores first and which queue this belongs on.
    ///
    /// The staging directory is the whole trick of adding a manifest to a
    /// `ditto` archive: `ditto` cannot exclude or inject, so the payload is
    /// assembled in a temp directory (per top-level entry for the Default
    /// user, so the multi-GB shared models directory is never even read for
    /// copy), the manifest is written beside it, and the staging root's
    /// CONTENTS are what `ditto -c` archives.
    static func export(
        profile: UserProfile,
        machineRoot: URL,
        turbosparkHome: URL,
        destination: URL,
        appVersion: String,
        exportedAt: Date = Date(),
        included: Set<String>? = nil
    ) async throws -> Manifest {
        let fileManager = FileManager.default
        let staging = fileManager.temporaryDirectory
            .appendingPathComponent("turbospark-backup-\(UUID().uuidString)", isDirectory: true)
        // Created eagerly so a poisoned temp directory fails here, at the
        // call site with context, rather than inside ditto.
        do {
            try fileManager.createDirectory(at: staging, withIntermediateDirectories: true)
        } catch {
            throw ProcessError.stagingFailed
        }
        defer { try? fileManager.removeItem(at: staging) }

        let layout: Layout
        if profile.id == UserProfileStore.defaultProfileID {
            layout = .defaultTwoRoot
            try await stageDefaultPayload(
                machineRoot: machineRoot, turbosparkHome: turbosparkHome, staging: staging,
                included: included)
        } else {
            layout = .profileFolder
            // Unreachable-nil is the same shape `userScopeSubdirectory` leans
            // on: storeDirectory only answers nil for the Default id.
            let source = UserProfileStore.storeDirectory(
                profileID: profile.id, machineRoot: machineRoot)!
            var isDirectory: ObjCBool = false
            guard fileManager.fileExists(atPath: source.path, isDirectory: &isDirectory),
                isDirectory.boolValue
            else { throw ProcessError.sourceMissing }
            try await stageProfileFolder(source: source, staging: staging, included: included)
        }

        let contents = try relativeContents(of: staging)
        let manifest = Manifest(
            formatVersion: formatVersion,
            kind: manifestKind,
            profileID: profile.id,
            profileName: profile.name,
            profileCreatedAt: profile.createdAt,
            isDefault: profile.id == UserProfileStore.defaultProfileID,
            layout: layout,
            exportedAt: exportedAt,
            appVersion: appVersion,
            contents: contents,
            includedCategories: included.map { $0.sorted() })
        let manifestURL = staging.appendingPathComponent(manifestFileName)
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(manifest).write(to: manifestURL, options: .atomic)

        try await runDitto(
            arguments: ["-c", "-k", "--sequesterRsrc", staging.path, destination.path],
            step: "archive")

        // A zero-byte archive means ditto failed to walk the staging tree and
        // still exited 0, which would otherwise read as success until the
        // import that cannot open it.
        let attributes = try fileManager.attributesOfItem(atPath: destination.path)
        if let size = attributes[.size] as? UInt64, size == 0 {
            throw ProcessError.processFailed(step: "archive", exitCode: 0, stderr: "archive is empty")
        }
        return manifest
    }

    /// Copies one profile folder's contents into the staging root, skipping
    /// top-level entries owned by categories the user left out.
    private static func stageProfileFolder(
        source: URL, staging: URL, included: Set<String>?
    ) async throws {
        for name in try topLevelEntries(of: source)
        where shouldInclude(topLevelName: name, included: included) {
            try await runDitto(
                arguments: [
                    source.appendingPathComponent(name).path,
                    staging.appendingPathComponent(name).path,
                ],
                step: "stage")
        }
    }

    /// Copies the Default user's two roots into `app-support/` and
    /// `dot-turbospark/`, skipping the excluded top-level entries per root so
    /// `models/` is never copied, and skipping entries owned by categories
    /// the user left out.
    private static func stageDefaultPayload(
        machineRoot: URL, turbosparkHome: URL, staging: URL, included: Set<String>?
    ) async throws {
        let fileManager = FileManager.default
        let roots: [(root: URL, name: String)] = [
            (machineRoot, appSupportDirectoryName),
            (turbosparkHome, dotTurbosparkDirectoryName),
        ]
        for (root, name) in roots {
            var isDirectory: ObjCBool = false
            // A machine without ~/.turbospark has an empty Default user-scope;
            // an absent root ships as an absent side of the layout rather
            // than failing the export.
            guard FileManager.default.fileExists(atPath: root.path, isDirectory: &isDirectory),
                isDirectory.boolValue
            else { continue }
            let target = staging.appendingPathComponent(name, isDirectory: true)
            do {
                try fileManager.createDirectory(at: target, withIntermediateDirectories: true)
            } catch {
                throw ProcessError.stagingFailed
            }
            for entry in try topLevelEntries(of: root)
                .filter({ !defaultExcludedTopLevelNames.contains($0) })
                .filter({ shouldInclude(topLevelName: $0, included: included) }) {
                try await runDitto(
                    arguments: [
                        root.appendingPathComponent(entry).path,
                        target.appendingPathComponent(entry).path,
                    ],
                    step: "stage")
            }
        }
    }

    /// Every file and directory under `root`, as `/`-separated paths relative
    /// to it, sorted. This is the manifest's contents list: what the archive
    /// actually carries, not what the copy intended to carry. Enumerates by
    /// PATH, whose entries are already relative: the URL enumerator's
    /// `relativePath` is unreliable across the /var vs /private/var symlink,
    /// where the base URL is unresolved and the items are resolved, and it
    /// silently degrades to absolute paths.
    static func relativeContents(of root: URL) throws -> [String] {
        let base = root.resolvingSymlinksInPath().path
        guard let enumerator = FileManager.default.enumerator(atPath: base) else {
            return []
        }
        return enumerator.compactMap { $0 as? String }.map { $0 as String }.sorted()
    }

    static func topLevelEntries(of directory: URL) throws -> [String] {
        try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0 != ".DS_Store" }
    }

    /// Runs `ditto` once. Internal because the importer reuses it for
    /// extraction and payload copies.
    static func runDitto(arguments: [String], step: String) async throws {
        do {
            let output = try await ProcessExecutor.run(
                executableURL: dittoURL,
                arguments: arguments,
                timeoutSeconds: 3_600)
            guard output.exitCode == 0, !output.timedOut else {
                throw ProcessError.processFailed(
                    step: step, exitCode: output.exitCode, stderr: output.stderr)
            }
        } catch let error as ProcessError {
            throw error
        } catch {
            // ProcessExecutor itself throws when the process could not so
            // much as spawn; surface that under the same shape.
            throw ProcessError.processFailed(step: step, exitCode: -1, stderr: error.localizedDescription)
        }
    }
}
