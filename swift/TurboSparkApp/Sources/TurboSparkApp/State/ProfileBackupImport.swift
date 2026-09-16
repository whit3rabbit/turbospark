import Foundation

/// The import half of profile backups: validate a `.zip` written by
/// `ProfileBackup` (or by hand to the same layout), then restore its payload
/// into a fresh profile folder. Deliberately the inverse of the export under
/// the same rules: the manifest is the contract, and anything that does not
/// meet it is refused BEFORE any real state is touched.
enum ProfileBackupImport {
    enum ImportError: Error, Equatable {
        /// Not a readable zip: `zipinfo` could not list it.
        case unreadableArchive
        /// The listing contains entries that would escape the extraction
        /// directory. Carries the offending paths.
        case unsafeEntries([String])
        /// The archive has no manifest where `ProfileBackup` writes one.
        case manifestMissing
        /// The manifest exists but does not decode, or names a different
        /// kind of backup.
        case manifestCorrupt
        /// A newer (or simply different) `formatVersion` than this build
        /// speaks.
        case unsupportedVersion(Int)
        /// A declared side of the layout is missing from the archive.
        case layoutCorrupt
    }

    /// Same fixed-path reasoning as `/usr/bin/ditto`.
    static let zipinfoURL = URL(fileURLWithPath: "/usr/bin/zipinfo")
    static let unzipURL = URL(fileURLWithPath: "/usr/bin/unzip")

    // MARK: - Archive entry validation

    /// Whether one archive entry is safe to extract into a chosen
    /// directory. Refuses absolute paths, `..` components (the zip-slip
    /// write-outside trick), backslash separators (entries crafted on
    /// Windows, which this tool would treat as one plain name but a
    /// hand-unzip may not), and control characters. A name that merely
    /// CONTAINS two dots, like `file..txt`, is fine; only a `..` path
    /// component escapes.
    ///
    /// Pure and directly unit-tested: the tests feed hostile strings here
    /// rather than crafting real malicious archives.
    static func isSafeArchiveEntry(_ entry: String) -> Bool {
        guard !entry.isEmpty, !entry.hasPrefix("/") else { return false }
        guard !entry.contains("\\") else { return false }
        guard !entry.split(separator: "/", omittingEmptySubsequences: false).contains("..")
        else { return false }
        let scalars = entry.unicodeScalars
        return !scalars.contains(where: { $0.value < 32 || $0.value == 127 })
    }

    static func validateArchiveEntries(_ entries: [String]) throws {
        let unsafe = entries.filter { !isSafeArchiveEntry($0) }
        guard unsafe.isEmpty else { throw ImportError.unsafeEntries(unsafe) }
    }

    // MARK: - Reading the summary (no extraction)

    /// Reads and validates the manifest WITHOUT extracting, so the import
    /// sheet can describe what it is about to restore and a hostile or
    /// foreign archive is refused before anything is offered.
    static func readSummary(archive: URL) async throws -> ProfileBackup.Manifest {
        try validateArchiveEntries(try await listEntries(archive: archive))
        return try ProfileBackup.validatedManifest(from: try await readManifestData(archive: archive))
    }

    private static func listEntries(archive: URL) async throws -> [String] {
        do {
            let output = try await ProcessExecutor.run(
                executableURL: zipinfoURL,
                arguments: ["-1", archive.path],
                timeoutSeconds: 120)
            guard output.exitCode == 0, !output.timedOut else {
                throw ImportError.unreadableArchive
            }
            return output.stdout.split(separator: "\n", omittingEmptySubsequences: true)
                .map(String.init)
        } catch let error as ImportError {
            throw error
        } catch {
            throw ImportError.unreadableArchive
        }
    }

    private static func readManifestData(archive: URL) async throws -> Data {
        do {
            // `unzip -p` writes one entry to stdout without extracting; ditto
            // has no single-entry read.
            let output = try await ProcessExecutor.run(
                executableURL: unzipURL,
                arguments: ["-p", archive.path, ProfileBackup.manifestFileName],
                timeoutSeconds: 120)
            guard output.exitCode == 0, !output.timedOut else {
                throw ImportError.manifestMissing
            }
            return Data(output.stdout.utf8)
        } catch let error as ImportError {
            throw error
        } catch {
            throw ImportError.manifestMissing
        }
    }

    // MARK: - Restore

    /// Extracts the archive and copies its payload into `destination` (a
    /// fresh `profiles/<id>/` folder; created here). Returns the manifest
    /// that governed the restore. The folder-first, registry-last ordering
    /// lives in the AppModel wrapper, mirroring delete's trash-first rule.
    ///
    /// Merge precedence for the two-root layout: `dot-turbospark/` is copied
    /// first and `app-support/` second, so the first-party stores win any
    /// file-level collision (`tools/` is the one directory both roots carry
    /// today).
    @discardableResult
    static func install(archive: URL, destination: URL) async throws -> ProfileBackup.Manifest {
        let fileManager = FileManager.default
        try validateArchiveEntries(try await listEntries(archive: archive))

        let extraction = fileManager.temporaryDirectory
            .appendingPathComponent("turbospark-import-\(UUID().uuidString)", isDirectory: true)
        try fileManager.createDirectory(at: extraction, withIntermediateDirectories: true)
        defer { try? fileManager.removeItem(at: extraction) }

        try await ProfileBackup.runDitto(
            arguments: ["-x", "-k", archive.path, extraction.path], step: "extract")

        let manifest = try ProfileBackup.validatedManifest(
            from: try manifestData(fromExtraction: extraction))
        try fileManager.createDirectory(at: destination, withIntermediateDirectories: true)

        switch manifest.layout {
        case .profileFolder:
            for entry in try ProfileBackup.topLevelEntries(of: extraction)
                where entry != ProfileBackup.manifestFileName && entry != "__MACOSX" {
                try await ProfileBackup.runDitto(
                    arguments: [
                        extraction.appendingPathComponent(entry).path,
                        destination.appendingPathComponent(entry).path,
                    ],
                    step: "restore")
            }
        case .defaultTwoRoot:
            for name in [ProfileBackup.dotTurbosparkDirectoryName,
                ProfileBackup.appSupportDirectoryName] {
                let source = extraction.appendingPathComponent(name, isDirectory: true)
                var isDirectory: ObjCBool = false
                guard fileManager.fileExists(atPath: source.path, isDirectory: &isDirectory),
                    isDirectory.boolValue
                else { throw ImportError.layoutCorrupt }
                for entry in try ProfileBackup.topLevelEntries(of: source) {
                    try await ProfileBackup.runDitto(
                        arguments: [
                            source.appendingPathComponent(entry).path,
                            destination.appendingPathComponent(entry).path,
                        ],
                        step: "restore")
                }
            }
        }
        return manifest
    }

    private static func manifestData(fromExtraction extraction: URL) throws -> Data {
        let url = extraction.appendingPathComponent(ProfileBackup.manifestFileName)
        guard let data = FileManager.default.contents(atPath: url.path) else {
            throw ImportError.manifestMissing
        }
        return data
    }
}
