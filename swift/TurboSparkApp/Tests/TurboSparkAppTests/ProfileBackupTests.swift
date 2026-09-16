import Foundation
@testable import TurboSparkApp
import XCTest

/// Profile backup export and import, driven through real temp directories
/// and the real archive tools (`ditto`, `zipinfo`, `unzip`): an archive is
/// verified by EXTRACTING it and comparing bytes, never by trusting the
/// manifest that travelled inside it.
final class ProfileBackupTests: XCTestCase {
    private var root: URL {
        URL(fileURLWithPath: NSTemporaryDirectory(), isDirectory: true)
            .appendingPathComponent(
                "ProfileBackupTests-\(ProcessInfo.processInfo.processIdentifier)",
                isDirectory: true)
    }

    override func setUp() {
        try? FileManager.default.removeItem(at: root)
        try? FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        super.setUp()
    }

    override func tearDown() {
        try? FileManager.default.removeItem(at: root)
        super.tearDown()
    }

    // MARK: - Fixtures

    private func makeFile(_ path: String, contents: String, under base: URL) throws {
        let url = base.appendingPathComponent(path)
        try FileManager.default.createDirectory(
            at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try Data(contents.utf8).write(to: url)
    }

    private func makeProfileFolder(id: String, name: String) throws -> (UserProfile, URL) {
        let machineRoot = root.appendingPathComponent("machine", isDirectory: true)
        let folder = machineRoot.appendingPathComponent("profiles/\(id)", isDirectory: true)
        try makeFile("settings.json", contents: "{\"theme\":\"dark\"}", under: folder)
        try makeFile("chats_archive.json", contents: "[1,2]", under: folder)
        try makeFile("skills/notes/SKILL.md", contents: "# notes", under: folder)
        let profile = UserProfile(id: id, name: name)
        return (profile, folder)
    }

    private func extract(_ archive: URL, into destination: URL) async throws {
        try await ProfileBackup.runDitto(
            arguments: ["-x", "-k", archive.path, destination.path], step: "test-extract")
    }

    private func bytes(_ url: URL) throws -> Data {
        try XCTUnwrap(FileManager.default.contents(atPath: url.path))
    }

    private func manifest(fromExtraction extraction: URL) throws -> ProfileBackup.Manifest {
        let data = try bytes(extraction.appendingPathComponent(ProfileBackup.manifestFileName))
        return try JSONDecoder().decode(ProfileBackup.Manifest.self, from: data)
    }

    /// Asserts every payload FILE under `extraction` (everything except the
    /// manifest) has a byte-identical counterpart under `source`. Directories
    /// are checked for existence only: they carry no bytes of their own.
    private func assertExtractionMatches(
        source: URL, extraction: URL
    ) throws {
        for relative in try ProfileBackup.relativeContents(of: extraction)
        where relative != ProfileBackup.manifestFileName {
            let inSource = source.appendingPathComponent(relative)
            let inExtraction = extraction.appendingPathComponent(relative)
            var isDirectory: ObjCBool = false
            let sourceIsDirectory =
                FileManager.default.fileExists(atPath: inSource.path, isDirectory: &isDirectory)
                && isDirectory.boolValue
            if sourceIsDirectory {
                XCTAssertTrue(
                    FileManager.default.fileExists(atPath: inExtraction.path),
                    "directory \(relative) is missing from the extraction")
                continue
            }
            XCTAssertEqual(
                try bytes(inSource),
                try bytes(inExtraction),
                "payload differs at \(relative)")
        }
    }

    // MARK: - Export: a non-default profile

    func testExportOfAProfileFolderRoundTripsThroughItsArchive() async throws {
        let (profile, folder) = try makeProfileFolder(id: "p1", name: "Work")
        let destination = root.appendingPathComponent("work-backup.zip")

        let exported = try await ProfileBackup.export(
            profile: profile,
            machineRoot: root.appendingPathComponent("machine"),
            turbosparkHome: root.appendingPathComponent("home"),
            destination: destination,
            appVersion: "test-1.2.3")

        XCTAssertEqual(exported.kind, ProfileBackup.manifestKind)
        XCTAssertEqual(exported.formatVersion, ProfileBackup.formatVersion)
        XCTAssertEqual(exported.layout, .profileFolder)
        XCTAssertFalse(exported.isDefault)
        XCTAssertEqual(exported.profileID, "p1")
        XCTAssertEqual(exported.profileName, "Work")
        XCTAssertEqual(exported.appVersion, "test-1.2.3")
        XCTAssertTrue(exported.contents.contains("settings.json"))
        XCTAssertTrue(exported.contents.contains("skills/notes/SKILL.md"))

        let extraction = root.appendingPathComponent("extracted-work", isDirectory: true)
        try FileManager.default.createDirectory(at: extraction, withIntermediateDirectories: true)
        try await extract(destination, into: extraction)
        try assertExtractionMatches(source: folder, extraction: extraction)
        let roundTripped = try manifest(fromExtraction: extraction)
        XCTAssertEqual(roundTripped, exported)
    }

    // MARK: - Export: the Default user

    func testExportOfTheDefaultUserExcludesSharedAndRegistryEntries() async throws {
        let machineRoot = root.appendingPathComponent("machine", isDirectory: true)
        let home = root.appendingPathComponent("home/.turbospark", isDirectory: true)
        try makeFile("settings.json", contents: "{}", under: machineRoot)
        try makeFile("other.txt", contents: "keep me", under: machineRoot)
        try makeFile("profiles.json", contents: "[]", under: machineRoot)
        try makeFile("profiles/p1/x.json", contents: "another user", under: machineRoot)
        try makeFile("models/big.bin", contents: "gigabytes", under: machineRoot)
        try makeFile("installed.json", contents: "[]", under: machineRoot)
        try makeFile("skills/s/SKILL.md", contents: "# s", under: home)
        try makeFile("agents/a.md", contents: "agent", under: home)
        try makeFile("models/m.bin", contents: "also gigabytes", under: home)
        try makeFile("installed.json", contents: "[]", under: home)

        let destination = root.appendingPathComponent("default-backup.zip")
        let exported = try await ProfileBackup.export(
            profile: UserProfileStore.defaultProfile,
            machineRoot: machineRoot,
            turbosparkHome: home,
            destination: destination,
            appVersion: "test")

        XCTAssertEqual(exported.layout, .defaultTwoRoot)
        XCTAssertTrue(exported.isDefault)

        let extraction = root.appendingPathComponent("extracted-default", isDirectory: true)
        try FileManager.default.createDirectory(at: extraction, withIntermediateDirectories: true)
        try await extract(destination, into: extraction)

        let kept = [
            "app-support/settings.json", "app-support/other.txt",
            "dot-turbospark/skills/s/SKILL.md", "dot-turbospark/agents/a.md",
        ]
        for relative in kept {
            XCTAssertTrue(
                FileManager.default.fileExists(atPath: extraction.appendingPathComponent(relative).path),
                "expected \(relative) in the archive")
        }
        let dropped = [
            "app-support/profiles.json", "app-support/profiles", "app-support/models",
            "dot-turbospark/models", "dot-turbospark/installed.json",
        ]
        for relative in dropped {
            XCTAssertFalse(
                FileManager.default.fileExists(atPath: extraction.appendingPathComponent(relative).path),
                "\(relative) is shared or machine-level and must not travel")
        }
        for relative in kept {
            let sourceSide = relative.hasPrefix("app-support/")
                ? machineRoot.appendingPathComponent(String(relative.dropFirst("app-support/".count)))
                : home.appendingPathComponent(String(relative.dropFirst("dot-turbospark/".count)))
            XCTAssertEqual(
                try bytes(sourceSide), try bytes(extraction.appendingPathComponent(relative)),
                "payload differs at \(relative)")
        }
    }

    func testAnAbsentTurbosparkHomeExportsAsAnEmptySide() async throws {
        let machineRoot = root.appendingPathComponent("machine", isDirectory: true)
        try makeFile("settings.json", contents: "{}", under: machineRoot)
        let destination = root.appendingPathComponent("lean-default.zip")
        let exported = try await ProfileBackup.export(
            profile: UserProfileStore.defaultProfile,
            machineRoot: machineRoot,
            turbosparkHome: root.appendingPathComponent("home/.turbospark"),
            destination: destination,
            appVersion: "test")
        XCTAssertEqual(exported.layout, .defaultTwoRoot)
        let extraction = root.appendingPathComponent("extracted-lean", isDirectory: true)
        try FileManager.default.createDirectory(at: extraction, withIntermediateDirectories: true)
        try await extract(destination, into: extraction)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: extraction.appendingPathComponent("app-support/settings.json").path))
    }

    // MARK: - Name sanitization

    func testSanitizedFileNameStripsHostileCharactersButKeepsEmoji() {
        let cleaned = ProfileBackup.sanitizedFileName("Work / Project: A*B?", fallback: "X")
        XCTAssertFalse(cleaned.contains(where: { "/:\\?%*|\"<>".contains($0) }))
        XCTAssertEqual(cleaned, "Work - Project- A-B-")

        XCTAssertEqual(
            ProfileBackup.sanitizedFileName("🎉 Release 🚀", fallback: "X"),
            "🎉 Release 🚀",
            "emoji are legal in file names and are part of the name")
        XCTAssertEqual(
            ProfileBackup.sanitizedFileName("Bad\u{7}Name", fallback: "X"),
            "Bad-Name",
            "control characters display as nothing or worse and become a dash")
        XCTAssertEqual(
            ProfileBackup.sanitizedFileName("..hidden name..", fallback: "X"),
            "hidden name",
            "edge dots make hidden files or trailing clutter")
        XCTAssertEqual(ProfileBackup.sanitizedFileName("   ", fallback: "Profile"), "Profile")
        XCTAssertEqual(ProfileBackup.sanitizedFileName("", fallback: "Profile"), "Profile")
    }

    func testSanitizedFileNameTruncatesOnWholeGraphemes() {
        let name = String(repeating: "🚀", count: 80)
        let capped = ProfileBackup.sanitizedFileName(name, fallback: "X")
        XCTAssertEqual(capped.count, 60, "the cap is 60 characters")
        XCTAssertEqual(capped, String(repeating: "🚀", count: 60), "no emoji is torn in half")
    }

    // MARK: - Archive entry validation

    func testTheZipSlipValidatorRefusesEscapeEntries() {
        XCTAssertTrue(ProfileBackupImport.isSafeArchiveEntry("settings.json"))
        XCTAssertTrue(ProfileBackupImport.isSafeArchiveEntry("skills/notes/SKILL.md"))
        XCTAssertTrue(ProfileBackupImport.isSafeArchiveEntry("folder/"))
        XCTAssertTrue(ProfileBackupImport.isSafeArchiveEntry("file..txt"),
                      "two dots inside a NAME are fine; only a .. component escapes")
        XCTAssertFalse(ProfileBackupImport.isSafeArchiveEntry(""))
        XCTAssertFalse(ProfileBackupImport.isSafeArchiveEntry("../evil.txt"))
        XCTAssertFalse(ProfileBackupImport.isSafeArchiveEntry("a/../../b"))
        XCTAssertFalse(ProfileBackupImport.isSafeArchiveEntry("/etc/passwd"),
                       "absolute entries extract outside the destination")
        XCTAssertFalse(ProfileBackupImport.isSafeArchiveEntry("back\\slash.txt"),
                       "backslash separators are a Windows-crafted entry")
        XCTAssertFalse(ProfileBackupImport.isSafeArchiveEntry("bad\u{1}name"))
    }

    // MARK: - Import: refusal cases

    func testImportRefusesAnArchiveWithoutAManifest() async throws {
        let staging = root.appendingPathComponent("bare", isDirectory: true)
        try makeFile("stray.txt", contents: "no manifest here", under: staging)
        let archive = root.appendingPathComponent("bare.zip")
        try await ProfileBackup.runDitto(
            arguments: ["-c", "-k", staging.path, archive.path], step: "test-zip")

        do {
            _ = try await ProfileBackupImport.readSummary(archive: archive)
            XCTFail("a manifest-less archive must be refused")
        } catch let error as ProfileBackupImport.ImportError {
            XCTAssertEqual(error, .manifestMissing)
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    func testImportRefusesAnUnsupportedFormatVersion() async throws {
        let (profile, _) = try makeProfileFolder(id: "p1", name: "Work")
        let destination = root.appendingPathComponent("future.zip")
        var exported = try await ProfileBackup.export(
            profile: profile,
            machineRoot: root.appendingPathComponent("machine"),
            turbosparkHome: root.appendingPathComponent("home"),
            destination: destination,
            appVersion: "test")
        exported.formatVersion = ProfileBackup.formatVersion + 1

        // Rewrite the manifest inside a copy of the archive's staging: build
        // a new archive around a foreign-version manifest.
        let staging = root.appendingPathComponent("future-staging", isDirectory: true)
        try FileManager.default.createDirectory(at: staging, withIntermediateDirectories: true)
        try JSONEncoder().encode(exported).write(
            to: staging.appendingPathComponent(ProfileBackup.manifestFileName))
        let archive = root.appendingPathComponent("future-manifest.zip")
        try await ProfileBackup.runDitto(
            arguments: ["-c", "-k", staging.path, archive.path], step: "test-zip")

        do {
            _ = try await ProfileBackupImport.readSummary(archive: archive)
            XCTFail("a future formatVersion must be refused")
        } catch let error as ProfileBackupImport.ImportError {
            XCTAssertEqual(
                error,
                .unsupportedVersion(ProfileBackup.formatVersion + 1))
        } catch {
            XCTFail("unexpected error: \(error)")
        }
    }

    // MARK: - Import: restore

    func testImportRestoresAProfileBackupIntoAFreshFolder() async throws {
        let (profile, folder) = try makeProfileFolder(id: "p1", name: "Work")
        let archive = root.appendingPathComponent("work-backup.zip")
        let exported = try await ProfileBackup.export(
            profile: profile,
            machineRoot: root.appendingPathComponent("machine"),
            turbosparkHome: root.appendingPathComponent("home"),
            destination: archive,
            appVersion: "test")

        let destination = root.appendingPathComponent("restored/p9", isDirectory: true)
        let restored = try await ProfileBackupImport.install(
            archive: archive, destination: destination)
        XCTAssertEqual(restored, exported)
        try assertExtractionMatches(source: destination, extraction: folder)
        XCTAssertTrue(FileManager.default.fileExists(
            atPath: destination.appendingPathComponent("settings.json").path))
    }

    func testTheTwoRootLayoutMergesIntoOneFolderWithAppSupportWinning() async throws {
        // Both roots of a Default backup carry `tools/`; the merge copies
        // dot-turbospark first so the first-party stores win collisions.
        let staging = root.appendingPathComponent("merge-staging", isDirectory: true)
        try makeFile(ProfileBackup.manifestFileName, contents: "manifest", under: staging)
        try makeFile("app-support/settings.json", contents: "STORES", under: staging)
        try makeFile("app-support/tools/t.json", contents: "FROM STORES", under: staging)
        try makeFile("dot-turbospark/skills/s/SKILL.md", contents: "# s", under: staging)
        try makeFile("dot-turbospark/tools/t.json", contents: "FROM HOME", under: staging)
        let manifest = ProfileBackup.Manifest(
            formatVersion: ProfileBackup.formatVersion,
            kind: ProfileBackup.manifestKind,
            profileID: UserProfileStore.defaultProfileID,
            profileName: "Default",
            profileCreatedAt: Date(timeIntervalSince1970: 0),
            isDefault: true,
            layout: .defaultTwoRoot,
            exportedAt: Date(timeIntervalSince1970: 0),
            appVersion: "test",
            contents: [])
        try JSONEncoder().encode(manifest).write(
            to: staging.appendingPathComponent(ProfileBackup.manifestFileName))
        let archive = root.appendingPathComponent("merge.zip")
        try await ProfileBackup.runDitto(
            arguments: ["-c", "-k", staging.path, archive.path], step: "test-zip")

        let destination = root.appendingPathComponent("restored-merge", isDirectory: true)
        let restored = try await ProfileBackupImport.install(
            archive: archive, destination: destination)
        XCTAssertEqual(restored.layout, .defaultTwoRoot)
        XCTAssertEqual(
            try bytes(destination.appendingPathComponent("settings.json")), Data("STORES".utf8))
        XCTAssertEqual(
            try bytes(destination.appendingPathComponent("skills/s/SKILL.md")), Data("# s".utf8))
        XCTAssertEqual(
            try bytes(destination.appendingPathComponent("tools/t.json")),
            Data("FROM STORES".utf8),
            "app-support is copied second, so the first-party stores win file collisions")
    }
}
