import XCTest

@testable import TurboSparkApp

/// B1/G4: the two halves of a JSON store's failure handling, both of which
/// lost user data silently.
///
/// A swallowed WRITE (`try? data.write(...)`) discarded the whole archive on
/// quit with no sign. A swallowed DECODE was reported to stderr and returned
/// the empty default -- correct as far as it goes, except the unreadable file
/// is still on disk and the next mutation overwrites it whole and atomically,
/// so by the time anyone reads the log the original is gone.
final class StoreDurabilityTests: XCTestCase {
    private func makeTempDir() throws -> URL {
        let dir = FileManager.default.temporaryDirectory
            .appendingPathComponent(UUID().uuidString, isDirectory: true)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        return dir
    }

    private struct Fixture: Codable, Equatable {
        var value: String
    }

    // MARK: - G4: an unreadable file is preserved, not overwritten

    func testACorruptFileIsQuarantinedRatherThanLeftToBeOverwritten() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("archive.json")
        try "{ this is not json at all".write(to: url, atomically: true, encoding: .utf8)

        let loaded = AppJSONStore.load(Fixture.self, from: url, label: "test archive")
        XCTAssertNil(loaded, "An unreadable file must not decode.")

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: url.path),
            "The unreadable file must be moved aside so the next save cannot overwrite it.")

        let siblings = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertTrue(
            siblings.contains { $0.contains("corrupt-") },
            "It must still be on disk under a quarantine name. Found: \(siblings)")

        // And the store is usable again immediately: the next save writes a
        // fresh file beside the quarantined one rather than failing forever.
        XCTAssertTrue(AppJSONStore.save(Fixture(value: "new"), to: url, label: "test archive"))
        XCTAssertEqual(
            AppJSONStore.load(Fixture.self, from: url, label: "test archive"),
            Fixture(value: "new"))
    }

    func testAMissingFileIsNotTreatedAsCorrupt() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        let url = dir.appendingPathComponent("absent.json")

        XCTAssertNil(AppJSONStore.load(Fixture.self, from: url, label: "absent"))
        let siblings = try FileManager.default.contentsOfDirectory(atPath: dir.path)
        XCTAssertTrue(
            siblings.isEmpty, "A first run has nothing to quarantine. Found: \(siblings)")
    }

    // MARK: - B1: a failed write is reported, not swallowed

    func testAWriteThatCannotSucceedIsReportedRatherThanReturningQuietly() {
        AppJSONStore.clearLastWriteError()
        // A path under a file rather than a directory: the write cannot
        // succeed and no amount of retrying will change that.
        let impossible = URL(fileURLWithPath: "/dev/null/nested/archive.json")

        let ok = AppJSONStore.save(Fixture(value: "x"), to: impossible, label: "Test archive")

        XCTAssertFalse(ok, "The caller must be able to tell a failed write from a successful one.")
        XCTAssertNotNil(
            AppJSONStore.lastWriteError,
            "A silent failure here loses the whole session on quit with no sign.")
        XCTAssertTrue(AppJSONStore.lastWriteError?.contains("Test archive") ?? false)
        AppJSONStore.clearLastWriteError()
    }

    func testASuccessfulWriteRecordsNoError() throws {
        let dir = try makeTempDir()
        defer { try? FileManager.default.removeItem(at: dir) }
        AppJSONStore.clearLastWriteError()

        XCTAssertTrue(
            AppJSONStore.save(
                Fixture(value: "y"), to: dir.appendingPathComponent("ok.json"), label: "Test"))
        XCTAssertNil(AppJSONStore.lastWriteError)
    }

    // MARK: - the real chat archive goes through it

    func testTheChatArchiveQuarantinesRatherThanLosingConversations() throws {
        // `AppStorageRoot` redirects to a per-process test directory
        // (swift/CLAUDE.md Gotcha 37), so this cannot touch real user data.
        let url = AppStorageRoot.file("chats_archive.json")
        let saved = try? Data(contentsOf: url)
        defer {
            try? FileManager.default.removeItem(at: url)
            if let saved { try? saved.write(to: url) }
        }

        try #"{"chats": "this used to be an array"}"#.write(
            to: url, atomically: true, encoding: .utf8)
        _ = AppChatFileStore.load()

        XCTAssertFalse(
            FileManager.default.fileExists(atPath: url.path),
            "The archive that would not decode must be moved aside before anything can save over it.")
        let siblings = try FileManager.default.contentsOfDirectory(
            atPath: AppStorageRoot.directory.path)
        XCTAssertTrue(siblings.contains { $0.hasPrefix("chats_archive.corrupt-") })

        // Clean up the quarantined copy this test created.
        for name in siblings where name.hasPrefix("chats_archive.corrupt-") {
            try? FileManager.default.removeItem(
                at: AppStorageRoot.directory.appendingPathComponent(name))
        }
    }
}
